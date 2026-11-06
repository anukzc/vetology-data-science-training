#!/usr/bin/env python3
"""Compare LLM-predicted disease labels against radiologist gold labels.

Usage:
    python radiology_eval.py <gold_labels.xlsx> <canine_thorax|canine_abdomen|feline_thorax> [options]

Pipeline:
    1. Read the gold-label Excel file (report text columns + one Positive/Negative column per condition).
    2. For each case, ask a local Ollama model (Qwen by default) to label every condition. The model is
       constrained to return JSON: {"<condition>": "Positive" | "Negative", ...}.
    3. Save the raw predictions to a JSON file (also used as a resume cache).
    4. Build a confusion-matrix Excel file (TP / FN / TN / FP, sensitivity, specificity, checks).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import requests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side

RADIOLOGY_TYPES = ("canine_thorax", "canine_abdomen", "feline_thorax")
LABELS = ("Positive", "Negative")

# Only explicit Positive/Negative cells are gold labels. Anything else (e.g. Normal/Abnormal on
# rows that were not manually labeled) is treated as unlabeled and ignored.
_POSITIVE = {"positive"}
_NEGATIVE = {"negative"}

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------------------
# Reading the gold-label file
# --------------------------------------------------------------------------------------------------

def normalize_label(value) -> str | None:
    """Map a cell value to 'Positive' / 'Negative', or None if blank/unrecognised."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if s in _POSITIVE:
        return "Positive"
    if s in _NEGATIVE:
        return "Negative"
    return None


def _find_column(columns, *keywords) -> str | None:
    """Find a column whose header contains a keyword, preferring the original radiologist report
    over AI-report columns (e.g. 'Findings (original radiologist report)' vs 'Findings (AI report)')."""
    matches = [c for c in columns if any(k in re.sub(r"[^a-z]", "", str(c).lower()) for k in keywords)]
    matches = [c for c in matches if "radiologist report" not in str(c).lower()]
    preferred = [c for c in matches if "ai report" in str(c).lower()]
    return (preferred or matches or [None])[0]


def load_gold_file(path: Path, sheet, max_rows: int | None) -> tuple[pd.DataFrame, str | None, list[str], list[str]]:
    """Return (dataframe, case_id_col, report_cols, condition_cols).

    Only the first `max_rows` data rows are read (the manually labeled set); the rest are ignored."""
    df = pd.read_excel(path, sheet_name=sheet, dtype=object, nrows=max_rows)
    df.columns = [str(c).strip() for c in df.columns]

    case_col = _find_column(df.columns, "caseid")
    report_cols = [c for c in (
        _find_column(df.columns, "finding"),
        _find_column(df.columns, "conclusion"),
        _find_column(df.columns, "recommendation"),
    ) if c is not None]
    if not report_cols:
        sys.exit("ERROR: could not find report text columns (expected headers containing "
                 "'Findings', 'Conclusions' and/or 'Recommendations').")

    # Condition columns: every other column that contains at least one Positive/Negative label.
    condition_cols = [
        col for col in df.columns
        if col not in report_cols and col != case_col
        and df[col].map(normalize_label).notna().any()
    ]
    if not condition_cols:
        sys.exit("ERROR: no condition columns with Positive/Negative values were found.")

    # Keep only cases that have at least one gold label; unlabeled rows are ignored entirely.
    labeled = df[condition_cols].apply(lambda col: col.map(normalize_label)).notna().any(axis=1)
    if not labeled.all():
        print(f"WARNING: {int((~labeled).sum())} of the first {len(df)} rows have no Positive/Negative "
              f"gold labels and will be skipped.")
    df = df[labeled]

    return df, case_col, report_cols, condition_cols


def build_report_text(row, report_cols) -> str:
    parts = []
    for col in report_cols:
        val = row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)) or not str(val).strip():
            continue
        text = str(val).replace("_x000D_", "").strip()  # Excel-escaped carriage returns
        parts.append(f"## {col}\n{text}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------------------
# Prompts + LLM call (Ollama)
# --------------------------------------------------------------------------------------------------

def load_prompt(prompts_path: Path, radiology_type: str, conditions: list[str]) -> tuple[str, str]:
    with open(prompts_path, encoding="utf-8") as f:
        prompts = json.load(f)
    if radiology_type not in prompts:
        sys.exit(f"ERROR: '{radiology_type}' not found in {prompts_path}. "
                 f"Available: {', '.join(prompts)}")
    cfg = prompts[radiology_type]
    # Simple format: the value is just the instruction string.
    if isinstance(cfg, str):
        cfg = {"system": cfg}
    definitions = cfg.get("condition_definitions", {}) or {}
    condition_lines = "\n".join(
        f"- {c}: {definitions[c]}" if definitions.get(c) else f"- {c}" for c in conditions
    )
    # str.replace (not str.format) so braces inside prompts/reports never break templating.
    system = cfg["system"]
    if "{conditions}" in system:
        system = system.replace("{conditions}", condition_lines)
    else:
        # Give the model the exact JSON keys (gold-file column names) it must use.
        system += f"\n\nUse exactly these JSON keys, one per condition:\n{condition_lines}"
    return system, cfg.get("user", "Radiology report:\n\n{report}")


def build_schema(conditions: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {c: {"type": "string", "enum": list(LABELS)} for c in conditions},
        "required": list(conditions),
        "additionalProperties": False,
    }


class OllamaClient:
    def __init__(self, host: str, model: str, think: bool, num_ctx: int, timeout: int, retries: int):
        self.url = host.rstrip("/") + "/api/chat"
        self.model = model
        self.think = think
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def check(self):
        base = self.url.rsplit("/api/", 1)[0]
        try:
            tags = self.session.get(base + "/api/tags", timeout=10).json()
        except requests.RequestException as e:
            sys.exit(f"ERROR: cannot reach Ollama at {base} ({e}). Is `ollama serve` running?")
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            sys.exit(f"ERROR: model '{self.model}' not found in Ollama. Installed: "
                     f"{', '.join(sorted(n for n in names if n))}. Run `ollama pull {self.model}`.")

    def predict(self, system: str, user: str, conditions: list[str], schema: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,          # Ollama structured outputs: constrains the reply to this JSON schema
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        last_err = None
        for _ in range(self.retries + 1):
            try:
                resp = self.session.post(self.url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                data = json.loads(content)
                result = {}
                for c in conditions:
                    label = normalize_label(data.get(c))
                    if label is None:
                        raise ValueError(f"missing/invalid label for '{c}': {data.get(c)!r}")
                    result[c] = label
                return result
            except (requests.RequestException, KeyError, ValueError) as e:  # JSONDecodeError is a ValueError
                last_err = e
        raise RuntimeError(f"Ollama call failed after {self.retries + 1} attempts: {last_err}")


# --------------------------------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------------------------------

def compute_confusion(df, keys, conditions, predictions) -> list[dict]:
    rows = []
    for cond in conditions:
        tp = fn = tn = fp = 0
        for idx, key in zip(df.index, keys):
            gold = normalize_label(df.at[idx, cond])
            pred = predictions.get(key, {}).get(cond)
            if gold is None or pred is None:
                continue  # unlabeled gold or failed LLM case -> excluded (shows up in the Check column)
            if gold == "Positive":
                tp += pred == "Positive"
                fn += pred == "Negative"
            else:
                tn += pred == "Negative"
                fp += pred == "Positive"
        rows.append({"condition": cond, "tp": tp, "fn": fn, "tn": tn, "fp": fp})
    return rows


def write_labels_excel(df, case_col, keys, conditions, predictions, out_path: Path):
    """One row per case (gold-file order): CaseID + the LLM's label for each condition.
    Cases the LLM failed on are left blank."""
    records = []
    for idx, key in zip(df.index, keys):
        pred = predictions.get(key, {})
        record = {"CaseID": df.at[idx, case_col] if case_col else idx + 1}
        record.update({c: pred.get(c) for c in conditions})
        records.append(record)
    pd.DataFrame(records, columns=["CaseID", *conditions]).to_excel(out_path, index=False, sheet_name="Results")


# Single-hue sequential blue ramp (light = near 0%, dark = near 100%).
_BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
_INK, _INK_MUTED = "#1f1f1f", "#5f5f5f"


def write_confusion_png(rows: list[dict], out_path: Path, title: str):
    """A single 2x2 confusion matrix pooled over all conditions.
    Rows = gold label, columns = LLM label; cells show counts. Shade = the cell's share of its
    gold-label row, so the rarer Positive row isn't washed out by the large Negative row."""
    tp, fn, tn, fp = (sum(r[k] for r in rows) for k in ("tp", "fn", "tn", "fp"))
    matrix = [[tp, fn], [fp, tn]]
    shares = [[v / sum(row) if sum(row) else 0 for v in row] for row in matrix]

    fig, ax = plt.subplots(figsize=(5, 4.6))
    fig.patch.set_facecolor("white")
    ax.imshow(shares, cmap=_BLUES, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if shares[i][j] > 0.55 else _INK
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    fontsize=22, fontweight="bold", color=color)
    ax.axhline(0.5, color="white", linewidth=3)  # gaps between cells
    ax.axvline(0.5, color="white", linewidth=3)
    ax.set_xticks([0, 1], ["Positive", "Negative"], fontsize=11, color=_INK_MUTED)
    ax.set_yticks([0, 1], ["Positive", "Negative"], fontsize=11, color=_INK_MUTED, rotation=90, va="center")
    ax.set_xlabel("LLM label", fontsize=11, color=_INK_MUTED)
    ax.set_ylabel("Gold label", fontsize=11, color=_INK_MUTED)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold", color=_INK, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_confusion_excel(rows: list[dict], out_path: Path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Confusion Matrix"

    headers = {
        "A": "condition", "B": "True Positive", "C": "False Negative", "D": "True Negative",
        "E": "False Positive", "F": "Sensitivity", "G": "Specificity", "H": "Check",
        "I": "Positive Ground Truth", "J": "Negative Ground Truth", "K": "Ground Truth Check",
        "M": "Radiologist Agreement Rate",
    }
    widths = {"A": 37, "B": 12, "C": 12, "D": 12, "E": 12, "F": 11.5, "G": 11.5, "H": 9.5,
              "I": 15, "J": 15, "K": 15, "L": 3, "M": 16.5}

    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for col, width in widths.items():
        ws.column_dimensions[col].width = width

    for col, text in headers.items():
        cell = ws[f"{col}1"]
        cell.value = text
        cell.font = bold
        cell.alignment = header_align
        cell.border = border
    ws.row_dimensions[1].height = 45

    for i, row in enumerate(rows, start=2):
        values = {
            "A": row["condition"],
            "B": row["tp"], "C": row["fn"], "D": row["tn"], "E": row["fp"],
            "F": f'=IFERROR(B{i}/(B{i}+C{i}),"N/A")',
            "G": f'=IFERROR(D{i}/(D{i}+E{i}),"N/A")',
            "H": f"=SUM(B{i}:E{i})",
            "I": f"=B{i}+C{i}",
            "J": f"=D{i}+E{i}",
            "K": f"=I{i}+J{i}",
            "M": None,  # Radiologist Agreement Rate intentionally left empty
        }
        for col, val in values.items():
            cell = ws[f"{col}{i}"]
            cell.value = val
            cell.border = border
            if col == "A":
                cell.alignment = Alignment(horizontal="left", vertical="center")
            elif col == "M":
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = "0%"
            else:
                cell.alignment = center
            if col in ("F", "G"):
                cell.number_format = "0.00%"

    ws.freeze_panes = "B2"
    wb.save(out_path)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", type=Path, help="Gold-label Excel file (.xlsx)")
    parser.add_argument("radiology_type", choices=RADIOLOGY_TYPES, help="Type of radiology report")
    parser.add_argument("--prompts", type=Path, default=SCRIPT_DIR / "prompts.json", help="Prompts JSON file")
    parser.add_argument("--sheet", default=0, help="Sheet name or index in the input file (default: first)")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name (default: qwen3:4b)")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--think", action="store_true", help="Enable model thinking (slower, may be more accurate)")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window (default: 8192)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel requests to Ollama (default: 1)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per case on bad output/errors")
    parser.add_argument("--rows", type=int, default=100,
                        help="Only the first N data rows are gold-labeled and sent to the LLM (default: 100)")
    parser.add_argument("--limit", type=int, help="Only process the first N cases (for quick testing)")
    args = parser.parse_args()

    if not args.input_file.exists():
        sys.exit(f"ERROR: input file not found: {args.input_file}")
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    # Outputs go to the directory the command is run from.
    out_path = Path.cwd() / f"{args.radiology_type}_confusion_matrix.xlsx"
    labels_path = Path.cwd() / f"{args.radiology_type}_qwen_labels.xlsx"
    png_path = Path.cwd() / f"{args.radiology_type}_confusion_matrix.png"

    df, case_col, report_cols, conditions = load_gold_file(args.input_file, sheet, args.rows)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} cases from {args.input_file.name}")
    print(f"Report columns: {report_cols}")
    print(f"Detected {len(conditions)} conditions: {', '.join(conditions)}")

    # Key each case by its Excel row number + case ID so duplicate IDs never collide.
    keys = [f"row{idx + 2}" + (f"|{df.at[idx, case_col]}" if case_col else "") for idx in df.index]

    system_prompt, user_template = load_prompt(args.prompts, args.radiology_type, conditions)
    schema = build_schema(conditions)

    predictions: dict[str, dict] = {}  # LLM JSON output per case: {condition: "Positive"/"Negative"}

    todo = []
    for idx, key in zip(df.index, keys):
        report = build_report_text(df.loc[idx], report_cols)
        if not report:
            print(f"  skipping {key}: empty report")
            continue
        todo.append((key, user_template.replace("{report}", report)))

    failures = []
    if todo:
        client = OllamaClient(args.host, args.model, args.think, args.num_ctx, args.timeout, args.retries)
        client.check()
        print(f"Calling {args.model} for {len(todo)} cases ({args.workers} worker(s))...")
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(client.predict, system_prompt, user, conditions, schema): key
                       for key, user in todo}
            for fut in as_completed(futures):
                key = futures[fut]
                done += 1
                try:
                    predictions[key] = fut.result()
                except RuntimeError as e:
                    failures.append(key)
                    print(f"  [{done}/{len(todo)}] FAILED {key}: {e}")
                    continue
                print(f"  [{done}/{len(todo)}] {key}")

    rows = compute_confusion(df, keys, conditions, predictions)
    write_confusion_excel(rows, out_path)
    write_confusion_png(rows, png_path, args.radiology_type.replace("_", " ").title())
    write_labels_excel(df, case_col, keys, conditions, predictions, labels_path)

    print(f"\nLLM labels:        {labels_path}")
    print(f"Confusion matrix:  {out_path}")
    print(f"Confusion image:   {png_path}")
    if failures:
        print(f"WARNING: {len(failures)} case(s) failed; their labels are blank and they are excluded "
              f"from the confusion matrix.")
    print(f"\n{'condition':40} {'TP':>5} {'FN':>5} {'TN':>5} {'FP':>5} {'Sens':>8} {'Spec':>8}")
    for r in rows:
        sens = f"{r['tp'] / (r['tp'] + r['fn']):.2%}" if r["tp"] + r["fn"] else "N/A"
        spec = f"{r['tn'] / (r['tn'] + r['fp']):.2%}" if r["tn"] + r["fp"] else "N/A"
        print(f"{r['condition']:40} {r['tp']:>5} {r['fn']:>5} {r['tn']:>5} {r['fp']:>5} {sens:>8} {spec:>8}")


if __name__ == "__main__":
    main()
