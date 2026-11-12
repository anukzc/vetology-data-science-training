# vetology-data-science-training

A project for the Vetology data science course.

`app.py` uses a local LLM (through [Ollama](https://ollama.com)) to label veterinary radiology reports as **Positive** or **Negative** for each condition. It then compares those labels against a gold-standard spreadsheet and writes confusion matrices Excel table.

Supported data types: `canine_thorax`, `canine_abdomen`, `feline_thorax`.

## How it works

1. Reads the gold Excel file once. Columns from index 8 onward are treated as condition labels.
2. Sends each radiologist report (Findings, Conclusions, Recommendations) to the LLM with the matching prompt from `prompts.json`. A JSON schema forces exactly one `Positive`/`Negative` answer per condition.
3. **Matrix 2:** compares the LLM's labels on the radiologist reports against the gold labels.
4. **Matrix 1** (optional, `--both_matrices true`): labels the AI-generated reports the same way, then compares them against the LLM's labels on the radiologist reports. This measures how well the AI report agrees with the radiologist report.

For each condition the Excel table reports TP, FP, TN, FN, sensitivity, specificity and row-count checks. 

## Setup

1. Install Python 3.10+ and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate   # Windows (Git Bash)
   # source .venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   ```
2. Install [Ollama](https://ollama.com/download) and pull the model:
   ```bash
   ollama pull qwen3:4b
   ```
3. Add the data files:
   - A `data/` folder holding the gold files, e.g. `data/canine_thorax_GOLD.xlsx`. Output files are written here too.

## Gold file format

| Column index | Contents |
|---|---|
| 0 | CaseID |
| 1–3 | Radiologist report: Findings, Conclusions, Recommendations |
| 4 | Original radiologist |
| 5–7 | AI report: Findings, Conclusions, Recommendations |
| 8+ | One column per condition, values `Positive` or `Negative` |

Only the first 100 rows are used, which is why `--rows` is capped at 100.

## Usage

Run from the repo root (paths to `data/` and `prompts.json` are relative):

```bash
python app.py --gold data/canine_thorax_GOLD.xlsx --data_type canine_thorax
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--gold` | yes | | Path to the gold labels Excel file |
| `--data_type` | yes | | `canine_thorax`, `canine_abdomen` or `feline_thorax` |
| `--model` | no | `qwen3:4b` | Any Ollama model name |
| `--rows` | no | `100` | Number of reports to label (1–100) |
| `--both_matrices` | no | `false` | `true` also builds matrix 1 (AI reports vs. radiologist reports) |

Each report is one LLM call, so a full 100-row run takes a while, and `--both_matrices true` doubles it. Try `--rows 5` first to check your setup.

## Outputs

All written to `data/`, where `<type>` is the `--data_type`:

| File | Contents |
|---|---|
| `<type>_llm_labels_on_radiologist_reports.xlsx` | LLM labels for each radiologist report |
| `<type>_llm_labels_on_AI_reports.xlsx` | LLM labels for each AI report (matrix 1 only) |
| `<type>_confusion_matrix_2.xlsx` / `.png` | Matrix 2: LLM on radiologist reports vs. gold |
| `<type>_confusion_matrix_1.xlsx` / `.png` | Matrix 1: AI reports vs. radiologist reports |

Re-running overwrites these files.
