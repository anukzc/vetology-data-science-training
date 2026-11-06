# Radiology LLM Label Evaluation

This tool runs a local Ollama model (Qwen) over radiologist reports. The model labels each condition as Positive or Negative, and the tool compares those labels with the gold labels to build a confusion matrix in Excel.

## Usage

```bash
pip install pandas openpyxl requests
ollama pull qwen3:4b

python radiology_eval.py <gold_labels.xlsx> <canine_thorax|canine_abdomen|feline_thorax>
```

Outputs (written to the directory you run the command from; existing files are overwritten):
- `<type>_qwen_labels.xlsx`: the LLM's labels, with a CaseID column and one column per condition.
- `<type>_confusion_matrix.xlsx`: TP / FN / TN / FP, sensitivity, specificity and the check columns. The Radiologist Agreement Rate column is left empty.
- `<type>_confusion_matrix.png`: a single 2×2 confusion matrix that adds up TP / FN / FP / TN across all conditions. Gold labels are the rows and LLM labels are the columns.

Useful options: `--model`, `--think`, `--workers N`, `--limit N` (test on a few cases first), `--sheet`, `--prompts`.

## Input file

- Report text comes from the columns whose headers contain `Findings`, `Conclusions` and `Recommendations`. Recommendations is optional.
- Only the first 100 data rows (the manually labeled set) are read and sent to the LLM. Change this with `--rows N`.
- Condition columns are detected automatically: any other column that contains Positive/Negative values. The header text is used as the condition name.
- Only `Positive` / `Negative` cells count as gold labels. Any other value (blank, Normal, Abnormal) skips that case for that condition only, which lowers the `Check` count.

## Prompts

`prompts.json` maps each radiology type to its instruction prompt, which is used as the system prompt. The report text is sent as the user message. The gold file's exact column names are added after your prompt as the required JSON keys, so the model knows that "liver mass" means `liver_mass`. The model's output is also constrained to a JSON schema with exactly those keys and `Positive`/`Negative` values.

An entry can instead be an object `{"system": ..., "user": ..., "condition_definitions": {...}}`. In that format, `{conditions}` and `{report}` are placeholders filled in at runtime.
