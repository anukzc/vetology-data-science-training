# For Vetology Data Science Training
# Date: 10/12/2026

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import ollama
import dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, Side
import json
import matplotlib
matplotlib.use('Agg')

dotenv.load_dotenv()

RADIOLOGY_TYPES = ("canine_thorax", "canine_abdomen", "feline_thorax")


def get_conditions(df):
    conditions = df.iloc[:0, 8:]
    return conditions


def format_confusion_matrix(conditions, data_type, matrix_type):
    # Creates an empty confusion matrix matchin the example
    cols = ['True Positive', 'False Positive', 'True Negative',
            'False Negative', 'Sensitivity', 'Specificity', 'Check',
            'Positive Ground Truth', 'Negative Ground Truth',
            'Ground Truth Check', '', 'Radiologist Agreement Rate']
    # Create the confusion matrix with the specified columns and index
    # (rows) as the conditions
    confusion_matrix = pd.DataFrame(columns=cols, index=conditions.columns)
    confusion_matrix.index.name = 'condition'
    # print(confusion_matrix)

    path = f"./data/{data_type}_confusion_matrix_{matrix_type}.xlsx"
    confusion_matrix.to_excel(path, index=True)

    # Format the excel file
    wb = load_workbook(path)
    ws = wb.active
    bold = Font(bold=True)
    thin = Side(style='thin', color='000000')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Border every cell in the table (header row + index column + body)
    skip_cols = [None, 'Radiologist Agreement Rate']
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row,
                            min_col=1, max_col=ws.max_column):
        for cell in row:
            if ws.cell(row=1, column=cell.column).value not in skip_cols:
                cell.border = border

    # Bold the header row
    for cell in ws[1]:
        cell.font = bold

    # Center the column headers
    center = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for cell in ws[1]:
        cell.alignment = center

    # Auto adjust column width
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = max(
            len(str(c.value or '')) for c in col) + 1

    wb.save(path)
    return path


def fill_confusion_matrix(gold_df, pred_df, cm_path,
                          conditions, data_type, matrix_type):
    # Load the confusion matrix template
    wb = load_workbook(cm_path)
    ws = wb.active
    for condition in conditions.columns:
        y_true = gold_df[condition]
        y_pred = pred_df[condition]

        # TP FN
        # FP TN
        cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"])

        tp, fn, fp, tn = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None

        # Fill in the confusion matrix
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=2, value=tp)  # True Positive
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=3, value=fp)  # False Positive
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=4, value=tn)  # True Negative
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=5, value=fn)  # False Negative
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=6, value=sensitivity)  # Sensitivity
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=7, value=specificity)  # Specificity
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=8, value=len(y_true))  # Check
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=9, value=tp + fn)  # Positive Ground Truth
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=10, value=tn + fp)  # Negative Ground Truth
        ws.cell(row=conditions.columns.get_loc(condition) + 2,
                column=11, value=tp + fn + tn + fp)  # Ground Truth Check
    wb.save(cm_path)

    # One overall confusion matrix across all conditions
    all_true = gold_df[conditions.columns].values.ravel()
    all_pred = pred_df[conditions.columns].values.ravel()
    overall_cm = confusion_matrix(all_true, all_pred,
                                  labels=["Positive", "Negative"])

    disp = ConfusionMatrixDisplay(overall_cm,
                                  display_labels=["Positive", "Negative"])
    disp.plot(cmap="Blues", colorbar=False)
    disp.ax_.set_title(f"{data_type}: all conditions")
    plt.savefig(f"./data/{data_type}_confusion_matrix_"
                f"{matrix_type}.png", bbox_inches="tight")
    plt.close()


def save_to_excel(df, data_type, matrix_type):
    if matrix_type == 2:
        report_type = "radiologist"
    else:
        report_type = "AI"
    output_filename = (f"./data/{data_type}_llm_labels_"
                       f"on_{report_type}_reports.xlsx")
    df.to_excel(output_filename, index=False)
    return output_filename


def get_prompt(data_type):
    # Load the appropriate prompt based on the type
    with open("prompts.json", "r") as prompts:
        prompt = json.load(prompts)[data_type]
    return prompt


def build_schema(condition_names):
    # Forces the llm to return exactly one Positive/Negative per condition
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "enum": ["Positive", "Negative"]}
            for name in condition_names
        },
        "required": list(condition_names),
    }


def call_llm(data_type, model, gold_df, condition_names, matrix_type):
    # Grabs radiologist reports and appropriate prompt
    if matrix_type == 2:
        reports = gold_df.iloc[:, 1:4]  # radiologist reports
    else:
        reports = gold_df.iloc[:, 5:8]  # AI reports
    prompt = get_prompt(data_type)
    schema = build_schema(condition_names)

    full_response = []
    for i in range(len(reports)):
        report = reports.iloc[[i]].to_string(index=False)
        # Call model through ollama
        response = ollama.chat(model=model, messages=[
            {
                'role': 'system',
                'content': "You are an expert radiologist assistanting "
                "in labeling data.",
            },
            {
                'role': 'user',
                'content': prompt + "\n\n" + report
            }
        ], format=schema, think=False)

        content = response['message']['content']
        print(f"Report {i+1} done")
        full_response.append(json.loads(content))

    pred_df = pd.DataFrame(full_response, columns=condition_names)
    save_to_excel(pred_df, data_type, matrix_type)

    return pred_df


def main():
    # Parse command line argument for excel file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str, required=True,
                        help='Your gold labels excel file path')
    parser.add_argument('--data_type', type=str,
                        required=True, choices=RADIOLOGY_TYPES,
                        help='The type of data (canine_abdomen, '
                        'canine_thorax, feline_thorax)')
    parser.add_argument("--model", default="qwen3:4b",
                        help="Ollama model name (default: qwen3:4b)")
    parser.add_argument("--rows", type=int, default=100,
                        choices=range(1, 101),
                        help="Only the first N data rows are sent "
                        "to the LLM (default: 100)")
    parser.add_argument('--both_matrices', default="no", type=str,
                        choices=["no", "yes"], help='Only matrix 2 or both')

    args = parser.parse_args()

    full_df = pd.read_excel(args.gold)
    conditions = get_conditions(full_df)

    gold_df = full_df[:args.rows]
    pred_df = call_llm(args.data_type, args.model, gold_df,
                       list(conditions.columns), matrix_type=2)

    cm_2_path = format_confusion_matrix(conditions,
                                        args.data_type, matrix_type=2)
    fill_confusion_matrix(gold_df, pred_df, cm_2_path,
                          conditions, args.data_type, matrix_type=2)

    if args.both_matrices == "yes":
        ai_pred = call_llm(args.data_type, args.model,
                           gold_df, list(conditions.columns), matrix_type=1)
        cm_1_path = format_confusion_matrix(conditions,
                                            args.data_type, matrix_type=1)
        fill_confusion_matrix(pred_df, ai_pred,
                              cm_1_path, conditions,
                              args.data_type, matrix_type=1)


if __name__ == '__main__':
    main()
