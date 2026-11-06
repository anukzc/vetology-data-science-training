# For Vetology Data Science Training
# Date: 10/12/2026

import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, Side
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ollama
import dotenv

dotenv.load_dotenv()

ROWS_TO_KEEP = 100


def get_conditions_from_excel(file_path):
    # Read the first ROWS_TO_KEEP rows of the excel file into a pandas
    # DataFrame, then select the columns for each condition
    full_df = pd.read_excel(file_path)
    first_rows = full_df.head(ROWS_TO_KEEP)
    conditions = first_rows.iloc[:, 8:]
    return conditions


def format_confusion_matrix(conditions):
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

    path = "confusion_matrix_template.xlsx"
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


def fill_confusion_matrix(gold_df, pred_df, cm_path, conditions, type):
    # Load the confusion matrix template
    wb = load_workbook(cm_path)
    ws = wb.active
    for condition in conditions.columns:
        y_true = gold_df[condition]
        y_pred = pred_df[condition]

        # TN FP
        # FN TP
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
    wb.save(f"./data/{type}_confusion_matrix.xlsx")
        # One overall confusion matrix across all conditions
    all_true = gold_df[conditions.columns].values.ravel()
    all_pred = pred_df[conditions.columns].values.ravel()
    overall_cm = confusion_matrix(all_true, all_pred, labels=["Positive", "Negative"])

    disp = ConfusionMatrixDisplay(overall_cm, display_labels=["Positive", "Negative"])
    disp.plot(cmap="Blues", colorbar=False)
    disp.ax_.set_title(f"{type}: all conditions")
    plt.savefig(f"./data/{type}_confusion_matrix_2.png", bbox_inches="tight")
    plt.close()



def save_to_excel(df, type):
    output_filename = f"./data/{type}_qwen_labels.xlsx"
    df.to_excel(output_filename, index=False)
    return output_filename


def get_prompt(type):
    # Load the appropriate prompt based on the type
    with open("prompts.json", "r") as prompts:
        if type == "canine_abdomen":
            prompt = json.load(prompts)["canine_abdomen"]
        elif type == "canine_thorax":
            prompt = json.load(prompts)["canine_thorax"]
        elif type == "feline_thorax":
            prompt = json.load(prompts)["feline_thorax"]
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


def call_llm(type, gold_df, condition_names):
    # Grabs radiologist reports and appropriate prompt
    reports = gold_df.iloc[:, 0:3]
    prompt = get_prompt(type)
    schema = build_schema(condition_names)

    full_response = []
    for i in range(len(reports)):
        report = reports.iloc[[i]].to_string(index=False)
        # Call model through ollama
        response = ollama.chat(model='qwen3:4b', messages=[
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
    save_to_excel(pred_df, type)

    return pred_df


def main():
    # Parse command line argument for excel file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str, required=True,
                        help='Your gold labels excel file path')
    # parser.add_argument('--pred', type=str, required=True,
    #                     help='Your predicted labelsexcel file path')
    parser.add_argument('--type', type=str, required=True,
                        help='The type of data (canine_abdomen, '
                        'canine_thorax, feline_thorax)')

    args = parser.parse_args()

    conditions = get_conditions_from_excel(args.gold)

    gold_df = pd.read_excel(args.gold)[:ROWS_TO_KEEP]
    pred_df = call_llm(args.type, gold_df, list(conditions.columns))

    cm_path = format_confusion_matrix(conditions)
    fill_confusion_matrix(gold_df, pred_df, cm_path, conditions, args.type)


if __name__ == '__main__':
    main()
