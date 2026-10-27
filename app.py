# For Vetology Data Science Training
# Date: 10/12/2026

import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, Side
import json
from google import genai
import dotenv

dotenv.load_dotenv()
client = genai.Client()

ROWS_TO_KEEP = 100


def get_conditions_from_excel(file_path):
    # Read the first ROWS_TO_KEEP rows of the excel file into a pandas
    # DataFrame, then select the columns for each condition
    full_df = pd.read_excel(file_path)
    first_rows = full_df.head(ROWS_TO_KEEP)
    conditions = first_rows.iloc[:, 8:]
    return conditions


def format_confusion_matrix(conditions):
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
    for cell in ws[1]:          # row 1: index header + column names
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


def fill_confusion_matrix(gold_df, pred_df, cm_path, conditions):
    # Load the confusion matrix template
    for condition in conditions.columns:
        y_true = gold_df[condition]
        y_pred = pred_df[condition]

        # TN FP
        # FN TP
        cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"])

        tp, fn, fp, tn = cm.ravel()

        sensitivity = tp / (tp + fn) or "N/A" # Recall
        specificity = tn / (tn + fp) or "N/A"

        # Fill in the confusion matrix
        wb = load_workbook(cm_path)
        ws = wb.active
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


def call_llm(type, gold_df):
    reports = gold_df.iloc[:, 0:3]

    with open("prompts.json", "r") as prompts:
        if type == "canine_abdomen":
                prompt = json.load(prompts)["canine_abdomen"]
        elif type == "canine_thorax":
                prompt = json.load(prompts)["canine_thorax"]
        elif type == "feline_thorax":
                prompt = json.load(prompts)["feline_thorax"]

    full_response = ""
    for i in range(0, len(reports), 5):  # Process in batches of 10
        reports_batch = reports.iloc[i:i + 5]
        print(reports_batch)
        # Call Gemini API
        interaction = client.interactions.create(
            model="gemini-3.5-flash",
            input=f"{prompt}\n\n{reports_batch.to_string(index=False)}"
        )
        full_response += interaction.output_text

        print(interaction.output_text)
        print(type(interaction.output_text))

    return full_response


def main():
    # Parse command line argument for excel file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str, required=True,
                        help='Your gold labels excel file path')
    parser.add_argument('--pred', type=str, required=True,
                        help='Your predicted labelsexcel file path')
    # parser.add_argument('--type', type=str, required=True,
    #                     help='The type of data (canine_abdomen, canine_thorax, feline_thorax)')
    args = parser.parse_args()

    gold_df = pd.read_excel(args.gold)[:ROWS_TO_KEEP]

    # print(gold_df)

    # pred = call_llm(args.type, gold_df)

    pred_df = pd.read_excel(args.pred)

    conditions = get_conditions_from_excel(args.gold)
    cm_path = format_confusion_matrix(conditions)
    fill_confusion_matrix(gold_df, pred_df, cm_path, conditions)

    print("Confusion matrix has been created")


if __name__ == '__main__':
    main()
