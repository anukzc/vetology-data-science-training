# For Vetology Data Science Training
# Date: 10/12/2026

import argparse
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, Side
import pandas as pd
from google import genai
import dotenv

dotenv.load_dotenv()
client = genai.Client()

ROWS_TO_KEEP = 101

def get_conditions_from_excel(file_path):
    # Read the first ROWS_TO_KEEP rows of the excel file into a pandas DataFrame, then select the columns for each condition
    full_df = pd.read_excel(file_path)
    first_rows = full_df.head(ROWS_TO_KEEP)
    conditions = first_rows.iloc[:, 8:]
    return conditions

def create_confusion_matrix(conditions):
    cols = ['True Positive', 'False Positive', 'True Negative', 'False Negative', 'Sensitivity', 'Specificity', 'Check', 'Positive Ground Truth', 'Negative Ground Truth', 'Ground Truth Check', '', 'Radiologist Agreement Rate']
    # Create the confusion matrix with the specified columns and index (rows) as the conditions
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
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
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
        ws.column_dimensions[col[0].column_letter].width = max(len(str(c.value or '')) for c in col) + 1
    
    wb.save(path)

def main():
    # Parse command line argument for excel file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Your excel file path')
    args = parser.parse_args()

    conditions = get_conditions_from_excel(args.file)
    create_confusion_matrix(conditions)

    print("Confusion matrix has been created")

    # Call Gemini API
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input="Tell the user that the confusion matrix has been created and saved as 'confusion_matrix_template.xlsx'."
    )

    print(interaction.output_text)

if __name__ == '__main__':
    main()