import pandas as pd

def parse_excel(file_path):

    text = ""

    sheets = pd.read_excel(file_path, sheet_name=None)

    for sheet_name, df in sheets.items():
        text += f"\nSheet: {sheet_name}\n"

        # Convert dataframe to readable text
        text += df.to_string(index=False)
        text += "\n"

    return text