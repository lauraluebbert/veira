import json
import pandas as pd
from transformers import AutoTokenizer

def data_to_use(df):
    """
    Function to remove columns that should not be included for predictions/clustering.
    """
    return df.loc[:,
                  (~df.columns.str.contains('infection_status')) &
                  (~df.columns.str.contains('label')) &
                  (~df.columns.str.contains('cluster'))
                #   (~df.columns.str.contains('record_id'))
                  ]


def explain_col(col, list_vals=True, data_date = "2025-05-02_1814"):
    dict_path = f"../data/SentinelNigeria_DataDictionary_{data_date.split('_')[0]}.csv"
    dict_df = pd.read_csv(dict_path)

    temp = dict_df[dict_df["Variable / Field Name"] == col]

    if list_vals:
        print(list(temp["Choices, Calculations, OR Slider Labels"]))
    else:
        print(temp)


def row_to_json(row):
    # Normalize input to dictionary
    if isinstance(row, pd.Series):
        row = row.to_dict()
    elif not isinstance(row, dict):
        raise TypeError("Input must be a pandas Series or a dict.")

    patient = {}

    for col, value in row.items():
        # Convert NaN to None
        if pd.isna(value):
            patient[col] = None
        # Split comma-separated strings into lists if they contain multiple items
        elif isinstance(value, str) and ',' in value:
            patient[col] = [v.strip() for v in value.split(',')]
        # Convert numbers
        elif isinstance(value, (int, float)) and pd.notna(value):
            patient[col] = int(value) if float(value).is_integer() else float(value)
        else:
            patient[col] = value

    return json.dumps(patient, indent=2)

# Load model tokenizer (Yi = gpt-oss) to estimate tokens
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-9B")
# Estimate token usage
def get_token_length(prompt):
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return len(tokens)
