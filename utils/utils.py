import os
import re
import json 
import pandas as pd


def extract_prob(text: str):
    PROB_RE = re.compile(r'(?mi)^\s*(?:p_virus|p_viral)\s*[:=]\s*([01](?:\.\d+)?|\.\d+)f?\s*$')
    m = PROB_RE.findall(text)
    if not m:
        print("PARSING FAILED DEFAULTING TO 0.25")
        return None  # <- return None, don't silently pick 0.25
    try:
        p = float(m[-1].rstrip('f'))
        return max(0.0, min(1.0, p))
    except Exception:
        print("PARSING FAILED DEFAULTING TO 0.25")
        return None

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