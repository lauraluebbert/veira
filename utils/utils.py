import os
import re
import json 
import pandas as pd
import pickle

import re

# First: bare decimal anywhere on a line by itself
BARE_DECIMAL_RE = re.compile(r'(?mi)^\s*([01](?:\.\d+)?|\.\d+)\s*$')

# Fallback: named fields: p_virus: 0.95, p_viral=0.8, etc.
PROB_RE = re.compile(
    r'(?mi)^\s*(?:p_virus|p_viral)\s*[:=]\s*([01](?:\.\d+)?|\.\d+)f?\s*$'
)

def extract_prob(text: str):
    # 1. Try bare decimal first
    m = BARE_DECIMAL_RE.findall(text)
    if m:
        try:
            p = float(m[-1])
            return max(0.0, min(1.0, p))
        except Exception:
            pass  # fall through to fallback parser

    # 2. Fallback: p_virus: XXX parsing
    m = PROB_RE.findall(text)
    if m:
        try:
            p = float(m[-1].rstrip("f"))
            return max(0.0, min(1.0, p))
        except Exception:
            pass

    # 3. Parse failed
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

