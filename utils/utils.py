import os
import re

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