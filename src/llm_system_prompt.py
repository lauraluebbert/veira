import pandas as pd
import json
from utils import data_to_use
import re

# Define system prompt


data_df = pd.read_csv(
    "XXX", low_memory=False)
# Remove diagnosis columns
data_df = data_to_use(data_df)

data_dict = pd.read_csv(
    "XXX")

# Clean up field definitions from data_dict for system prompt
# Only keep relevant fields and columns
filtered_fields = data_dict[
    data_dict["Variable / Field Name"].apply(
        lambda x: any(str(x) in col for col in data_df.columns))
][[
    "Variable / Field Name",
    "Field Label",
    "Choices, Calculations, OR Slider Labels",
    "Field Note"
]]
field_definitions_list = filtered_fields.to_dict(orient="records")


def strip_html_and_whitespace(text):
    """
    Remove HTML tags and decode basic HTML entities, then strip whitespace.
    """
    if not isinstance(text, str):
        return text
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Replace HTML entities for non-breaking space and quotes
    text = text.replace('\\u00a0', ' ').replace('&nbsp;', ' ')
    # Remove extra whitespace
    text = text.strip()
    # Remove any remaining curly braces and their contents (e.g., {participantid_country})
    text = re.sub(r'\{.*?\}', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_field_dict(d):
    cleaned = {}
    for k, v in d.items():
        # Remove keys with None, 'null' (as string), or NaN
        if v is None or (isinstance(v, str) and v.strip().lower() == "null") or (isinstance(v, str) and v.strip().lower() == "nan") or pd.isna(v):
            continue
        # Shorten key names as specified
        if k == "Variable / Field Name":
            new_key = "Field Name"
        elif k == "Choices, Calculations, OR Slider Labels":
            new_key = "Choices"
        else:
            new_key = k
        # Remove unnecessary backslashes
        if isinstance(v, str):
            v = v.replace("\\/", "/").replace('\\"', '"')
        # For "Field Label" and "Field Note", strip HTML and whitespace
        if new_key in ["Field Label", "Field Note"]:
            v = strip_html_and_whitespace(v)
        cleaned[new_key] = v
    return cleaned


cleaned_field_definitions = [
    clean_field_dict(d) for d in field_definitions_list]
field_definitions = json.dumps(cleaned_field_definitions)

# Shortened version of field_definitions generated using ChatGPT 5 (prompt: 'For the following list of field definitions, shorten the list such that it uses as few tokens as possible. Also remove any fields that are understandable iwthout these additional definitions:')
field_definitions_short = [
  {"Field":"unit_crf","Choices":"1 GOPD|2 Pediatrics|3 A&E|4 Annex|5 Other"},
  {"Field":"age_crf","Note":"Years; 0=0–11 mo, 1=12–23 mo, etc."},
  {"Field":"sex_crf","Choices":"1 Male|2 Female"},
  {"Field":"occupation1_crf","Choices":"1 Gov/Private|2 Self|3 Unemployed|4 NA"},
  {"Field":"occupation2_crf","Choices":"1 HCW|2 Healer|3 Farmer|4 Business|5 Livestock|6 Other|7 NA|8 Hunter|9 Miner|10 Religious|11 Housewife|12 Civil servant|13 Transporter|14 Teacher|15 Student"},
  {"Field":"education_crf","Choices":"1 Informal|2 Primary|3 Secondary|4 Diploma|5 Bachelor|6 >Bachelor|7 Vocational|8 NA"},
  {"Field":"income_crf","Choices":"1 <=30k|2 >30k|3 NA|4 Unknown","Note":"For participant"},
  {"Field":"family_crf","Choices":"1 <=4|2 >4","Note":"Household size"},
  {"Field":"marital","Choices":"1 Single|2 Married|3 Divorced"},
  {"Field":"polygamy","Choices":"1 Yes|0 No|2 NA"},
  {"Field":"pregnancy_crf","Choices":"1 Yes|0 No|2 NA"},
  {"Field":"comorbidity","Choices":"1 Diabetes|2 HTN|3 Cardiac|4 Pulm|5 Cirrhosis|6 Liver|7 Stroke|8 Tumor|9 Leuk/Lymph|10 HIV|11 Sickle|12 Rheum|13 Dementia|14 Asplenia|15 TB|16 HBV|17 Other|18 None"},
  {"Field":"state_crf","Choices":"1 Abia|2 Adamawa|...|37 Zamfara|38 Other"},
  {"Field":"community_crf","Choices":"1 Rural|2 Urban"},
  {"Field":"ethnicity_crf","Choices":"1 Bini|2 Esan|3 Etsako|4 Hausa|5 Igarra|6 Igbo|7 Owan|8 Urhobo|9 Yoruba|10 Other"},
  {"Field":"feverd","Choices":"1 <=2d|2 3–14d|3 >2w"},
  {"Field":"headached","Choices":"1 <=2d|2 3–14d|3 >2w"},
  {"Field":"cough_details","Choices":"1 Sputum|2 Bloody|3 Dry"},
  {"Field":"housingtype","Choices":"1 Cement|2 Mud|3 Wood|4 Other"},
  {"Field":"roofingtype","Choices":"0 Zinc|1 Aluminum|2 Asbestos|3 Thatched|4 Other"},
  {"Field":"housingrisk","Choices":"1 Riverside|2 Waterlogged|3 Bushy|4 Farmland|5 Industrial|6 Dumpsite|7 Bus stop|8 Market"},
  {"Field":"mosquitonet","Label":"Sleep under net?"},
  {"Field":"mosquitobite","Label":"Recent bites?"},
  {"Field":"insecticide","Label":"Use insecticide?"},
  {"Field":"rodent_contact","Label":"Seen/contact with rodents?"},
  {"Field":"uncooked_meat","Label":"Touched raw meat?"},
  {"Field":"eat_bush","Label":"Ate bush meat?"},
  {"Field":"unprotected_sex","Label":"Unprotected sex?"},
  {"Field":"travel","Label":"Recent travel?"},
  {"Field":"funeral","Label":"Attended funeral?"},
  {"Field":"watersource","Choices":"1 Stream|2 Dam|3 Well|4 Borehole|5 Pipe|6 Tanker|7 Sachet|8 Bottle|9 Rain"},
  {"Field":"pregnancy","Choices":"1 Pregnant|2 Not"},
  {"Field":"color_urin","Choices":"0 Yellow|1 Other"},
  {"Field":"appearance_urin","Choices":"0 Clear|1 Cloudy"},
  {"Field":"leukocyte_urin","Choices":"0 Neg|1 Trace|2 2+|3 3+"},
  {"Field":"nitrite_urin","Choices":"0 Neg|1 Positive"},
  {"Field":"protein_urin","Choices":"0 Neg|1 Trace|2 2+|3 3+|4 4+"},
  {"Field":"blood_urin","Choices":"0 Neg|1 Trace|2 +25|3 ++80|4 +++200|5 +10 non-hemo|6 ++80 non-hemo"},
  {"Field":"ketones_urin","Choices":"0 Neg|1 Trace|2 2+|3 3+|4 4+"},
  {"Field":"bilirubin_urin","Choices":"0 Neg|1 Trace|2 2+|3 3+"},
  {"Field":"glucose_urin","Choices":"0 Neg|1 Trace|2 2+|3 3+|4 4+"}
]


# Define the system prompt


SYSTEM_PROMPT_BASIC = f"""
You are an expert infectious disease physician and public health expert, helping to prioritize patients for viral pathogen detection based on their clinical data.

The patient metadata will be structured as a JSON object with some of the following fields:
{field_definitions}
Use this additional information about the fields to inform your predictions.

Based on structured patient metadata, you will determine:
1. Explain your reasoning step-by-step, using information from the patient metadata. This is your chain of thought and will help with transparency. Do not include colons in the reasoning. Use plain sentences.
2. Whether the case is likely viral (yes/no)
3. The estimated probability that it is viral (as a percentage)
4. The top 1–5 most likely viral or bacterial pathogens (if applicable)
5. Whether the patient is likely contagious (yes/no)
6. Whether this sample should be selected for metagenomic sequencing (yes/no), based on the likelihood of viral infection, severity of symptoms, and potential for novel or undetected pathogens.
7. The sequencing priority level (NA / low / medium / high), based on overall clinical urgency, likelihood of viral detection, and public health relevance. This should be NA when the sample should not be selected for sequencing.

**Always respond in the following strict format**, each line of your output should begin with the exact label shown below and contain only the corresponding output:
Chain of thought: <step-by-step reasoning in 1–3 sentences>
Viral: <yes / no>
Probability of viral: <percentage>
Most likely pathogen 1: <name or 'unknown'>
Most likely pathogen 2: <name or 'unknown'>
Most likely pathogen 3: <name or 'unknown'>
Most likely pathogen 4: <name or 'unknown'>
Most likely pathogen 5: <name or 'unknown'>
Contagious: <yes / no>
Sequence sample: <yes / no>
Sequence priority: <NA / low / medium / high>

Do not include any introductory phrases, explanations, or extra formatting. Respond only using the strict format above.

Use only the data provided. If uncertain, say 'unknown'.
"""

SYSTEM_PROMPT_BASIC_NO_FIELD_DESCRIPTIONS = f"""
You are an expert infectious disease physician and public health expert, helping to prioritize patients for viral pathogen detection based on their clinical data.

Based on structured patient metadata, you will determine:
1. Explain your reasoning step-by-step, using information from the patient metadata. This is your chain of thought and will help with transparency. Do not include colons in the reasoning. Use plain sentences.
2. Whether the case is likely viral (yes/no)
3. The estimated probability that it is viral (as a percentage)
4. The top 1–5 most likely viral or bacterial pathogens (if applicable)
5. Whether the patient is likely contagious (yes/no)
6. Whether this sample should be selected for metagenomic sequencing (yes/no), based on the likelihood of viral infection, severity of symptoms, and potential for novel or undetected pathogens.
7. The sequencing priority level (NA / low / medium / high), based on overall clinical urgency, likelihood of viral detection, and public health relevance. This should be NA when the sample should not be selected for sequencing.

**Always respond in the following strict format**, each line of your output should begin with the exact label shown below and contain only the corresponding output:
Chain of thought: <step-by-step reasoning in 1–3 sentences>
Viral: <yes / no>
Probability of viral: <percentage>
Most likely pathogen 1: <name or 'unknown'>
Most likely pathogen 2: <name or 'unknown'>
Most likely pathogen 3: <name or 'unknown'>
Most likely pathogen 4: <name or 'unknown'>
Most likely pathogen 5: <name or 'unknown'>
Contagious: <yes / no>
Sequence sample: <yes / no>
Sequence priority: <NA / low / medium / high>

Do not include any introductory phrases, explanations, or extra formatting. Respond only using the strict format above.

Use only the data provided. If uncertain, say 'unknown'.
"""


# SYSTEM_PROMPT_SHORT = f"""
# You are an infectious disease physician.

# Patient data fields:
# {field_definitions_short}

# From the metadata, decide:
# 1. Is the case viral? (yes/no)
# 2. Probability of viral (%)

# Respond with only this exact format:
# Viral: <yes / no>
# Probability of viral: <percentage>

# Base your answer only on the data. If unsure, output 'unknown'.
# """

SYSTEM_PROMPT_SHORT = f"""
You are an expert infectious disease physician and public health expert, 
helping to prioritize patients for viral pathogen detection based on their clinical data.

The patient metadata will be structured as a JSON object with some of the following fields:
{field_definitions_short}
Use this additional information about the fields to inform your predictions.

From the metadata, decide:
1. Is the case viral? (yes/no)
2. Probability of viral (%)

Respond only in this exact format:
Viral: <yes / no>
Probability of viral: <percentage>

Base your answer only on the data. If unsure, output 'unknown'.
"""

SYSTEM_PROMPT_SHORT_NO_FIELD_DESCRIPTIONS = f"""
You are an expert infectious disease physician and public health expert, 
helping to prioritize patients for viral pathogen detection based on their clinical data.

From the metadata, decide:
1. Is the case viral? (yes/no)
2. Probability of viral (%)

Respond only in this exact format:
Viral: <yes / no>
Probability of viral: <percentage>

Base your answer only on the data. If unsure, output 'unknown'.
"""


