import json
import math
import requests
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import argparse

from utils import row_to_json
from llm_system_prompt import field_definitions

MODEL = "gpt-oss:120b"

# Summarize training data in batches that fit in the model context window,
# then use a separate model call to merge the different summaries

parser = argparse.ArgumentParser(description="Generate LLM training summary.")
parser.add_argument("--include_rf", action="store_true",
                    help="Include random forest predictions in the training data.")
n_patients = 50
parser.add_argument("--subset", action="store_true",
                    help=f"Only including a random selection of {n_patients} positive and {n_patients} negative patients.")
args = parser.parse_args()

# Load model tokenizer (Yi = gpt-oss) to estimate tokens
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-9B")
TOKEN_LIMIT = 8192

# Prepare training data
data_path = "XXX"
if args.include_rf:
    training_df = pd.read_pickle(
        f"{data_path}/test_train_splits/X_train_all-viral_rf.pkl")
else:
    training_df = pd.read_pickle(
        f"{data_path}/test_train_splits/X_train_all-viral.pkl")

ground_truth_df = pd.read_pickle(
    f"{data_path}/test_train_splits/y_train_all-viral.pkl")

# Replace 0/1 with no/yes labels and add to training_df
ground_truth_df = ground_truth_df.replace({0: "no", 1: "yes"})
training_df['viral_diagnosis'] = ground_truth_df

if args.subset:
    # Randomly select [n_patients] positive and [n_patients] negative rows
    pos_df = training_df[training_df['viral_diagnosis'] == "yes"].sample(
        n=n_patients, random_state=42).reset_index(drop=True)
    neg_df = training_df[training_df['viral_diagnosis'] == "no"].sample(
        n=n_patients, random_state=42).reset_index(drop=True)
    # Alternate positive and negative rows
    alternating_rows = []
    for i in range(n_patients):
        alternating_rows.append(pos_df.iloc[i])
        alternating_rows.append(neg_df.iloc[i])
    training_df = pd.DataFrame(alternating_rows).reset_index(drop=True)
    print(
        f"Only including a random selection of {n_patients} positive and {n_patients} negative patients.")

rows = training_df.to_dict(orient="records")

# Group patient data into batches that fit into context window

# Estimate token usage for a string or dict (prompt or row)


def get_token_length(obj):
    """
    Estimate token usage for a string prompt or a dict (row).
    If a dict is provided, it is converted to JSON.
    """
    if isinstance(obj, dict):
        text = json.dumps(obj, indent=2)
    else:
        text = str(obj)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

# Build batches that fit in context window


def batch_rows(rows, token_limit=TOKEN_LIMIT):
    batches = []
    current_batch = []
    current_tokens = 0

    for row in rows:
        if args.include_rf:
            buffer = 100  # buffer for explanation
        else:
            buffer = 100
        t = get_token_length(row) + buffer
        if current_tokens + t > token_limit:
            batches.append(current_batch)
            current_batch = [row]
            current_tokens = t
        else:
            current_batch.append(row)
            current_tokens += t

    if current_batch:
        batches.append(current_batch)

    return batches


batches = batch_rows(rows)
print(f"Split {len(rows)} rows into {len(batches)} context-safe batches")


# Summarize each batch


# LLM prompts for summarizing a batch
if args.include_rf:
    feature_importance_df = pd.read_csv(
        f"{data_path}/model_rf/all-viral_feature_importances_rf.csv")
    # Convert the top10 rows of feature_importance_df to a JSON dictionary
    feature_importance_json = feature_importance_df.head(
        10).to_dict(orient="records")
    feature_importance_text = "\n".join(
        f"- {row['Feature_clean']}: importance={row['Importance']:.4f}"
        for row in feature_importance_json
    )

    # print(feature_importance_text)

SYSTEM_PROMPT = f"""
        The structured patient metadata will be structured as a JSON object with some of the following fields:
        {field_definitions}
        Use this additional information about the fields to inform your predictions.
        """


def make_batch_summary_prompt(batch):
    formatted = "\n\n".join(
        f"Patient data:\n{row_to_json(row)}" for row in batch
    )
    if args.include_rf:
        # return f"""
        # You are an expert clinical reasoning engine analyzing a dataset of labeled patient cases.

        # Each case includes:
        # - Structured patient metadata (symptoms, demographics, exposures, comorbidities)
        # - A ground truth diagnosis ('viral_diagnosis')
        # - Predictions from a random forest model ('probability_of_viral_rf' and 'viral_rf')

        # Note that the RF model found the following features to be highly predictive, but critically assess their clinical plausibility:
        # {feature_importance_text}

        # Your task is to distill actionable insights and patterns that connect patient features to viral diagnoses, with a focus on how the random forest model's predictions relate to the true outcomes.

        # Please pay close attention to the following:
        # - Identify key symptom patterns, combinations of symptoms, travel, and exposures that are strongly associated with viral infection.
        # - Highlight common scenarios or feature combinations that are typically **not** associated with a viral diagnosis.
        # - Analyze the random forest model's predictions:
        #     - When do its predictions agree or disagree with the true diagnosis?
        #     - How should another model use the random forest's outputs to improve its own diagnostic accuracy?

        # Your output should be a clear, structured summary that includes:
        # - Main symptom and feature patterns indicating viral infection
        # - Typical non-viral cases and their distinguishing features
        # - Insights about the random forest model's strengths, weaknesses, and the features that influence its performance
        # - Practical guidance for leveraging both clinical data and random forest predictions in future diagnostic models

        # Please write a concise, structured summary that would help another model make more accurate diagnostic predictions from similar patient metadata, using both the clinical features and the random forest results.

        # Here is the training data:

        # {formatted}
        # """
        return f"""
        You are an expert clinical reasoning engine analyzing a dataset of labeled patient cases.

        Each case includes:
        - Structured patient metadata (symptoms, demographics, exposures, comorbidities)
        - A ground truth diagnosis ('viral_diagnosis')
        - Predictions from a random forest model ('probability_of_viral_rf' and 'viral_rf')

        The RF model found the following features to be highly predictive. Critically assess their clinical plausibility:
        {feature_importance_text}

        Your task:
        Distill only the most essential, high-value insights that will help another model predict viral vs. non-viral cases. Keep your summary short. Focus on what is consistently important and ignore less relevant patterns.

        Specifically:
        - Key symptom/exposure patterns strongly linked to viral infection.
        - Key patterns strongly linked to non-viral cases.
        - How RF predictions align or misalign with true outcomes.
        - When to trust or override the RF outputs.

        Output format:
        A concise bullet-point list grouped into:
        1. "Viral indicators"
        2. "Non-viral indicators"
        3. "RF interpretation rules"

        Do not include explanations, background context, or verbose narrative. Only include the minimal set of rules and patterns essential for training the next model.

        Here is the training data:

        {formatted}
        """
    else:
        return f"""
        You are an expert clinical reasoning engine analyzing a dataset of labeled patient cases.

        Each case includes:
        - Structured patient metadata (symptoms, demographics, exposures, comorbidities)
        - A ground truth diagnosis ('viral_diagnosis')

        Your task:
        Distill only the most essential, high-value insights that will help another model predict viral vs. non-viral cases. Keep your summary short. Focus on what is consistently important and ignore less relevant patterns.

        Specifically:
        - Key symptom/exposure patterns strongly linked to viral infection.
        - Key patterns strongly linked to non-viral cases.

        Output format:
        A concise bullet-point list grouped into:
        1. "Viral indicators"
        2. "Non-viral indicators"

        Do not include explanations, background context, or verbose narrative. Only include the minimal set of rules and patterns essential for training the next model.

        Here is the training data:

        {formatted}
        """

# Call Ollama for batch summary


def run_ollama_chat(model, user_prompt, system_prompt=None):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "options": {"temperature": 0.3},
            "stream": False
        }
    )
    return response.json()["message"]["content"]


# Generate summaries for all batches


batch_summaries = []
for i, batch in enumerate(tqdm(batches, desc="Summarizing batches")):
    prompt = make_batch_summary_prompt(batch)

    # Double-check that we're within token limit
    prompt_tokens = get_token_length(prompt)
    if prompt_tokens > TOKEN_LIMIT:
        print(
            f"ERROR: prompt size ({prompt_tokens} tokens) exceeds token limit ({TOKEN_LIMIT}).")

    summary = run_ollama_chat(MODEL, prompt, SYSTEM_PROMPT)
    batch_summaries.append(summary)


# Merge all summaries into final distilled knowledge summary

def make_merging_prompt(summaries):
    """
    Create a merging prompt that fits within the context window.
    If the joined summaries are too long, trim from the end until the prompt fits.
    """
    # Compose the static part of the prompt (everything except the joined summaries)
    # static_prompt = f"""
    #     You are an expert in infectious diseases and clinical data analysis.

    #     Your task is to synthesize the following batch summaries into a single, unified clinical knowledge base. Your summary should:
    #     - Clearly capture consistent relationships between exposures, symptoms, and pathogens
    #     - Highlight common patterns, notable exceptions, and rare or edge cases
    #     - Distill actionable insights that can guide future model predictions based on patient metadata
    #     {'- Integrate and clearly explain insights from Random Forest model predictions and the most important features' if args.include_rf else ''}

    #     Please provide a concise, well-organized, and clinically relevant summary of key findings. Avoid unnecessary repetition, and structure your response for clarity and practical use.

    #     Below are the summaries generated from different batches of patient data:

    #     """
    static_prompt = f"""
        You are an expert in infectious diseases and clinical data analysis.

        Your task:
        Merge the following batch summaries into one concise, unified clinical knowledge base that will help another model make accurate viral vs. non-viral predictions.

        Include only:
        - Consistent, high-value relationships between exposures, symptoms, and pathogens.
        - Key patterns that strongly indicate viral or non-viral status.
        - Notable exceptions and rare but important edge cases.
        {'- Key takeaways from Random Forest predictions: when to trust or override them, and the most important features.' if args.include_rf else ''}

        Output format:
        A short bullet-point list grouped into:
        1. "Viral indicators"
        2. "Non-viral indicators"
        {'3. RF interpretation rules' if args.include_rf else ''}

        Instructions:
        - Eliminate repetition from the batch summaries.
        - Ignore low-value or inconsistent observations.
        - Do not include background explanations or long narratives.
        - Focus only on rules and patterns essential for model training.

        Below are the batch summaries:
        
        """

    # Now, join summaries and check token length
    joined = "\n\n".join(
        f"Summary {i+1}:\n{summary}" for i, summary in enumerate(summaries)
    )

    # Compose full prompt and check length
    def full_prompt(joined_summaries):
        return static_prompt + joined_summaries

    # Estimate token length for the full prompt
    prompt = full_prompt(joined)
    prompt_tokens = get_token_length(prompt)

    # If too long, trim summaries from the end until it fits
    if prompt_tokens > TOKEN_LIMIT:
        print(
            f"WARNING: merge_prompt size ({prompt_tokens} tokens) exceeds token limit ({TOKEN_LIMIT}). Trimming summaries...")
        # Try removing summaries from the end one by one
        trimmed_summaries = list(summaries)
        while trimmed_summaries:
            joined_trimmed = "\n\n".join(
                f"Summary {i+1}:\n{summary}" for i, summary in enumerate(trimmed_summaries)
            )
            prompt = full_prompt(joined_trimmed)
            prompt_tokens = get_token_length(prompt)
            if prompt_tokens <= TOKEN_LIMIT:
                print(
                    f"Trimmed to {len(trimmed_summaries)} summaries to fit context window.")
                return prompt
            trimmed_summaries = trimmed_summaries[:-1]
        # If even one summary is too long, just use the first summary
        print("Even one summary is too long for the context window. Using the first summary only.")
        joined_trimmed = f"Summary 1:\n{summaries[0]}"
        return full_prompt(joined_trimmed)
    else:
        return prompt


merge_prompt = make_merging_prompt(batch_summaries)
# Double-check that we're within token limit (should always be true now)
merge_prompt_tokens = get_token_length(merge_prompt)
if merge_prompt_tokens > TOKEN_LIMIT:
    print(
        f"ERROR: merge_prompt size ({merge_prompt_tokens} tokens) still exceeds token limit ({TOKEN_LIMIT}) after trimming.")

final_knowledge_summary = run_ollama_chat(MODEL, merge_prompt)

# Save to file

if args.include_rf:
    saving_path = f"{data_path}/test_train_splits/llm_training_summary{'_subset' if args.subset else ''}_rf_v3.txt"
else:
    saving_path = f"{data_path}/test_train_splits/llm_training_summary{'_subset' if args.subset else ''}_v3.txt"

with open(saving_path, "w") as f:
    f.write(final_knowledge_summary)
