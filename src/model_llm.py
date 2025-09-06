import pandas as pd
import requests
import sys
import json
import pickle
import os
import re
import time
import argparse
import psutil
import pdb
import pynvml
from tqdm import tqdm

from llm_system_prompt import SYSTEM_PROMPT_BASIC, SYSTEM_PROMPT_SHORT, SYSTEM_PROMPT_BASIC_NO_FIELD_DESCRIPTIONS, SYSTEM_PROMPT_SHORT_NO_FIELD_DESCRIPTIONS
from utils import data_to_use, row_to_json, get_token_length
from model_llm_rag_funcs import load_vectorstore_for_rag, retrieve_context

sys.path.append("./grpo")
from create_vignette import generate_patient_data_vignette

# Ollama v0.11.4
# https://github.com/ollama/ollama/releases/tag/v0.11.4

# Set the interval for saving progress (e.g., every 100 rows)
SAVE_INTERVAL = 100

TEMPERATURE = 0.5

# https://ollama.com/library/gpt-oss

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process data with LLM')
parser.add_argument('--model', type=str, help='Model to use', required=True)

parser.add_argument('--data', type=str, default='all',
                    choices=['all', 'test', 'test_rf'],
                    help='Which data to use: all, test, test_rf')
parser.add_argument('--test_mode', action='store_true',
                    help='Run on first few patients only')

parser.add_argument('--run_once', action='store_true',
                    default=False, help='Run only one prediction per patient (default runs everything 3 times and saves as run01, run02, run03)')

parser.add_argument('--system_prompt', type=str, default='basic',
                    choices=['basic', 'short'],
                    help='Which system prompt to use: basic, short')

parser.add_argument('--knowledge_summary', type=str, default=None,
                    help='Path to a file containing a distilled knowledge summary to include in the user prompt')

# Set combine_prompts to True for DeepSeek models
parser.add_argument('--combine_prompts', action='store_true', default=False,
                    help='Combine system and user prompts into a single user message')

parser.add_argument('--patient_json_to_text', action='store_true', default=False,
                    help='Generate text from patient dictionary instead of passing it to the model in json format.')

# RAG options
parser.add_argument('--rag_dir', type=str, default=None,
                    help='Path to the vectorstore directory (index.faiss, texts.jsonl, metadata.jsonl, model.json)')
parser.add_argument('--rag_top_k', type=int, default=20, help='Top-K retrieved contexts to include')
parser.add_argument('--rag_max_tokens', type=int, default=1200, help='Max tokens to spend on retrieved context')

args = parser.parse_args()

# Initialize RAG
rag_state = None
if args.rag_dir:
    index, texts, metas, model_info = load_vectorstore_for_rag(args.rag_dir)
    rag_state = {"index": index, "texts": texts, "metas": metas, "model_info": model_info}
    print(f"RAG loaded from {args.rag_dir} — {len(texts)} docs, embedder={model_info.get('model_name')}")


# Define system prompt map (args.system_prompt,  args.patient_json_to_text)
system_prompt_map = {
    ('basic', False): SYSTEM_PROMPT_BASIC,
    ('short', False): SYSTEM_PROMPT_SHORT,
    ('basic', True): SYSTEM_PROMPT_BASIC_NO_FIELD_DESCRIPTIONS,
    ('short', True): SYSTEM_PROMPT_SHORT_NO_FIELD_DESCRIPTIONS,
}

try:
    SYSTEM_PROMPT = system_prompt_map[(args.system_prompt, args.patient_json_to_text)]
except KeyError:
    raise ValueError(
        f"Unknown system prompt: {args.system_prompt} (must be one of: basic, short) "
        f"and patient_json_to_text: {args.patient_json_to_text}"
    )

MODEL = args.model

# Define token limit per model for warning (actual limit defined by ollama)
if MODEL == "gpt-oss:120b" or MODEL == "gpt-oss:20b":
    TOKEN_LIMIT = 8192
elif MODEL == "deepseek-r1:70b" or MODEL == "deepseek-r1:671b":
    TOKEN_LIMIT = 4096
else:
    TOKEN_LIMIT = 4096
    print(f"Token limit not know for model {MODEL}. Defaulting to {TOKEN_LIMIT}.")

# Define knowledge summary
knowledge_summary = None
if args.knowledge_summary:
    with open(args.knowledge_summary, "r", encoding="utf-8") as f:
        knowledge_summary = f.read()

# Define output directory
base_output_dir = os.path.join(
    "XXX", "data", "model_llm", MODEL.replace(':', '-'), f"data_{args.data}", f"system_prompt_{args.system_prompt }", f"temp_{str(TEMPERATURE).replace(".", "-")}"
)

if args.knowledge_summary:
    knowledge_name = args.knowledge_summary.split("/")[-1].split(".")[0]
else:
    knowledge_name = "no_knowledge"
if args.rag_dir:
    knowledge_name = knowledge_name + "_rag"
if args.patient_json_to_text:
    knowledge_name = knowledge_name + "_json2text"

output_dir = os.path.join(base_output_dir, knowledge_name)


data_dir = "XXX"


# Load data


if args.data == 'all':
    # Load data
    data_df = pd.read_csv(
        f"{data_dir}/SentinelNigeria_DATA_2025-05-02_1814_clean.csv", low_memory=False)
    data_df = data_to_use(data_df)

elif args.data == 'test':
    train_test_data_folder = f"{data_dir}/test_train_splits"
    pathogen = "all-viral"

    with open(f"{train_test_data_folder}/X_test_{pathogen}.pkl", "rb") as f:
        data_df = pickle.load(f)

elif args.data == 'test_rf':
    train_test_data_folder = f"{data_dir}/test_train_splits"
    pathogen = "all-viral"

    with open(f"{train_test_data_folder}/X_test_{pathogen}_rf.pkl", "rb") as f:
        data_df = pickle.load(f)

else:
    raise ValueError(
        f"Unknown system prompt: {args.data} (must be one of: all, test, test_rf)")

# Use test data if requested
if args.test_mode:
    data_df = data_df.head(2)
    print(f"Running in test mode with {len(data_df)} rows")


# Define model calls


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    class Chat:
        class Completions:
            @staticmethod
            def create(model, messages, temperature):
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "options": {
                            "temperature": temperature,
                            # "n_probs": 3,
                            # "num_predict": 100,
                        },
                        "stream": False
                    }
                )

                data = response.json()

                # Handle structure from /api/chat
                choice = data.get("message", {})

                return type("OllamaResponse", (), {
                    "choices": [type("Choice", (), {
                        "message": type("Message", (), {
                            "content": choice.get("content"),
                            "thinking": choice.get("thinking"),
                            "tool_calls": choice.get("tool_calls", []),
                            # "logprobs": choice.get("logprobs")
                        })
                    })]
                })

                # full_text = []
                # token_probs = []  # will hold per-token top-k probs

                # for line in response.iter_lines(decode_unicode=True):
                #     if not line:
                #         continue
                #     try:
                #         chunk = json.loads(line)
                #     except json.JSONDecodeError:
                #         continue

                #     # accumulate generated text
                #     msg = chunk.get("message", {})
                #     if (piece := msg.get("content")):
                #         full_text.append(piece)

                #     # collect per-token probabilities (if present on this chunk)
                #     if "completion_probabilities" in chunk:
                #         # each entry: {"content": "tok", "probs": [{"tok_str": "...", "prob": ...}, ...]}
                #         token_probs.extend(chunk["completion_probabilities"])

                #     if chunk.get("done", False):
                #         break

                # content = "".join(full_text)

                # # pdb.set_trace()

                # return type("OllamaResponse", (), {
                #     "choices": [type("Choice", (), {
                #         "message": type("Message", (), {
                #             "content": content,
                #             "thinking": None,
                #             "tool_calls": [],
                #             # expose the collected probs so your parser can read them
                #             "completion_probabilities": token_probs,
                #         })
                #     })]
                # })


# Parse model output


def parse_model_output(response) -> dict:
    message = response.choices[0].message

    thinking = getattr(message, "thinking", None)
    tool_calls = getattr(message, "tool_calls", [])
    # completion_probabilities = getattr(message, "completion_probabilities", None)

    response_text = getattr(message, "content", "") or ""

    # Regex to find start of structured output (Chain of thought)
    pattern = re.compile(r"(?i)^Chain of thought\s*:", re.MULTILINE)
    match = pattern.search(response_text)

    # Initialize extra text containers
    result = {
        "prelude": None,
        "epilogue": None
    }

    if match:
        structured_start = match.start()
        result["prelude"] = response_text[:structured_start].strip()
        structured_block = response_text[structured_start:].strip()
    else:
        # Couldn't find the start — treat the whole thing as unstructured
        structured_block = response_text.strip()

    # pdb.set_trace()

    # Parse structured block
    lines = structured_block.splitlines()
    parsed = {}
    remaining_lines = []

    expected_keys = [
        "chain_of_thought",
        "viral",
        "probability_of_viral",
        "most_likely_pathogen_1",
        "most_likely_pathogen_2",
        "most_likely_pathogen_3",
        "most_likely_pathogen_4",
        "most_likely_pathogen_5",
        "contagious",
        "sequence_sample",
        "sequence_priority"
    ]

    for line in lines:
        if ':' not in line:
            remaining_lines.append(line)
            continue

        key, value = line.split(':', 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()

        if key in expected_keys:
            # Normalize common value types
            if value.lower() in ["unknown"]:
                value = "unknown"
                parsed[key] = value
                continue
            if value.lower() in ["n/a", "na", "none"]:
                value = None
                parsed[key] = value
                continue
            if "probability" in key:
                value = re.sub(r'[^0-9.]', '', value)
                try:
                    value = float(value)
                except ValueError:
                    value = None
            parsed[key] = value
        else:
            remaining_lines.append(line)

    # Fill in any missing expected fields
    for key in expected_keys:
        parsed.setdefault(key, None)

    result.update(parsed)

    # Save any remaining lines after the structured output
    trailing_lines = "\n".join(remaining_lines).strip()
    if trailing_lines:
        result["epilogue"] = trailing_lines

    # Add structured metadata if present
    result["thinking"] = thinking
    result["tool_calls"] = tool_calls
    # result["completion_probabilities"] = completion_probabilities

    return result


# Model call

CLIENT = OllamaClient()


def ask_llm(system_prompt, prompt, combine_prompts):

    if combine_prompts:
        full_prompt = f"{system_prompt}\n\n{prompt}"
        response = CLIENT.Chat.Completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=TEMPERATURE
        )

        # Double-check that we're within token limit
        prompt_tokens = get_token_length(full_prompt)
        if prompt_tokens > TOKEN_LIMIT:
            print(f"WARNING: User prompt size ({prompt_tokens} tokens) exceeds context window token limit ({TOKEN_LIMIT}).") 


    else:
        response = CLIENT.Chat.Completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE
        )
        
        # Double-check that we're within token limit
        prompt_tokens = get_token_length(prompt)
        if prompt_tokens > TOKEN_LIMIT:
            print(f"WARNING: User prompt size ({prompt_tokens} tokens) exceeds context window token limit ({TOKEN_LIMIT}).") 

    # pdb.set_trace()
    return parse_model_output(response)



def get_ollama_usage():
    for proc in psutil.process_iter(['pid', 'name']):
        if "ollama" in proc.info['name'].lower():
            try:
                mem = proc.memory_info().rss / 1024 / 1024
                cpu = proc.cpu_percent(interval=1)
                return mem, cpu, proc.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return None, None, None


# For each run_id, call the model for every row and save results separately
if args.run_once:
    run_ids = ["01"]
else:
    run_ids = ["01", "02", "03"]

os.makedirs(output_dir, exist_ok=True)

for run_id in run_ids:
    start_time = time.time()
    output_filename = f"{MODEL.replace(':', '-')}_run{run_id}_predictions.csv"
    output_csv_path = os.path.join(output_dir, output_filename)
    run_results = []

    idx = 0
    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"{MODEL} (run {run_id}): Processing patients", leave=True):
        row_text = row_to_json(row)

        retrieved_block = None
        if rag_state:
            query_text = row_text
            retrieved_block = retrieve_context(
                query_text, rag_state, top_k=args.rag_top_k, max_ctx_tokens=args.rag_max_tokens
            )

            # print(retrieved_block)
            # if retrieved_block is None or retrieved_block.strip() == "":
            #     print("WARNING: Retrieved context block is empty for this row.")

        if args.patient_json_to_text:
            patient_vignette = generate_patient_data_vignette(eval(row_text.replace('null','None')))
            patient_text = patient_vignette
        else:
            patient_text = row_text

        if knowledge_summary and retrieved_block:
            prompt = f"""
            Use the distilled knowledge base and retrieved context to guide your prediction.

            Knowledge base:
            {knowledge_summary}

            Viral diagnoses (1=pos, 0=neg) of the top {args.rag_top_k} most similar patients:
            {retrieved_block}

            Now, analyze the current patient:
            {patient_text}
            """.strip()

        elif knowledge_summary:
            prompt = f"""
            Use the distilled knowledge base to guide your prediction.

            Knowledge base:
            {knowledge_summary}

            Now, analyze the current patient:
            {patient_text}
            """.strip()

        elif retrieved_block:
            prompt = f"""
            Viral diagnoses (1=pos, 0=neg) of the top {args.rag_top_k} most similar patient:
            {retrieved_block}

            Now, analyze the current patient:
            {patient_text}
            """.strip()

        else:
            prompt = f"Patient data:\n{patient_text}".strip()

        run_results.append(
            ask_llm(SYSTEM_PROMPT, prompt, args.combine_prompts))

        idx += 1
        # Save progress at intervals
        if (idx + 1) % SAVE_INTERVAL == 0 or (idx + 1) == len(data_df):
            results_df = pd.DataFrame(run_results)
            results_df.to_csv(output_csv_path, index=False,
                              encoding="utf-8-sig")
            print(f"Progress saved to {output_csv_path} at row {idx + 1}")

    end_time = time.time()

    # Final save (in case not already saved at the last row)
    results_df = pd.DataFrame(run_results)
    results_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Results saved to {output_csv_path}")

    # Create runtime log file

    runtime_seconds = end_time - start_time
    avg_time_per_patient = runtime_seconds / len(data_df)

    priority_counts = results_df["sequence_priority"].value_counts().to_dict()
    cot_lengths = results_df["chain_of_thought"].dropna().apply(
        lambda x: len(str(x).split()))
    avg_cot_words = cot_lengths.mean()

    # Track Python script
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / 1024 / 1024
    cpu_usage_percent = process.cpu_percent(interval=1)

    # Track Ollama (external) process
    ollama_mem, ollama_cpu, ollama_pid = get_ollama_usage()

    # GPU usage (Ollama process only)
    gpu_usage = "GPU usage not available"
    if ollama_pid:
        try:
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if proc.pid == ollama_pid:
                        mem_used = proc.usedGpuMemory / 1024 / 1024
                        gpu_name = pynvml.nvmlDeviceGetName(handle)
                        gpu_usage = f"GPU {i} ({gpu_name}): {mem_used:.2f} MB used by Ollama (PID {ollama_pid})"
            pynvml.nvmlShutdown()
        except Exception as e:
            gpu_usage = f"Error collecting GPU usage: {e}"

    runtime_log_path = os.path.join(
        output_dir, f"{MODEL.replace(':', '-')}_run{run_id}_runtime_log.txt")
    with open(runtime_log_path, "w") as f:
        if args.knowledge_summary:
            f.write(
                f"Knowledge summary added to user prompt:\n{args.knowledge_summary}\n\n")
        if args.rag_dir:
            f.write(
                f"Path to RAG vectorstore:\n{args.rag_dir}\n\n")
        f.write(f"Total runtime: {runtime_seconds / 3600:.2f} hours\n")
        f.write(
            f"Average time per patient: {avg_time_per_patient:.2f} seconds\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Temperature: {TEMPERATURE}\n\n")

        f.write(f"Script memory usage (RSS): {memory_usage_mb:.2f} MB\n")
        f.write(f"Script CPU usage: {cpu_usage_percent:.2f}%\n")

        if ollama_pid:
            f.write(f"Ollama memory usage: {ollama_mem:.2f} MB\n")
            f.write(f"Ollama CPU usage: {ollama_cpu:.2f}%\n")
            f.write(f"{gpu_usage}\n")
        else:
            f.write("Ollama process not found.\n")

        f.write("\n--- Output Summary ---\n")
        f.write(f"Avg CoT length: {avg_cot_words:.2f} words\n")
        for prio in ["high", "medium", "low", "NA"]:
            f.write(
                f"Sequence priority {prio}: {priority_counts.get(prio, 0)}\n")
