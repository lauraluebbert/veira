import yaml
import sys
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from data.dataset import InferenceDataset
from tqdm import tqdm
from utils.utils import extract_prob
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import glob
from peft import get_peft_model, LoraConfig

os.environ["TORCHDYNAMO_DISABLE"] = "1"
import pickle

# ---- Accept config path as CLI argument (default: neurips_config.yaml) ----
config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/neurips_config.yaml'
print(f"Loading config from: {config_path}")
configs = yaml.load(open(config_path), Loader=yaml.FullLoader)

###Step 1: Load model (prefer local .ckpt via load_model, otherwise HF + LoRA repo)
cache_dir = configs['model']['cache_dir']

tokenizer = AutoTokenizer.from_pretrained(configs['model']['base_model'])
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
PAD = tokenizer.pad_token_id

policy_model = AutoModelForCausalLM.from_pretrained(
    configs['model']['base_model'],
    torch_dtype=torch.bfloat16,
    device_map=None,
    cache_dir=configs['model']['cache_dir']
)

lora_config = LoraConfig(
    r=configs['model']['lora_r'],
    lora_alpha=configs['model']['lora_alpha'],
    lora_dropout=configs['model']['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
)

model = get_peft_model(policy_model, lora_config)

# Check for a local .ckpt first, then download from HuggingFace repo
ckpts = sorted(glob.glob(os.path.join(cache_dir, "*.ckpt")))

if ckpts:
    ckpt_path = ckpts[-1]
    print(f"Loading local checkpoint: {ckpt_path}")
else:
    # Download .ckpt from HuggingFace repo
    from huggingface_hub import hf_hub_download, list_repo_files
    repo_id = configs['model']['lora_repo']
    print(f"Downloading checkpoint from HuggingFace: {repo_id}")
    repo_files = list_repo_files(repo_id)
    ckpt_files = [f for f in repo_files if f.endswith('.ckpt')]
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in HuggingFace repo: {repo_id}")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_files[-1], cache_dir=cache_dir)
    print(f"Downloaded checkpoint to: {ckpt_path}")

state_dict = torch.load(ckpt_path, map_location=configs['model']['device'])
# Handle Lightning checkpoint format (keys prefixed with 'model.')
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
    state_dict = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)

model_device = configs['model']['device']
model.eval().to(model_device)

stop_ids = {tokenizer.eos_token_id}
eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id and eot_id >= 0:
    stop_ids.add(eot_id)

###Step 2: Load in eval data in Inference DataLoader

test_dataloader = InferenceDataset(train_test_data_folder = configs['data']['train_test_data_folder'],
                                   pathogen = configs['data']['pathogen'],
                                   sheet_name = configs['data']['sheet_name'],
                                   batch_size_train = configs['data']['batch_size_train'],
                                   batch_size_eval = configs['data']['batch_size_eval'],
                                   num_workers=configs['data']['num_workers'],
                                   data_path = configs['data']['data_path'],
                                   col_meta_path = configs['data']['col_meta_path'],
                                   data_dictionary_path = configs['data']['data_dictionary_path'],
                                   new_data = configs['data'].get('new_data', False),
                                   no_malaria = configs['data'].get('no_malaria', False),
                                   tokenizer=tokenizer).test_dataloader()

###Step 3: Set up results file and see if any results have already been generated

name_result = configs['eval']['eval_name']

prob_preds = []
binary_preds = []
labels = []
stability_scores = []
modified_probability = []

if os.path.exists(name_result):
    pred_prob = []
    pred_binary = []
    label = []
    finished_files = []
    for line in open(name_result, 'r'):
        parts = line.strip().split('\t')
        if len(parts) < 4:
            continue
        id = parts[0]
        prob = parts[1]
        binary = parts[2]
        true_label = parts[3]
        finished_files.append(id)
        pred_prob.append(float(prob))
        pred_binary.append(int(float(binary)))
        label.append(int(true_label))

    if len(set(label)) == len(test_dataloader):
        #Calculate AUC
        auc = roc_auc_score(label, pred_prob)
        accuracy = accuracy_score(label, pred_binary)
        print(f'AUC: {auc}, Accuracy: {accuracy}')
        # import pdb; pdb.set_trace()

    result_file = open(name_result, 'a')
else:
    finished_files = []
    result_file = open(name_result, 'w')  

###Step 4: Inference!

for step, (prompts, answers) in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
        id = answers[0][1]
        keep_idx = [i for i, (_, rec_id) in enumerate(answers) if rec_id not in finished_files]
        if not keep_idx:
            continue

        prompt_enc = tokenizer(
            prompts,
            return_tensors = 'pt',
            padding = True,
            padding_side = 'left',
            truncation = True
        )

        input_ids = prompt_enc["input_ids"][keep_idx].to(model_device, non_blocking=True)
        attention_mask = prompt_enc["attention_mask"][keep_idx].to(model_device, non_blocking=True)
        kept_answers = [answers[i] for i in keep_idx]

        gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=configs['model']['max_new_tokens'],
                do_sample=False,
                eos_token_id=list(stop_ids),
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, 
                output_scores = True      
            )

        explore_generations = gen_out.sequences
        prompt_len = input_ids.shape[1]
        gen_ids  = explore_generations[:, prompt_len:]          # [B*K, L]
        batch_responses = tokenizer.batch_decode(gen_ids, skip_special_tokens = True)

        raw_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens = True)
        raw_outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens = True)
        scores = gen_out.scores
        logits_t_b_v = torch.stack(scores, dim=0)
        logits_b_t_v = logits_t_b_v.permute(1, 0, 2)
        probs_b_t_v = logits_b_t_v.softmax(dim=-1)
        max_probs_b_t = probs_b_t_v.max(dim=-1).values

        threshold = 0.99
        # mask: positions where model was NOT trivially confident
        interesting_mask_b_t = max_probs_b_t < threshold  # [B, T] bool

        # To avoid division by zero if an example has all steps > threshold:
        masked_max = torch.where(
            interesting_mask_b_t,
            max_probs_b_t,
            torch.nan,   # mark "uninteresting" positions as NaN
        )  # [B, T]

        # Mean over only "interesting" positions for each example
        stability_score = torch.nanmean(masked_max, dim=-1)  # [B]
        # If an example had no interesting positions, nanmean will give NaN; you can replace:
        stability_score = torch.nan_to_num(stability_score, nan=1.0)

        for (y_true, rec_id), txt, ques, out, stability in zip(kept_answers, batch_responses, raw_inputs, raw_outputs, stability_score):
            p = extract_prob(txt)
            if p is None:
                p = 0.5
            b = 0
            try:
                b = int(p > 0.5)
            except Exception:
                import pdb
                pdb.set_trace()

            prob_preds.append(p)
            binary_preds.append(b)
            labels.append(int(y_true))
            w = stability.item()
            stability_scores.append(w)
            modified_probability.append(p * w + (1 - w) * 0.9)

            result_file.write(f'{rec_id}\t{p}\t{b}\t{int(y_true)}\t{w}\t{modified_probability[-1]}\t{ques.rstrip()}\t{out.rstrip()}\n')
            finished_files.append(rec_id)

        result_file.flush()

result_file.close()

# Save all preds
pickle.dump(
    {'prob_preds': prob_preds, 'binary_preds': binary_preds, 'labels': labels, 'stability_scores': stability_scores, 'modified_probability': modified_probability},
    open(f'{configs["eval"]["eval_name"]}.pkl', 'wb')
)

# ---- Harmonised summary (matches llm_training eval output) ----
total_examples = len(labels)
total_failed = sum(1 for p in prob_preds if p == 0.5)  # failed parses defaulted to 0.5
failure_rate = total_failed / total_examples if total_examples else 0.0

if total_examples > 0 and len(set([int(x) for x in labels])) > 1:
    clean_probs = [x for x in prob_preds if isinstance(x, float)]
    int_labels = [int(x) for x in labels]
    bin_preds = [int(x) for x in binary_preds]

    auc = roc_auc_score(int_labels, prob_preds)
    acc = accuracy_score(int_labels, bin_preds)
    brier = brier_score_loss(int_labels, prob_preds)
    pos = [p for p, l in zip(prob_preds, int_labels) if l == 1]
    neg = [p for p, l in zip(prob_preds, int_labels) if l == 0]

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total examples : {total_examples}")
    print(f"  Parse failures : {total_failed} ({failure_rate:.2%})")
    print(f"  AUC            : {auc:.4f}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Brier Score    : {brier:.4f}")
    print(f"  Mean P (pos)   : {np.mean(pos):.4f}" if pos else "  Mean P (pos)   : N/A")
    print(f"  Mean P (neg)   : {np.mean(neg):.4f}" if neg else "  Mean P (neg)   : N/A")
    print(f"  Pos-Neg Gap    : {np.mean(pos) - np.mean(neg):.4f}" if (pos and neg) else "  Pos-Neg Gap    : N/A")
    print(f"  Failure Rate   : {failure_rate:.4f}")
    print("=" * 60)

    # Modified AUC with stability scores
    print(f"\n  Modified AUC   : {roc_auc_score(int_labels, modified_probability):.4f}")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        filtered_probs = [mp for mp, ss in zip(modified_probability, stability_scores) if ss >= threshold]
        filtered_labels = [lbl for mp, ss, lbl in zip(modified_probability, stability_scores, labels) if ss >= threshold]
        if len(set(filtered_labels)) > 1:
            print(f"  Filtered AUC (stability >= {threshold}): {roc_auc_score(filtered_labels, filtered_probs):.4f}")
    print("=" * 60 + "\n")
