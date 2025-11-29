import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from data.dataset import InferenceDataset
from tqdm import tqdm
from utils.utils import extract_prob
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import pickle

configs = yaml.load(open('configs/neurips_config.yaml'), Loader=yaml.FullLoader)

###Step 1: Load in model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(configs['model']['base_model'])
stop_ids = {tokenizer.eos_token_id}
eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id and eot_id >= 0:
    stop_ids.add(eot_id)  

policy_model = AutoModelForCausalLM.from_pretrained(
    configs['model']['base_model'],
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(
    policy_model,
    configs['model']['lora_repo'],
)
model_device = configs['model']['device']
model.eval().to(model_device)

###Step 2: Load in eval data in Inference DataLoader

test_dataloader = InferenceDataset(configs['data']['train_test_data_folder']).test_dataloader()

###Step 3: Set up results file and see if any results have already been generated

name_result = configs['eval']['eval_name']

prob_preds = []
binary_preds = []
labels = []

if os.path.exists(name_result):
    pred_prob = []
    pred_binary = []
    label = []
    finished_files = []
    for line in open(name_result, 'r'):
        id, prob, binary, true_label = line.split('\t')
        finished_files.append(id)
        pred_prob.append(float(prob))
        pred_binary.append(int(float(binary)))
        label.append(int(true_label))
    
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

        gen_out = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=configs['model']['max_new_tokens'],
                do_sample=False,
                eos_token_id=list(stop_ids),
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,       
            )

        explore_generations = gen_out.sequences
        prompt_len = input_ids.shape[1]
        gen_ids  = explore_generations[:, prompt_len:]          # [B*K, L]
        batch_responses = tokenizer.batch_decode(gen_ids, skip_special_tokens = True)

        for (y_true, rec_id), txt in zip(kept_answers, batch_responses):
            p = extract_prob(txt)
            b = int(p > 0.5)

            prob_preds.append(p)
            binary_preds.append(b)
            labels.append(int(y_true))

            result_file.write(f'{rec_id}\t{p}\t{b}\t{int(y_true)}\n')
            finished_files.append(rec_id)

        result_file.flush()

result_file.close()

# Save all preds
pickle.dump(
    {'prob_preds': prob_preds, 'binary_preds': binary_preds, 'labels': labels},
    open(f'{configs['eval']['eval_name']}.pkl', 'wb')
)

if len(labels) > 0 and len(set([int(x) for x in labels])) > 1:
    clean_probs = [x for x in prob_preds if isinstance(x, float)]
    if len(clean_probs) == len(prob_preds):
        print("AUC:", roc_auc_score(labels, prob_preds))
    print("Accuracy:", accuracy_score(labels, [int(x) for x in binary_preds]))



