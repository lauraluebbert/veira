import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

configs = yaml.load(open('configs/neurips_config.yaml'), Loader=yaml.FullLoader)

tokenizer = AutoTokenizer.from_pretrained(configs['model']['base_model'])
policy_model = AutoModelForCausalLM.from_pretrained(
    configs['model']['base_model'],
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(
    policy_model,
    configs['model']['lora_repo'],
)

model.eval()

