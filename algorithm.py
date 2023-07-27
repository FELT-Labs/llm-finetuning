import os
import tarfile
import json

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer


# Dataset load
dids = json.loads(os.getenv('DIDS', None))
file_paths = []
for did in dids:
    file_paths.append(f'/data/inputs/{did}/0')
dataset = load_dataset('json', data_files=file_paths, split="train")

# Hyper parameters
with open("/data/inputs/algoCustomData.json", "r") as f:
    params = json.load(f)

# Model load
model_name = "ybelkada/falcon-7b-sharded-bf16"

# LoRA adapters settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

# Tokenizer for preprocesing dataset
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


lora_alpha = params.get("lora_alpha", 16)
lora_dropout = params.get("lora_dropout", 0.1)
lora_r = params.get("lora_r", 64)

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)


output_dir = "/data/outputs"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10000 # Ideally don't save at all
logging_steps = 5
learning_rate = params.get("learning_rate", 2e-4)
max_grad_norm = params.get("max_grad_norm", 0.3)
max_steps = params.get("max_steps", 50)
warmup_ratio = params.get("warmup_ratio", 0.03)
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    report_to="none",
    # logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    push_to_hub=False,
)

max_seq_length = 512
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)


for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)


trainer.train()
trainer.save_model("/data/model")

with tarfile.open("/data/outputs/model", "w:gz") as tar:
    tar.add("/data/model", arcname=os.path.basename("/data/model"))