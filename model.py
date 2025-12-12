import time
global_start_time = time.time()

import subprocess
import sys

print("📦 Upgrading pip to latest version...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"])
print("✅ Pip upgraded!\n")

print("📦 Installing required packages:")
packages = [
    "transformers==4.45.2",
    "datasets==3.1.0",
    "accelerate==1.1.1",
    "peft==0.13.2",
    "trl==0.11.4",
    "pandas==2.2.0",
    "openpyxl==3.1.2",
]

for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("✅ Packages installed!\n")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import datasets
import pandas as pd
import json
import torch
import os
import gc

from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

print("Logged into Hugging Face!\n")

config = {
    'base_model': "meta-llama/Meta-Llama-3-70B-Instruct",
    
    'excel_file': "enterprise-attack-v18.1.xlsx",
    'additional_datasets': {
        'bash_commands': "aelhalili/bash-commands-dataset",
        # 'powershell_offensive': "dessertlab/offensive-powershell",
    },
    
    # Training parameters
    'max_steps': 750,
    'batch_size': 1,
    'gradient_accumulation_steps': 4,
    'learning_rate': 2e-4,
    'lora_r': 16,
    'lora_alpha': 16,
    'seed': 42,
    
    # Generation settings
    'max_commands': 700,
}


print("🚀 Starting training...")
training_start_time = time.time()

print(f"📦 Loading model: {config['base_model']}")

model = AutoModelForCausalLM.from_pretrained(
    config['base_model'],
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()

print("Base model loaded!\n")

print("Loading datasets...\n")

all_datasets = []

# Load bash commands dataset
if 'bash_commands' in config['additional_datasets']:
    dataset_name = config['additional_datasets']['bash_commands']
    print(f"  Loading: {dataset_name}")
    bash_dataset = datasets.load_dataset(dataset_name, split="train")
    
    def format_bash_commands(example):
        prompt = f"### Task: {example['prompt']}\n\n### Command:\n{example['response']}"
        return {"text": prompt}
    
    bash_dataset = bash_dataset.map(format_bash_commands, remove_columns=bash_dataset.column_names)
    all_datasets.append(bash_dataset)
    print(f"    ✅ Loaded {len(bash_dataset)} bash examples")

# Load PowerShell dataset
if 'powershell_offensive' in config['additional_datasets']:
    dataset_name = config['additional_datasets']['powershell_offensive']
    print(f"  Loading: {dataset_name}")
    powershell_dataset = datasets.load_dataset(dataset_name, split="train")
    
    def format_powershell_commands(example):
        prompt = f"### Task: {example['n1']}\n\n### Command:\n{example['code']}"
        return {"text": prompt}
    
    powershell_dataset = powershell_dataset.map(format_powershell_commands, remove_columns=powershell_dataset.column_names)
    all_datasets.append(powershell_dataset)
    print(f"    ✅ Loaded {len(powershell_dataset)} PowerShell examples")

# Load MITRE ATT&CK Excel file
print(f"\n  Loading: {config['excel_file']}")
df = pd.read_excel(config['excel_file'])
columns_to_use = ["ID", "STIX ID", "name", "description", "url", "tactics", "platforms"]
df = df[columns_to_use]

print(f"    Found {len(df)} MITRE techniques")

techniques_dataset = datasets.Dataset.from_pandas(df)

def format_technique_prompt(example):
    prompt = f"### Task: Create a command to detect {example['name']} on {example['platforms']}\n\n### Command:\n"
    return {"text": prompt}

techniques_dataset = techniques_dataset.map(format_technique_prompt)
all_datasets.append(techniques_dataset)
print(f"    ✅ Loaded {len(techniques_dataset)} technique examples")

# Combine datasets
from datasets import concatenate_datasets
combined_dataset = concatenate_datasets(all_datasets)
combined_dataset = combined_dataset.shuffle(seed=config['seed'])

print(f"\nTotal training examples: {len(combined_dataset)}")


print("\nAdding LoRA adapters...")

lora_config = LoraConfig(
    r=config['lora_r'],
    lora_alpha=config['lora_alpha'],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

training_args = TrainingArguments(
    output_dir="./my-trained-model",
    per_device_train_batch_size=config['batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    max_steps=config['max_steps'],
    logging_steps=10,
    save_steps=50,
    warmup_ratio=0.1,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
    formatting_func=lambda x: x["text"],
    max_seq_length=512,
    tokenizer=tokenizer,
)

print("\nTraining...")
trainer.train()

print("\nSaving model...")
model.save_pretrained("./my-trained-model")
tokenizer.save_pretrained("./my-trained-model")

training_end_time = time.time()
training_time = training_end_time - training_start_time
training_hours = int(training_time // 3600)
training_minutes = int((training_time % 3600) // 60)
training_seconds = int(training_time % 60)

print("Training complete!")
print(f"⏱️  Training time: {training_hours}h {training_minutes}m {training_seconds}s ({training_time:.2f}s)\n")

model_path = "./my-trained-model"

print("\nTesting the model...")
test_start_time = time.time()

test_prompt = "### Task: Create a bash command to detect Process Injection\n\n### Command:\n"

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

test_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(test_result)

test_end_time = time.time()
test_time = test_end_time - test_start_time
test_minutes = int(test_time // 60)
test_seconds = int(test_time % 60)

print(f"\n⏱️  Testing time: {test_minutes}m {test_seconds}s ({test_time:.2f}s)\n")


print("\nGenerating commands for MITRE techniques...")
generation_start_time = time.time()

df = pd.read_excel(config['excel_file'])

all_commands = []
max_commands = config['max_commands']

# Platform to shell type mapping
PLATFORM_SHELL_MAP = {
    'windows': 'PowerShell',
    'linux': 'bash',
    'macos': 'Unix',
    'mac': 'Unix',
}

def get_platforms_and_shells(platform_string):
    """Extract individual platforms and their corresponding shell types"""
    if pd.isna(platform_string):
        return []
    
    platform_string = str(platform_string).lower()
    platforms_found = []
    
    for platform, shell in PLATFORM_SHELL_MAP.items():
        if platform in platform_string:
            # Avoid duplicates (mac and macos both map to Unix)
            if (platform, shell) not in platforms_found:
                platforms_found.append((platform, shell))
    
    return platforms_found

for idx, row in df.iterrows():
    if len(all_commands) >= max_commands:
        print(f"\n✅ Reached {max_commands} commands! Stopping early.")
        break
    
    platforms_and_shells = get_platforms_and_shells(row['platforms'])
    
    if not platforms_and_shells:
        print(f"Skipping {row['name']} - Platform: {row['platforms']}")
        continue
    
    print(f"Processing {idx + 1}/{len(df)}: {row['name']} (Commands: {len(all_commands)}/{max_commands})")
    print(f"  Platforms detected: {[p[0] for p in platforms_and_shells]}")
    
    for platform, shell_type in platforms_and_shells:
        if len(all_commands) >= max_commands:
            print(f"\n✅ Reached {max_commands} commands! Stopping early.")
            break
        
        description = str(row['description'])[:500]
        prompt = f"### Task: Create a one line {shell_type} command to execute {row['name']}. Ensure that the command is valid, and can be run with no issues. Do NOT assume that there are already files/executables available to be used. Do not have comments in the command you generate.\n\n### Description: {description}\n\n### Command:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Command:" in generated:
            command = generated.split("### Command:")[-1].strip()
        else:
            command = generated.strip()
        
        command = command.split('\n')[0].strip()
        
        if command and len(command) > 5:
            all_commands.append({
                "technique_id": row["ID"],
                "stix_id": row["STIX ID"],
                "technique_name": row["name"],
                "tactic": row["tactics"],
                "platform": platform.capitalize(),
                "all_platforms": row["platforms"],
                "shell_type": shell_type,
                "description": row["description"],
                "url": row["url"],
                "command": command
            })
            print(f"  ✅ Generated {shell_type} command for {platform}")

output_file = "mitre_with_commands.json"
with open(output_file, 'w') as f:
    json.dump(all_commands, f, indent=2)

generation_end_time = time.time()
generation_time = generation_end_time - generation_start_time
generation_hours = int(generation_time // 3600)
generation_minutes = int((generation_time % 3600) // 60)
generation_seconds = int(generation_time % 60)

print(f"\n✅ Generated {len(all_commands)} unique commands!")
print(f"💾 Saved to {output_file}")
print(f"⏱️  Generation time: {generation_hours}h {generation_minutes}m {generation_seconds}s ({generation_time:.2f}s)\n")