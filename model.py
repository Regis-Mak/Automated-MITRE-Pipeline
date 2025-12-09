# Modal Notebook - Direct Execution
# No decorators, no .remote() calls needed!

# STEP 0: Install required packages (RUN ONCE, then comment out)
# Uncomment the section below if packages aren't installed yet


import subprocess
import sys

print("📦 Installing required packages...")
packages = [
    "torch==2.5.1",
    "torchvision==0.20.1",
    "transformers==4.46.0",
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
import re

# ==========================================
# STEP 1: TRAIN THE MODEL
# ==========================================

print("🚀 Starting training...")

# Configuration
config = {
    'base_model': "microsoft/phi-2",
    'bash_dataset': "aelhalili/bash-commands-dataset",
    'excel_file': "enterprise-attack-v18.1.xlsx",
    'max_steps': 200,
    'batch_size': 1,  # Reduced from 4 to 1
    'gradient_accumulation_steps': 4,  # Simulate larger batch
    'learning_rate': 2e-4,
    'lora_r': 16,
    'lora_alpha': 16,
    'seed': 42,
}

print(f"📦 Loading model: {config['base_model']}")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    config['base_model'],
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
tokenizer.pad_token = tokenizer.eos_token

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Add LoRA adapters
lora_config = LoraConfig(
    r=config['lora_r'],
    lora_alpha=config['lora_alpha'],
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

print(f"📦 Loading bash commands dataset: {config['bash_dataset']}")

# Load bash commands dataset
bash_dataset = datasets.load_dataset(config['bash_dataset'], split="train")

def format_bash_commands(example):
    prompt = f"### Task: {example['prompt']}\n\n### Command:\n{example['response']}"
    return {"text": prompt}

bash_dataset = bash_dataset.map(format_bash_commands, remove_columns=bash_dataset.column_names)

print(f"📦 Loading Excel file: {config['excel_file']}")

# Load Excel file
df = pd.read_excel(config['excel_file'])
columns_to_use = ["ID", "STIX ID", "name", "description", "url", "tactics", "platforms"]
df = df[columns_to_use]

print(f"Found {len(df)} techniques in Excel file")

techniques_dataset = datasets.Dataset.from_pandas(df)

def format_technique_prompt(example):
    prompt = f"""### Cybersecurity Technique: {example['name']}

Description: {example['description']}
Tactics: {example['tactics']}
Platforms: {example['platforms']}

### Suggested Commands for Detection/Investigation:
Based on the technique description, relevant commands include:
- Command to detect this activity
- Command to investigate indicators
- Command to analyze affected systems"""
    
    return {"text": prompt}

techniques_dataset = techniques_dataset.map(format_technique_prompt)

# COMBINE BOTH DATASETS
print(f"Combining {len(bash_dataset)} bash examples + {len(techniques_dataset)} technique examples")
from datasets import concatenate_datasets
dataset = concatenate_datasets([bash_dataset, techniques_dataset])
dataset = dataset.shuffle(seed=config['seed'])

# Training settings
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
    train_dataset=dataset,
    formatting_func=lambda x: x["text"],
    max_seq_length=512,
    tokenizer=tokenizer,
)

print("🏋️ Training...")
trainer.train()

print("💾 Saving model...")
model.save_pretrained("./my-trained-model")
tokenizer.save_pretrained("./my-trained-model")

print("✅ Training complete!")
model_path = "./my-trained-model"

# ==========================================
# STEP 2: TEST THE MODEL
# ==========================================

print("\n🧪 Testing the model...")

# Reload the trained model properly
del model  # Free up memory
import torch
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

test_prompt = "### Cybersecurity Technique: Process Injection\n\nDescription: Inject malicious code into legitimate process\n\n### Suggested Commands for Detection/Investigation:\n"

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
)

test_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(test_result)

# ==========================================
# STEP 3: GENERATE COMMANDS FOR ALL TECHNIQUES
# ==========================================

print("\n🔄 Generating commands for ALL techniques...")

# Model is already loaded from step 2

# Reload Excel
df = pd.read_excel(config['excel_file'])

all_commands = []

for idx, row in df.iterrows():
    print(f"Processing {idx + 1}/{len(df)}: {row['name']}")
    
    prompt = f"""### Cybersecurity Technique: {row['name']}

Description: {row['description']}
Tactics: {row['tactics']}
Platforms: {row['platforms']}

### Suggested Commands for Detection/Investigation:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the commands part
    if "### Suggested Commands" in generated:
        commands_text = generated.split("### Suggested Commands for Detection/Investigation:")[-1].strip()
    else:
        commands_text = generated
    
    # Split commands by newlines or bullet points
    command_lines = re.split(r'\n[-•*]\s*|\n', commands_text)
    command_lines = [cmd.strip().lstrip('-•* ') for cmd in command_lines if cmd.strip()]
    
    # Create separate entry for each command
    for command in command_lines:
        if command:  # Only add non-empty commands
            all_commands.append({
                "technique_id": row["ID"],
                "stix_id": row["STIX ID"],
                "technique_name": row["name"],
                "tactic": row["tactics"],
                "platform": row["platforms"],
                "description": row["description"],
                "url": row["url"],
                "command": command
            })

# Save results as JSON
output_file = "mitre_with_commands.json"
with open(output_file, 'w') as f:
    json.dump(all_commands, f, indent=2)

print(f"✅ Generated {len(all_commands)} unique commands from {len(df)} techniques!")
print(f"💾 Saved to {output_file}")
print(f"\n🎉 DONE!")
