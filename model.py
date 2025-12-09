# STEP 0: Install required packages (RUN ONCE, then comment out)
# Uncomment the section below if packages aren't installed yet

import subprocess
import sys

print("📦 Upgrading pip to latest version...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"])
print("✅ Pip upgraded!\n")

print("📦 Installing required packages:")
packages = [
    "torch==2.8.0",
    "torchvision==0.23.0",
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

# ==========================================
# HUGGINGFACE LOGIN (Modal Secret)
# ==========================================

from huggingface_hub import login

# Get token from Modal secret named "llama-secret"
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

print("🔐 Logged into Hugging Face!\n")

# ==========================================
# CONFIGURATION
# ==========================================

config = {
    'base_model': "meta-llama/Llama-3.1-8B-Instruct",  # Pre-trained model
    'excel_file': "enterprise-attack-v18.1.xlsx",
    
    # Additional datasets to incorporate (add your own here!)
    'additional_datasets': [
        "aelhalili/bash-commands-dataset",  # Bash commands
        "dessertlab/offensive-powershell",  # Powershell
    ],
    
    # Training parameters
    'max_steps': 500,
    'batch_size': 1,
    'gradient_accumulation_steps': 4,
    'learning_rate': 2e-4,
    'lora_r': 16,
    'lora_alpha': 16,
    'seed': 42,
    
    # Output
    'max_commands': 100,
}

# ==========================================
# STEP 1: LOAD PRE-TRAINED MODEL
# ==========================================

print(f"📦 Loading pre-trained model: {config['base_model']}")

model = AutoModelForCausalLM.from_pretrained(
    config['base_model'],
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
tokenizer.pad_token = tokenizer.eos_token

print("✅ Base model loaded!\n")

# ==========================================
# STEP 2: PREPARE ADDITIONAL TRAINING DATA
# ==========================================

print("📊 Loading additional datasets...\n")

all_datasets = []

# Load additional HuggingFace datasets
for dataset_name in config['additional_datasets']:
    print(f"  Loading: {dataset_name}")
    try:
        ds = datasets.load_dataset(dataset_name, split="train")
        
        # Format bash commands dataset
        if "bash" in dataset_name.lower():
            def format_bash(example):
                prompt = f"### Task: {example['prompt']}\n\n### Command:\n{example['response']}"
                return {"text": prompt}
            ds = ds.map(format_bash, remove_columns=ds.column_names)
        
        all_datasets.append(ds)
        print(f"    ✅ Loaded {len(ds)} examples")
    except Exception as e:
        print(f"    ⚠️ Failed to load {dataset_name}: {e}")

# Load MITRE ATT&CK techniques from Excel
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

# Combine all datasets
from datasets import concatenate_datasets
combined_dataset = concatenate_datasets(all_datasets)
combined_dataset = combined_dataset.shuffle(seed=config['seed'])

print(f"\n📊 Total training examples: {len(combined_dataset)}")

# ==========================================
# STEP 3: ADD LORA ADAPTERS & FINE-TUNE
# ==========================================

print("\n🔧 Adding LoRA adapters for efficient fine-tuning...")

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Add LoRA adapters (only trains a small portion of parameters)
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

# Training settings
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
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

print("\n🏋️ Fine-tuning model with additional data...")
trainer.train()

print("\n💾 Saving fine-tuned model...")
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

print("✅ Fine-tuning complete!")
model_path = "./fine-tuned-model"

# ==========================================
# STEP 4: TEST THE FINE-TUNED MODEL
# ==========================================

print("\n🧪 Testing the fine-tuned model...")

# Clear memory and reload
del model
del trainer
import gc
torch.cuda.empty_cache()
gc.collect()

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

test_prompt = "### Task: Create a bash command to detect Process Injection\n\n### Command:\n"

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    temperature=0.7,
    do_sample=True,
)

test_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(test_result)
print("\n" + "="*80 + "\n")

# ==========================================
# STEP 5: GENERATE COMMANDS FOR ALL TECHNIQUES
# ==========================================

print("🔄 Generating commands for ALL MITRE ATT&CK techniques...\n")

# Reload Excel
df = pd.read_excel(config['excel_file'])

all_commands = []
max_commands = config['max_commands']

for idx, row in df.iterrows():
    if len(all_commands) >= max_commands:
        print(f"\n✅ Reached {max_commands} commands! Stopping early.")
        break
        
    print(f"Processing {idx + 1}/{len(df)}: {row['name']} (Commands: {len(all_commands)}/{max_commands})")
    
    # Generate multiple command types for each technique
    command_types = [
        ("detect", f"### Task: Create a bash command to detect {row['name']}\n\n### Command:\n"),
        ("investigate", f"### Task: Create a bash command to investigate {row['name']}\n\n### Command:\n"),
        ("monitor", f"### Task: Create a bash command to monitor for {row['name']}\n\n### Command:\n"),
    ]
    
    for cmd_type, prompt in command_types:
        if len(all_commands) >= max_commands:
            break
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract command
            if "### Command:" in generated:
                command = generated.split("### Command:")[-1].strip()
            else:
                command = generated.strip()
            
            # Take first line as the command
            command = command.split('\n')[0].strip()
            
            # Only add valid commands
            if command and len(command) > 5:
                all_commands.append({
                    "technique_id": row["ID"],
                    "stix_id": row["STIX ID"],
                    "technique_name": row["name"],
                    "command_type": cmd_type,
                    "tactic": row["tactics"],
                    "platform": row["platforms"],
                    "description": row["description"],
                    "url": row["url"],
                    "command": command
                })
        
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            continue

# ==========================================
# STEP 6: SAVE RESULTS
# ==========================================

output_file = "mitre_with_commands.json"
with open(output_file, 'w') as f:
    json.dump(all_commands, f, indent=2)

print(f"\n✅ Generated {len(all_commands)} unique commands!")
print(f"💾 Saved to {output_file}")

# Show sample commands
print("\n📋 Sample commands:")
for i, cmd in enumerate(all_commands[:5]):
    print(f"\n{i+1}. {cmd['technique_name']} ({cmd['command_type']}):")
    print(f"   {cmd['command']}")

print(f"\n🎉 DONE!")