# Modal Notebook - Using Pre-trained Model
# No training needed - just load and generate!

# STEP 0: Install required packages (RUN ONCE, then comment out)
# Uncomment the section below if packages aren't installed yet

import subprocess
import sys

print("📦 Installing required packages:")
packages = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "transformers==4.45.2",
    "pandas==2.2.0",
    "openpyxl==3.1.2",
]

for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("✅ Packages installed!\n")

from transformers import AutoModelForCausalLM, AutoTokenizer
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
    'model_name': "meta-llama/Llama-3.1-8B-Instruct",  # Pre-trained model
    'excel_file': "enterprise-attack-v18.1.xlsx",
    'max_commands': 750,
}

# ==========================================
# STEP 1: LOAD PRE-TRAINED MODEL
# ==========================================

print(f"📦 Loading pre-trained model: {config['model_name']}")

model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded!\n")

# ==========================================
# STEP 2: TEST THE MODEL
# ==========================================

print("🧪 Testing the model...")

test_prompt = """You are a security analyst. Create a bash command to detect Process Injection attacks.

Command:"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

test_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(test_result)
print("\n" + "="*80 + "\n")

# ==========================================
# STEP 3: GENERATE COMMANDS FOR ALL TECHNIQUES
# ==========================================

print("🔄 Generating commands for MITRE ATT&CK techniques...")

# Load Excel file
print(f"📦 Loading Excel file: {config['excel_file']}")
df = pd.read_excel(config['excel_file'])
columns_to_use = ["ID", "STIX ID", "name", "description", "url", "tactics", "platforms"]
df = df[columns_to_use]

print(f"Found {len(df)} techniques in Excel file\n")

all_commands = []
max_commands = config['max_commands']

for idx, row in df.iterrows():
    if len(all_commands) >= max_commands:
        print(f"\n✅ Reached {max_commands} commands! Stopping early.")
        break
        
    print(f"Processing {idx + 1}/{len(df)}: {row['name']} (Commands: {len(all_commands)}/{max_commands})")
    
    # Generate multiple command types for each technique
    command_types = [
        ("detect", f"You are a security analyst. Create a bash command to detect {row['name']} on {row['platforms']} systems.\n\nCommand:"),
        ("investigate", f"You are a security analyst. Create a bash command to investigate {row['name']} activity.\n\nCommand:"),
        ("monitor", f"You are a security analyst. Create a bash command to monitor for {row['name']}.\n\nCommand:"),
    ]
    
    for cmd_type, prompt in command_types:
        if len(all_commands) >= max_commands:
            break
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract command after the prompt
            if "Command:" in generated:
                command = generated.split("Command:")[-1].strip()
            else:
                command = generated[len(prompt):].strip()
            
            # Take first non-empty line as the command
            command_lines = [line.strip() for line in command.split('\n') if line.strip()]
            if command_lines:
                command = command_lines[0]
            
            # Only add valid commands
            if command and len(command) > 10 and not command.startswith("You are"):
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
            print(f"  ⚠️ Error generating command: {e}")
            continue

# ==========================================
# STEP 4: SAVE RESULTS
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
