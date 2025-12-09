import modal
from dataclasses import dataclass

# 1. SET UP YOUR ENVIRONMENT
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.54.0",
        "datasets==3.6.0",
        "torch==2.7.0",
        "accelerate==1.9.0",
        "peft==0.16.0",
        "trl==0.19.1",
        "pandas==2.2.0",
        "openpyxl==3.1.2",
    )
)

# Storage for your trained models
model_storage = modal.Volume.from_name("my-models", create_if_missing=True)

# 2. CONFIGURE YOUR TRAINING
@dataclass
class TrainingConfig:
    base_model: str = "microsoft/phi-2"
    bash_dataset: str = "aelhalili/bash-commands-dataset"
    excel_file: str = "enterprise-attack-v18.1.xlsx"
    columns_to_use: list = None
    max_steps: int = 200
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 16
    seed: int = 42
    
    def __post_init__(self):
        if self.columns_to_use is None:
            self.columns_to_use = ["ID", "STIX ID", "name", "description", "url", "tactics", "platforms"]

# 3. TRAINING FUNCTION
@modal.function(
    image=train_image,
    gpu="T4",
    volumes={"/models": model_storage},
    timeout=3600,
)
def train_model(config: TrainingConfig):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    import datasets
    
    print(f"🚀 Loading model: {config.base_model}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    print(f"📦 Loading bash commands dataset: {config.bash_dataset}")
    
    # Load bash commands dataset
    bash_dataset = datasets.load_dataset(config.bash_dataset, split="train")
    
    def format_bash_commands(example):
        prompt = f"### Task: {example['prompt']}\n\n### Command:\n{example['response']}"
        return {"text": prompt}
    
    bash_dataset = bash_dataset.map(format_bash_commands, remove_columns=bash_dataset.column_names)
    
    print(f"📦 Loading Excel file: {config.excel_file}")
    
    # Load Excel file
    import pandas as pd
    df = pd.read_excel(config.excel_file)
    df = df[config.columns_to_use]
    
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
    dataset = dataset.shuffle(seed=config.seed)
    
    # Training settings
    training_args = TrainingArguments(
        output_dir="/models/my-trained-model",
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        logging_steps=10,
        save_steps=50,
        warmup_ratio=0.1,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
    )
    
    print("🏋️ Training...")
    trainer.train()
    
    print("💾 Saving model...")
    model.save_pretrained("/models/my-trained-model")
    tokenizer.save_pretrained("/models/my-trained-model")
    
    model_storage.commit()
    
    print("✅ Training complete!")
    return "/models/my-trained-model"

# 4. INFERENCE FUNCTION
@modal.function(
    image=train_image,
    gpu="T4",
    volumes={"/models": model_storage},
)
def generate_text(prompt: str, model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. AUTO-GENERATE COMMANDS FOR ALL TECHNIQUES
@modal.function(
    image=train_image,
    gpu="T4",
    volumes={"/models": model_storage},
    timeout=7200,
)
def generate_all_commands(excel_file: str, model_path: str, output_file: str = "generated_commands.json"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd
    import json
    import re
    
    print(f"🔄 Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"📂 Loading techniques from {excel_file}")
    df = pd.read_excel(excel_file)
    
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
    with open(output_file, 'w') as f:
        json.dump(all_commands, f, indent=2)
    
    print(f"✅ Generated {len(all_commands)} unique commands from {len(df)} techniques!")
    print(f"💾 Saved to {output_file}")
    
    return output_file

# ==========================================
# NOTEBOOK EXECUTION CELLS
# Run these in separate cells in your Modal notebook
# ==========================================

# CELL 1: Configure and start training
config = TrainingConfig(
    base_model="microsoft/phi-2",
    bash_dataset="aelhalili/bash-commands-dataset",
    excel_file="enterprise-attack-v18.1.xlsx",
    max_steps=200,
)

print("🚀 Starting training...")
model_path = train_model.remote(config)
print(f"✅ Model saved to: {model_path}")

# CELL 2: Test the model
print("\n🧪 Testing the model...")
test_result = generate_text.remote(
    prompt="### Cybersecurity Technique: Process Injection\n\nDescription: Inject malicious code into legitimate process\n\n### Suggested Commands for Detection/Investigation:\n",
    model_path=model_path
)
print(test_result)

# CELL 3: Generate commands for ALL techniques
print("\n🔄 Generating commands for ALL techniques...")
output_file = generate_all_commands.remote(
    excel_file="enterprise-attack-v18.1.xlsx",
    model_path=model_path,
    output_file="mitre_with_commands.json"
)

print(f"\n🎉 DONE! All commands generated and saved to: {output_file}")
