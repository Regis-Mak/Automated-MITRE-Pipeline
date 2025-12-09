import modal
from dataclasses import dataclass

# 1. SET UP YOUR ENVIRONMENT
app = modal.App("my-model-trainer")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.54.0",
        "datasets==3.6.0",
        "torch==2.7.0",
        "accelerate==1.9.0",
        "peft==0.16.0",
        "trl==0.19.1",
        "pandas==2.2.0",  # For reading Excel
        "openpyxl==3.1.2",  # Excel support
    )
)

# Storage for your trained models
model_storage = modal.Volume.from_name("my-models", create_if_missing=True)

# 2. CONFIGURE YOUR TRAINING
@dataclass
class TrainingConfig:
    # What model to start with
    base_model: str = "microsoft/phi-2"  # Small, fast model (2.7B params)
    
    # Dataset 1: Bash commands from HuggingFace
    bash_dataset: str = "your-username/linux-command-automation"  # UPDATE THIS with the exact dataset name
    
    # Dataset 2: Your Excel file with techniques
    excel_file: str = "my_data.xlsx"
    columns_to_use: list = None
    
    # Training settings
    max_steps: int = 200  # More steps for two datasets
    batch_size: int = 4
    learning_rate: float = 2e-4
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    
    def __post_init__(self):
        if self.columns_to_use is None:
            self.columns_to_use = ["ID", "STIX ID", "name", "description", "url", "tactics", "platforms"]

# 3. TRAINING FUNCTION
@app.function(
    image=train_image,
    gpu="T4",  # Cheapest GPU option
    volumes={"/models": model_storage},
    timeout=3600,  # 1 hour
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
    
    # Add LoRA adapters (trains only small % of parameters)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # Adjust for your model
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    print(f"📦 Loading bash commands dataset: {config.bash_dataset}")
    
    # Load bash commands dataset from HuggingFace
    bash_dataset = datasets.load_dataset(config.bash_dataset, split="train")
    
    # Format bash dataset (adjust based on your dataset's column names)
    def format_bash_commands(example):
        # Your dataset has "prompt" and "response" columns - perfect!
        prompt = f"### Task: {example['prompt']}\n\n### Command:\n{example['response']}"
        return {"text": prompt}
    
    bash_dataset = bash_dataset.map(format_bash_commands, remove_columns=bash_dataset.column_names)
    
    print(f"📦 Loading Excel file: {config.excel_file}")
    
    # Load Excel file with cybersecurity techniques
    import pandas as pd
    df = pd.read_excel(config.excel_file)
    df = df[config.columns_to_use]
    
    print(f"Found {len(df)} techniques in Excel file")
    
    # Convert to dataset format
    techniques_dataset = datasets.Dataset.from_pandas(df)
    
    # Format your techniques data
    def format_technique_prompt(example):
        # The model will learn to suggest commands based on technique description
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
    
    # COMBINE BOTH DATASETS!
    print(f"Combining {len(bash_dataset)} bash examples + {len(techniques_dataset)} technique examples")
    from datasets import concatenate_datasets
    dataset = concatenate_datasets([bash_dataset, techniques_dataset])
    
    # Shuffle so bash and techniques are mixed
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
        fp16=True,  # Faster training
        optim="adamw_torch",
        report_to="none",
    )
    
    # Create trainer
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
@app.function(
    image=train_image,
    gpu="T4",
    volumes={"/models": model_storage},
)
def generate_text(prompt: str, model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load your trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. RUN IT!
@app.local_entrypoint()
def main():
    # Configure your training
    config = TrainingConfig(
        base_model="microsoft/phi-2",
        bash_dataset="aelhalili/bash-commands-dataset",  # UPDATE with exact HuggingFace dataset name
        excel_file="enterprise-attack-v18.1.xlsx",
        max_steps=200,
    )
    
    print("Starting training with bash commands + cybersecurity techniques...")
    model_path = train_model.remote(config)
    
    print(f"\n✅ Model trained and saved to: {model_path}")
    
    # Test it
    print("\n🧪 Testing your model:")
    result = generate_text.remote(
        prompt="### Cybersecurity Technique: Process Injection\n\nDescription: Inject malicious code into legitimate process\n\n### Suggested Commands for Detection/Investigation:\n",
        model_path=model_path
    )
    print(result)

# TO RUN:
# 1. Install Modal: pip install modal
# 2. Setup: modal setup
# 3. Put your Excel file in the same folder
# 4. Update bash_dataset name (line 152) with your HuggingFace dataset
# 5. Check the column names in your bash dataset and update format_bash_commands (lines 72-80)
# 6. Run: modal run train.py