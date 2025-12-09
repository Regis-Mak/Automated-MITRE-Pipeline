import modal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

with train_image.imports():
    import unsloth
    import datasets
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

model_cache_volume = modal.Volume.from_name(
    "bash-model-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "bash-checkpoints", create_if_missing=True
)
dataset_volume = modal.Volume.from_name(
    "bash-datasets", create_if_missing=True
)

@dataclass
class BashTrainingConfig:
    model_name: str
    dataset_name: str = "aelhalili/bash-commands-dataset"
    max_seq_length: int = 512
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    
    # Training config
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 200
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10
    
    seed: int = 42
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.experiment_name = f"bash-{model_short}-{timestamp}"

@modal.function(
    image=train_image,
    gpu="L40S",
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=2 * 60 * 60,  # 2 hours
    retries=modal.Retries(max_retries=1),
)
def train_bash_model(config: BashTrainingConfig):
    """Train a model to generate bash commands"""
    
    print(f"🚀 Starting training: {config.experiment_name}")
    print(f"📦 Model: {config.model_name}")
    
    # Load model with LoRA
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    
    # Load and format dataset
    print(f"Loading dataset: {config.dataset_name}")
    dataset = datasets.load_dataset(config.dataset_name, split="train")
    
    # Format as chat messages
    def format_bash_instruction(example):
        messages = [
            {"role": "system", "content": "You are a helpful bash command generator. Convert natural language descriptions into bash commands. Only output the command, nothing else."},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    
    dataset = dataset.map(format_bash_instruction, remove_columns=dataset.column_names)
    
    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=config.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"📊 Train examples: {len(train_dataset)}")
    print(f"📊 Eval examples: {len(eval_dataset)}")
    
    # Training arguments
    checkpoint_path = Path("/checkpoints") / config.experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_ratio=0.1,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        do_eval=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        output_dir=str(checkpoint_path),
        seed=config.seed,
        report_to="none",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )
    
    # Train
    print("🏋️ Training...")
    trainer.train()
    
    # Save final model
    print("💾 Saving model...")
    final_model_path = checkpoint_path / "final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Commit to volume
    checkpoint_volume.commit()
    
    print(f"Training complete! Model saved to: {final_model_path}")
    return config.experiment_name

config = BashTrainingConfig(
    model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    max_steps=200,
    batch_size=4,
    learning_rate=2e-4,
)

print(f"🎯 Training configuration:")
print(f"   Model: {config.model_name}")
print(f"   Steps: {config.max_steps}")
print(f"   Batch size: {config.batch_size}")

# Run training
experiment_name = train_bash_model.remote(config)
print(f"\n🎉 Training complete!")
print(f"📁 Experiment: {experiment_name}")

# Save this for later!
trained_model_path = f"/checkpoints/{experiment_name}/final_model"
print(f"📍 Model path: {trained_model_path}")


inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.54.0",
        "torch==2.7.0",
        "unsloth[cu128-torch270]==2025.7.8",
        "hf-transfer==0.1.9",
        "pandas==2.3.3",
        "pyarrow==21.0.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

with inference_image.imports():
    import torch
    from unsloth import FastLanguageModel

@modal.function(
    image=inference_image,
    gpu="T4",
    volumes={"/checkpoints": checkpoint_volume},
)
def generate_command(prompt: str, model_path: str, temperature: float = 0.1):
    """Generate a bash command from natural language"""
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Generate
    messages = [
        {
            "role": "system",
            "content": "You are a helpful bash command generator. Convert natural language descriptions into bash commands. Only output the command, nothing else.",
        },
        {"role": "user", "content": prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    outputs = model.generate(
        inputs,
        max_new_tokens=128,
        temperature=temperature,
        top_p=0.9,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the command
    if "assistant" in full_text.lower():
        parts = full_text.split("assistant")
        command = parts[-1].strip()
    else:
        command = full_text.split("\n")[-1].strip()
    
    return command

test_prompts = [
    "list all python files in current directory",
    "find files larger than 100MB",
    "count lines in file.txt",
    "search for TODO in all files",
    "show disk usage sorted by size",
]

print("🧪 Testing your trained model...\n")
for prompt in test_prompts:
    command = generate_command.remote(prompt, trained_model_path)
    print(f"{prompt}")
    print(f"{command}\n")

@modal.function(
    image=inference_image,
    gpu="T4",
    volumes={
        "/checkpoints": checkpoint_volume,
        "/datasets": dataset_volume,
    },
    timeout=3600,
)
def generate_synthetic_dataset(
    num_examples: int,
    model_path: str,
    output_file: str,
):
    """Generate synthetic bash command dataset"""
    import pandas as pd
    
    # Load model
    print(f"📦 Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Template prompts
    prompt_templates = [
        "List all {filetype} files in current directory",
        "Find all {filetype} files recursively",
        "Count number of {filetype} files",
        "Search for '{keyword}' in all {filetype} files",
        "Delete all {filetype} files in current directory",
        "Copy all {filetype} files to {destination}",
        "Find {filetype} files larger than {size}",
        "Find {filetype} files modified in last {time}",
        "Show disk usage sorted by size",
        "Kill process on port {port}",
        "Count lines in {filename}",
        "Show first {n} lines of {filename}",
        "Replace '{old}' with '{new}' in {filename}",
        "Download file from {url}",
        "Extract {archive}",
        "Create tar archive of {directory}",
    ]
    
    # Variables for substitution
    variables = {
        "filetype": ["python", "javascript", "text", "log", "json", "csv", "md"],
        "keyword": ["TODO", "FIXME", "error", "warning", "function"],
        "destination": ["backup", "archive", "output", "~/Documents"],
        "size": ["100MB", "1GB", "10KB", "50MB"],
        "time": ["7 days", "24 hours", "1 week", "30 days"],
        "port": ["8080", "3000", "5000", "80"],
        "filename": ["file.txt", "data.csv", "log.txt", "config.json"],
        "n": ["10", "20", "50", "100"],
        "old": ["old_text", "deprecated", "v1"],
        "new": ["new_text", "current", "v2"],
        "url": ["https://example.com/file.zip", "https://api.example.com/data"],
        "archive": ["archive.tar.gz", "backup.zip", "data.tar"],
        "directory": ["logs", "backups", "data", "src"],
    }
    
    # Generate prompts with random substitutions
    prompts = []
    for _ in range(num_examples):
        template = random.choice(prompt_templates)
        prompt = template
        
        # Substitute variables
        for var_name, var_options in variables.items():
            if f"{{{var_name}}}" in prompt:
                prompt = prompt.replace(f"{{{var_name}}}", random.choice(var_options))
        
        prompts.append(prompt)
    
    # Generate commands
    print(f"🔄 Generating {num_examples} bash commands...")
    results = []
    
    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            print(f"  Progress: {i}/{num_examples}")
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful bash command generator. Convert natural language descriptions into bash commands. Only output the command, nothing else.",
            },
            {"role": "user", "content": prompt},
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        
        outputs = model.generate(
            inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract command
        if "assistant" in full_text.lower():
            parts = full_text.split("assistant")
            command = parts[-1].strip()
        else:
            command = full_text.split("\n")[-1].strip()
        
        results.append({
            "prompt": prompt,
            "command": command,
            "model": model_path,
            "temperature": 0.7,
            "generated_at": datetime.now().isoformat(),
        })
    
    # Save as parquet
    df = pd.DataFrame(results)
    print(f"Saving {len(df)} examples to {output_file}")
    df.to_parquet(output_file, index=False, engine='pyarrow')
    
    dataset_volume.commit()
    
    print(f"Dataset saved!")
    print(f"\nFirst 3 examples:")
    print(df.head(3).to_string())
    
    return output_file

timestamp = time.strftime("%Y%m%d-%H%M%S")
output_file = f"/datasets/bash_commands_{timestamp}.parquet"

result_path = generate_synthetic_dataset.remote(
    num_examples=100,
    model_path=trained_model_path,
    output_file=output_file,
)

print(f"\nSynthetic dataset generated!")
print(f"@modal.function(
    volumes={"/checkpoints": checkpoint_volume},
)
def list_checkpoints():
    """List available model checkpoints"""
    import os
    checkpoints = []
    for item in os.listdir("/checkpoints"):
        path = os.path.join("/checkpoints", item)
        if os.path.isdir(path):
            checkpoints.append(item)
    return checkpoints

checkpoints = list_checkpoints.remote()
print("Available model checkpoints:")
for i, cp in enumerate(checkpoints, 1):
    print(f"   {i}. {cp}")Saved to: {result_path}")

your_prompts = [
    "compress all logs into a zip file",
    "find python files modified today",
    "show me the 10 largest files",
    "kill all processes using port 8080",
    "download a file and extract it",
]

print("Interactive Bash Command Generator")
print("=" * 60)

for prompt in your_prompts:
    command = generate_command.remote(
        prompt=prompt,
        model_path=trained_model_path,
        temperature=0.1
    )
    print(f"\n💭 {prompt}")
    print(f"⚡ {command}")