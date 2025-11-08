"""DPO Training Script"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import yaml
import argparse
import os
import wandb
import time
from dotenv import load_dotenv
from src.utils.chat_templates import LLAMA3_CHAT_TEMPLATE

load_dotenv()


def format_dpo_data(example, tokenizer):
    """Format HH-RLHF data for DPO training"""
    
    def parse_conversation(text):
        messages = []
        turns = text.split('\n\n')
        for turn in turns:
            turn = turn.strip()
            if turn.startswith('Human:'):
                messages.append({"role": "user", "content": turn.replace('Human:', '').strip()})
            elif turn.startswith('Assistant:'):
                messages.append({"role": "assistant", "content": turn.replace('Assistant:', '').strip()})
        return messages
    
    # Parse chosen and rejected conversations
    chosen_messages = parse_conversation(example["chosen"])
    rejected_messages = parse_conversation(example["rejected"])
    
    # Extract prompt (all messages except last assistant response)
    if chosen_messages and chosen_messages[-1]["role"] == "assistant":
        prompt_messages = chosen_messages[:-1]
        chosen_response = chosen_messages[-1]["content"]
    else:
        prompt_messages = chosen_messages
        chosen_response = ""
    
    if rejected_messages and rejected_messages[-1]["role"] == "assistant":
        rejected_response = rejected_messages[-1]["content"]
    else:
        rejected_response = ""
    
    # Format prompt
    if prompt_messages:
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = ""
    
    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response
    }


def main(config_path):
    start_time = time.time()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*80)
    print("DPO TRAINING")
    print("="*80 + "\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template for Llama-3
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    
    # Load base model with quantization
    print("Loading base model with 4-bit quantization...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Load QLoRA adapter if specified
    if config.get('adapter_path'):
        print(f"Loading QLoRA adapter from {config['adapter_path']}...")
        model = PeftModel.from_pretrained(model, config['adapter_path'])
        model = model.merge_and_unload()
    
    # Configure new LoRA for DPO
    print("Configuring LoRA for DPO...")
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset(config['dataset_name'], split="train")
    
    num_train = min(config.get('num_train_samples', 5000), len(dataset))
    train_dataset = dataset.shuffle(seed=42).select(range(num_train))
    
    # Format dataset for DPO
    print("Formatting dataset for DPO...")
    train_dataset = train_dataset.map(
        lambda x: format_dpo_data(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    
    print(f"Training on {len(train_dataset)} samples")

    # Calculate total steps
    total_steps = (len(train_dataset) // 
                   (config['per_device_train_batch_size'] * 
                    config['gradient_accumulation_steps'])) * config['num_train_epochs']
    
    # Initialize W&B
    wandb.init(
        project=config.get('wandb_project', 'responsible-ai-alignment'),
        name=f"dpo-llama3-{num_train}samples-{config['num_train_epochs']}epochs",
        config={
            **config,
            'total_steps': total_steps,
            'effective_batch_size': (config['per_device_train_batch_size'] * 
                                    config['gradient_accumulation_steps'])
        }
    )

    print(f"\n{'='*80}")
    print("DPO CONFIGURATION")
    print(f"{'='*80}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Epochs: {config['num_train_epochs']}")
    print(f"Batch size: {config['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Beta: {config.get('beta', 0.1)}")
    print(f"Total steps: {total_steps}")
    print(f"{'='*80}\n")

    # Create DPO config
    dpo_config = DPOConfig(
        output_dir="./outputs/dpo",
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_steps=config.get('warmup_steps', 10),
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(total_steps // 5, 50),
        save_total_limit=2,
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=config.get('max_grad_norm', 1.0),
        report_to="wandb",
        logging_first_step=True,
        eval_strategy="no",
        seed=3407,
        beta=config.get('beta', 0.1),
        loss_type=config.get('loss_type', 'sigmoid'),
        max_length=config.get('max_seq_length', 2048),
        max_prompt_length=config.get('max_seq_length', 2048) // 2,
        padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        truncation_mode='keep_end'
    )

    # Initialize DPO Trainer
    print("Initializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference model
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    dpo_trainer.train()
    
    elapsed_time = time.time() - start_time
    print(f"\nDPO training completed in {elapsed_time/60:.2f} minutes")

    # Save final model
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving DPO adapter to {output_dir}...")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("DPO adapter saved successfully")
    
    # Log metrics
    wandb.log({"dpo_training_time_minutes": elapsed_time/60})
    wandb.finish()
    
    print("\n" + "="*80)
    print("DPO TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
    args = parser.parse_args()
    main(args.config)