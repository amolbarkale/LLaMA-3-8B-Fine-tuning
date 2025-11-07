"""QLoRA Training Script using Unsloth"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import yaml
import argparse
import os
import wandb
import time
from dotenv import load_dotenv

load_dotenv()


def formatting_func_hhrlhf(examples, tokenizer):
    """
    Format HH-RLHF 'chosen' conversations for training.
    Converts multi-turn dialogues to Llama-3 chat format.
    """
    texts = []
    for conversation in examples["chosen"]:
        messages = []
        
        # Parse conversation into turns
        turns = conversation.split('\n\n')
        for turn in turns:
            turn = turn.strip()
            if turn.startswith('Human:'):
                content = turn.replace('Human:', '').strip()
                messages.append({"role": "user", "content": content})
            elif turn.startswith('Assistant:'):
                content = turn.replace('Assistant:', '').strip()
                messages.append({"role": "assistant", "content": content})
        
        # Apply chat template
        if messages:
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(formatted_text)
        else:
            texts.append(conversation + tokenizer.eos_token)
    
    return texts


def main(config_path):
    start_time = time.time()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*80)
    print("QLoRA TRAINING WITH UNSLOTH")
    print("="*80 + "\n")

    print("Loading base model and tokenizer...")
    max_seq_length = config.get('max_seq_length', 2048)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model_name'],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Set chat template for Llama-3
    from src.utils.chat_templates import LLAMA3_CHAT_TEMPLATE
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora_r'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("Loading training dataset...")
    dataset = load_dataset(config['dataset_name'], split="train")
    
    num_train = min(config.get('num_train_samples', 10000), len(dataset))
    train_dataset = dataset.shuffle(seed=42).select(range(num_train))
    
    print(f"Training on {len(train_dataset)} samples")

    # Create formatting function with tokenizer
    def format_func(examples):
        return formatting_func_hhrlhf(examples, tokenizer)

    print("Setting up trainer...")
    total_steps = (len(train_dataset) // 
                   (config['per_device_train_batch_size'] * 
                    config['gradient_accumulation_steps'])) * config['num_train_epochs']
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        formatting_func=format_func,
        max_seq_length=max_seq_length,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config['per_device_train_batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            warmup_steps=config.get('warmup_steps', 10),
            num_train_epochs=config['num_train_epochs'],
            learning_rate=config['learning_rate'],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            max_grad_norm=config.get('max_grad_norm', 1.0),
            seed=3407,
            output_dir="./outputs/qlora",
            save_strategy="steps",
            save_steps=max(total_steps // 5, 50),
            save_total_limit=2,
            report_to="wandb",
            logging_first_step=True,
            eval_strategy="no",
        ),
    )
    
    # Initialize W&B
    wandb.init(
        project=config.get('wandb_project', 'responsible-ai-alignment'),
        name=f"qlora-llama3-{num_train}samples-{config['num_train_epochs']}epochs",
        config={
            **config,
            'total_steps': total_steps,
            'effective_batch_size': (config['per_device_train_batch_size'] * 
                                    config['gradient_accumulation_steps'])
        }
    )

    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Epochs: {config['num_train_epochs']}")
    print(f"Batch size: {config['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"LoRA rank: {config['lora_r']}")
    print(f"Total steps: {total_steps}")
    print(f"{'='*80}\n")

    print("Starting training...")
    trainer.train()
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.2f} minutes")

    # Save final model
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving LoRA adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Adapter saved successfully")
    
    # Log metrics
    wandb.log({"training_time_minutes": elapsed_time/60})
    wandb.finish()
    
    print("\n" + "="*80)
    print("QLORA TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    args = parser.parse_args()
    main(args.config)

