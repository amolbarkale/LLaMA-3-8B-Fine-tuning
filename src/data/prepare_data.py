"""Prepare evaluation dataset from HH-RLHF"""

import json
import argparse
from datasets import load_dataset
import os


def prepare_evaluation_set(num_samples=100, output_file="./data/evaluation_set.jsonl"):
    """
    Prepare a fixed evaluation set from HH-RLHF test split
    
    Args:
        num_samples: Number of samples to extract
        output_file: Output JSONL file path
    """
    print(f"Loading HH-RLHF test dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    
    # Sample evaluation examples
    eval_samples = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save as JSONL
    with open(output_file, 'w') as f:
        for example in eval_samples:
            json.dump({
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }, f)
            f.write('\n')
    
    print(f"✅ Saved {len(eval_samples)} evaluation samples to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="./data/evaluation_set.jsonl")
    args = parser.parse_args()
    
    prepare_evaluation_set(args.num_samples, args.output_file)

