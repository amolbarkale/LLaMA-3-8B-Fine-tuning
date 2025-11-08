"""Compare evaluation results across different models"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

sns.set_style("whitegrid")


def main(baseline_file, qlora_file, dpo_file=None):
    print("\n" + "="*80)
    print("MODEL COMPARISON ANALYSIS")
    print("="*80 + "\n")
    
    # Load results
    baseline_df = pd.read_csv(baseline_file)
    qlora_df = pd.read_csv(qlora_file)
    
    has_dpo = dpo_file is not None and os.path.exists(dpo_file)
    if has_dpo:
        dpo_df = pd.read_csv(dpo_file)
    
    # Calculate metrics
    models = ['Baseline', 'QLoRA']
    if has_dpo:
        models.append('DPO')
    
    metrics = {
        'Model': models,
        'Avg Toxicity': [
            baseline_df['toxicity_score'].mean(),
            qlora_df['toxicity_score'].mean(),
        ],
        'Avg Helpfulness': [
            baseline_df['helpfulness_score'].mean(),
            qlora_df['helpfulness_score'].mean(),
        ],
        'Avg Harmlessness': [
            baseline_df['harmlessness_score'].mean(),
            qlora_df['harmlessness_score'].mean(),
        ],
        'High Toxicity %': [
            (baseline_df['toxicity_score'] > 0.5).mean() * 100,
            (qlora_df['toxicity_score'] > 0.5).mean() * 100,
        ]
    }
    
    if has_dpo:
        metrics['Avg Toxicity'].append(dpo_df['toxicity_score'].mean())
        metrics['Avg Helpfulness'].append(dpo_df['helpfulness_score'].mean())
        metrics['Avg Harmlessness'].append(dpo_df['harmlessness_score'].mean())
        metrics['High Toxicity %'].append((dpo_df['toxicity_score'] > 0.5).mean() * 100)
    
    comparison_df = pd.DataFrame(metrics)
    
    # Add improvement columns
    comparison_df['Toxicity Δ'] = comparison_df['Avg Toxicity'] - comparison_df['Avg Toxicity'].iloc[0]
    comparison_df['Helpfulness Δ'] = comparison_df['Avg Helpfulness'] - comparison_df['Avg Helpfulness'].iloc[0]
    comparison_df['Harmlessness Δ'] = comparison_df['Avg Harmlessness'] - comparison_df['Avg Harmlessness'].iloc[0]
    
    print(comparison_df.to_string(index=False))
    print("\n")
    
    # Create visualizations
    num_models = len(models)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison: Baseline vs QLoRA' + (' vs DPO' if has_dpo else ''), 
                 fontsize=16, fontweight='bold')
    
    colors = ['#e74c3c', '#27ae60', '#3498db'][:num_models]
    
    # Toxicity comparison
    axes[0, 0].bar(models, comparison_df['Avg Toxicity'], color=colors)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Average Toxicity (Lower is Better)')
    axes[0, 0].set_ylim(0, max(comparison_df['Avg Toxicity']) * 1.2)
    for i, v in enumerate(comparison_df['Avg Toxicity']):
        axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Helpfulness comparison
    axes[0, 1].bar(models, comparison_df['Avg Helpfulness'], color=colors)
    axes[0, 1].set_ylabel('Score (out of 5)')
    axes[0, 1].set_title('Average Helpfulness (Higher is Better)')
    axes[0, 1].set_ylim(0, 5)
    for i, v in enumerate(comparison_df['Avg Helpfulness']):
        axes[0, 1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Harmlessness comparison
    axes[1, 0].bar(models, comparison_df['Avg Harmlessness'], color=colors)
    axes[1, 0].set_ylabel('Score (out of 5)')
    axes[1, 0].set_title('Average Harmlessness (Higher is Better)')
    axes[1, 0].set_ylim(0, 5)
    for i, v in enumerate(comparison_df['Avg Harmlessness']):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Toxicity distribution
    if has_dpo:
        axes[1, 1].hist([baseline_df['toxicity_score'], qlora_df['toxicity_score'], 
                        dpo_df['toxicity_score']], 
                       bins=20, label=models, alpha=0.6, color=colors)
    else:
        axes[1, 1].hist([baseline_df['toxicity_score'], qlora_df['toxicity_score']], 
                       bins=20, label=models, alpha=0.7, color=colors)
    axes[1, 1].set_xlabel('Toxicity Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Toxicity Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    os.makedirs("./results", exist_ok=True)
    viz_file = './results/model_comparison.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {viz_file}")
    
    # Save comparison metrics
    metrics_file = './results/comparison_metrics.csv'
    comparison_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")
    
    # Side-by-side examples
    print("\n" + "="*80)
    print("SIDE-BY-SIDE RESPONSE EXAMPLES (First 3)")
    print("="*80 + "\n")
    
    for i in range(min(3, len(baseline_df))):
        print(f"{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")
        print(f"\nPROMPT:\n{baseline_df.iloc[i]['prompt_history'][:300]}...\n")
        
        print(f"BASELINE RESPONSE:")
        print(f"{baseline_df.iloc[i]['response']}")
        print(f"[Toxicity: {baseline_df.iloc[i]['toxicity_score']:.4f}, "
              f"Helpfulness: {baseline_df.iloc[i]['helpfulness_score']}, "
              f"Harmlessness: {baseline_df.iloc[i]['harmlessness_score']}]\n")
        
        print(f"QLORA RESPONSE:")
        print(f"{qlora_df.iloc[i]['response']}")
        print(f"[Toxicity: {qlora_df.iloc[i]['toxicity_score']:.4f}, "
              f"Helpfulness: {qlora_df.iloc[i]['helpfulness_score']}, "
              f"Harmlessness: {qlora_df.iloc[i]['harmlessness_score']}]\n")
        
        if has_dpo:
            print(f"DPO RESPONSE:")
            print(f"{dpo_df.iloc[i]['response']}")
            print(f"[Toxicity: {dpo_df.iloc[i]['toxicity_score']:.4f}, "
                  f"Helpfulness: {dpo_df.iloc[i]['helpfulness_score']}, "
                  f"Harmlessness: {dpo_df.iloc[i]['harmlessness_score']}]\n")
        
        print()
    
    # Analysis Summary
    print("="*80)
    print("ANALYSIS SUMMARY")
    print("="*80 + "\n")
    
    toxicity_improvement_qlora = (baseline_df['toxicity_score'].mean() - 
                                  qlora_df['toxicity_score'].mean())
    helpfulness_change_qlora = (qlora_df['helpfulness_score'].mean() - 
                                baseline_df['helpfulness_score'].mean())
    
    print(f"QLoRA Impact:")
    print(f"  Toxicity: {'REDUCED' if toxicity_improvement_qlora > 0 else 'INCREASED'} by {abs(toxicity_improvement_qlora):.4f}")
    print(f"  Helpfulness: {'IMPROVED' if helpfulness_change_qlora > 0 else 'DECREASED'} by {abs(helpfulness_change_qlora):.2f}")
    
    if has_dpo:
        toxicity_improvement_dpo = (baseline_df['toxicity_score'].mean() - 
                                    dpo_df['toxicity_score'].mean())
        helpfulness_change_dpo = (dpo_df['helpfulness_score'].mean() - 
                                  baseline_df['helpfulness_score'].mean())
        
        print(f"\nDPO Impact:")
        print(f"  Toxicity: {'REDUCED' if toxicity_improvement_dpo > 0 else 'INCREASED'} by {abs(toxicity_improvement_dpo):.4f}")
        print(f"  Helpfulness: {'IMPROVED' if helpfulness_change_dpo > 0 else 'DECREASED'} by {abs(helpfulness_change_dpo):.2f}")
        
        print(f"\nBest Model: ", end="")
        best_idx = comparison_df['Avg Toxicity'].idxmin()
        print(f"{comparison_df.iloc[best_idx]['Model']} (Lowest toxicity)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Baseline results CSV")
    parser.add_argument("--qlora", type=str, required=True, help="QLoRA results CSV")
    parser.add_argument("--dpo", type=str, default=None, help="DPO results CSV (optional)")
    args = parser.parse_args()
    
    main(args.baseline, args.qlora, args.dpo)

