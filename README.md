# QLoRA + DPO fine-tuning on Llama-3-8B with Anthropic HH-RLHF dataset.

## Technical Report: Large Language Model Alignment Study

---

## Executive Summary

This study investigates the effectiveness of two prominent parameter-efficient fine-tuning (PEFT) techniques—**QLoRA** (Quantized Low-Rank Adaptation) and **DPO** (Direct Preference Optimization)—for aligning the Llama-3-8B base model with human preferences using the Anthropic HH-RLHF dataset. The research evaluates model performance across three critical dimensions: toxicity reduction, helpfulness enhancement, and harmlessness preservation.

**Key Findings:**
- **QLoRA** achieved the most significant toxicity reduction (-44.3%) while maintaining helpfulness
- **DPO** showed moderate toxicity reduction (-30.9%) but experienced some helpfulness trade-off
- **Training disparities significantly influenced results** - QLoRA received 2× epochs and 2× data compared to DPO
- Both methods successfully improved model safety without catastrophic forgetting
- **Resource constraints** (GPU memory, training time) impacted DPO's training configuration and potentially limited its performance

---

## Methodology

### Dataset
- **Training Data**: Anthropic HH-RLHF dataset (10,000 samples for QLoRA, 5,000 for DPO)
- **Evaluation Data**: Curated subset of 100 challenging prompts covering safety-critical scenarios
- **Evaluation Metrics**:
  - **Toxicity**: Automated scoring using Detoxify BERT model
  - **Helpfulness**: 5-point Likert scale via GPT-4o-mini judge
  - **Harmlessness**: 5-point Likert scale via GPT-4o-mini judge

### Model Architecture

| Component | Specification |
|-----------|---------------|
| **Base Model** | Meta Llama-3-8B (8B parameters) |
| **QLoRA Config** | r=16, α=32, 4-bit quantization |
| **DPO Config** | β=0.1, sigmoid loss, reference model |
| **Hardware** | NVIDIA A100 40GB GPU |
| **Training Time** | QLoRA: ~25 min, DPO: ~45 min |

---

## Technical Implementation

### Two-Stage Alignment Pipeline

1. **Supervised Fine-Tuning (SFT) with QLoRA**
   - Parameter-efficient fine-tuning using 4-bit quantization
   - LoRA rank 16 with alpha 32 for optimal parameter utilization
   - 2 epochs with optimized learning rate scheduling

2. **Preference Optimization with DPO**
   - Direct optimization on preference pairs from HH-RLHF
   - Reference model preservation for stability
   - Single epoch with conservative learning rate (5e-7)

### Evaluation Framework

The evaluation system employs a multi-metric approach:

```python
# Automated toxicity assessment
toxicity_score = detoxify_bert(response)

# Human-aligned quality metrics via LLM judge
helpfulness_score = gpt4o_judge(prompt, response, "helpfulness")
harmlessness_score = gpt4o_judge(prompt, response, "harmlessness")
```

---

## Results & Analysis

### Performance Comparison

| Metric | Baseline | QLoRA | DPO | QLoRA Δ | DPO Δ |
|--------|----------|-------|-----|---------|-------|
| **Avg Toxicity** | 0.0486 | 0.0271 | 0.0336 | **-44.3%** | -30.9% |
| **High Toxicity %** | 3.0% | 1.0% | 2.0% | **-67%** | -33% |
| **Avg Helpfulness** | 2.41 | 2.60 | 2.15 | **+7.9%** | -10.8% |
| **Avg Harmlessness** | 4.35 | 4.45 | 4.31 | +2.3% | -0.9% |

### Model Comparison Visualization

![Model Comparison Results](./results/model_comparison.png)

**Figure 1: Multi-dimensional performance comparison showing toxicity reduction effectiveness, helpfulness preservation, and safety improvements across all three models.**

### Detailed Performance Analysis

#### Toxicity Reduction
- **QLoRA** demonstrated superior toxicity mitigation with 44.3% average reduction
- **High-toxicity instances** (>0.5 score) reduced by 67% with QLoRA vs 33% with DPO
- Both methods successfully moved toxicity distribution toward safer responses

#### Helpfulness Trade-offs
- **QLoRA** improved helpfulness by 7.9% while reducing toxicity
- **DPO** showed a 10.8% decrease in helpfulness, indicating potential over-caution
- The helpfulness-toxicity trade-off suggests different alignment strategies

#### Safety Preservation
- **Harmlessness** scores remained stable across all models (4.3-4.45 range)
- No evidence of catastrophic forgetting or safety degradation
- Both methods maintained core model capabilities

---

## Technical Insights

### QLoRA Effectiveness
The quantized low-rank adaptation proved highly effective for safety alignment:
- **Memory Efficiency**: 4-bit quantization reduced memory footprint by ~75%
- **Training Stability**: Gradient accumulation and warmup prevented optimization issues
- **Parameter Efficiency**: Only 0.1% of parameters modified (LoRA rank 16)

### DPO Characteristics
Direct preference optimization revealed interesting behavioral patterns:
- **Conservative Alignment**: Lower learning rate prevented overfitting but limited gains
- **Reference Model Dependency**: Performance tied to SFT quality
- **Preference Sensitivity**: More responsive to nuanced preference signals

### Comparative Analysis

#### Training Configuration Disparity

**Critical Training Differences:**
- **QLoRA (SFT)**: 2 epochs, 10,000 samples, LR=2.0e-4, effective batch size=16
- **DPO**: 1 epoch, 5,000 samples, LR=5.0e-7, effective batch size=16

*Note: QLoRA received significantly more training exposure (2× epochs, 2× data) compared to DPO due to GPU memory constraints and training time limitations. This disparity likely contributed to QLoRA's superior performance.*

**QLoRA Advantages:**
- Superior toxicity reduction without helpfulness loss
- More extensive training (2 epochs vs DPO's 1 epoch)
- Higher learning rate enabling faster convergence (2.0e-4 vs 5.0e-7)
- Larger dataset exposure (10,000 vs 5,000 samples)
- More stable optimization landscape with gradient accumulation

**DPO Limitations (Current Setup):**
- Conservative learning rate (5.0e-7) limited adaptation speed
- Reduced training data (50% of QLoRA) due to memory constraints
- Single epoch training insufficient for full preference learning
- Reference model dependency amplified training sensitivity

**Technical Configuration Details:**

From `configs/training_config.yaml`:
```yaml
# QLoRA Training Hyperparameters
num_train_epochs: 2
per_device_train_batch_size: 2
gradient_accumulation_steps: 8  # Effective batch size: 16
learning_rate: 2.0e-4
num_train_samples: 10000
```

From `configs/dpo_config.yaml`:
```yaml
# DPO Training Hyperparameters
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # Effective batch size: 16
learning_rate: 5.0e-7
num_train_samples: 5000
beta: 0.1
loss_type: "sigmoid"
```

**Methodological Considerations:**
Given equivalent computational budgets, DPO might demonstrate different performance characteristics with:
- Extended training epochs (2-3 epochs)
- Larger preference datasets (10,000+ pairs)
- Optimized learning rate scheduling
- Multi-stage DPO training pipeline

---

## Usage Instructions

### Quick Start
```bash
# Complete pipeline (3-4 hours)
python run_pipeline.py --include_dpo
```

### Step-by-Step Execution
```bash
# 1. Data preparation
python src/data/prepare_data.py

# 2. Baseline evaluation
python src/evaluation/evaluate_model.py --config configs/training_config.yaml --num_samples 100

# 3. QLoRA training
python src/training/train_qlora.py --config configs/training_config.yaml

# 4. QLoRA evaluation
python src/evaluation/evaluate_model.py --config configs/training_config.yaml --model_path ./models/llama3-qlora-adapter --num_samples 100

# 5. DPO training
python src/training/train_dpo.py --config configs/dpo_config.yaml

# 6. DPO evaluation
python src/evaluation/evaluate_model.py --config configs/dpo_config.yaml --model_path ./models/llama3-dpo-adapter --num_samples 100

# 7. Comprehensive comparison
python src/evaluation/compare_results.py \
    --baseline ./results/evaluation_baseline_100samples.csv \
    --qlora ./results/evaluation_qlora_100samples.csv \
    --dpo ./results/evaluation_dpo_100samples.csv
```

### Monitoring & Visualization
- **GPU Monitoring**: `watch -n 1 nvidia-smi`
- **Training Logs**: [Weights & Biases Dashboard](https://wandb.ai/barkaleamol7/Responsible-AI-Alignment-A100-please-final?nw=nwuserbarkaleamol)
- **Results**: `./results/comparison_metrics.csv`
- **Visualizations**: `./results/model_comparison.png`

---

## Inference Optimization & Performance

### Implemented Optimizations

The evaluation pipeline includes several inference optimizations for efficient deployment:

#### 1. 4-bit Quantization
```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # 4-bit NormalFloat quantization
    bnb_4bit_compute_dtype=torch.bfloat16 # Mixed precision for speed
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,     # 4-bit quantization enabled
    device_map="auto",                   # Automatic device placement
    torch_dtype=torch.bfloat16           # Memory efficient dtype
)
```

#### 2. Adapter Merging for Faster Inference
```python
# Load adapter if provided
if model_path:
    model = PeftModel.from_pretrained(model, model_path)
    print("Merging adapter for faster inference...")
    model = model.merge_and_unload()  # Merging for speed optimization
```

#### 3. Memory & Speed Optimizations
- **Device Mapping**: `device_map="auto"` for optimal GPU placement across available devices
- **Mixed Precision**: `torch.bfloat16` for faster computation while maintaining model quality
- **Efficient Generation**: Proper `pad_token_id` and `max_new_tokens` settings for optimal token generation

#### Performance Benefits
- **Memory Reduction**: ~75% memory savings with 4-bit quantization
- **Inference Speed**: Faster response generation through adapter merging
- **Scalability**: Better GPU utilization with automatic device mapping

---

## Safety Analysis & Red-Teaming

### Current Safety Evaluation

**Implemented Safety Metrics:**
- **Toxicity Detection**: Using Detoxify BERT model for automated toxicity scoring
- **LLM-as-Judge**: GPT-4o-mini evaluation for helpfulness and harmlessness
- **Trade-off Analysis**: Quantitative comparison of safety vs. capability metrics

### Toxicity Measurement Implementation

**Method**: Detoxify BERT (not Perspective API)

```python
# In evaluate_model.py - Toxicity detection implementation
toxicity_model = Detoxify('original', device='cuda' if torch.cuda.is_available() else 'cpu')

# For each response, calculate toxicity score
toxicity_scores = toxicity_model.predict(response)
toxicity_score = toxicity_scores['toxicity']
```

**Why Detoxify instead of Perspective API:**
- **Self-contained**: No external API dependencies or rate limits
- **BERT-based**: Uses pre-trained transformer model for consistent scoring
- **Multi-class**: Detects toxicity, severe_toxicity, obscene, threat, insult, identity_attack
- **Offline processing**: Works without internet connectivity

### Safety vs. Capability Trade-offs

**Analyzed Trade-offs:**
- **QLoRA**: Achieved 44.3% toxicity reduction while improving helpfulness by 7.9%
- **DPO**: 30.9% toxicity reduction but 10.8% helpfulness decrease
- **Training Constraints**: Limited DPO effectiveness due to resource constraints

---

## Model Artifacts

### Available Models
- **QLoRA Adapter**: `./models/llama3-qlora-adapter/` (Best toxicity reduction)
- **DPO Adapter**: `./models/llama3-dpo-adapter/` (Balanced safety alignment)

### Performance Files
- **Baseline Results**: `./results/evaluation_baseline_100samples.csv`
- **QLoRA Results**: `./results/evaluation_qlora_100samples.csv`
- **DPO Results**: `./results/evaluation_dpo_100samples.csv`
- **Comparison Metrics**: `./results/comparison_metrics.csv`

---

## Conclusions & Recommendations

### Key Insights
1. **QLoRA outperforms DPO** for toxicity reduction while preserving helpfulness under current training constraints
2. **Both methods achieve meaningful safety improvements** without catastrophic forgetting
3. **Training configuration significantly impacts results** - QLoRA's 2× epochs and 2× data exposure likely contributed to superior performance
4. **Resource constraints influenced methodology** - DPO used conservative hyperparameters due to GPU memory limitations
5. **Trade-off exists** between toxicity reduction and helpfulness preservation, amplified by training disparities

### Recommendations
- **Current Setup**: Deploy QLoRA for maximum toxicity reduction with maintained performance given existing training constraints
- **Extended Training**: Provide DPO equivalent training exposure (2-3 epochs, 10,000+ samples) for fair comparison
- **Hybrid Approaches**: Consider QLoRA → DPO pipelines leveraging SFT initialization for DPO stability
- **Constitutional AI Integration**: Incorporate Anthropic's Constitutional AI principles by adding explicit safety instructions and self-critique mechanisms to the model prompts
- **Resource Planning**: Allocate additional GPU memory and training time for comprehensive DPO evaluation
- **Future Research**: Investigate multi-turn conversation safety, ensemble methods, and scaled preference optimization

### Future Work
- Extend evaluation to multi-turn conversations
- Investigate ensemble methods combining QLoRA and DPO
- Scale to larger model sizes (30B, 70B parameters)
- Explore alternative preference datasets and optimization objectives

---

## References

- **QLoRA Paper**: [Quantized Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2305.14314)
- **DPO Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- **HH-RLHF Dataset**: [Anthropic's Helpful and Harmless Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- **Llama-3-8B**: [Meta's Llama 3 8B Parameter Model](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

---

**Report**: https://docs.google.com/document/d/1u3ADrxbRu5Vde7ZZxjlX1yVEUUYj7JqT-YTngcdEico/edit?usp=sharing

**Author**: Amol S B
**Version**: 1.0.0
**Last Updated**: November 2025
