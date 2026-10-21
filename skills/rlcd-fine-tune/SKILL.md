---
name: rlcd-fine-tune
description: Fine-tune and calibrate open-weight System One decision models (Laya, Von, Decision 1.0, Typical, OpenDecision) on any dataset using Reinforcement Learning with Calibrated Decisions (RLCD). Use whenever the user asks to fine-tune, train, calibrate, or benchmark a decision model on a dataset via the `rlcd-fine-tune` CLI or `rlcd` Python SDK.
---

# RLCD Decision Model Fine-Tuning Skill (`rlcd-fine-tune`)

## 1. Overview & When to Use
Use this skill whenever you need to fine-tune, calibrate, or evaluate an open-weight System One decision model on a Hugging Face or local dataset. The pipeline optimizes both task accuracy and probability calibration (Expected Calibration Error / ECE and Brier Score) across three decision primitives:
- **`choice`**: Multi-class classification or tool/function selection.
- **`noul`**: Binary / boolean decisions (`True`/`False`, `yes`/`no`, safety guardrails, or `chosen`/`rejected` preference pairs).
- **`score`**: Ordinal rating decisions over a bounded numeric scale `[min, max]`.

---

## 2. Supported Models (`--model` / `model_name`)

| Model Family | Identifier (`--model`) | Architecture Notes |
| :--- | :--- | :--- |
| **Laya** *(default)* | `convaiinnovations/laya` | 421M ModernBERT-large bidirectional encoder with native Laya decision head. |
| **Von** | `wfzyx/von` | 395M ModernBERT-large bidirectional encoder with option marker vectors. |
| **Decision 1.0** | `vllm-sr/Decision-1.0-Lex-0.6B` | 600M Vela/ModernBERT bidirectional decision encoder. |
| **OpenDecision** | `open-decision-150m` *(or `Tokz-labs/OpenDecision-Small`)* | 150M ModernBERT-base (or DeBERTa-v2) encoder wrapped with `MarkerDecisionModelWrapper`. |
| **Typical** | `OzLabs/typical-small` | Qwen3-1.7B tap-truncated backbone wrapped with `MarkerDecisionModelWrapper`. |

---

## 3. Running via the Single CLI Command: `rlcd-fine-tune`

The entire pipeline is invoked through a single CLI command: **`rlcd-fine-tune`**. Every hyperparameter and variable is optional; omitted flags automatically use the pipeline defaults.

### Quick Examples

```bash
# 1. Run with all defaults (Laya on Salesforce/xlam-function-calling-60k)
rlcd-fine-tune

# 2. Fine-tune Von on a 5,000-row slice of Tweet Sentiment with custom hyperparameters
rlcd-fine-tune \
  --model wfzyx/von \
  --dataset mteb/tweet_sentiment_extraction \
  --subset 5000 \
  --epochs 3 \
  --micro-batch-size 8 \
  --lr-head 1e-4 \
  --lr-encoder 2.5e-5 \
  --output-dir ./outputs/von_sentiment

# 3. Fine-tune Decision 1.0 on LMSYS Toxic-Chat with an explicit column_map for 'score'
rlcd-fine-tune \
  --model vllm-sr/Decision-1.0-Lex-0.6B \
  --dataset lmsys/toxic-chat \
  --subset 2000 \
  --column-map '{"context": ["user_input", "model_output"], "target": "openai_moderation", "decision_type": "score", "scale": [1, 5]}' \
  --output-dir ./outputs/decision1_toxic_score

# 4. Multi-GPU Distributed Data Parallel (DDP) via torchrun
torchrun --nproc_per_node=2 main.py --model wfzyx/von --dataset Anthropic/hh-rlhf --subset 10000 --output-dir ./outputs/von_hhrlhf
```

---

## 4. Complete CLI & Python SDK Reference

In Python, call **`rlcd.fine_tune(...)`** (or `from main import fine_tune`):

```python
import rlcd

result = rlcd.fine_tune(
    model_name="vllm-sr/Decision-1.0-Lex-0.6B",
    dataset_name="Nexusflow/NexusRaven_API_evaluation",
    dataset_subset=318,
    epochs=2,
    micro_batch_size=4,
    lr_head=1e-4,
    lr_encoder=2.5e-5,
    output_dir="./outputs/decision1_nexusraven",
)
print(result["metrics"])
```

### All Passable Arguments & Defaults

| Python SDK Keyword (`rlcd.fine_tune`) | CLI Flag (`rlcd-fine-tune`) | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `--model` | `"convaiinnovations/laya"` | Hugging Face model ID or local checkpoint directory. |
| `output_dir` | `--output-dir` | `"./laya_finetuned_xlam_rl"` | Directory to export the final calibrated model & metrics. |
| `cache_dir` | `--cache-dir` | `"./.cache"` | Directory to cache downloaded base checkpoints. |
| `dataset_name` | `--dataset` | `"Salesforce/xlam-function-calling-60k"` | Hugging Face dataset identifier. |
| `dataset_subset` | `--subset` | `None` *(full dataset)* | Integer row cap for fast testing/benchmarking, or `None`. |
| `column_map` | `--column-map` | `None` *(auto-detect)* | Dict (or JSON string on CLI) overriding `context`, `target`, `options`, `chosen`/`rejected`, `decision_type`, or `scale`. |
| `max_len` | `--max-len` | `1024` | Maximum total token sequence length. |
| `head_max_len` | `--head-max-len` | `256` | Maximum token budget for the question header and options. |
| `max_tokens_per_batch` | `--max-tokens-per-batch` | `4096` | Maximum token budget per batch stored in agent config. |
| `gradient_checkpointing` | `--gradient-checkpointing` / `--no-gradient-checkpointing` | `True` | Enable/disable non-reentrant gradient checkpointing. |
| `epochs` | `--epochs` | `2` | Number of RLCD training epochs. |
| `micro_batch_size` | `--micro-batch-size` | `4` | Sequences processed per forward pass. |
| `grad_accum_steps` | `--grad-accum-steps` | `2` | Micro-batches accumulated before each optimizer step. |
| `group_size` | `--group-size` | `4` | Number of GRPO Gaussian logit perturbations per sample ($G$). |
| `lr_head` | `--lr-head` | `1.0e-4` | Learning rate for the decision head. |
| `lr_encoder` | `--lr-encoder` | `2.5e-5` | Learning rate for the pretrained encoder backbone. |
| `min_lr` | `--min-lr` | `1.0e-6` | Minimum learning rate floor at the end of cosine annealing. |
| `weight_decay` | `--weight-decay` | `0.01` | AdamW L2 weight decay penalty. |
| `max_grad_norm` | `--max-grad-norm` | `1.0` | Gradient norm clipping threshold. |
| `sigma_start` | `--sigma-start` | `0.4` | Initial GRPO logit exploration noise at epoch 0. |
| `sigma_end` | `--sigma-end` | `0.1` | Final GRPO logit exploration noise at the final epoch. |
| `ce_weight` | `--ce-weight` | `1.0` | Weight for the soft cross-entropy guidance loss. |
| `act_reg_weight` | `--act-reg-weight` | `0.0` | Penalty weight on head activation summaries. |
| `reward_strategy` | `--reward-strategy` | `"brier"` | Proper scoring rule strategy (`"brier"`). |
| `test_ratio` | `--test-ratio` | `0.20` | Fraction of items held out for pre/post-training evaluation. |
| `calib_ratio` | `--calib-ratio` | `0.10` | Fraction of items held out for post-hoc calibration. |
| `calib_max` | `--calib-max` | `400` | Maximum items held out for post-hoc calibration. |
| `calib_seed` | `--calib-seed` | `20260922` | Random seed for calibration holdout selection. |
| `calib_batch_size` | `--calib-batch-size` | `16` | Batch size during post-hoc calibration inference. |
| `seed` | `--seed` | `42` | Random seed for dataset splitting and training shuffle. |
| `device` | `--device` | `"cuda"` *(if available)* | Target compute device (`"cuda"` or `"cpu"`). |
| `fp16` | `--fp16` / `--no-fp16` | `True` | Enable/disable CUDA Automatic Mixed Precision (AMP). |
| `run_sweep` | `--sweep` | `False` | Run hyperparameter search across candidate configurations. |
| `param_grid` | `--param-grid` | `None` | List of candidate hyperparameter dicts (or JSON string on CLI) when `--sweep` is enabled. |

---

## 5. Verification Checklist After Running
1. Confirm `<output_dir>/model.safetensors`, `<output_dir>/rl_agent_config.json`, and `<output_dir>/test_evaluation_metrics.json` were created.
2. Inspect `<output_dir>/test_evaluation_metrics.json` and report:
   - `baseline` vs. `final` **Accuracy**, **Precision**, **Recall**, and **F1**
   - `baseline` vs. `final` **Mean Brier Score** (closer to `0.0` is better)
   - `baseline` vs. `final` **Expected Calibration Error / ECE** (lower is better)
   - `post_hoc_calibration` summary (selected calibration strategy and fitted temperatures)
