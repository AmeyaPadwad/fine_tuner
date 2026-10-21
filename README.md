# RLCD Fine-Tuner: Model- & Dataset-Agnostic Decision Model Pipeline

An end-to-end fine-tuning pipeline that takes **any dataset** and **any open-weight System One decision model** as input and fine-tunes it using **Reinforcement Learning with Calibrated Decisions (RLCD)**.

Unlike standard supervised fine-tuning (SFT), which optimizes solely for token-level cross-entropy and often produces overconfident probability estimates, RLCD combines **Group Relative Policy Optimization (GRPO)** with **strictly proper scoring rules** (such as the negative Brier score) and post-training temperature scaling. The result is a decision model optimized for both **high predictive accuracy** and **well-calibrated confidence scores** across discrete choices (`choice`), binary guardrails (`noul`), and continuous ratings (`score`).

For a deep dive into how dataset preprocessing, model wrapping, the RLCD training loop, evaluation, and the repository directory structure work under the hood, see **[`rlcd_README.md`](./rlcd_README.md)**.

---

## Supported Models & Datasets

- **Decision Models (`models.py`)**:
  - **Laya** (`convaiinnovations/laya`)
  - **Von** (`wfzyx/von`)
  - **OpenDecision** (`open-decision-150m`, `Tokz-labs/OpenDecision-Small`, `Tokz-labs/OpenDecision-Large`)
  - **Typical** (`OzLabs/typical-small`)
  - **Decision 1.0** (`vllm-sr/Decision-1.0-Lex-0.6B`)
  - Any local checkpoint directory or compatible Hugging Face model ID.
- **Datasets (`processing.py`)**:
  - Automatically detects dataset schemas (standard context $\to$ target classification/regression, tool/function-calling datasets, contrastive preference pairs like `chosen`/`rejected`, or native decision formats) and maps them into the target model's marker-token sequence format across all three decision primitives (`choice`, `noul`, `score`).

---

## How to Use the Repo

### 1. Installation

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Note: To train on a CUDA GPU, install a CUDA-enabled build of PyTorch matching your system before or after installing `requirements.txt`.)*

Optionally, copy `.env.example` to `.env` and set `HF_TOKEN` if you are downloading gated models or datasets from the Hugging Face Hub:

```bash
cp .env.example .env
```

### 2. Running the Fine-Tuning Pipeline

The primary entry point is `main.py`. You can run it directly with the default configuration at the top of `main.py`:

```bash
python main.py
```

Or override the model, dataset, subset size, column mapping, output directory, hardware device, or hyperparameter sweep via command-line arguments:

```bash
python main.py \
  --model convaiinnovations/laya \
  --dataset Salesforce/xlam-function-calling-60k \
  --subset 600 \
  --output-dir ./outputs/laya_xlam \
  --device cuda
```

Using a custom column mapping via CLI:

```bash
python main.py \
  --model wfzyx/von \
  --dataset mteb/tweet_sentiment_extraction \
  --column-map '{"context": "text", "target": "label_text", "decision_type": "choice"}' \
  --output-dir ./outputs/von_sentiment
```

Running a multi-GPU distributed job with `torchrun`:

```bash
torchrun --nproc_per_node=2 main.py --model OzLabs/typical-small --dataset lmsys/toxic-chat
```

### 3. Running Ablation & Smoke Tests

To run the 3-way ablation study comparing the untouched base model, standard SFT (cross-entropy only, no RLCD), and full RLCD Strategy 1:

```bash
python run_ablation_comparison.py
```

To run the standalone unit and integration smoke test suite verifying all pipeline modules:

```bash
python smoke_tests.py
```

---

## Variables in `main.py`

To customize default runs without passing CLI flags, edit the configuration block at the top of `main.py` (or pass `--model`, `--dataset`, `--subset`, `--column-map`, `--output-dir`, `--device`, `--sweep` on the CLI):

- `MODEL_NAME_OR_PATH`: The Hugging Face repo ID or local path of the decision model (e.g., `"convaiinnovations/laya"`, `"wfzyx/von"`, `"open-decision-150m"`, `"OzLabs/typical-small"`, or `"vllm-sr/Decision-1.0-Lex-0.6B"`).
- `OUTPUT_DIR`: The directory where the final fine-tuned model, evaluation metrics, and rolling checkpoints will be saved.
- `DATASET`: The Hugging Face dataset identifier or local dataset path (e.g., `"Salesforce/xlam-function-calling-60k"` or `"Nexusflow/NexusRaven_API_evaluation"`).
- `DATASET_SUBSET`: An integer to limit the number of dataset examples (useful for fast local development or smoke tests), or `None` to train on the full dataset.
- `COLUMN_MAP`: An optional dictionary to explicitly map dataset columns or override the decision type (e.g., `{"context": "text", "target": "label_text", "decision_type": "choice"}`), or `None` for automatic schema and primitive detection.
- `DEVICE`: Automatically detects `"cuda"` if available and falls back to `"cpu"`, or can be overridden via the `DEVICE` environment variable or `--device` flag.
- `TEST_RATIO`, `CALIB_RATIO`, `CALIB_MAX`, `SPLIT_SEED`: Control the deterministic train, calibration, and held-out test split proportions.

---

## Hyperparameters

The `RLTrainingConfig` dataclass in `training.py` exposes the following parameters to control sequence packing, optimization, exploration, and calibration:

### Sequence Lengths

- `max_len`: Maximum total token sequence length.
- `head_max_len`: Maximum token budget specifically allocated for the question and option strings.
- `max_tokens_per_batch`: The absolute maximum number of tokens allowed per batch.
- `gradient_checkpointing`: Enable gradient checkpointing to save memory at the cost of computation speed.

### RL & Optimization Hyperparameters

- `epochs`: Total number of training passes over the dataset.
- `micro_batch_size`: Number of sequences processed per forward pass.
- `grad_accum_steps`: Number of micro-batches to process before updating the model weights.
- `group_size`: Number of noisy logit perturbations to sample per sequence (for GRPO exploration).
- `lr_encoder`: Learning rate for the pretrained language encoder (kept very small to avoid catastrophic forgetting).
- `lr_head`: Learning rate for the decision head (kept higher for fast adaptation).
- `min_lr`: The minimum learning rate to decay down to at the end of the cosine schedule.
- `weight_decay`: The L2 penalty applied to model weights during the AdamW optimizer step.
- `max_grad_norm`: The clipping threshold for gradient norms to prevent exploding gradients.

### Exploration Noise Schedule (`sigma`)

- `sigma_start`: Initial exploration noise multiplier for logits at epoch 0.
- `sigma_end`: Final exploration noise multiplier for logits at the last epoch.

### Loss Weights

- `ce_weight`: The multiplier weight for the soft cross-entropy guidance loss.
- `act_reg_weight`: Penalty weight applied to the model's head activation summaries.

### Reward Strategy

- `reward_strategy`: The scoring rule used to calculate rewards (set to `"brier"`).

### Calibration Holdout Settings

- `calib_max`: The maximum number of dataset items to withhold from training for post-hoc temperature calibration.
- `calib_ratio`: The percentage of the dataset to withhold for calibration if `calib_max` isn't hit first.
- `calib_seed`: The fixed random seed used to consistently shuffle and select the calibration holdout items.
- `calib_batch_size`: The batch size to use during the post-training calibration inference loop.

### Environment

- `device`: Optionally force a specific device (e.g., `"cuda"`, `"cpu"`); `None` auto-detects.
- `fp16`: Enables Automatic Mixed Precision (AMP) to speed up math on CUDA GPUs.
- `seed`: The random seed used for shuffling items during the training loop.
