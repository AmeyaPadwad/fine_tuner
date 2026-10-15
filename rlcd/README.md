# Reinforcement Learning with Calibrated Decisions (RLCD)

## 1. The Big Picture

The `rlcd` module provides a comprehensive Reinforcement Learning (RL) fine-tuning runtime for **open-weight System One decision models** (including **Laya**, **Von**, **OpenDecision**, **Typical**, and **Decision 1.0**). Its goal is to take a base decision model and optimize it to produce highly calibrated confidence scores alongside its predictions.

The pipeline handles the entire end-to-end process autonomously:

1. **Loading Data**: It fetches instruction or conversational datasets from Hugging Face or local storage.
2. **Preprocessing**: It maps raw text and dictionaries into the mathematically structured format (sequences and `[MASK]` tokens) that the Laya architecture requires, and splits them into train, calibration, and test sets.
3. **Loading the Model**: It downloads the specified base model weights and configuration, dynamically assembling the neural network.
4. **Configuring the Trainer**: It sets up a tailored RL environment with specific hyperparameters for learning rates, exploration noise, and reward strategies.
5. **Training & Evaluation**: It evaluates pre-training baseline metrics, executes Group Relative Policy Optimization (GRPO) to fine-tune the model, fits post-training temperature calibration, and evaluates post-training performance on the held-out test set.
6. **Saving**: It compiles the updated weights, the tokenizer, the calibrated configurations, and evaluation metrics into a production-ready Laya Agent package.

## 2. How to Run This

**Before running:** Make sure to install all the dependencies from `requirements.txt`.

```bash
pip install -r rlcd/requirements.txt
```

_(Note: To train on a GPU, you must install a GPU-enabled version of PyTorch compatible with your CUDA version! Default installations from `requirements.txt` only install the CPU version.)_

The primary entry point to start the training process is `main.py`. You can run it via the command line using its default top-of-file variables or pass CLI overrides:

```bash
python rlcd/main.py
# Or override model, dataset, subset, column_map, and output_dir via CLI:
python rlcd/main.py --model vllm-sr/Decision-1.0-Lex-0.6B --dataset Nexusflow/NexusRaven_API_evaluation --subset 500 --output-dir ./outputs/decision1_nexusraven
```

_(Note: For distributed multi-GPU training, use `torchrun --nproc_per_node=<GPUs> rlcd/main.py`)_

To run the ablation study comparing the base model, standard SFT (without RLCD), and RLCD fine-tuning:

```bash
python rlcd/run_ablation_comparison.py
```

### Variables in `main.py`

To customize your run, edit the top section of `main.py` (or pass `--model`, `--dataset`, `--subset`, `--column-map`, `--output-dir`, `--sweep`):

- `MODEL_NAME_OR_PATH`: The Hugging Face repo ID or local path of the decision model (e.g., `"convaiinnovations/laya"`, `"wfzyx/von"`, `"open-decision-150m"`, `"OzLabs/typical-small"`, or `"vllm-sr/Decision-1.0-Lex-0.6B"`).
- `OUTPUT_DIR`: The directory where the final fine-tuned model and rolling checkpoints will be saved.
- `DATASET`: The Hugging Face dataset identifier (e.g., `"Salesforce/xlam-function-calling-60k"` or `"Nexusflow/NexusRaven_API_evaluation"`).
- `DATASET_SUBSET`: An integer to limit the number of training examples (useful for quick tests), or `None` to use the full dataset.
- `COLUMN_MAP`: An optional dictionary to explicitly map dataset columns or override the decision type (e.g., `{"context": "text", "target": "label_text", "decision_type": "choice"}`), or `None` for automatic schema detection.
- `DEVICE`: Set to `"cuda"` for GPU acceleration or `"cpu"` for local testing.
- `TEST_RATIO`, `CALIB_RATIO`, `CALIB_MAX`, `SPLIT_SEED`: Control the train, calibration, and held-out test split proportions.

## 3. Hyperparameters

The `RLTrainingConfig` dataclass exposes the following parameters to control the training loop:

Sequence lengths

- `max_len`: Maximum total token sequence length.
- `head_max_len`: Maximum token budget specifically allocated for the question and option strings.
- `max_tokens_per_batch`: The absolute maximum number of tokens allowed per batch.
- `gradient_checkpointing`: Enable gradient checkpointing to save memory at the cost of computation speed.

RL & Optimization Hyperparameters

- `epochs`: Total number of training passes over the dataset.
- `micro_batch_size`: Number of sequences processed per forward pass.
- `grad_accum_steps`: Number of micro-batches to process before updating the model weights.
- `group_size`: Number of noisy logit perturbations to sample per sequence (for GRPO exploration).
- `lr_encoder`: Learning rate for the pretrained language encoder (kept very small to avoid catastrophic forgetting).
- `lr_head`: Learning rate for the new decision head (kept higher for fast adaptation).
- `min_lr`: The minimum learning rate to decay down to at the end of the cosine schedule.
- `weight_decay`: The L2 penalty applied to model weights during the AdamW optimizer step.
- `max_grad_norm`: The clipping threshold for gradient norms to prevent exploding gradients.

Exploration Noise Schedule (sigma)

- `sigma_start`: Initial exploration noise multiplier for logits at epoch 0.
- `sigma_end`: Final exploration noise multiplier for logits at the last epoch.

Loss Weights

- `ce_weight`: The multiplier weight for the soft cross-entropy guidance loss.
- `act_reg_weight`: Penalty weight applied to the model's head activation summaries.

Reward Strategy

- `reward_strategy`: The scoring rule used to calculate rewards (set to `"brier"`).

Calibration Holdout Settings

- `calib_max`: The maximum number of dataset items to withhold from training for post-hoc temperature calibration.
- `calib_ratio`: The percentage of the dataset to withhold for calibration if `calib_max` isn't hit first.
- `calib_seed`: The fixed random seed used to consistently shuffle and select the calibration holdout items.
- `calib_batch_size`: The batch size to use during the post-training calibration inference loop.

Environment

- `device`: Optionally force a specific device (e.g. `"cuda"`, `"cpu"`); `None` auto-detects.
- `fp16`: Enables Automatic Mixed Precision (AMP) to speed up math on CUDA GPUs.
- `seed`: The random seed used for shuffling items during the training loop.

## 4. Directory Structure

- **`main.py`**: The execution script that ties all components together, loads the dataset, configures the trainer, and starts the fine-tuning loop (and optional hyperparameter sweep).
- **`training.py`**: Contains the core `RLTrainer` class, the `RLTrainingConfig` dataclass, GRPO step and loss calculation, held-out evaluation (`evaluate`), and hyperparameter search (`hyperparameter_sweep`).
- **`processing.py`**: Model- and dataset-agnostic preprocessing module that handles automatic schema and primitive detection (`auto_prepare_dataset`), optional user `column_map` overrides, built-in sequence tokenization and `[MASK]`/marker alignment (`build_sequence`, `render_options`, `prepare_dataset_items`), train/calibration/test splitting (`split_dataset_items`), and collation into padded GPU tensors.
- **`models.py`**: Provides dedicated loader classes (`Laya`, `Von`, `OpenDecision`, `Typical`, `Decision1`), a unified `DecisionModel` router, and `MarkerDecisionModelWrapper` to download, cache, and adapt any supported decision model into a unified RLCD forward interface.
- **`helper.py`**: Provides utility functions for Hugging Face Hub authentication, dataset loading, and exporting model checkpoints to disk.
- **`reward_strategies/brier.py`**: Implements the vectorized Brier Score function used to calculate the reward during the RL step.
- **`run_ablation_comparison.py`**: Runs a controlled 3-way ablation study comparing the untouched base model, standard SFT (cross-entropy only, no RLCD), and RLCD fine-tuning, saving metrics to `ablation_no_rlcd_results.json`.
- **`rlcd_kaggle_benchmark.ipynb`**: Ready-to-run Kaggle Notebook (2x T4 GPU) benchmarking `Decision 1.0` and `Typical` across all four target datasets (`Nexusflow/NexusRaven_API_evaluation`, `mteb/tweet_sentiment_extraction`, `Anthropic/hh-rlhf`, and `lmsys/toxic-chat`).
- **`smoke_tests.py`**: A comprehensive, standalone unit testing suite that verifies the mathematical and structural integrity of every module above without requiring a full training run.

## 5. Data Preprocessing (Dataset- & Model-Agnostic)

Before training, raw datasets are transformed into the standardized `[MASK]`/marker sequence format required by decision models. The preprocessing pipeline in `processing.py` operates in four phases:

1. **Dataset-Agnostic Schema & Primitive Resolution (`auto_prepare_dataset`)**:
   - **How it works**: `resolve_schema_and_columns` and `infer_decision_primitive` inspect a sample of rows and Hugging Face `features` metadata to classify the dataset into one of four structural modes: `"native_decision"`, `"rollout"`, `"contrastive_pair"` (e.g., `chosen`/`rejected` preference pairs in `Anthropic/hh-rlhf`), or `"standard"` (context $\to$ target datasets). It then automatically infers the decision primitive—**`choice`** (extracting per-row tool/function options or collecting global class labels when no options column exists), **`noul`** (boolean/binary targets), or **`score`** (continuous/ordinal ratings mapped to a `(min, max)` scale).
   - **Optional User Overrides (`column_map`)**: Users can optionally pass `COLUMN_MAP` to explicitly specify any subset of `context` (single column or list of columns), `target`, `options` (column name or static list), `chosen`/`rejected`, `decision_type`, or `scale`; any omitted fields fall back to automatic detection.
2. **Model-Agnostic Tokenization and Marker Packing (`build_training_item` / `prepare_dataset_items`)**: Built-in `render_options` and `build_sequence` pack the instructions, candidate options, and context into a single sequence. If the loaded model specifies a custom `marker_token_id` via `models.py`, that token ID is placed before each option; otherwise, it falls back to `tokenizer.mask_token_id` (`[MASK]`).
3. **Dataset Splitting (`split_dataset_items`)**: The tokenized items are deterministically partitioned into non-overlapping **Train**, **Calibration**, and **Test** splits.
4. **Collation (`collate_train_batch`)**: Because sentences have different lengths, this function pads the sequences with zeros so they form perfectly rectangular PyTorch tensors, creating boolean masks so the RL loop knows to ignore the padded areas during matrix multiplication.

## 6. The Training Loop

The reinforcement learning engine relies on **Group Relative Policy Optimization (GRPO)**. The loop inside `training.py` executes the following steps for every batch:

1. **Forward Pass**: The model evaluates the padded input tokens and extracts the hidden states precisely at the `[MASK]` marker positions to generate raw prediction logits.
2. **Exploration Noise**: Instead of relying solely on the model's raw prediction, the script clones the logits several times (`group_size`) and injects zero-mean Gaussian noise. This forces the model to explore different probability distributions around its initial guess.
3. **Reward Calculation**: The noisy predictions are evaluated against the ground-truth target using the Brier score. A statistical "advantage" is calculated by normalizing the rewards, identifying which of the noisy guesses performed better than average.
4. **Policy Gradient Loss**: The model's weights are penalized or rewarded based on the advantage score combined with a soft cross-entropy guidance loss, pushing the model toward accurate, calibrated probabilities.

## 7. Evaluation, Calibration & Ablation Study

- **Pre- and Post-Training Evaluation (`RLTrainer.evaluate`)**: Before and after fine-tuning, the model is evaluated on the held-out test split using **Accuracy**, **Precision**, **Recall**, **F1**, **Mean Brier Score**, and 10-bin **Expected Calibration Error (ECE)**. Results and improvement deltas are saved to `test_evaluation_metrics.json`.
- **Temperature Calibration**: Post-training, the held-out calibration split is used to fit optimal temperature scaling parameters per decision type via L-BFGS before exporting the final model package.
- **Hyperparameter Sweep (`RLTrainer.hyperparameter_sweep`)**: Supports evaluating a grid of candidate hyperparameters (`run_sweep=True` in `main.py`) and ranking trials by a combined Brier and ECE score.
- **Ablation Study (`run_ablation_comparison.py`)**: Benchmarks the untouched base model against standard SFT (pure cross-entropy) and RLCD Strategy 1 on a 150-example slice to isolate the calibration and accuracy gains of RLCD.
