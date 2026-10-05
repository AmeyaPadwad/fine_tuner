# Reinforcement Learning with Calibrated Decisions (RLCD)

## 1. The Big Picture

The `rlcd` module provides a comprehensive Reinforcement Learning (RL) fine-tuning runtime built specifically for the **Laya model family**. Its goal is to take a base decision model and optimize it to produce highly calibrated confidence scores alongside its predictions.

The pipeline handles the entire end-to-end process autonomously:

1. **Loading Data**: It fetches instruction or conversational datasets from Hugging Face or local storage.
2. **Preprocessing**: It maps raw text and dictionaries into the mathematically structured format (sequences and `[MASK]` tokens) that the Laya architecture requires.
3. **Loading the Model**: It downloads the specified base Laya weights and configuration, dynamically assembling the neural network.
4. **Configuring the Trainer**: It sets up a tailored RL environment with specific hyperparameters for learning rates, exploration noise, and reward strategies.
5. **Training**: It executes Group Relative Policy Optimization (GRPO) to fine-tune the model, updating weights based on how well the model's confidence matches the ground-truth outcomes.
6. **Saving**: It compiles the updated weights, the tokenizer, and the calibrated configurations into a production-ready Laya Agent package.

## 2. How to Run This

**Before running:** Make sure to install all the dependencies from `requirements.txt`.

```bash
pip install -r rlcd/requirements.txt
```

_(Note: To train on a GPU, you must install a GPU-enabled version of PyTorch compatible with your CUDA version! Default installations from `requirements.txt` only install the CPU version.)_

The primary entry point to start the training process is `main.py`. You can run it via the command line:

```bash
python rlcd/main.py
```

_(Note: For distributed multi-GPU training, use `torchrun --nproc_per_node=<GPUs> rlcd/main.py`)_

### Variables in `main.py`

To customize your run, edit the top section of `main.py`:

- `MODEL_NAME_OR_PATH`: The Hugging Face repo ID or local path of the base Laya model (e.g., `"convaiinnovations/laya"`).
- `OUTPUT_DIR`: The directory where the final fine-tuned model and rolling checkpoints will be saved.
- `DATASET`: The Hugging Face dataset identifier (e.g., `"Salesforce/xlam-function-calling-60k"`).
- `DATASET_SUBSET`: An integer to limit the number of training examples (useful for quick tests), or `None` to use the full dataset.
- `DEVICE`: Set to `"cuda"` for GPU acceleration or `"cpu"` for local testing.

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

- **`main.py`**: The execution script that ties all components together, loads the dataset, configures the trainer, and starts the fine-tuning loop.
- **`training.py`**: Contains the core `LayaRLTrainer` class, the `RLTrainingConfig` dataclass, and the mathematical implementation of the GRPO step and loss calculation.
- **`processing.py`**: Handles dataset mapping, sequence tokenization, `[MASK]` token alignment, and the collation of variable-length items into padded GPU tensors.
- **`models.py`**: Manages the downloading, caching, instantiation, and parameter-loading of the base Laya neural network architecture.
- **`helper.py`**: Provides utility functions for Hugging Face Hub authentication, dataset loading, and exporting model checkpoints to disk.
- **`reward_strategies/brier.py`**: Implements the vectorized Brier Score function used to calculate the reward during the RL step.
- **`smoke_tests.py`**: A comprehensive, standalone unit testing suite that verifies the mathematical and structural integrity of every module above without requiring a full training run.

## 5. Data Preprocessing

Before training, raw datasets must be transformed into the precise format Laya expects. The preprocessing pipeline operates in three phases:

1. **Format Mapping**: Functions like `prepare_xlam_for_laya` take raw JSON datasets and extract the `context`, the `decision_type` (choice, score, or noul), the available `options`, and the ground-truth `target`.
2. **Tokenization and Masking (`build_training_item`)**: The text is passed through the tokenizer. The function calculates exactly where the `[MASK]` tokens belong in the sequence to represent the available options and converts the string target into a normalized mathematical probability distribution (e.g., `[0.0, 1.0, 0.0]`).
3. **Collation (`collate_train_batch`)**: Because sentences have different lengths, this function pads the sequences with zeros so they form perfectly rectangular PyTorch tensors, creating boolean masks so the RL loop knows to ignore the padded areas during matrix multiplication.

## 6. The Training Loop

The reinforcement learning engine relies on **Group Relative Policy Optimization (GRPO)**. The loop inside `training.py` executes the following steps for every batch:

1. **Forward Pass**: The model evaluates the padded input tokens and extracts the hidden states precisely at the `[MASK]` marker positions to generate raw prediction logits.
2. **Exploration Noise**: Instead of relying solely on the model's raw prediction, the script clones the logits several times (`group_size`) and injects zero-mean Gaussian noise. This forces the model to explore different probability distributions around its initial guess.
3. **Reward Calculation**: The noisy predictions are evaluated against the ground-truth target using the Brier score. A statistical "advantage" is calculated by normalizing the rewards, identifying which of the noisy guesses performed better than average.
4. **Policy Gradient Loss**: The model's weights are penalized or rewarded based on the advantage score combined with a soft cross-entropy guidance loss, pushing the model toward accurate, calibrated probabili
