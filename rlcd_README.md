# Reinforcement Learning with Calibrated Decisions (RLCD) — Architecture & Internals

This document explains how the RLCD pipeline works end to end—covering **dataset processing**, **model loading and wrapping**, **RLCD training with proper scoring rules**, **evaluation & calibration**, and the **repository directory structure**.

For high-level usage instructions, CLI flags, `main.py` configuration variables, and `RLTrainingConfig` hyperparameters, see the root **[`README.md`](./README.md)**.

---

## 1. Directory Structure

All core pipeline modules live at the root of the repository for direct execution and flat imports:

```text
.
├── main.py                        # End-to-end CLI entry point (dataset load -> preprocess -> train -> evaluate)
├── processing.py                  # Dataset- & model-agnostic schema resolution, tokenization, marker alignment & splitting
├── models.py                      # Unified DecisionModel router, model loaders (Laya, Von, OpenDecision, Typical, Decision1) & wrapper
├── training.py                    # RLTrainer, RLTrainingConfig, GRPO step, temperature calibration, and evaluation
├── helper.py                      # Hugging Face Hub auth, multi-config dataset loading, and checkpoint/final model export
├── reward_strategies/             # Proper scoring rule reward implementations
│   ├── brier.py                   # Vectorized negative Brier score reward (choice, noul, score)
│   ├── RLCD Reward Strategies.md  # Mathematical notes on proper scoring rules for RLCD
│   └── smoke_tests.py             # Unit tests for the Brier reward function
├── run_ablation_comparison.py     # Controlled 3-way ablation study (Untouched Base vs. Standard SFT vs. Full RLCD)
├── smoke_tests.py                 # Standalone unit & integration test suite covering every pipeline module
├── data/
│   └── samples/                   # Sample JSONL datasets (e.g., clinical_triage_dataset.jsonl)
├── docs/                          # Empirical validation reports, cross-model benchmark tables, and test plans
├── results/                       # Saved benchmark CSVs and ablation JSON outputs
├── requirements.txt               # Python package dependencies
└── .env.example                   # Template for environment variables (HF_TOKEN, DEFAULT_CACHE_DIR, etc.)
```

### Module Responsibilities

- **`main.py`**: Orchestrates the pipeline lifecycle—authenticates with Hugging Face Hub, initializes `RLTrainer` (and any distributed process group), loads and subsets the dataset safely across ranks, runs `auto_prepare_dataset` and `prepare_dataset_items`, splits data into Train / Calibration / Test sets, and launches either `trainer.train()` or `trainer.hyperparameter_sweep()`.
- **`processing.py`**: Converts arbitrary raw datasets into padded PyTorch tensors with `[MASK]` or model-specific option markers.
- **`models.py`**: Abstracts architectural differences across bidirectional encoders (ModernBERT/BERT) and causal backbones (Qwen/Lex), exposing a single forward signature for training and inference.
- **`training.py`**: Implements the GRPO reinforcement learning loop, Gaussian logit exploration, advantage normalization, soft cross-entropy blending, L-BFGS post-hoc temperature scaling, and held-out test evaluation.
- **`helper.py`**: Manages Hugging Face Hub authentication (`hf_login`), dataset fetching with automatic config selection for multi-config repositories (`load_dataset`), rolling checkpoint management (`save_rolling_checkpoint`), and final package export (`save_final_model`).

---

## 2. How the Pipeline Works

### Stage 1: Dataset Processing (`helper.py` & `processing.py`)

Decision models do not generate free-form text token by token; instead, they pack the prompt, candidate options, and context into a single input sequence containing special **marker tokens** (such as `[MASK]`, `<<choice>>`, `<<score>>`, or `<<noul>>`) and read out option logits directly from the hidden states at those marker positions.

`processing.py` automates this transformation in four steps:

1. **Schema & Decision Primitive Resolution (`auto_prepare_dataset`)**:
   - `resolve_schema_and_columns` and `infer_decision_primitive` inspect dataset features and sample rows to classify the dataset into one of four structural modes:
     - **`native_decision`**: Already formatted with `qtype`/`decision_type`, `state`/`context`, `options`, and `target`.
     - **`rollout`**: Agent trajectory logs containing step-level observations, candidate actions, and outcome rewards.
     - **`contrastive_pair`**: Preference datasets containing `chosen` and `rejected` responses (e.g., `Anthropic/hh-rlhf`).
     - **`standard`**: General classification, regression, or tool-calling datasets mapping context columns to a target column.
   - It then resolves the task into one of three decision primitives:
     - **`choice`**: Multi-class or binary selection over K candidate options (extracting per-row tool/function schemas when present, or building a global label set across the dataset).
     - **`noul`**: Binary / boolean judgement (`Yes` vs. `No`, plus an abstention/uncertainty slot).
     - **`score`**: Continuous or ordinal rating normalized to a `[0, 1]` target across a `(min, max)` scale.
   - **Manual Overrides (`column_map`)**: Users can override any subset of `context`, `target`, `options`, `chosen`/`rejected`, `decision_type`, or `scale`; unspecified fields remain auto-detected.

2. **Tokenization & Marker Sequence Packing (`build_training_item` / `prepare_dataset_items`)**:
   - `render_options` formats the candidate options and prefixes each option with the active model's marker token (`[MASK]` for Laya/Von/OpenDecision, or custom decision markers for Typical/Decision 1.0).
   - `build_sequence` allocates the token budget between the question/options header (`head_max_len`) and the context body (`max_len`), recording the exact token index (`marker_pos`) of every option marker.

3. **Deterministic Train / Calibration / Test Splitting (`split_dataset_items`)**:
   - Partitioning is seeded (`SPLIT_SEED = 42`) so every run and ablation study evaluates on the exact same held-out **Test** split (`TEST_RATIO = 0.20`), reserves a **Calibration** holdout (`CALIB_RATIO = 0.10`, capped at `CALIB_MAX`), and uses the remainder for **Train**.

4. **Batch Collation (`collate_train_batch`)**:
   - Pads variable-length sequences in a micro-batch to a uniform length and produces rectangular tensors (`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`, and `target`) ready for vectorized GPU execution.

---

### Stage 2: Model Loading & Unified Decision Wrapping (`models.py`)

To support diverse model families without changing the training loop, `models.py` provides a central router (`DecisionModel.load_model(model_name_or_path, device)`) backed by five loader classes:

1. **`Laya` (`convaiinnovations/laya`)**:
   - Loads the native 421M ModernBERT-large bidirectional encoder and Laya's multi-primitive decision heads (`choice`, `score`, `noul`).
2. **`Von` (`wfzyx/von`)**:
   - Loads the non-autoregressive System One Von checkpoint and adapts it through `MarkerDecisionModelWrapper`.
3. **`OpenDecision` (`open-decision-150m`, `Tokz-labs/OpenDecision-Small`, `Tokz-labs/OpenDecision-Large`)**:
   - Resolves `open-decision-150m` to its `answerdotai/ModernBERT-base` backbone or downloads Hugging Face OpenDecision checkpoints (`backbone_config.json` + `model.safetensors`).
4. **`Typical` (`OzLabs/typical-small`)**:
   - Loads the Qwen-based Typical decision model, registers its decision marker tokens, and truncates mismatched `layer_types` metadata on save/load for Hugging Face compatibility.
5. **`Decision1` (`vllm-sr/Decision-1.0-Lex-0.6B`)**:
   - Loads the 600M Lex-backbone Decision 1.0 model and maps its choice/score/noul markers into the unified wrapper.

#### `MarkerDecisionModelWrapper`
For models that do not natively expose Laya's internal training signature, `MarkerDecisionModelWrapper` wraps the underlying Hugging Face backbone and attaches lightweight projection heads for `choice`, `score`, and `noul`. During the forward pass:
1. It runs the encoder on `(input_ids, attention_mask)` and extracts the last hidden state `h` (cast to `float32` for numerical stability).
2. It gathers the hidden representations at `marker_pos` (clamped safely to valid sequence indices).
3. It routes each sample by `qtype` (`choice = 0`, `score = 1`, `noul = 2`) to produce raw option `logits` and an activation regularization scalar `act_reg`.

---

### Stage 3: RLCD Training Loop (`training.py` & `reward_strategies/brier.py`)

Standard cross-entropy training pushes predicted probabilities toward $1.0$ or $0.0$, often degrading calibration on ambiguous inputs. RLCD instead trains the decision head and encoder using **Group Relative Policy Optimization (GRPO)** guided by a **strictly proper scoring rule** (**Strategy 1: Negative Brier Score**).

For every micro-batch in `RLTrainer.train()`:

1. **Base Forward Pass**:
   - The model computes raw option `logits` at the marker positions and an activation regularization term `act_reg`.
2. **Group Exploration with Annealed Gaussian Noise (`compute_rl_step`)**:
   - Rather than taking a single point prediction, the trainer clones the raw logits `group_size` times (default `G = 4`) and adds zero-mean Gaussian perturbations scaled by $\sigma(t)$:
     $$\tilde{z}^{(g)} = z + \sigma(t)\,\epsilon^{(g)}, \quad \epsilon^{(g)} \sim \mathcal{N}(0, I)$$
   - The exploration noise $\sigma(t)$ linearly anneals from `sigma_start` ($0.4$) at the beginning of training down to `sigma_end` ($0.1$) at the final step, encouraging broad probability exploration early and fine calibration convergence late.
3. **Proper Scoring Rule Reward (`reward_strategies/brier.py`)**:
   - Each perturbed logit vector $\tilde{z}^{(g)}$ is converted into a valid probability distribution $q^{(g)} = \text{softmax}(\tilde{z}^{(g)})$ (with invalid marker slots masked out) and scored against the ground-truth target $y$ using the negative Brier score:
     $$R\big(q^{(g)}, y\big) = -\sum_{k} \big(q_k^{(g)} - y_k\big)^2$$
   - Because the Brier score is a strictly proper scoring rule, its expected reward is uniquely maximized when the model outputs its true posterior probability distribution $q = P(y \mid x)$.
4. **Group-Normalized Advantage Calculation**:
   - Across the `G` exploration samples for each item, rewards are standardized into relative advantages:
     $$A^{(g)} = \frac{R^{(g)} - \mu_R}{\sigma_R + \epsilon}$$
   - Perturbations that yielded better-calibrated probabilities than the group mean receive positive advantage ($A^{(g)} > 0$); worse perturbations receive negative advantage.
5. **Combined Policy Gradient + Soft Cross-Entropy Loss**:
   - The trainer combines the advantage-weighted policy gradient loss with a soft cross-entropy guidance term (`ce_weight`) and activation regularization (`act_reg_weight`):
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PG}}(A, \tilde{z}, z) + \lambda_{\text{CE}}\,\mathcal{L}_{\text{CE}}(z, y) + \lambda_{\text{reg}}\,\mathcal{L}_{\text{act}}$$
   - Gradients are accumulated over `grad_accum_steps`, clipped to `max_grad_norm`, and applied using AdamW with **differential learning rates**—a higher learning rate (`lr_head = 1e-4`) for the decision head and a conservative learning rate (`lr_encoder = 2.5e-5`) with cosine decay for the pretrained backbone.
6. **Post-Training Temperature Calibration (`calibrate`)**:
   - After epoch training finishes, the trainer runs inference on the held-out **Calibration** split (`calib_items`) and uses L-BFGS to fit per-primitive temperature scalars (`T_choice`, `T_score`, `T_noul`) that minimize NLL/Brier error on unseen data without altering the relative ranking of options.

---

### Stage 4: Evaluation, Checkpointing & Ablation (`training.py` & `run_ablation_comparison.py`)

1. **Pre- and Post-Training Held-Out Evaluation (`RLTrainer.evaluate`)**:
   - Before step 1 (`baseline`) and after post-hoc calibration (`final`), the trainer evaluates the model on the held-out **Test** split (`test_items`).
   - It computes six metrics:
     - **Accuracy** (top-1 exact match on `choice`/`noul`, or thresholded accuracy on `score`)
     - **Weighted Precision, Recall, and F1-Score**
     - **Mean Brier Score** (measuring overall probabilistic accuracy)
     - **Expected Calibration Error (ECE)** across 10 confidence bins (measuring the gap between predicted confidence and empirical accuracy)
   - Both baseline and post-training metrics—along with improvement deltas—are printed to the console and saved to `<OUTPUT_DIR>/test_evaluation_metrics.json`.

2. **Rolling Checkpoints & Final Export (`helper.py`)**:
   - After each epoch, `save_rolling_checkpoint` writes the latest weights and tokenizer to `<OUTPUT_DIR>/checkpoint_latest/`.
   - Once final evaluation completes, `save_final_model` exports the calibrated model weights (`model.safetensors`), encoder configuration, tokenizer, and `laya_config.json` to `<OUTPUT_DIR>/`, and automatically removes the temporary `checkpoint_latest/` folder to conserve disk space.

3. **Hyperparameter Sweep (`RLTrainer.hyperparameter_sweep`)**:
   - Passing `--sweep` to `main.py` runs multiple candidate configurations (`lr_head`, `group_size`, `ce_weight`, etc.) and ranks trials by a composite calibration-and-accuracy score on the test split.

4. **Controlled SFT vs. RLCD Ablation (`run_ablation_comparison.py`)**:
   - Evaluates three models on the exact same held-out test split (`seed=42`):
     1. **Untouched Base Model** (zero fine-tuning)
     2. **Standard SFT** (`ce_weight = 1.0`, pure cross-entropy with RL policy gradient disabled)
     3. **Full RLCD Strategy 1** (GRPO with Brier proper scoring rule reward + CE guidance + temperature calibration)
   - Saves the side-by-side comparison metrics to `results/ablation_no_rlcd_results.json`.
