import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional

import torch
from helper import hf_login, load_dataset
from processing import auto_prepare_dataset, prepare_dataset_items, split_dataset_items
from training import RLTrainingConfig, RLTrainer

#### Model & Path variables ####
MODEL_NAME_OR_PATH = os.getenv("DEFAULT_LAYA_MODEL", "convaiinnovations/laya")
OUTPUT_DIR = "./laya_finetuned_xlam_rl"
CACHE_DIR = os.getenv("DEFAULT_CACHE_DIR", "./.cache")
################################

#### Dataset variables ####
DATASET = os.getenv("DEFAULT_DATASET_NAME", "Salesforce/xlam-function-calling-60k")
DATASET_SUBSET = None  # Set to an int for fast testing/dev (e.g., 30-60), or None for full dataset
COLUMN_MAP = None      # Optional dict to override column roles, e.g. {"context": "text", "target": "label_text", "decision_type": "choice"}
###########################

#### Environment variables ####
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
FP16 = True
###############################

#### Sequence & Memory Hyperparameters ####
MAX_LEN = 1024
HEAD_MAX_LEN = 256
MAX_TOKENS_PER_BATCH = 4096
GRADIENT_CHECKPOINTING = True
###########################################

#### Active RL & Optimization Hyperparameters ####
EPOCHS = 2
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
GROUP_SIZE = 4        # Number of GRPO exploration perturbations per sample
LR_HEAD = 1.0e-4      # Learning rate for the decision head
LR_ENCODER = 2.5e-5   # Learning rate for the pretrained encoder
MIN_LR = 1.0e-6       # Minimum learning rate at the end of cosine schedule
WEIGHT_DECAY = 0.01   # AdamW L2 weight decay penalty
MAX_GRAD_NORM = 1.0   # Gradient clipping threshold
SIGMA_START = 0.4     # Initial exploration noise
SIGMA_END = 0.1       # Final exploration noise
CE_WEIGHT = 1.0       # Weight for soft cross-entropy guidance loss
ACT_REG_WEIGHT = 0.0  # Penalty weight on head activation summaries
REWARD_STRATEGY = "brier"
##################################################

#### Split & Calibration settings ####
TEST_RATIO = 0.20          # 20% held out for evaluation (Accuracy, Brier, ECE)
CALIB_RATIO = 0.10         # 10% held out for post-hoc calibration
CALIB_MAX = 400            # Max items to withhold for calibration
CALIB_SEED = 20260922      # Seed for calibration holdout selection
CALIB_BATCH_SIZE = 16      # Batch size during post-hoc calibration inference
SPLIT_SEED = 42            # Seed for dataset splitting and training shuffle
######################################


def main(
    run_sweep: bool = False,
    param_grid: Optional[List[Dict[str, Any]]] = None,
    # Model & Paths
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = CACHE_DIR,
    # Dataset & Schema
    dataset_name: Optional[str] = None,
    dataset_subset: Optional[int] = DATASET_SUBSET,
    column_map: Optional[Dict[str, Any]] = COLUMN_MAP,
    # Sequence & Memory
    max_len: int = MAX_LEN,
    head_max_len: int = HEAD_MAX_LEN,
    max_tokens_per_batch: int = MAX_TOKENS_PER_BATCH,
    gradient_checkpointing: bool = GRADIENT_CHECKPOINTING,
    # RL & Optimization Hyperparameters
    epochs: int = EPOCHS,
    micro_batch_size: int = MICRO_BATCH_SIZE,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    group_size: int = GROUP_SIZE,
    lr_head: float = LR_HEAD,
    lr_encoder: float = LR_ENCODER,
    min_lr: float = MIN_LR,
    weight_decay: float = WEIGHT_DECAY,
    max_grad_norm: float = MAX_GRAD_NORM,
    # Exploration & Loss Weights
    sigma_start: float = SIGMA_START,
    sigma_end: float = SIGMA_END,
    ce_weight: float = CE_WEIGHT,
    act_reg_weight: float = ACT_REG_WEIGHT,
    reward_strategy: str = REWARD_STRATEGY,
    custom_reward_fn: Optional[Callable] = None,
    # Split & Calibration Settings
    test_ratio: float = TEST_RATIO,
    calib_ratio: float = CALIB_RATIO,
    calib_max: int = CALIB_MAX,
    calib_seed: int = CALIB_SEED,
    calib_batch_size: int = CALIB_BATCH_SIZE,
    seed: int = SPLIT_SEED,
    # Hardware & Precision
    device: Optional[str] = None,
    fp16: bool = FP16,
) -> Dict[str, Any]:
    """End-to-end RLCD SDK entry point to fine-tune and calibrate a decision model on any dataset.

    Every hyperparameter and pipeline variable is optional and defaults to the
    module-level configuration when omitted.
    """
    active_model = model_name or MODEL_NAME_OR_PATH
    active_dataset = dataset_name or DATASET
    active_output_dir = output_dir or OUTPUT_DIR
    active_device = device or DEVICE

    # 1. Authenticate with Hugging Face Hub if credentials exist
    hf_login()

    # 2. Configure active hyperparameters (initializes distributed process group if running under torchrun)
    print("\n=== Configuring RL Training Hyperparameters ===")
    config = RLTrainingConfig(
        model_name_or_path=active_model,
        output_dir=active_output_dir,
        cache_dir=cache_dir,
        max_len=max_len,
        head_max_len=head_max_len,
        max_tokens_per_batch=max_tokens_per_batch,
        gradient_checkpointing=gradient_checkpointing,
        epochs=epochs,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        group_size=group_size,
        lr_head=lr_head,
        lr_encoder=lr_encoder,
        min_lr=min_lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        sigma_start=sigma_start,
        sigma_end=sigma_end,
        ce_weight=ce_weight,
        act_reg_weight=act_reg_weight,
        reward_strategy=reward_strategy,
        calib_max=calib_max,
        calib_ratio=calib_ratio,
        calib_seed=calib_seed,
        calib_batch_size=calib_batch_size,
        device=active_device,
        fp16=fp16,
        seed=seed,
    )

    # 3. Initialize trainer first so DDP process group & rank are active before dataset download
    trainer = RLTrainer(config=config, custom_reward_fn=custom_reward_fn)

    # 4. Load dataset safely across DDP ranks
    print(f"\n=== Loading '{active_dataset}' dataset ===")
    if trainer.is_distributed:
        import torch.distributed as dist
        if trainer.rank == 0:
            dataset = load_dataset(active_dataset)
        dist.barrier()
        if trainer.rank != 0:
            dataset = load_dataset(active_dataset)
    else:
        dataset = load_dataset(active_dataset)

    split_data = (
        dataset["train"]
        if isinstance(dataset, dict) or (hasattr(dataset, "keys") and "train" in dataset)
        else dataset
    )

    if dataset_subset is not None:
        raw_data = split_data.select(range(min(dataset_subset, len(split_data))))
    else:
        raw_data = split_data

    # 5. Preprocess the dataset into the model's format (auto-adapts any schema or uses column_map)
    print("\n=== Preprocessing dataset into model format ===")
    prepared_dataset = auto_prepare_dataset(raw_data, column_map=column_map)

    print("\n=== Tokenizing sequences and creating [MASK] alignments ===")
    all_items = prepare_dataset_items(
        prepared_dataset,
        trainer.tokenizer,
        max_len=config.max_len,
        head_max_len=config.head_max_len,
        marker_token_id=trainer.marker_token_id,
    )

    # 6. Partition into Train, Calibration, and Test splits
    train_items, calib_items, test_items = split_dataset_items(
        all_items,
        test_ratio=test_ratio,
        calib_ratio=calib_ratio,
        calib_max=config.calib_max,
        seed=config.seed,
    )

    if run_sweep:
        sweep_grid = param_grid or [
            {"lr_head": 1.0e-4, "group_size": 4, "ce_weight": 1.0},
            {"lr_head": 2.0e-4, "group_size": 4, "ce_weight": 0.5},
            {"lr_head": 5.0e-5, "group_size": 2, "ce_weight": 1.0},
        ]
        return trainer.hyperparameter_sweep(
            train_items=train_items,
            calib_items=calib_items,
            test_items=test_items,
            param_grid=sweep_grid,
        )

    # 7. Run the RL fine-tuning loop with post-hoc calibration and held-out evaluation
    print("\n=== Starting Reinforcement Learning Fine-Tuning Loop ===")
    result = trainer.train(
        train_items=train_items,
        calib_items=calib_items,
        test_items=test_items,
    )
    return result


# SDK alias
fine_tune = main


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the `rlcd-fine-tune` CLI command."""
    parser = argparse.ArgumentParser(
        prog="rlcd-fine-tune",
        description="Fine-tune and calibrate open-weight System One decision models using RLCD.",
    )

    # Model & Paths
    parser.add_argument("--model", dest="model_name", type=str, default=None, help=f"Model ID or local path (default: {MODEL_NAME_OR_PATH})")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None, help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--cache-dir", dest="cache_dir", type=str, default=CACHE_DIR, help=f"Model cache directory (default: {CACHE_DIR})")

    # Dataset & Schema
    parser.add_argument("--dataset", dest="dataset_name", type=str, default=None, help=f"Hugging Face dataset ID (default: {DATASET})")
    parser.add_argument("--subset", dest="dataset_subset", type=int, default=DATASET_SUBSET, help="Dataset subset size (int) or None for full dataset")
    parser.add_argument("--column-map", dest="column_map", type=str, default=None, help="JSON string for column_map override")

    # Sequence & Memory
    parser.add_argument("--max-len", dest="max_len", type=int, default=MAX_LEN, help=f"Max token sequence length (default: {MAX_LEN})")
    parser.add_argument("--head-max-len", dest="head_max_len", type=int, default=HEAD_MAX_LEN, help=f"Max token budget for question & options (default: {HEAD_MAX_LEN})")
    parser.add_argument("--max-tokens-per-batch", dest="max_tokens_per_batch", type=int, default=MAX_TOKENS_PER_BATCH, help=f"Max tokens per batch (default: {MAX_TOKENS_PER_BATCH})")
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=GRADIENT_CHECKPOINTING,
        help="Enable/disable gradient checkpointing (default: enabled)",
    )

    # RL & Optimization Hyperparameters
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--micro-batch-size", dest="micro_batch_size", type=int, default=MICRO_BATCH_SIZE, help=f"Micro-batch size per forward pass (default: {MICRO_BATCH_SIZE})")
    parser.add_argument("--grad-accum-steps", dest="grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS, help=f"Gradient accumulation steps (default: {GRAD_ACCUM_STEPS})")
    parser.add_argument("--group-size", dest="group_size", type=int, default=GROUP_SIZE, help=f"GRPO exploration group size G (default: {GROUP_SIZE})")
    parser.add_argument("--lr-head", dest="lr_head", type=float, default=LR_HEAD, help=f"Decision head learning rate (default: {LR_HEAD})")
    parser.add_argument("--lr-encoder", dest="lr_encoder", type=float, default=LR_ENCODER, help=f"Encoder learning rate (default: {LR_ENCODER})")
    parser.add_argument("--min-lr", dest="min_lr", type=float, default=MIN_LR, help=f"Minimum cosine schedule learning rate (default: {MIN_LR})")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=WEIGHT_DECAY, help=f"AdamW weight decay (default: {WEIGHT_DECAY})")
    parser.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=MAX_GRAD_NORM, help=f"Gradient norm clipping threshold (default: {MAX_GRAD_NORM})")

    # Exploration & Loss Weights
    parser.add_argument("--sigma-start", dest="sigma_start", type=float, default=SIGMA_START, help=f"Initial GRPO exploration noise sigma (default: {SIGMA_START})")
    parser.add_argument("--sigma-end", dest="sigma_end", type=float, default=SIGMA_END, help=f"Final GRPO exploration noise sigma (default: {SIGMA_END})")
    parser.add_argument("--ce-weight", dest="ce_weight", type=float, default=CE_WEIGHT, help=f"Soft cross-entropy guidance weight (default: {CE_WEIGHT})")
    parser.add_argument("--act-reg-weight", dest="act_reg_weight", type=float, default=ACT_REG_WEIGHT, help=f"Activation regularization weight (default: {ACT_REG_WEIGHT})")
    parser.add_argument("--reward-strategy", dest="reward_strategy", type=str, default=REWARD_STRATEGY, help=f"Reward scoring rule strategy (default: {REWARD_STRATEGY})")

    # Split & Calibration Settings
    parser.add_argument("--test-ratio", dest="test_ratio", type=float, default=TEST_RATIO, help=f"Held-out test split ratio (default: {TEST_RATIO})")
    parser.add_argument("--calib-ratio", dest="calib_ratio", type=float, default=CALIB_RATIO, help=f"Held-out calibration split ratio (default: {CALIB_RATIO})")
    parser.add_argument("--calib-max", dest="calib_max", type=int, default=CALIB_MAX, help=f"Maximum calibration holdout items (default: {CALIB_MAX})")
    parser.add_argument("--calib-seed", dest="calib_seed", type=int, default=CALIB_SEED, help=f"Calibration split seed (default: {CALIB_SEED})")
    parser.add_argument("--calib-batch-size", dest="calib_batch_size", type=int, default=CALIB_BATCH_SIZE, help=f"Calibration inference batch size (default: {CALIB_BATCH_SIZE})")
    parser.add_argument("--seed", dest="seed", type=int, default=SPLIT_SEED, help=f"Random seed for dataset split & training (default: {SPLIT_SEED})")

    # Hardware & Sweep
    parser.add_argument("--device", type=str, default=DEVICE, help=f"Device override: cuda or cpu (default: {DEVICE})")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action=argparse.BooleanOptionalAction,
        default=FP16,
        help="Enable/disable CUDA FP16 mixed precision (default: enabled)",
    )
    parser.add_argument("--sweep", dest="run_sweep", action="store_true", help="Run hyperparameter sweep across candidate configurations")
    parser.add_argument("--param-grid", dest="param_grid", type=str, default=None, help="Optional JSON list of parameter dicts for --sweep")

    return parser


def cli_entry(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entry point registered as `rlcd-fine-tune`."""
    parser = build_arg_parser()
    cli_args, _ = parser.parse_known_args(argv)

    parsed_col_map = json.loads(cli_args.column_map) if cli_args.column_map else COLUMN_MAP
    parsed_grid = json.loads(cli_args.param_grid) if cli_args.param_grid else None

    return main(
        run_sweep=cli_args.run_sweep,
        param_grid=parsed_grid,
        model_name=cli_args.model_name,
        output_dir=cli_args.output_dir,
        cache_dir=cli_args.cache_dir,
        dataset_name=cli_args.dataset_name,
        dataset_subset=cli_args.dataset_subset,
        column_map=parsed_col_map,
        max_len=cli_args.max_len,
        head_max_len=cli_args.head_max_len,
        max_tokens_per_batch=cli_args.max_tokens_per_batch,
        gradient_checkpointing=cli_args.gradient_checkpointing,
        epochs=cli_args.epochs,
        micro_batch_size=cli_args.micro_batch_size,
        grad_accum_steps=cli_args.grad_accum_steps,
        group_size=cli_args.group_size,
        lr_head=cli_args.lr_head,
        lr_encoder=cli_args.lr_encoder,
        min_lr=cli_args.min_lr,
        weight_decay=cli_args.weight_decay,
        max_grad_norm=cli_args.max_grad_norm,
        sigma_start=cli_args.sigma_start,
        sigma_end=cli_args.sigma_end,
        ce_weight=cli_args.ce_weight,
        act_reg_weight=cli_args.act_reg_weight,
        reward_strategy=cli_args.reward_strategy,
        test_ratio=cli_args.test_ratio,
        calib_ratio=cli_args.calib_ratio,
        calib_max=cli_args.calib_max,
        calib_seed=cli_args.calib_seed,
        calib_batch_size=cli_args.calib_batch_size,
        seed=cli_args.seed,
        device=cli_args.device,
        fp16=cli_args.fp16,
    )


if __name__ == "__main__":
    cli_entry()
