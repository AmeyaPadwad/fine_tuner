import os
from helper import hf_login, load_dataset
from processing import auto_prepare_dataset, prepare_dataset_items, split_dataset_items
from training import RLTrainingConfig, RLTrainer

#### Model variables ####
MODEL_NAME_OR_PATH = os.getenv("DEFAULT_LAYA_MODEL", "convaiinnovations/laya")
OUTPUT_DIR = "./laya_finetuned_xlam_rl"
#########################

#### Dataset variables ####
DATASET = "Salesforce/xlam-function-calling-60k"
DATASET_SUBSET = None  # Set to an int for fast testing/dev (e.g., 30-60), or None for full dataset
COLUMN_MAP = None    # Optional dict to override column roles, e.g. {"context": "text", "target": "label_text", "decision_type": "choice"}
###########################

#### Environment variables ####
import torch
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
###############################

#### Active Hyperparameters ####
EPOCHS = 2
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
GROUP_SIZE = 4       # Number of GRPO exploration perturbations per sample
LR_HEAD = 1.0e-4     # Learning rate for the decision head
LR_ENCODER = 2.5e-5  # Learning rate for the ModernBERT encoder
SIGMA_START = 0.4    # Initial exploration noise
SIGMA_END = 0.1      # Final exploration noise
CE_WEIGHT = 1.0      # Weight for soft cross-entropy guidance loss
REWARD_STRATEGY = "brier"

#### Split ratios ####
TEST_RATIO = 0.20    # 20% held out for evaluation (Accuracy, Brier, ECE)
CALIB_RATIO = 0.10   # 10% held out for temperature calibration
CALIB_MAX = 20       # Max items to withhold for calibration
SPLIT_SEED = 42
################################


def main(
    run_sweep: bool = False,
    model_name: str | None = None,
    dataset_name: str | None = None,
    dataset_subset: int | None = DATASET_SUBSET,
    column_map: dict | None = COLUMN_MAP,
    output_dir: str | None = None,
    device: str | None = None,
):
    """
    End-to-end pipeline to fine-tune a decision model on a dataset using RLCD with Brier reward strategy.
    """
    active_model = model_name or MODEL_NAME_OR_PATH
    active_dataset = dataset_name or DATASET
    active_output_dir = output_dir or OUTPUT_DIR
    active_device = device or DEVICE

    # 1. Authenticate with Hugging Face Hub if credentials exist
    hf_login()

    # 4. Configure active hyperparameters (initializes distributed process group if running under torchrun)
    print("\n=== Configuring RL Training Hyperparameters ===")
    config = RLTrainingConfig(
        model_name_or_path=active_model,
        output_dir=active_output_dir,
        epochs=EPOCHS,
        micro_batch_size=MICRO_BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        group_size=GROUP_SIZE,
        lr_head=LR_HEAD,
        lr_encoder=LR_ENCODER,
        sigma_start=SIGMA_START,
        sigma_end=SIGMA_END,
        ce_weight=CE_WEIGHT,
        reward_strategy=REWARD_STRATEGY,
        device=active_device,
        calib_max=CALIB_MAX,
        seed=SPLIT_SEED,
    )

    # 5. Initialize trainer first so DDP process group & rank are active before dataset download
    trainer = RLTrainer(config=config)

    # 2. Load dataset safely across DDP ranks
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

    split_data = dataset["train"] if isinstance(dataset, dict) or (hasattr(dataset, "keys") and "train" in dataset) else dataset

    if dataset_subset is not None:
        raw_data = split_data.select(range(min(dataset_subset, len(split_data))))
    else:
        raw_data = split_data

    # 3. Preprocess the dataset into the model's format (auto-adapts any schema or uses column_map)
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
        test_ratio=TEST_RATIO,
        calib_ratio=CALIB_RATIO,
        calib_max=config.calib_max,
        seed=config.seed,
    )

    if run_sweep:
        # Example candidate range for hyperparameter exploration
        param_grid = [
            {"lr_head": 1.0e-4, "group_size": 4, "ce_weight": 1.0},
            {"lr_head": 2.0e-4, "group_size": 4, "ce_weight": 0.5},
            {"lr_head": 5.0e-5, "group_size": 2, "ce_weight": 1.0},
        ]
        return trainer.hyperparameter_sweep(
            train_items=train_items,
            calib_items=calib_items,
            test_items=test_items,
            base_config=config,
            param_grid=param_grid,
        )

    # 7. Run the RL fine-tuning loop with held-out evaluation
    print("\n=== Starting Reinforcement Learning Fine-Tuning Loop ===")
    result = trainer.train(
        train_items=train_items,
        calib_items=calib_items,
        test_items=test_items,
    )
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="RLCD Fine-Tuning Pipeline")
    parser.add_argument("--model", type=str, default=None, help="Model name or path override")
    parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset ID override")
    parser.add_argument("--subset", type=int, default=DATASET_SUBSET, help="Dataset subset size (int) or None")
    parser.add_argument("--column-map", type=str, default=None, help="JSON string for column_map override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device override (cuda/cpu)")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    cli_args, _ = parser.parse_known_args()

    parsed_col_map = json.loads(cli_args.column_map) if cli_args.column_map else COLUMN_MAP
    main(
        run_sweep=cli_args.sweep,
        model_name=cli_args.model,
        dataset_name=cli_args.dataset,
        dataset_subset=cli_args.subset,
        column_map=parsed_col_map,
        output_dir=cli_args.output_dir,
        device=cli_args.device,
    )
