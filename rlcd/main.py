import os
from helper import hf_login, load_dataset
from processing import auto_prepare_dataset_for_laya, prepare_dataset_items, split_dataset_items
from training import RLTrainingConfig, RLTrainer

#### Model variables ####
MODEL_NAME_OR_PATH = os.getenv("DEFAULT_LAYA_MODEL", "convaiinnovations/laya")
OUTPUT_DIR = "./laya_finetuned_xlam_rl"
#########################

#### Dataset variables ####
DATASET = "Salesforce/xlam-function-calling-60k"
DATASET_SUBSET = 10  # Set to an int for fast testing/dev (e.g., 30-60), or None for full dataset
###########################

#### Environment variables ####
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"  # Change to "cuda" if testing on a GPU
###############################

#### Active Hyperparameters ####
EPOCHS = 2
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
GROUP_SIZE = 4       # Number of GRPO exploration perturbations per sample
LR_HEAD = 1.0e-4     # Learning rate for the Laya decision head
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


def main(run_sweep: bool = False):
    """
    End-to-end pipeline to fine-tune Laya on Salesforce xLAM using RLCD Strategy 1.
    """
    # 1. Authenticate with Hugging Face Hub if credentials exist
    hf_login()

    # 2. Load dataset
    print(f"\n=== Loading '{DATASET}' dataset ===")
    dataset = load_dataset(DATASET)

    if DATASET_SUBSET is not None:
        raw_data = dataset["train"].select(range(min(DATASET_SUBSET, len(dataset["train"]))))
    else:
        raw_data = dataset["train"]

    # 3. Preprocess the dataset into Laya's format (auto-adapts any schema)
    print("\n=== Preprocessing dataset into Laya format ===")
    prepared_dataset = auto_prepare_dataset_for_laya(raw_data)

    # 4. Configure active hyperparameters
    print("\n=== Configuring RL Training Hyperparameters ===")
    config = RLTrainingConfig(
        model_name_or_path=MODEL_NAME_OR_PATH,
        output_dir=OUTPUT_DIR,
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
        device=DEVICE,
        calib_max=CALIB_MAX,
        seed=SPLIT_SEED,
    )

    # 5. Initialize trainer and tokenize sequences
    trainer = RLTrainer(config=config)

    print("\n=== Tokenizing sequences and creating [MASK] alignments ===")
    all_items = prepare_dataset_items(
        prepared_dataset,
        trainer.tokenizer,
        max_len=config.max_len,
        head_max_len=config.head_max_len,
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
    main()
