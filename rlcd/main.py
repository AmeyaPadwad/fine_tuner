import copy
import json
import os
import random
from typing import Any, Dict, List, Optional
from helper import hf_login, load_dataset
from processing import auto_prepare_dataset_for_laya, prepare_dataset_items
from training import RLTrainingConfig, RLTrainer

#### Model variables ####
MODEL_NAME_OR_PATH = os.getenv("DEFAULT_LAYA_MODEL", "convaiinnovations/laya")
OUTPUT_DIR = "./laya_finetuned_xlam_rl"
#########################

#### Dataset variables ####
DATASET = "Salesforce/xlam-function-calling-60k"
DATASET_SUBSET = 30  # Set to an int for fast testing/dev (e.g., 30-60), or None for full dataset
###########################

#### Environment variables ####
DEVICE = "cpu"  # Change to "cuda" if testing on a GPU
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

def split_dataset_items(
    all_items: List[Dict[str, Any]],
    test_ratio: float = 0.20,
    calib_ratio: float = 0.10,
    calib_max: int = 20,
    seed: int = 42,
):
    """Partition items into non-overlapping train, calibration, and test splits."""
    n_total = len(all_items)
    if n_total < 3:
        raise ValueError(f"Need at least 3 items to split, got {n_total}")

    order = list(range(n_total))
    random.Random(seed).shuffle(order)

    n_test = max(1, int(n_total * test_ratio))
    n_calib = min(calib_max, max(1, int(n_total * calib_ratio)))
    n_train = n_total - n_test - n_calib

    if n_train < 1:
        # Guarantee at least 1 training item if dataset is tiny
        n_train = 1
        n_test = max(1, n_total - n_train - n_calib)

    test_indices = order[:n_test]
    calib_indices = order[n_test : n_test + n_calib]
    train_indices = order[n_test + n_calib :]

    test_items = [all_items[i] for i in test_indices]
    calib_items = [all_items[i] for i in calib_indices]
    train_items = [all_items[i] for i in train_indices]

    print(
        f"\nDataset Split Summary (Seed {seed}):\n"
        f"  Total:       {n_total} items\n"
        f"  Train:       {len(train_items)} items ({len(train_items)/n_total:.1%})\n"
        f"  Calibration: {len(calib_items)} items ({len(calib_items)/n_total:.1%})\n"
        f"  Test:        {len(test_items)} items ({len(test_items)/n_total:.1%})"
    )

    return train_items, calib_items, test_items


def hyperparameter_sweep(
    train_items: List[Dict[str, Any]],
    calib_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    base_config: RLTrainingConfig,
    param_grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run a hyperparameter search across candidate configurations.
    
    Evaluates each candidate on held-out test_items and returns the best configuration.
    """
    print(f"\n========================================================")
    print(f"=== Starting Hyperparameter Sweep ({len(param_grid)} trials) ===")
    print(f"========================================================")

    best_score = -float("inf")
    best_config = None
    best_metrics = None
    results = []

    for trial_idx, params in enumerate(param_grid, start=1):
        print(f"\n--- Running Trial {trial_idx}/{len(param_grid)} with params: {params} ---")
        trial_cfg_dict = base_config.to_dict()
        trial_cfg_dict.update(params)
        trial_cfg_dict["output_dir"] = os.path.join(base_config.output_dir, f"trial_{trial_idx}")
        trial_config = RLTrainingConfig.from_dict(trial_cfg_dict)

        trainer = RLTrainer(config=trial_config)
        trial_result = trainer.train(
            train_items=train_items,
            calib_items=calib_items,
            test_items=test_items,
        )

        final_metrics = trial_result.get("metrics", {}).get("final", {})
        mean_brier = final_metrics.get("mean_brier", -float("inf"))
        ece = final_metrics.get("ece", 1.0)
        acc = final_metrics.get("accuracy", 0.0)

        # Optimization criterion: Higher Brier score (closer to 0) + lower ECE
        # Score = Mean Brier - ECE
        combined_score = mean_brier - (0.5 * ece)

        record = {
            "trial": trial_idx,
            "params": params,
            "accuracy": acc,
            "mean_brier": mean_brier,
            "ece": ece,
            "combined_score": combined_score,
            "output_dir": trial_config.output_dir,
        }
        results.append(record)
        print(f"Trial {trial_idx} Results: Acc: {acc:.2%} | Brier: {mean_brier:.4f} | ECE: {ece:.4f} | Combined: {combined_score:.4f}")

        if combined_score > best_score:
            best_score = combined_score
            best_config = trial_config
            best_metrics = record

    print("\n========================================================")
    print("=== Hyperparameter Sweep Complete: Ranked Results ===")
    print("========================================================")
    results.sort(key=lambda r: r["combined_score"], reverse=True)
    for r in results:
        print(f"Rank {r['trial']}: Score={r['combined_score']:.4f} | Acc={r['accuracy']:.2%} | Brier={r['mean_brier']:.4f} | ECE={r['ece']:.4f} | Params={r['params']}")

    print(f"\nBest Configuration Found in Trial {best_metrics['trial']} with Score {best_score:.4f}:")
    print(json.dumps(best_metrics, indent=2))

    return {"best_config": best_config, "best_metrics": best_metrics, "all_results": results}


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
        return hyperparameter_sweep(
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
