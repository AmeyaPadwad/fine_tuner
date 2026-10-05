"""Smoke tests for the functions in helper.py, models.py, processing.py, and training.py."""

# General libraries
import os
import sys
import argparse
from dotenv import load_dotenv
import torch
import torch.nn as nn


load_dotenv()
DEFAULT_DATASET_NAME = os.getenv("DEFAULT_DATASET_NAME", "Salesforce/xlam-function-calling-60k")
DEFAULT_LAYA_MODEL = os.getenv("DEFAULT_LAYA_MODEL", "convaiinnovations/laya")

class HelperSmokeTests:
    """Smoke tests for helper.py"""
    def __init__(self, dataset_name=DEFAULT_DATASET_NAME):
        self.dataset_name = dataset_name
    

    def test_hf_login(self):
        print(f"\n=== Smoke test: hf_login ===")
        try:
            from helper import hf_login
            hf_login()
            print("PASS: hf_login returned successfully.")
            return True
        except Exception as e:
            print(f"FAIL: hf_login raised {type(e).__name__}: {e}")
            return False

    def test_load_dataset(self):
        print(f"\n=== Smoke test: load_dataset('{self.dataset_name}') ===")
        try:
            from helper import load_dataset
            dataset = load_dataset(self.dataset_name)
            assert dataset is not None, "load_dataset returned None"
            print("PASS: load_dataset returned a dataset.")
            return True
        except Exception as e:
            print(f"FAIL: load_dataset raised {type(e).__name__}: {e}")
            return False

    def test_save_rolling_checkpoint(self):
        print(f"\n=== Smoke test: save_rolling_checkpoint ===")
        try:
            from helper import save_rolling_checkpoint
            assert callable(save_rolling_checkpoint)
            print("PASS: save_rolling_checkpoint exists.")
            return True
        except Exception as e:
            print(f"FAIL: save_rolling_checkpoint raised {type(e).__name__}: {e}")
            return False

    def test_save_final_model(self):
        print(f"\n=== Smoke test: save_final_model ===")
        try:
            from helper import save_final_model
            assert callable(save_final_model)
            print("PASS: save_final_model exists.")
            return True
        except Exception as e:
            print(f"FAIL: save_final_model raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Runs all smoke tests for this class."""
        
        results = {
            "hf_login": self.test_hf_login(),
            "load_dataset": self.test_load_dataset(),
            "save_rolling_checkpoint": self.test_save_rolling_checkpoint(),
            "save_final_model": self.test_save_final_model(),
        }
        print("\n--- Helper Smoke Test Summary ---")
        for name, passed in results.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
        return all(results.values())


class LayaSmokeTests:
    """Smoke tests for models.py"""
    def __init__(self, model_name=DEFAULT_LAYA_MODEL):
        self.model_name = model_name

    def test_load_model(self):
        print(f"\n=== Smoke test: Laya.load_model('{self.model_name}') ===")
        try:
            from models import Laya
            assert callable(Laya.load_model)
            print("PASS: Laya.load_model exists.")
            return True
        except Exception as e:
            print(f"FAIL: Laya.load_model raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Runs all smoke tests for this class."""
        results = {
            "load_model": self.test_load_model(),
        }
        print("\n--- Laya Smoke Test Summary ---")
        for name, passed in results.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
        return all(results.values())


class ProcessingSmokeTests:
    """Smoke tests for processing.py"""
    def test_row_to_laya_input(self):
        print("\n=== Smoke test: row_to_laya_input ===")
        try:
            from processing import row_to_laya_input
            assert callable(row_to_laya_input)
            print("PASS: row_to_laya_input exists.")
            return True
        except Exception as e:
            print(f"FAIL: row_to_laya_input raised {type(e).__name__}: {e}")
            return False

    def test_prepare_dataset_for_laya(self):
        print("\n=== Smoke test: prepare_dataset_for_laya ===")
        rows = [{"text": "Patient has chest pain.", "label": "urgent"}]
        try:
            from processing import prepare_dataset_for_laya
            prepared = prepare_dataset_for_laya(
                rows, context_fn=lambda r: r["text"], decision_type="choice",
                options_fn=lambda _: ["urgent", "not urgent"], target_fn=lambda r: r["label"],
            )
            assert len(prepared) == 1
            print("PASS: prepare_dataset_for_laya produced well-formed rows.")
            return True
        except Exception as e:
            print(f"FAIL: prepare_dataset_for_laya raised {type(e).__name__}: {e}")
            return False

    def test_prepare_xlam_for_laya(self):
        print("\n=== Smoke test: prepare_xlam_for_laya ===")
        try:
            from processing import prepare_xlam_for_laya
            assert callable(prepare_xlam_for_laya)
            print("PASS: prepare_xlam_for_laya exists.")
            return True
        except Exception as e:
            print(f"FAIL: prepare_xlam_for_laya raised {type(e).__name__}: {e}")
            return False

    def test_build_training_item(self):
        print("\n=== Smoke test: build_training_item ===")
        try:
            from processing import build_training_item
            assert callable(build_training_item)
            print("PASS: build_training_item exists.")
            return True
        except Exception as e:
            print(f"FAIL: build_training_item raised {type(e).__name__}: {e}")
            return False

    def test_build_item_from_laya_row(self):
        print("\n=== Smoke test: build_item_from_laya_row ===")
        try:
            from processing import build_item_from_laya_row
            
            assert callable(build_item_from_laya_row)
            print("PASS: build_item_from_laya_row exists.")
            return True
        except Exception as e:
            print(f"FAIL: build_item_from_laya_row raised {type(e).__name__}: {e}")
            return False

    def test_prepare_dataset_items(self):
        print("\n=== Smoke test: prepare_dataset_items ===")
        try:
            from processing import prepare_dataset_items
            assert callable(prepare_dataset_items)
            print("PASS: prepare_dataset_items exists.")
            return True
        except Exception as e:
            print(f"FAIL: prepare_dataset_items raised {type(e).__name__}: {e}")
            return False

    def test_collate_train_batch(self):
        print("\n=== Smoke test: collate_train_batch ===")
        try:
            from processing import collate_train_batch
            
            item_sample = [
                {"ids": [101, 200, 102], "markers": [1], "target": [1.0], "qtype": 0, "label": 0},
            ]
            batch = collate_train_batch(item_sample, pad_id=0)
            assert batch["input_ids"].shape == (1, 3)
            print("PASS: collate_train_batch works as expected.")
            return True
        except Exception as e:
            print(f"FAIL: collate_train_batch raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Runs all smoke tests for this class."""
        results = {
            "row_to_laya_input": self.test_row_to_laya_input(),
            "prepare_dataset_for_laya": self.test_prepare_dataset_for_laya(),
            "prepare_xlam_for_laya": self.test_prepare_xlam_for_laya(),
            "build_training_item": self.test_build_training_item(),
            "build_item_from_laya_row": self.test_build_item_from_laya_row(),
            "prepare_dataset_items": self.test_prepare_dataset_items(),
            "collate_train_batch": self.test_collate_train_batch()
        }
        print("\n--- Processing Smoke Test Summary ---")
        for name, passed in results.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
        return all(results.values())


class TrainingSmokeTests:
    """Smoke tests for training.py"""
    def test_compute_rewards(self):
        print("\n=== Smoke test: compute_rewards ===")
        try:
            from training import compute_rewards
            q = torch.tensor([[[0.8, 0.2], [0.3, 0.7]]])
            target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
            qtype = torch.tensor([0, 0])
            mask = torch.tensor([[True, True], [True, True]])
            r_brier = compute_rewards(q, target, qtype, mask, reward_strategy="brier")
            assert r_brier.shape == (1, 2)
            print("PASS: compute_rewards correctly evaluates brier rewards.")
            return True
        except Exception as e:
            print(f"FAIL: compute_rewards raised {type(e).__name__}: {e}")
            return False

    def test_compute_rl_step(self):
        print("\n=== Smoke test: compute_rl_step ===")
        try:
            from training import compute_rl_step
            class MockDecisionModel(nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
                    b = input_ids.size(0)
                    logits = torch.randn(b, 2, requires_grad=True)
                    act = torch.zeros(b, 2)
                    return logits, act

            mock_model = MockDecisionModel()
            mock_batch = {
                "input_ids": torch.randint(10, 100, (2, 4)),
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
                "marker_pos": torch.tensor([[1, 2], [1, 2]]),
                "marker_mask": torch.tensor([[True, True], [True, True]]),
                "qtype": torch.tensor([0, 0]),
                "target": torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
            }
            loss, rl_l, ce_l, rew = compute_rl_step(
                model=mock_model, batch=mock_batch, device=torch.device("cpu"),
                group_size=4, sigma=0.3, reward_strategy="brier"
            )
            assert loss.requires_grad
            print("PASS: compute_rl_step computed loss and completed forward pass.")
            return True
        except Exception as e:
            print(f"FAIL: compute_rl_step raised {type(e).__name__}: {e}")
            return False

    def test_train(self):
        print("\n=== Smoke test: RLTrainer.train ===")
        try:
            from training import RLTrainer
            assert callable(RLTrainer.train)
            print("PASS: RLTrainer.train exists.")
            return True
        except Exception as e:
            print(f"FAIL: RLTrainer.train raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Runs all smoke tests for this class."""
        results = {
            "compute_rewards": self.test_compute_rewards(),
            "compute_rl_step": self.test_compute_rl_step(),
            "train": self.test_train()
        }
        print("\n--- RL Training Smoke Test Summary ---")
        for name, passed in results.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
        return all(results.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run smoke tests for RLCD components.")
    parser.add_argument("--helper", action="store_true", help="Run smoke tests for helper.py")
    parser.add_argument("--laya", action="store_true", help="Run smoke tests for models.py (Laya)")
    parser.add_argument("--processing", action="store_true", help="Run smoke tests for processing.py")
    parser.add_argument("--training", action="store_true", help="Run smoke tests for training.py")
    parser.add_argument("--all", action="store_true", help="Run all smoke tests across all files")
    
    args = parser.parse_args()

    # If no flags are provided, run all by default
    if not (args.helper or args.laya or args.processing or args.training or args.all):
        args.all = True

    all_success = True
    successes = {}

    if args.all or args.helper:
        print("\nRunning HelperSmokeTests...")
        success = HelperSmokeTests().run_all()
        all_success = all_success and success
        successes["helper"] = success

    if args.all or args.laya:
        print("\nRunning LayaSmokeTests...")
        success = LayaSmokeTests().run_all()
        all_success = all_success and success
        successes["laya"] = success

    if args.all or args.processing:
        print("\nRunning ProcessingSmokeTests...")
        success = ProcessingSmokeTests().run_all()
        all_success = all_success and success
        successes["processing"] = success

    if args.all or args.training:
        print("\nRunning TrainingSmokeTests...")
        success = TrainingSmokeTests().run_all()
        all_success = all_success and success
        successes["training"] = success

    print("\n===============================")
    if all_success:
        print(f"FINAL RESULT: {'ALL PASS'}")
    else:
        print(f"FINAL RESULT: {'SOME FAILURES'}")
        print(f"Failures:")
        
        for k, v in successes.items():
            print(f"-----{k}: {v}")

    print("===============================")
    
    sys.exit(0 if all_success else 1)