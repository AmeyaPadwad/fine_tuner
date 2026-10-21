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


class ModelsSmokeTests:
    """Smoke tests for models.py (Laya, Von, OpenDecision, Typical, Decision1, DecisionModel)."""
    def __init__(self, model_name=DEFAULT_LAYA_MODEL):
        self.model_name = model_name

    def test_laya_load_model(self):
        print(f"\n=== Smoke test: Laya.load_model('{self.model_name}') ===")
        try:
            from models import Laya
            assert callable(Laya.load_model)
            assert Laya.DEFAULT_MODEL == "convaiinnovations/laya"
            print("PASS: Laya.load_model exists and is configured.")
            return True
        except Exception as e:
            print(f"FAIL: Laya.load_model raised {type(e).__name__}: {e}")
            return False

    def test_von_load_model(self):
        print("\n=== Smoke test: Von.load_model('wfzyx/von') ===")
        try:
            from models import Von
            assert callable(Von.load_model)
            assert Von.DEFAULT_MODEL == "wfzyx/von"
            print("PASS: Von.load_model exists and is configured.")
            return True
        except Exception as e:
            print(f"FAIL: Von.load_model raised {type(e).__name__}: {e}")
            return False

    def test_open_decision_load_model(self):
        print("\n=== Smoke test: OpenDecision.load_model('open-decision-150m') ===")
        try:
            from models import OpenDecision
            assert callable(OpenDecision.load_model)
            assert OpenDecision.DEFAULT_MODEL == "open-decision-150m"
            print("PASS: OpenDecision.load_model exists and is configured.")
            return True
        except Exception as e:
            print(f"FAIL: OpenDecision.load_model raised {type(e).__name__}: {e}")
            return False

    def test_typical_load_model(self):
        print("\n=== Smoke test: Typical.load_model('OzLabs/typical-small') ===")
        try:
            from models import Typical
            assert callable(Typical.load_model)
            assert Typical.DEFAULT_MODEL == "OzLabs/typical-small"
            print("PASS: Typical.load_model exists and is configured.")
            return True
        except Exception as e:
            print(f"FAIL: Typical.load_model raised {type(e).__name__}: {e}")
            return False

    def test_decision1_load_model(self):
        print("\n=== Smoke test: Decision1.load_model('vllm-sr/Decision-1.0-Lex-0.6B') ===")
        try:
            from models import Decision1
            assert callable(Decision1.load_model)
            assert Decision1.DEFAULT_MODEL == "vllm-sr/Decision-1.0-Lex-0.6B"
            print("PASS: Decision1.load_model exists and is configured.")
            return True
        except Exception as e:
            print(f"FAIL: Decision1.load_model raised {type(e).__name__}: {e}")
            return False

    def test_decision_model_router_and_wrapper(self):
        print("\n=== Smoke test: DecisionModel router & MarkerDecisionModelWrapper ===")
        try:
            from models import DecisionModel, Laya, Von, OpenDecision, Typical, Decision1, MarkerDecisionModelWrapper

            assert DecisionModel.resolve_loader_class("convaiinnovations/laya") is Laya
            assert DecisionModel.resolve_loader_class("wfzyx/von") is Von
            assert DecisionModel.resolve_loader_class("open-decision-150m") is OpenDecision
            assert DecisionModel.resolve_loader_class("ozlabs/typical") is Typical
            assert DecisionModel.resolve_loader_class("vllm-sr/Decision-1.0-Lex-0.6B") is Decision1

            # Verify MarkerDecisionModelWrapper forward pass
            class DummyEncoder(nn.Module):
                def __init__(self, hidden_size=16):
                    super().__init__()
                    self.emb = nn.Embedding(100, hidden_size)
                def forward(self, input_ids, attention_mask=None):
                    return (self.emb(input_ids),)

            wrapper = MarkerDecisionModelWrapper(DummyEncoder(16), hidden_size=16, option_marker_vec=torch.ones(16))
            input_ids = torch.randint(0, 100, (2, 6))
            att = torch.ones((2, 6), dtype=torch.long)
            mpos = torch.tensor([[1, 3], [2, 4]])
            mmask = torch.tensor([[True, True], [True, True]])
            qtype = torch.tensor([0, 2])
            logits, act = wrapper(input_ids, att, mpos, mmask, qtype)
            assert logits.shape == (2, 2) and act.shape == (2, 1)

            print("PASS: DecisionModel router and MarkerDecisionModelWrapper forward pass verified.")
            return True
        except Exception as e:
            print(f"FAIL: DecisionModel router/wrapper raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Runs all smoke tests for this class."""
        results = {
            "Laya.load_model": self.test_laya_load_model(),
            "Von.load_model": self.test_von_load_model(),
            "OpenDecision.load_model": self.test_open_decision_load_model(),
            "Typical.load_model": self.test_typical_load_model(),
            "Decision1.load_model": self.test_decision1_load_model(),
            "DecisionModel_router_and_wrapper": self.test_decision_model_router_and_wrapper(),
        }
        print("\n--- Models Smoke Test Summary ---")
        for name, passed in results.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
        return all(results.values())


# Backward-compatible alias
LayaSmokeTests = ModelsSmokeTests


class ProcessingSmokeTests:
    """Smoke tests for processing.py"""
    def test_auto_prepare_dataset(self):
        print("\n=== Smoke test: auto_prepare_dataset (all 4 dataset types + column_map) ===")
        try:
            from processing import auto_prepare_dataset

            # 1. NexusRaven-V2 / Function Calling (choice)
            nexus_data = [
                {
                    "prompt": "Function:\ndef get_weather(city: str):\n    '''Get weather'''\n\nFunction:\ndef book_flight(dest: str):\n    '''Book flight'''\n\nUser Query: Weather in Paris?",
                    "answer": "Call: get_weather(city='Paris')",
                },
                {
                    "prompt": "Function:\ndef get_weather(city: str):\n    '''Get weather'''\n\nFunction:\ndef book_flight(dest: str):\n    '''Book flight'''\n\nUser Query: Fly to Tokyo",
                    "answer": "Call: book_flight(dest='Tokyo')",
                },
            ]
            prep_nexus = auto_prepare_dataset(nexus_data)
            assert prep_nexus[0]["decision_type"] == "choice"
            assert prep_nexus[0]["options"] == ["get_weather", "book_flight"]
            assert prep_nexus[0]["target"] == "get_weather"

            # 2. mteb/tweet_sentiment_extraction (choice: auto-detected global labels & explicit column_map)
            tweet_data = [
                {"text": "I love this!", "label": 2, "label_text": "positive"},
                {"text": "It is okay.", "label": 1, "label_text": "neutral"},
                {"text": "Terrible service.", "label": 0, "label_text": "negative"},
            ]
            prep_tweet = auto_prepare_dataset(tweet_data)
            assert prep_tweet[0]["decision_type"] == "choice"
            assert sorted(prep_tweet[0]["options"]) == ["negative", "neutral", "positive"]
            assert prep_tweet[0]["target"] == "positive"

            prep_tweet_custom = auto_prepare_dataset(
                tweet_data,
                column_map={
                    "context": "text",
                    "target": "label",
                    "options": ["negative", "neutral", "positive"],
                    "decision_type": "choice",
                },
            )
            assert prep_tweet_custom[0]["target"] == "positive"
            assert prep_tweet_custom[2]["target"] == "negative"

            # 3. Anthropic/hh-rlhf (noul: contrastive chosen/rejected pairs)
            hh_data = [
                {"chosen": "Human: Hi\nAssistant: Hello!", "rejected": "Human: Hi\nAssistant: Go away."},
            ]
            prep_hh = auto_prepare_dataset(hh_data)
            assert len(prep_hh) == 2
            assert prep_hh[0]["decision_type"] == "noul" and prep_hh[0]["target"] is True
            assert prep_hh[1]["decision_type"] == "noul" and prep_hh[1]["target"] is False

            # 4. lmsys/toxic-chat (auto-detected noul on toxicity & explicit column_map score on moderation)
            toxic_data = [
                {
                    "user_input": "Hello friend",
                    "model_output": "Hi there!",
                    "toxicity": 0,
                    "openai_moderation": '[["harassment", 0.01], ["hate", 0.002]]',
                },
                {
                    "user_input": "Say something rude",
                    "model_output": "You are awful",
                    "toxicity": 1,
                    "openai_moderation": '[["harassment", 0.92], ["hate", 0.45]]',
                },
            ]
            prep_toxic_auto = auto_prepare_dataset(toxic_data)
            assert prep_toxic_auto[0]["decision_type"] == "noul" and prep_toxic_auto[0]["target"] is False
            assert prep_toxic_auto[1]["decision_type"] == "noul" and prep_toxic_auto[1]["target"] is True

            prep_toxic_score = auto_prepare_dataset(
                toxic_data,
                column_map={
                    "context": ["user_input", "model_output"],
                    "target": "openai_moderation",
                    "decision_type": "score",
                    "scale": (1, 5),
                },
            )
            assert prep_toxic_score[0]["decision_type"] == "score"
            assert prep_toxic_score[0]["scale"] == [1, 5]

            print("PASS: auto_prepare_dataset works across all 4 dataset types and column_map overrides.")
            return True
        except Exception as e:
            print(f"FAIL: auto_prepare_dataset raised {type(e).__name__}: {e}")
            return False

    def test_build_training_item(self):
        print("\n=== Smoke test: build_training_item (default & custom marker_token_id) ===")
        try:
            from processing import build_training_item, build_sequence, render_options
            from models import Laya

            loaded = Laya.load_model(DEFAULT_LAYA_MODEL, device="cpu")
            tok = loaded["tokenizer"]
            q = {"type": "choice", "instructions": "Pick one", "criteria": {"opt_a": "first", "opt_b": "second"}}
            gold = {"probabilities": {"opt_a": 1.0, "opt_b": 0.0}, "label": "opt_a"}

            # 1. Default packing rule (marker_token_id=None -> uses tok.mask_token_id)
            item_default = build_training_item("sample state", q, gold, tok, marker_token_id=None)
            assert item_default is not None
            for m_pos in item_default["markers"]:
                assert item_default["ids"][m_pos] == tok.mask_token_id

            # 2. Custom marker_token_id override
            custom_marker_id = 999
            item_custom = build_training_item("sample state", q, gold, tok, marker_token_id=custom_marker_id)
            assert item_custom is not None
            for m_pos in item_custom["markers"]:
                assert item_custom["ids"][m_pos] == custom_marker_id

            print("PASS: build_training_item works with both default mask_token_id and custom marker_token_id.")
            return True
        except Exception as e:
            print(f"FAIL: build_training_item raised {type(e).__name__}: {e}")
            return False

    def test_build_item_from_row(self):
        print("\n=== Smoke test: build_item_from_row ===")
        try:
            from processing import build_item_from_row
            
            assert callable(build_item_from_row)
            print("PASS: build_item_from_row exists.")
            return True
        except Exception as e:
            print(f"FAIL: build_item_from_row raised {type(e).__name__}: {e}")
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

    def test_split_dataset_items(self):
        print("\n=== Smoke test: split_dataset_items ===")
        try:
            from processing import split_dataset_items

            dummy_items = [{"ids": [1, 2], "markers": [1], "target": [1.0], "qtype": 0, "label": 0} for _ in range(10)]
            train_items, calib_items, test_items = split_dataset_items(
                dummy_items, test_ratio=0.2, calib_ratio=0.1, calib_max=5, seed=42
            )
            assert len(train_items) == 7 and len(calib_items) == 1 and len(test_items) == 2
            print("PASS: split_dataset_items partitions train, calib, and test splits accurately.")
            return True
        except Exception as e:
            print(f"FAIL: split_dataset_items raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Runs all smoke tests for this class."""
        results = {
            "auto_prepare_dataset": self.test_auto_prepare_dataset(),
            "build_training_item": self.test_build_training_item(),
            "build_item_from_row": self.test_build_item_from_row(),
            "prepare_dataset_items": self.test_prepare_dataset_items(),
            "collate_train_batch": self.test_collate_train_batch(),
            "split_dataset_items": self.test_split_dataset_items(),
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
    parser.add_argument("--models", "--laya", dest="models", action="store_true", help="Run smoke tests for models.py")
    parser.add_argument("--processing", action="store_true", help="Run smoke tests for processing.py")
    parser.add_argument("--training", action="store_true", help="Run smoke tests for training.py")
    parser.add_argument("--all", action="store_true", help="Run all smoke tests across all files")
    
    args = parser.parse_args()

    # If no flags are provided, run all by default
    if not (args.helper or args.models or args.processing or args.training or args.all):
        args.all = True

    all_success = True
    successes = {}

    if args.all or args.helper:
        print("\nRunning HelperSmokeTests...")
        success = HelperSmokeTests().run_all()
        all_success = all_success and success
        successes["helper"] = success

    if args.all or args.models:
        print("\nRunning ModelsSmokeTests...")
        success = ModelsSmokeTests().run_all()
        all_success = all_success and success
        successes["models"] = success

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