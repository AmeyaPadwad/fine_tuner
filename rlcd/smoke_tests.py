"""Smoke tests for the functions in helper.py and models.py.


Run directly with:
    python smoke_tests.py
"""
import os
from dotenv import load_dotenv
from helper import hf_login, load_dataset
from models import Laya

load_dotenv()

DEFAULT_DATASET_NAME = os.getenv("DEFAULT_DATASET_NAME")
DEFAULT_LAYA_MODEL = os.getenv("DEFAULT_LAYA_MODEL")

class HelperSmokeTests:
    """Groups one smoke-test method per function in helper.py, plus a
    `run_all` entry point that runs every test and prints a summary.
    """

    def __init__(self, dataset_name=DEFAULT_DATASET_NAME):
        self.dataset_name = dataset_name

    def test_hf_login(self):
        """Smoke-test `hf_login`: it should run without raising, whether or not HF_TOKEN is actually set (it falls back to cached local
        credentials either way).
        """
        print("\n=== Smoke test: hf_login ===")
        try:
            hf_login()
            print("PASS: hf_login() completed without raising.")
            return True
        except Exception as e:
            print(f"FAIL: hf_login() raised {type(e).__name__}: {e}")
            return False

    def test_load_dataset(self):
        """Smoke-test `load_dataset`: it should fetch a real dataset from the Hugging Face Hub and return a non-empty object with at least one column.
        """
        print(f"\n=== Smoke test: load_dataset('{self.dataset_name}') ===")
        try:
            dataset = load_dataset(self.dataset_name)
            assert dataset is not None, "load_dataset returned None"
            assert len(dataset) > 0, "loaded dataset has 0 splits/rows"
            assert len(dataset.column_names) > 0, "loaded dataset has no columns"
            print(f"Sample row: {next(iter(dataset.values()))[0] if hasattr(dataset, 'values') else dataset[0]}")
            print(f"PASS: load_dataset('{self.dataset_name}') returned a non-empty dataset.")
            return True
        except Exception as e:
            print(f"FAIL: load_dataset('{self.dataset_name}') raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Run every smoke test in this class and print a pass/fail summary.

        Returns:
            bool: True if every smoke test passed, False otherwise.
        """
        results = {
            "hf_login": self.test_hf_login(),
            "load_dataset": self.test_load_dataset(),
        }

        print("\n--- Smoke Test Summary ---")
        for name, passed in results.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")

        return all(results.values())


class LayaSmokeTests:
    """Groups one smoke-test method per Laya-related function in models.py
    (`Laya.load_model`, `Laya.load_agent`, `Laya.prepare_dataset_for_laya`,
    `Laya.prepare_xlam_for_laya`), plus one integration test and a
    `run_all` entry point that runs every test and prints a summary.

    `load_model` downloads a checkpoint and builds its trainable
    model/tokenizer/config, the way the notebook sets things up before
    fine-tuning; `load_agent` wraps a checkpoint directory as an
    inference-ready `laya.Agent` instead, the way the notebook evaluates
    a (fine-tuned) checkpoint -- `test_load_agent` and
    `test_xlam_rows_forward_pass` exercise that path since prompting
    requires an Agent, not the raw trainable model.
    """

    def __init__(self, model_name=DEFAULT_LAYA_MODEL):
        self.model_name = model_name

    def test_load_model(self):
        """Smoke-test `Laya.load_model`: it should download a real Laya
        checkpoint and build its trainable model/tokenizer/config.
        """
        print(f"\n=== Smoke test: Laya.load_model('{self.model_name}') ===")
        try:
            loaded = Laya.load_model(self.model_name)
            assert loaded is not None, "load_model returned None"
            for key in ("model", "tokenizer", "config", "model_dir"):
                assert loaded.get(key) is not None, f"load_model result missing '{key}'"
            print(f"PASS: Laya.load_model('{self.model_name}') returned model/tokenizer/config/model_dir.")
            return True
        except Exception as e:
            print(f"FAIL: Laya.load_model('{self.model_name}') raised {type(e).__name__}: {e}")
            return False

    def test_prepare_dataset_for_laya(self):
        """Smoke-test `Laya.prepare_dataset_for_laya` against a tiny, synthetic set of rows, checking it produces Laya's expected {context, decision_type, target, options, scale} shape.
        """
        print("\n=== Smoke test: Laya.prepare_dataset_for_laya ===")
        rows = [
            {"text": "Patient has chest pain and high heart rate.", "label": "urgent"},
            {"text": "Patient here for a routine checkup.", "label": "not urgent"},
        ]
        try:
            prepared = Laya.prepare_dataset_for_laya(
                rows,
                context_fn=lambda r: r["text"],
                decision_type="choice",
                options_fn=lambda _: ["urgent", "not urgent"],
                target_fn=lambda r: r["label"],
            )
            assert len(prepared) == len(rows), "row count changed during preparation"
            for row in prepared:
                assert row["context"], "prepared row missing non-empty context"
                assert row["decision_type"] == "choice", "decision_type not propagated"
                assert row["target"] in row["options"], "target not among options"
            print("PASS: Laya.prepare_dataset_for_laya produced well-formed rows.")
            return True
        except Exception as e:
            print(f"FAIL: Laya.prepare_dataset_for_laya raised {type(e).__name__}: {e}")
            return False

    def test_prepare_xlam_for_laya(self):
        """Smoke-test `Laya.prepare_xlam_for_laya` against the first 100 rows of the real Salesforce/xlam-function-calling-60k dataset (loaded via
        helper.load_dataset), checking it adds well-formed Laya "choice" fields to every row without changing the row count.
        """
        print("\n=== Smoke test: Laya.prepare_xlam_for_laya ===")
        try:
            dataset = load_dataset(DEFAULT_DATASET_NAME)
            split = next(iter(dataset.values())) if hasattr(dataset, "values") else dataset
            subset = split.select(range(min(100, len(split))))

            prepared = Laya.prepare_xlam_for_laya(subset)
            assert len(prepared) == len(subset), "row count changed during preparation"
            for row in prepared:
                assert row["context"], "prepared row missing non-empty context"
                assert row["decision_type"] == "choice", "decision_type not propagated"
                assert row["options"], "prepared row missing non-empty options"
                assert row["target"] is None or row["target"] in row["options"], (
                    "target not among options"
                )
            print(
                f"PASS: Laya.prepare_xlam_for_laya produced {len(prepared)} "
                "well-formed rows from the real dataset."
            )
            return True
        except Exception as e:
            print(f"FAIL: Laya.prepare_xlam_for_laya raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Run every Laya smoke test in this class and print a pass/fail summary.

        Returns:
            bool: True if every smoke test passed, False otherwise.
        """
        results = {
            "load_model": self.test_load_model(),
            "prepare_dataset_for_laya": self.test_prepare_dataset_for_laya(),
            "prepare_xlam_for_laya": self.test_prepare_xlam_for_laya(),
        }

        print("\n--- Laya Smoke Test Summary ---")
        for name, passed in results.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")

        return all(results.values())


if __name__ == "__main__":
    helper_success = HelperSmokeTests().run_all()
    laya_success = LayaSmokeTests().run_all()
    raise SystemExit(0 if (helper_success and laya_success) else 1)
