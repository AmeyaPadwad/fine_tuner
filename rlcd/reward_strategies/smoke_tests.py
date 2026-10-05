"""Smoke tests for reward strategies.

Run directly with:
    python smoke_tests.py
"""
from brier import Brier


class BrierSmokeTests:
    """Groups one smoke-test method per function on `Brier`, plus a
    `run_all` entry point that runs every test and prints a summary.

    These are pure-function checks -- no network, no Laya/model
    dependency, no dataset download -- just the Brier formula applied to
    the exact worked example from "Idea 1: Proper Scoring Rule as Reward"
    in RLCD Reward Strategies.md, so results here are deterministic and
    checked against the values documented there.
    """

    # The worked example rollouts from RLCD Reward Strategies.md, with
    # their documented rewards.
    WORKED_EXAMPLE = [
        {"context": "Case #1023", "decision": "urgent", "confidence": 0.92, "outcome": 1, "expected_reward": -0.0064},
        {"context": "Case #1024", "decision": "not urgent", "confidence": 0.60, "outcome": 1, "expected_reward": -0.16},
        {"context": "Case #1025", "decision": "urgent", "confidence": 0.55, "outcome": 0, "expected_reward": -0.3025},
        {"context": "Case #1026", "decision": "urgent", "confidence": 0.90, "outcome": 1, "expected_reward": -0.01},
        {"context": "Case #1027", "decision": "not urgent", "confidence": 0.85, "outcome": 0, "expected_reward": -0.7225},
    ]

    def test_brier_reward(self):
        """Smoke-test `Brier.tensor_brier_reward`: it should compute vectorized
        Brier scores for a PyTorch batch matching the per-sample scalar math.
        """
        import torch
        print("\n=== Smoke test: Brier.tensor_brier_reward ===")
        try:
            q = torch.tensor([
                [0.92, 0.08],
                [0.40, 0.60],
                [0.55, 0.45],
                [0.90, 0.10],
                [0.15, 0.85]
            ])
            target = torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0]
            ])
            # mask the second option for binary decisions like 'noul'
            mask = torch.tensor([
                [True, False],
                [False, True],
                [True, False],
                [True, False],
                [False, True]
            ])
            
            # The mathematical expectation from the markdown doc is calculated on a 
            # single scalar confidence value, whereas our tensor reward sums squared errors 
            # over all valid options. So we check equality with the expected scalar values.
            expected_scores = [-0.0064, -0.16, -0.3025, -0.01, -0.7225]
            
            got = Brier.brier_reward(q, target, mask)
            
            for i, expected in enumerate(expected_scores):
                assert abs(got[i].item() - expected) < 1e-4, f"Row {i} got {got[i].item()} expected {expected}"
            
            print("PASS: Brier.brier_reward matched expected batch calculations.")
            return True
        except Exception as e:
            print(f"FAIL: Brier.brier_reward raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Run every smoke test in this class and print a pass/fail summary.

        Returns:
            bool: True if every smoke test passed, False otherwise.
        """
        results = {
            "brier_reward": self.test_brier_reward(),
        }

        print("\n--- Brier Smoke Test Summary ---")
        for name, passed in results.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")

        return all(results.values())

if __name__ == "__main__":
    success = BrierSmokeTests().run_all()
    raise SystemExit(0 if success else 1)
