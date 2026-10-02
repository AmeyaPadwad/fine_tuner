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

    def test_reward(self):
        """Smoke-test `Brier.reward`: it should reproduce the exact
        rewards from RLCD Reward Strategies.md's worked example, and
        reject out-of-range confidence/outcome values.
        """
        print("\n=== Smoke test: Brier.reward ===")
        try:
            for row in self.WORKED_EXAMPLE:
                got = Brier.reward(row["confidence"], row["outcome"])
                assert abs(got - row["expected_reward"]) < 1e-9, (
                    f"reward({row['confidence']}, {row['outcome']}) = {got}, "
                    f"expected {row['expected_reward']}"
                )

            try:
                Brier.reward(1.5, 1)
                raise AssertionError("reward() should reject confidence outside [0, 1]")
            except ValueError:
                pass

            try:
                Brier.reward(0.5, 2)
                raise AssertionError("reward() should reject an outcome that isn't 0/1")
            except ValueError:
                pass

            print("PASS: Brier.reward matched the documented worked example and validated inputs.")
            return True
        except Exception as e:
            print(f"FAIL: Brier.reward raised {type(e).__name__}: {e}")
            return False

    def test_batch_reward(self):
        """Smoke-test `Brier.batch_reward`: it should score every rollout
        in a batch, matching `Brier.reward` row-by-row, without losing or
        reordering any rollout.
        """
        print("\n=== Smoke test: Brier.batch_reward ===")
        try:
            scored = Brier.batch_reward(self.WORKED_EXAMPLE)
            assert len(scored) == len(self.WORKED_EXAMPLE), "row count changed"
            for original, row in zip(self.WORKED_EXAMPLE, scored):
                assert row["context"] == original["context"], "row order/context changed"
                assert abs(row["reward"] - original["expected_reward"]) < 1e-9, (
                    f"batch_reward row {row} did not match expected {original['expected_reward']}"
                )
            print(f"PASS: Brier.batch_reward scored all {len(scored)} rollouts correctly.")
            return True
        except Exception as e:
            print(f"FAIL: Brier.batch_reward raised {type(e).__name__}: {e}")
            return False

    def test_mean_reward(self):
        """Smoke-test `Brier.mean_reward`: it should match the average of
        the documented per-row rewards, and return 0.0 for an empty batch.
        """
        print("\n=== Smoke test: Brier.mean_reward ===")
        try:
            expected_mean = sum(r["expected_reward"] for r in self.WORKED_EXAMPLE) / len(self.WORKED_EXAMPLE)
            got_mean = Brier.mean_reward(self.WORKED_EXAMPLE)
            assert abs(got_mean - expected_mean) < 1e-9, (
                f"mean_reward() = {got_mean}, expected {expected_mean}"
            )
            assert Brier.mean_reward([]) == 0.0, "mean_reward([]) should be 0.0"
            print(f"PASS: Brier.mean_reward returned {got_mean:.5f}, matching the documented average.")
            return True
        except Exception as e:
            print(f"FAIL: Brier.mean_reward raised {type(e).__name__}: {e}")
            return False

    def run_all(self):
        """Run every smoke test in this class and print a pass/fail summary.

        Returns:
            bool: True if every smoke test passed, False otherwise.
        """
        results = {
            "reward": self.test_reward(),
            "batch_reward": self.test_batch_reward(),
            "mean_reward": self.test_mean_reward(),
        }

        print("\n--- Brier Smoke Test Summary ---")
        for name, passed in results.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")

        return all(results.values())


if __name__ == "__main__":
    success = BrierSmokeTests().run_all()
    raise SystemExit(0 if success else 1)
