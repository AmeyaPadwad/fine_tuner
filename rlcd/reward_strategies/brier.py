"""Brier-score reward strategy for RLCD fine-tuning.

"""


class Brier:
    """Namespace for the Brier-score reward strategy."""

    @staticmethod
    def reward(confidence, outcome):
        """Compute the Brier reward for one decision.

        Args:
            confidence (float): The model's reported confidence in its
                decision, in [0, 1].
            outcome (int | bool): Whether the decision was actually
                correct (1 / True) or wrong (0 / False), per ground truth.

        Returns:
            float: `-(confidence - outcome) ** 2`. 0 for a perfectly
            calibrated, correct call (confidence=1, outcome=1) or a
            perfectly calibrated, wrong call (confidence=0, outcome=0);
            increasingly negative the more over/under-confident the call
            was.

        Raises:
            ValueError: If confidence isn't in [0, 1], or outcome isn't
                0/1 (or False/True).
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")

        outcome = int(outcome)
        if outcome not in (0, 1):
            raise ValueError(f"outcome must be 0/1 (or False/True), got {outcome!r}")

        return -((confidence - outcome) ** 2)

    @staticmethod
    def batch_reward(rollouts):
        """Compute the Brier reward for a batch of logged rollouts.

        Args:
            rollouts (list[dict]): Logged rollouts

        Returns:
            list[dict]: The same rollouts, each with a new "reward" key
            holding `Brier.reward(confidence, outcome)`.
        """
        print(f"Scoring {len(rollouts)} rollout(s) with the Brier reward...")

        scored = [
            {**rollout, "reward": Brier.reward(rollout["confidence"], rollout["outcome"])}
            for rollout in rollouts
        ]

        print("Done.")
        return scored

    @staticmethod
    def mean_reward(rollouts):
        """Mean Brier reward across a batch of logged rollouts.

        Useful as a single running calibration metric during RLCD
        fine-tuning: it should trend toward 0 (its maximum) as the policy
        becomes better calibrated.

        Args:
            rollouts (list[dict]): Logged rollouts, same shape as
                `batch_reward` expects.

        Returns:
            float: The average of `Brier.reward(...)` across all
            rollouts. 0.0 if `rollouts` is empty.
        """
        if not rollouts:
            return 0.0

        rewards = [Brier.reward(r["confidence"], r["outcome"]) for r in rollouts]
        return sum(rewards) / len(rewards)
