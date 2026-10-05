"""Brier-score reward strategy for RLCD fine-tuning.

"""


class Brier:
    """Namespace for the Brier-score reward strategy."""

    @staticmethod
    def brier_reward(q, target, mask):
        """Compute the Brier reward over batched PyTorch tensors for the RL loop.

        Args:
            q (torch.Tensor): The model's predicted probability distribution.
            target (torch.Tensor): The ground truth probability distribution.
            mask (torch.Tensor): Boolean mask indicating valid options.

        Returns:
            torch.Tensor: The computed Brier reward per sequence.
        """
        diff = (q - target) * mask
        return -(diff ** 2).sum(-1)
    