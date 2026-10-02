"""Tests for Strategy 1: Proper Scoring Rule (Negative Brier Score) as Reward.

Can run immediately without waiting for Ameya's training loop implementation,
validating the mathematical contract, worked examples, and properness properties.
"""

import math
import pytest


def brier_score_reward(confidence: float, outcome: int) -> float:
    """Strategy 1 reward formula: reward = -(confidence - outcome)^2"""
    return -((confidence - outcome) ** 2)


def multiclass_brier_reward(probabilities: list[float], target_idx: int) -> float:
    """Multi-class extension: reward = - sum((q_k - y_k)^2)"""
    reward = 0.0
    for idx, prob in enumerate(probabilities):
        y = 1.0 if idx == target_idx else 0.0
        reward -= (prob - y) ** 2
    return reward


def normalize_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Standardize rewards to advantages: (r - mean) / (std + eps)"""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance)
    return [(r - mean) / (std + eps) for r in rewards]


def compute_sigma(sigma_init: float, sigma_final: float, step: int, total_steps: int) -> float:
    """Linear sigma annealing schedule across training steps."""
    progress = step / max(total_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return sigma_init + (sigma_final - sigma_init) * progress


def blend_losses(rl_loss: float, ce_loss: float, fraction: float) -> float:
    """Blend RL and CE losses: fraction * rl_loss + (1 - fraction) * ce_loss"""
    return fraction * rl_loss + (1.0 - fraction) * ce_loss


# ==============================================================================
# Tier 3.1: Worked Examples from README Table
# ==============================================================================

@pytest.mark.parametrize(
    "case_id,decision,confidence,outcome,expected_reward",
    [
        ("Case #1023", "urgent", 0.92, 1, -0.0064),
        ("Case #1024", "not urgent", 0.60, 1, -0.1600),
        ("Case #1025", "urgent", 0.55, 0, -0.3025),
        ("Case #1026", "urgent", 0.90, 1, -0.0100),
        ("Case #1027", "not urgent", 0.85, 0, -0.7225),
    ],
)
def test_worked_example_rewards(case_id, decision, confidence, outcome, expected_reward):
    """Verifies each exact reward value from the README's worked example table."""
    calc_reward = brier_score_reward(confidence, outcome)
    assert pytest.approx(calc_reward, abs=1e-4) == expected_reward, (
        f"{case_id} failed: got {calc_reward}, expected {expected_reward}"
    )


# ==============================================================================
# Tier 3.2: Strictly Proper Scoring Rule Property
# ==============================================================================

def test_strictly_proper_property():
    """Mathematically proves the Brier score is strictly proper.
    
    If the true probability of event occurrence is p_true, the expected reward
    E[R(q)] = p_true * -(q - 1)^2 + (1 - p_true) * -(q - 0)^2
    must reach its unique global maximum at q = p_true.
    """
    for p_true in [0.2, 0.5, 0.75, 0.9]:
        best_q = None
        max_expected_reward = -float("inf")

        # Test candidate predicted confidences q from 0.01 to 0.99
        for step in range(1, 100):
            q = step / 100.0
            # E[R] = p_true * R(q, 1) + (1 - p_true) * R(q, 0)
            expected_reward = p_true * brier_score_reward(q, 1) + (1 - p_true) * brier_score_reward(q, 0)
            if expected_reward > max_expected_reward:
                max_expected_reward = expected_reward
                best_q = q

        # Peak must occur at q == p_true (within grid resolution)
        assert pytest.approx(best_q, abs=0.02) == p_true, (
            f"Properness failed for p_true={p_true}: peak was at {best_q}"
        )


# ==============================================================================
# Tier 3.3: Multi-class Brier Extension
# ==============================================================================

def test_multiclass_brier_reward():
    """Tests multi-class extension when choosing among N tools."""
    # Tool 0 selected with 0.8 prob, Tool 0 is indeed the target (correct)
    reward_correct = multiclass_brier_reward([0.8, 0.1, 0.1], target_idx=0)
    # -( (0.8-1)^2 + (0.1-0)^2 + (0.1-0)^2 ) = -(0.04 + 0.01 + 0.01) = -0.06
    assert pytest.approx(reward_correct, abs=1e-4) == -0.06

    # Tool 1 selected with 0.8 prob, but Tool 0 was the target (wrong)
    reward_wrong = multiclass_brier_reward([0.1, 0.8, 0.1], target_idx=0)
    # -( (0.1-1)^2 + (0.8-0)^2 + (0.1-0)^2 ) = -(0.81 + 0.64 + 0.01) = -1.46
    assert pytest.approx(reward_wrong, abs=1e-4) == -1.46
    assert reward_correct > reward_wrong


# ==============================================================================
# Tier 3.4: Advantage Normalization
# ==============================================================================

def test_advantage_normalization_properties():
    """Advantages should have mean ~ 0 and std ~ 1."""
    rewards = [-0.0064, -0.1600, -0.3025, -0.0100, -0.7225]
    advantages = normalize_advantages(rewards)

    mean_adv = sum(advantages) / len(advantages)
    variance_adv = sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)

    assert pytest.approx(mean_adv, abs=1e-4) == 0.0
    assert pytest.approx(variance_adv, abs=1e-4) == 1.0


def test_advantage_normalization_zero_variance():
    """If all sampled rewards are identical, advantage normalization should not crash."""
    identical_rewards = [-0.1, -0.1, -0.1, -0.1]
    advantages = normalize_advantages(identical_rewards)
    assert len(advantages) == 4
    for a in advantages:
        assert pytest.approx(a, abs=1e-3) == 0.0


# ==============================================================================
# Tier 3.5: Sigma Annealing Schedule
# ==============================================================================

def test_sigma_annealing_monotonic_decay():
    """Sigma must anneal monotonically from sigma_init down to sigma_final."""
    sigma_init = 1.0
    sigma_final = 0.1
    total_steps = 100

    sigmas = [compute_sigma(sigma_init, sigma_final, step, total_steps) for step in range(total_steps + 1)]

    assert pytest.approx(sigmas[0], abs=1e-5) == sigma_init
    assert pytest.approx(sigmas[-1], abs=1e-5) == sigma_final

    for i in range(len(sigmas) - 1):
        assert sigmas[i] >= sigmas[i + 1], f"Sigma did not decrease at step {i}: {sigmas[i]} < {sigmas[i+1]}"


# ==============================================================================
# Tier 3.6: Loss Blending
# ==============================================================================

def test_loss_blending_weights():
    """Verifies blending of policy gradient loss and cross-entropy loss."""
    rl = 2.0
    ce = 1.0

    assert pytest.approx(blend_losses(rl, ce, fraction=0.0)) == 1.0
    assert pytest.approx(blend_losses(rl, ce, fraction=1.0)) == 2.0
    assert pytest.approx(blend_losses(rl, ce, fraction=0.5)) == 1.5
    assert pytest.approx(blend_losses(rl, ce, fraction=0.25)) == 1.25
