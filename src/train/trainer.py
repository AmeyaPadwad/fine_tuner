"""RLCD training loop for a Laya checkpoint.

REWRITTEN Sept 28 for the Laya/RLCD scope correction.

======================================================================
IMPORTANT — confidence level of what's in this file
======================================================================
TypeSafe/Convai haven't published an official RLCD training reference that
I've found yet (the convaiinnovations site mentions Kaggle notebooks for
fine-tuning but I haven't pulled them). What's implemented here is a
best-effort reconstruction of the approach used in a third-party
open-source repo (https://github.com/shyamsridhar123/LAYA-RLCD), and even
that repo's content reached me through an AI-summarized fetch rather than
the literal source file — so treat the algorithm shape below (perturbed-
logit sampling, proper-scoring-rule reward, REINFORCE-style policy
gradient blended with cross-entropy) as a plausible, structurally sound
starting point, NOT a verified match to any official spec. Each place this
matters is marked "NEEDS CONFIRMATION" below. Before trusting a real
training run's output, get the official reference and diff this file
against it.
======================================================================

The shape of the loop, per the reference:
  1. Encode context -> logits over the decision's output space.
  2. Sample `reward_samples` Gaussian perturbations of those logits (noise
     scale `sigma`, annealed from sigma_init to sigma_final over training).
  3. Turn each perturbed logit vector into a probability distribution `q`
     (softmax, masked to valid options).
  4. Score each `q` against the target with a proper scoring rule
     (spherical score + ranked probability score) to get a reward.
  5. Normalize rewards into advantages, use them to weight a policy-
     gradient loss, and blend that with a straightforward cross-entropy
     loss against the target.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .schema import DecisionType, JobSpec


def build_target_tensor(example, decision_type: DecisionType, num_classes: int):
    """Turn a DecisionExample's target into a tensor the loss functions can
    use: a one-hot vector for CHOICE/NOUL, a soft/binned distribution for
    SCORE (bucketed into num_classes bins across its scale).
    """
    import torch

    if decision_type == DecisionType.CHOICE:
        target = torch.zeros(num_classes)
        if example.target is not None:
            idx = example.options.index(example.target)
            target[idx] = 1.0
        # else: abstain — an all-zero target is the "no option is correct"
        # signal; NEEDS CONFIRMATION whether Laya expects an explicit extra
        # "none" class instead of an all-zero vector.
        return target

    if decision_type == DecisionType.NOUL:
        return torch.tensor([1.0, 0.0] if not example.target else [0.0, 1.0])

    if decision_type == DecisionType.SCORE:
        lo, hi = example.scale
        bin_width = (hi - lo) / num_classes
        bin_idx = min(int((example.target - lo) / bin_width), num_classes - 1)
        target = torch.zeros(num_classes)
        target[bin_idx] = 1.0
        return target

    raise ValueError(f"unsupported decision_type: {decision_type}")


def proper_reward(q, target, w_spherical: float, w_rps: float):
    """Score a predicted distribution `q` against a one-hot-ish `target`
    with a blend of the spherical scoring rule and the ranked probability
    score (RPS). Both are standard proper scoring rules for probabilistic
    forecasts — proper here means a model can't game the score by
    predicting anything other than its true belief.

    NEEDS CONFIRMATION: this is a from-scratch reimplementation, not a
    call into `laya.common.proper_reward` (unclear if that's importable
    from the installed package — try that first in a real environment and
    prefer it over this if it exists, since the exact weighting/sign
    conventions matter and haven't been verified here).
    """
    import torch

    # Spherical score: q_true / ||q||_2, in [0, 1], higher is better.
    spherical = (q * target).sum(dim=-1) / (q.norm(dim=-1) + 1e-8)

    # Ranked probability score: sum of squared differences between
    # cumulative distributions. This is a loss (lower is better), so it's
    # negated below to combine as a reward.
    q_cumsum = q.cumsum(dim=-1)
    target_cumsum = target.cumsum(dim=-1)
    rps = ((q_cumsum - target_cumsum) ** 2).sum(dim=-1)

    return w_spherical * spherical - w_rps * rps


def rlcd_step(
    logits,
    target,
    mask,
    job: JobSpec,
    step: int,
    total_steps: int,
) -> dict[str, Any]:
    """One RLCD update's loss computation (not including the optimizer step
    itself, which build_trainer's loop below handles so gradient
    accumulation composes cleanly).

    Returns a dict with `loss`, `rl_loss`, `ce_loss`, `reward_mean` — all
    tensors except reward_mean (float), matching StepEvent's fields.
    """
    import torch
    import torch.nn.functional as F

    spec = job.training
    progress = step / max(total_steps, 1)
    sigma = spec.sigma_init + (spec.sigma_final - spec.sigma_init) * progress

    eps = torch.randn((spec.reward_samples,) + logits.shape, device=logits.device) * sigma
    perturbed = logits.detach().unsqueeze(0) + eps
    if mask is not None:
        perturbed = perturbed.masked_fill(~mask, -1e4)
    q = torch.softmax(perturbed, dim=-1)

    reward = proper_reward(q, target.unsqueeze(0), spec.w_spherical, spec.w_rps)
    advantage = (reward - reward.mean()) / (reward.std() + 1e-6)

    # log-prob of the perturbed sample's argmax class under the *current*
    # (unperturbed) model distribution — the policy-gradient term.
    # NEEDS CONFIRMATION: this is the most uncertain part of the whole
    # reconstruction (see this file's module docstring).
    log_probs = F.log_softmax(logits, dim=-1)
    sampled_classes = q.argmax(dim=-1)
    logp = log_probs.gather(-1, sampled_classes.unsqueeze(-1)).squeeze(-1)

    rl_loss = -(advantage.detach() * logp).mean()
    ce_loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    fraction = spec.rl_ce_fraction
    loss = fraction * rl_loss + (1 - fraction) * ce_loss

    return {
        "loss": loss,
        "rl_loss": rl_loss.item(),
        "ce_loss": ce_loss.item(),
        "reward_mean": reward.mean().item(),
    }


def run_training(
    job: JobSpec,
    agent,
    train_examples: list,
    eval_examples: list,
) -> dict[str, Any]:
    """Full training loop: encode each example, run rlcd_step, accumulate
    gradients, step the optimizer. Returns final eval metrics.

    NEEDS CONFIRMATION: how to actually get `logits` for an example out of
    a Laya Agent — this assumes `agent.encoder`/`agent.scorer` (see
    model.py) expose something forward()-able the way a standard PyTorch
    module does. If Laya's Agent wraps that differently (e.g. only exposes
    a high-level `.predict()` that isn't autograd-friendly), this loop
    needs to change to call whatever the real gradient-friendly entry
    point is.
    """
    import torch

    from . import model as model_mod
    from .callbacks import JsonlTelemetryCallback

    encoder, head = model_mod.get_trainable_components(agent)
    spec = job.training

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": spec.encoder_lr},
            {"params": head.parameters(), "lr": spec.head_lr},
        ]
    )

    steps_per_epoch = math.ceil(len(train_examples) / spec.batch_size)
    total_steps = max(1, steps_per_epoch * math.ceil(spec.epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=spec.lr_eta_min
    )

    telemetry = JsonlTelemetryCallback(job.output.output_dir, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=spec.mixed_precision)

    step = 0
    for _epoch in range(math.ceil(spec.epochs)):
        for example in train_examples:
            # NEEDS CONFIRMATION: replace with the real Laya forward pass.
            # This is a placeholder shape assumption (num_classes inferred
            # from the example) so the rest of the loop is at least
            # structurally complete and testable against a mock encoder.
            num_classes = len(example.options) if example.options else 2
            logits = encoder(example.context, head=head, num_classes=num_classes)
            target = build_target_tensor(example, example.decision_type, num_classes)

            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=spec.mixed_precision):
                result = rlcd_step(logits, target, None, job, step, total_steps)

            scaler.scale(result["loss"]).backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), spec.grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if step % job.output.logging_steps == 0:
                event = telemetry.build_event(
                    step=step,
                    loss=result["loss"].item(),
                    lr=scheduler.get_last_lr()[0],
                )
                event.rl_loss = result["rl_loss"]
                event.ce_loss = result["ce_loss"]
                event.reward_mean = result["reward_mean"]
                telemetry.write_event(event)

            step += 1

    metrics = evaluate(agent, eval_examples, job)
    metrics_path = Path(job.output.output_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    return {"agent": agent, "metrics": metrics}


def evaluate(agent, eval_examples: list, job: JobSpec) -> dict[str, Any]:
    """Accuracy on CHOICE examples (does argmax match target), mean squared
    error on SCORE, and accuracy on NOUL (thresholded at 0.5). Simple,
    interpretable metrics rather than trying to reconstruct Laya's own
    calibration metric without confirmed access to it.
    """
    from . import model as model_mod

    encoder, head = model_mod.get_trainable_components(agent)
    correct = 0
    total = 0

    for example in eval_examples:
        num_classes = len(example.options) if example.options else 2
        logits = encoder(example.context, head=head, num_classes=num_classes)
        target = build_target_tensor(example, example.decision_type, num_classes)
        if logits.argmax().item() == target.argmax().item():
            correct += 1
        total += 1

    accuracy = correct / total if total else None
    return {"eval_accuracy": accuracy, "eval_count": total, "job_id": job.job_id}
