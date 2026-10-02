"""Loads a Laya checkpoint for fine-tuning (planning doc's Laya/RLCD scope
correction — see schema.py's module docstring).

REWRITTEN Sept 28. Laya ships as an installable package (`pip install
laya`, PyPI, confirmed real) built around an `Agent` class wrapping a
ModernBERT-family encoder plus a decision head. That package's *inference*
surface (Agent, Router, load()) is well-documented; its *fine-tuning*
surface is not, what's here is built from a third-party reference
implementation that accessed `agent.encoder` / `agent.scorer` directly
(see trainer.py's docstring for the same caveat). Treat the exact
attribute names below as a best-effort placeholder to confirm against the
real package before running anything for real, not as verified fact.

Heavy imports (`laya`, `torch`) stay deferred into function bodies, same
pattern as before, so this file still imports cleanly with neither
installed.
"""

from __future__ import annotations

from .schema import JobSpec


def load_agent(job: JobSpec):
    """Load a Laya Agent for the checkpoint named in job.model.base_model_id.

    NEEDS CONFIRMATION: `laya.load()` is documented for inference
    (`laya.load()` / `Router` / `Agent`), but not explicitly for getting
    back a checkpoint in *training* mode (e.g. does it need a `train=True`
    flag, or a separate loader?). Using `laya.load()` here as the most
    likely entry point; flag this the first time it's actually run.
    """
    import laya

    return laya.load(job.model.base_model_id, auth_token=job.model.auth_token)


def get_trainable_components(agent):
    """Return (encoder, head) from a loaded Agent, the two things the RLCD
    loop in trainer.py updates with separate learning rates.

    NEEDS CONFIRMATION: attribute names `encoder` / `scorer` come from the
    third-party reference script (read via an AI-summarized fetch, not the
    literal source — see trainer.py's docstring). If the real `laya`
    package names these differently, this is the one place to fix it.
    """
    encoder = getattr(agent, "encoder", None)
    head = getattr(agent, "scorer", None) or getattr(agent, "head", None)

    if encoder is None or head is None:
        raise AttributeError(
            "Could not find encoder/scorer(head) attributes on the loaded "
            "Agent — the reference implementation this was modeled on used "
            "agent.encoder and agent.scorer, but that hasn't been confirmed "
            "against the actual installed `laya` package version. Run "
            "`python -c \"import laya; a = laya.load('laya'); print(dir(a))\"` "
            "and update this function with the real attribute names."
        )
    return encoder, head


def count_parameters(agent) -> int:
    """Total parameter count, useful for pre-flight memory estimates."""
    encoder, head = get_trainable_components(agent)
    total = 0
    for module in (encoder, head):
        total += sum(p.numel() for p in module.parameters())
    return total
