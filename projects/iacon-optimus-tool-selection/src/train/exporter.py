"""Checkpoint saving and ONNX export for a fine-tuned Laya agent.

REWRITTEN Sept 28. The old exporter.py (adapter save / weight merge / GGUF)
was entirely LoRA-and-causal-LM specific and doesn't apply to Laya's
architecture. What does apply: a plain state-dict checkpoint, and ONNX
export, which is Laya's own documented alternate deployment path
(`laya.ONNXAgent`, confirmed via the PyPI package listing — unlike GGUF,
which was never a real Laya feature to begin with).
"""

from __future__ import annotations

from pathlib import Path

from .schema import JobSpec


def save_checkpoint(agent, job: JobSpec) -> Path:
    import torch

    from . import model as model_mod

    checkpoint_dir = Path(job.output.output_dir) / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    encoder, head = model_mod.get_trainable_components(agent)
    torch.save(
        {"encoder": encoder.state_dict(), "head": head.state_dict(), "job_id": job.job_id},
        checkpoint_dir / "model.pt",
    )
    return checkpoint_dir


def export_onnx(agent, job: JobSpec) -> Path:
    """Export via Laya's own ONNX path.

    NEEDS CONFIRMATION: the PyPI listing confirms an `ONNXAgent` class
    exists as an inference-time ONNX Runtime alternative, but not whether
    it also provides an export/conversion utility versus just *loading* an
    already-converted ONNX file. If there's no built-in converter, this
    needs a standard `torch.onnx.export` call against the encoder+head
    instead — check the installed package before relying on this.
    """
    import laya

    onnx_dir = Path(job.output.output_dir) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    out_path = onnx_dir / f"{job.job_id}.onnx"

    if not hasattr(laya, "export_onnx"):
        raise RuntimeError(
            "laya.export_onnx not found on the installed package — "
            "ONNXAgent may only load pre-converted ONNX files rather than "
            "producing them. Check the installed `laya` package's actual "
            "API before relying on export_onnx: true in a job spec."
        )

    laya.export_onnx(agent, str(out_path))
    return out_path


def run_export(agent, job: JobSpec) -> dict[str, str]:
    outputs: dict[str, str] = {}

    if job.output.save_checkpoint:
        outputs["checkpoint_dir"] = str(save_checkpoint(agent, job))

    if job.output.export_onnx:
        outputs["onnx_path"] = str(export_onnx(agent, job))

    return outputs
