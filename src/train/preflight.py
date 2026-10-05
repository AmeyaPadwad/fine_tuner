"""Hardware / VRAM feasibility check, run before any heavy dependency loads.

REWRITTEN Sept 28. The old version's estimate was built around LoRA/QLoRA
on a causal LM (bytes/param by quantization method, a LoRA-trainable-
fraction constant). None of that applies to Laya: the reference fine-tune
runs (see trainer.py's docstring) did full fine-tuning of a 421M-parameter
encoder on 2xT4 GPUs, not a quantized/adapter-based approach. So this
estimate is now much simpler: full fp32 model weights + full AdamW
optimizer state (no LoRA fraction) + a coarse activation-memory term.

`torch` stays optional at import time — this sandbox has no GPU and no
torch installed, so `check_hardware()` degrades to reporting that it can't
verify hardware rather than crashing. `estimate_vram_gb` has no torch
dependency and is unit-testable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import JobSpec

# fp32 weights (4 bytes/param) — the reference implementation didn't
# mention quantization or half precision for the base weights, only mixed
# precision (autocast) during the forward/backward pass.
BYTES_PER_PARAM_WEIGHTS = 4

# Full AdamW optimizer state: two fp32 moment buffers per parameter.
OPTIMIZER_STATE_BYTES_PER_PARAM = 8


@dataclass
class VRAMEstimate:
    model_weights_gb: float
    optimizer_state_gb: float
    activation_gb: float
    total_gb: float

    def __str__(self) -> str:
        return (
            f"model={self.model_weights_gb:.2f}GB  "
            f"optimizer={self.optimizer_state_gb:.2f}GB  "
            f"activations(est.)={self.activation_gb:.2f}GB  "
            f"-> total~={self.total_gb:.2f}GB"
        )


def estimate_vram_gb(
    num_params: int,
    job: JobSpec,
    activation_overhead_multiplier: float = 1.3,
) -> VRAMEstimate:
    """Estimate peak VRAM for a full fine-tune of a Laya checkpoint.

    Deliberately simple (see module docstring) — good enough to catch an
    obviously oversized batch/sequence-length config, not to size a real
    run precisely. Calibrate against an actual run once there's a GPU to
    test on (log what's off in docs/failures.md, same as before).
    """
    model_weights_gb = (num_params * BYTES_PER_PARAM_WEIGHTS) / 1e9
    optimizer_state_gb = (num_params * OPTIMIZER_STATE_BYTES_PER_PARAM) / 1e9

    seq_factor = job.dataset.max_seq_length / 512
    batch_factor = job.training.batch_size
    activation_gb = 0.3 * seq_factor * batch_factor

    subtotal = model_weights_gb + optimizer_state_gb + activation_gb
    total_gb = subtotal * activation_overhead_multiplier

    return VRAMEstimate(
        model_weights_gb=model_weights_gb,
        optimizer_state_gb=optimizer_state_gb,
        activation_gb=activation_gb,
        total_gb=total_gb,
    )


@dataclass
class HardwareReport:
    torch_available: bool
    cuda_available: bool
    device_count: int
    devices: list[dict]
    warnings: list[str]

    def summary(self) -> str:
        if not self.torch_available:
            return "torch is not installed in this environment — cannot verify hardware."
        if not self.cuda_available:
            return "No CUDA device visible — training will run on CPU (fine for the smoke test, not for a real run)."
        lines = [f"{self.device_count} CUDA device(s):"]
        for d in self.devices:
            lines.append(f"  - {d['name']}: {d['total_vram_gb']:.1f} GB total")
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def check_hardware() -> HardwareReport:
    """Inspect available CUDA devices. Safe to call with no torch installed."""
    try:
        import torch
    except ImportError:
        return HardwareReport(
            torch_available=False,
            cuda_available=False,
            device_count=0,
            devices=[],
            warnings=["torch not installed — install requirements.txt on a GPU host before training."],
        )

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return HardwareReport(
            torch_available=True,
            cuda_available=False,
            device_count=0,
            devices=[],
            warnings=[],
        )

    device_count = torch.cuda.device_count()
    devices = []
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        devices.append({"name": props.name, "total_vram_gb": props.total_memory / 1e9})

    return HardwareReport(
        torch_available=True,
        cuda_available=True,
        device_count=device_count,
        devices=devices,
        warnings=[],
    )


def run_preflight(job: JobSpec, num_params: int) -> tuple[HardwareReport, VRAMEstimate, list[str]]:
    hw = check_hardware()
    estimate = estimate_vram_gb(num_params, job)
    advice: list[str] = []

    if hw.cuda_available and hw.devices:
        smallest_device_gb = min(d["total_vram_gb"] for d in hw.devices)
        if estimate.total_gb > smallest_device_gb * 0.9:
            advice.append(
                f"Estimated VRAM ({estimate.total_gb:.2f} GB) is close to or over the "
                f"smallest visible device's capacity ({smallest_device_gb:.1f} GB). "
                "Consider lowering batch_size or max_seq_length."
            )
    elif not hw.torch_available:
        advice.append(
            "Could not verify hardware in this environment. Re-run pre-flight on the "
            "actual training host before starting a real (non-smoke-test) job."
        )

    return hw, estimate, advice
