"""Structured JSONL telemetry callback (planning doc Section 5.3.4).

Emits one `StepEvent` (schema.py) per logging step to `events.jsonl` inside
the job's output_dir, so a dashboard or another process can tail training
progress without needing to know anything about the Trainer's internals.

`transformers` is optional at import time (see model.py for why this
sandbox needs that): if it isn't installed, `_CallbackBase` falls back to
plain `object`, so this module still imports and the JSONL-writing logic
below can be unit tested without pulling in the real Trainer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .schema import StepEvent

try:
    from transformers import TrainerCallback as _CallbackBase
except ImportError:  # pragma: no cover - exercised in the no-torch sandbox
    _CallbackBase = object


class JsonlTelemetryCallback(_CallbackBase):
    def __init__(self, output_dir: str, total_steps: int):
        self.events_path = Path(output_dir) / "events.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps
        self._start_time = time.monotonic()
        # Truncate any previous run's telemetry for this output_dir.
        self.events_path.write_text("")

    def _vram_used_gb(self) -> Optional[float]:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
        except ImportError:
            pass
        return None

    def build_event(self, step: int, loss: float, lr: float) -> StepEvent:
        elapsed = time.monotonic() - self._start_time
        steps_per_sec = step / elapsed if elapsed > 0 and step > 0 else None
        remaining_steps = max(self.total_steps - step, 0)
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec else None

        return StepEvent(
            step=step,
            total_steps=self.total_steps,
            train_loss=loss,
            lr=lr,
            vram_used_gb=self._vram_used_gb(),
            eta_seconds=eta_seconds,
        )

    def write_event(self, event: StepEvent) -> None:
        with self.events_path.open("a") as f:
            f.write(event.model_dump_json() + "\n")

    # transformers.TrainerCallback hook — only ever invoked when transformers
    # is actually installed and a real Trainer is driving the loop.
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:
            return control
        event = self.build_event(
            step=state.global_step,
            loss=logs["loss"],
            lr=logs.get("learning_rate", 0.0),
        )
        self.write_event(event)
        return control
