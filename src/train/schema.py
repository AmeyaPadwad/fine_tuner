"""Pydantic models for the training job specification and event schema.

REWRITTEN Sept 28 for the corrected scope from Dev's feedback: the target
is the Laya/Jev family of "System One" decision models, fine-tuned with
RLCD (Reinforcement Learning for Calibrated Decisions) — not a causal LLM
fine-tuned with SFT. See docs/decisions.md for the full context on why this
replaced the earlier FunctionGemma/SFT schema.

Laya (the open-source, fine-tunable member of the family — Jev is
closed-source/API-only, per https://wilsonwu.me/en/blog/2026/jev-vs-laya/)
is a bidirectional encoder (ModernBERT-based), not autoregressive. It
never generates text. It answers one of three fixed "decision primitives"
(https://laya.convaiinnovations.com/):

  - choice: pick one of N options, returns a probability distribution
  - score:  place input on an ordinal scale
  - noul:   a yes/no question, returns a calibrated P(true)

Tool selection (this project's actual research question) is a `choice`
problem: pick one tool from the available set, or abstain.

CAVEAT: the RLCD hyperparameters below (reward_samples, sigma schedule,
the spherical/RPS scoring-rule weights, the rl/ce loss blend) are modeled
on a third-party open-source implementation
(https://github.com/shyamsridhar123/LAYA-RLCD), read via an AI-summarized
fetch rather than the literal source — not an official TypeSafe/Convai
spec. Treat these as a reasonable starting point, not confirmed defaults.
Revisit once we have the official Laya fine-tuning reference (the
convaiinnovations site mentions Kaggle notebooks for this that haven't
been pulled yet).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Base English checkpoint — the flagship, most likely fit for tool selection.
# Free-form string field below (not this constant) is what actually makes
# the job model-agnostic within the family: laya-multilingual and
# laya-typed-decisions are equally valid values, per Dev's "any model in
# the Laya/Jev family" instruction.
DEFAULT_BASE_MODEL = "laya"


class DecisionType(str, Enum):
    """Which of Laya's three fixed output shapes this job trains for.
    Determines both the dataset row shape and how the reward is scored.
    """

    CHOICE = "choice"  # tool selection lives here
    SCORE = "score"
    NOUL = "noul"


class DatasetFormat(str, Enum):
    DECISION = "decision"  # rows already shaped as {context, options?, target}
    CUSTOM = "custom"      # arbitrary column mapping onto the same shape


class LRSchedulerType(str, Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"


class ModelSpec(BaseModel):
    # Any checkpoint in the Laya family (laya / laya-multilingual /
    # laya-typed-decisions), loaded via the `laya` package's Agent/load().
    # Jev is deliberately not a valid target here — it's closed-source and
    # not fine-tunable, per the scope correction.
    base_model_id: str = DEFAULT_BASE_MODEL
    auth_token: Optional[str] = None


class DatasetSpec(BaseModel):
    source_path: str
    format: DatasetFormat = DatasetFormat.DECISION
    decision_type: DecisionType = DecisionType.CHOICE
    max_seq_length: int = Field(default=512, gt=0, le=8192)
    eval_split_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    column_map: Optional[dict[str, str]] = None  # required when format == CUSTOM

    @field_validator("source_path")
    @classmethod
    def path_must_exist(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"dataset source_path does not exist: {v}")
        return v

    @model_validator(mode="after")
    def custom_needs_column_map(self) -> "DatasetSpec":
        if self.format == DatasetFormat.CUSTOM and not self.column_map:
            raise ValueError("dataset.format == 'custom' requires a column_map")
        return self


class TrainingSpec(BaseModel):
    """RLCD training config. Field names and defaults follow the
    perturbed-logit / proper-scoring-rule approach described in this
    file's module docstring — see the caveat there before trusting a
    default value in a real run.
    """

    epochs: float = Field(default=4, gt=0)  # the reference run selected epoch 4 by eval accuracy
    batch_size: int = Field(default=8, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)

    # Separate learning rates for the frozen-ish encoder vs. the decision head,
    # matching the reference implementation's optimizer setup.
    encoder_lr: float = Field(default=1e-5, gt=0)
    head_lr: float = Field(default=1e-4, gt=0)
    lr_scheduler_type: LRSchedulerType = LRSchedulerType.COSINE
    lr_eta_min: float = Field(default=1e-6, ge=0)

    # RLCD-specific: how many noise-perturbed output distributions to sample
    # per step, and how that noise is annealed across training.
    reward_samples: int = Field(default=8, gt=0)
    sigma_init: float = Field(default=1.0, gt=0)
    sigma_final: float = Field(default=0.1, gt=0)

    # Proper scoring rule weights (spherical score + ranked probability score)
    # used to turn a perturbed distribution into a reward against the target.
    w_spherical: float = Field(default=0.75, ge=0)
    w_rps: float = Field(default=1.0, ge=0)

    # Blend between the RL (policy-gradient-style) loss and a supervised
    # cross-entropy loss against the target distribution.
    rl_ce_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    grad_clip_norm: float = Field(default=1.0, gt=0)
    mixed_precision: bool = True

    @model_validator(mode="after")
    def sigma_final_not_above_init(self) -> "TrainingSpec":
        if self.sigma_final > self.sigma_init:
            raise ValueError("sigma_final should anneal down from sigma_init, not up")
        return self


class OutputSpec(BaseModel):
    output_dir: str = "outputs/run_01"
    save_checkpoint: bool = True
    # ONNX is Laya's actual documented alternate deployment path
    # (laya.ONNXAgent) — the causal-LM-specific merge_weights/export_gguf
    # fields from the old schema don't apply to this architecture at all.
    export_onnx: bool = False
    logging_steps: int = Field(default=5, gt=0)


class JobSpec(BaseModel):
    job_id: str
    model: ModelSpec = Field(default_factory=ModelSpec)
    dataset: DatasetSpec
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "JobSpec":
        import json

        with open(path) as f:
            return cls.model_validate(json.load(f))


class StepEvent(BaseModel):
    """One line of the events.jsonl telemetry stream. Extended from the SFT
    version with the RL/CE loss breakdown and mean reward, since RLCD's
    combined loss is less informative on its own than watching both terms.
    """

    event: str = "step_completed"
    step: int
    total_steps: int
    train_loss: float  # combined (rl + ce) * fraction, matching trainer.py
    rl_loss: Optional[float] = None
    ce_loss: Optional[float] = None
    reward_mean: Optional[float] = None
    lr: float
    vram_used_gb: Optional[float] = None
    eta_seconds: Optional[float] = None
