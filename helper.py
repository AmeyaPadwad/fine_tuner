"""Utility functions for RLCD fine-tuning pipeline.

Provides foundational tools like Hugging Face Hub authentication, loading
datasets, and checkpoint/model saving functions to orchestrate RL training.
"""

import os
import json
import time
import datasets
from dotenv import load_dotenv
from huggingface_hub import login
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoTokenizer
from reward_strategies.brier import Brier

load_dotenv()

def hf_login():
    """Authenticate with the Hugging Face Hub using a token from the environment.

    """
    # Authenticate with Hugging Face if HF_TOKEN is defined
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token != "your_huggingface_token_here":
        login(token=hf_token)
        print("Logged in with HF_TOKEN from .env file.")
    else:
        print("No HF_TOKEN found in .env; using cached local Hugging Face credentials if available.")

def load_dataset(dataset_name=None):
    """Load a dataset from the Hugging Face Hub and print a summary of it.

    Args:
        dataset_name (str, optional): Name/identifier of the dataset on the Hugging
            Face Hub (e.g. ``"org/dataset"``), passed directly to
            ``datasets.load_dataset``.

    Returns:
        datasets.DatasetDict or datasets.Dataset: The loaded dataset object,
        as returned by ``datasets.load_dataset``.
    """

    if dataset_name is None:
        dataset_name = os.getenv("DEFAULT_DATASET_NAME", "Salesforce/xlam-function-calling-60k")

    # Default config names for multi-config Hugging Face datasets
    default_configs = {
        "lmsys/toxic-chat": "toxicchat0124",
        "Nexusflow/NexusRaven_API_evaluation": "standardized_queries",
    }
    cfg_name = default_configs.get(dataset_name)

    try:
        dataset = datasets.load_dataset(dataset_name, cfg_name) if cfg_name else datasets.load_dataset(dataset_name)
    except Exception as e:
        if dataset_name == "Salesforce/xlam-function-calling-60k":
            print(f"Notice: '{dataset_name}' is gated on HF Hub ({e}).")
            print("Falling back to public verified mirror 'lockon/xlam-function-calling-60k'...")
            dataset = datasets.load_dataset("lockon/xlam-function-calling-60k")
        else:
            raise e

    print(f"Loaded {dataset_name}...")

    return dataset

# ---------------------------------------------------------------------------
# Model and Checkpoint Saving functions
# ---------------------------------------------------------------------------

def save_rolling_checkpoint(
    model: nn.Module,
    tok: AutoTokenizer,
    output_dir: str,
    epoch: int,
    total_epochs: int,
    avg_loss: float,
) -> str:
    """Save a rolling intermediate checkpoint (checkpoint_latest).

    Args:
        model (nn.Module): The neural network model to save.
        tok (AutoTokenizer): The tokenizer to save.
        output_dir (str): Base output directory. The checkpoint will be saved in `output_dir/checkpoint_latest/`.
        epoch (int): The current training epoch.
        total_epochs (int): The total number of epochs planned.
        avg_loss (float): The average loss for the current epoch.

    Returns:
        str: The path to the saved checkpoint directory.
    """
    ckpt_dir = os.path.join(output_dir, "checkpoint_latest")
    os.makedirs(ckpt_dir, exist_ok=True)

    base_model = model.module if hasattr(model, "module") else model
    sd = {k: v.half().contiguous().cpu() for k, v in base_model.state_dict().items()}
    save_file(sd, os.path.join(ckpt_dir, "model.safetensors"))

    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "config"):
        enc_cfg = base_model.encoder.config
        if getattr(enc_cfg, "layer_types", None) is not None and getattr(enc_cfg, "num_hidden_layers", None) is not None:
            enc_cfg.layer_types = list(enc_cfg.layer_types)[: enc_cfg.num_hidden_layers]
        enc_cfg.save_pretrained(os.path.join(ckpt_dir, "encoder"))

    tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))

    meta = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "avg_loss": avg_loss,
        "timestamp": time.time(),
    }
    with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return ckpt_dir


def save_final_model(
    model: nn.Module,
    tok: AutoTokenizer,
    cfg: Dict[str, Any],
    output_dir: str,
    fitted_temperatures: List[float],
    calibrator: Any = None,
    calibration_info: Dict[str, Any] | None = None,
) -> str:
    """Export the fully fine-tuned model package in Laya Agent format.

    Creates the required artifacts for a Laya agent to be instantly loaded for inference:
        - model.safetensors
        - encoder/
        - tokenizer/
        - rl_agent_config.json (with updated temperatures, post-hoc calibration metadata, fine_tuned=True)
        - post_hoc_calibrator.pt (when a post-hoc calibrator module is provided)

    Args:
        model (nn.Module): The fully fine-tuned model.
        tok (AutoTokenizer): The tokenizer.
        cfg (Dict[str, Any]): The original `rl_agent_config.json` content as a dictionary.
        output_dir (str): Directory to save the final agent package into.
        fitted_temperatures (List[float]): The temperatures calibrated post-training for each decision type.
        calibrator (nn.Module, optional): Fitted post-hoc calibrator module (Temperature/Vector/Matrix scaling).
        calibration_info (Dict[str, Any], optional): Summary dictionary from holdout-validated post-hoc calibration.

    Returns:
        str: The path to the final output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_model = model.module if hasattr(model, "module") else model

    # Save weights
    sd = {k: v.half().contiguous().cpu() for k, v in base_model.state_dict().items()}
    save_file(sd, os.path.join(output_dir, "model.safetensors"))

    # Save encoder config and tokenizer
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "config"):
        enc_cfg = base_model.encoder.config
        if getattr(enc_cfg, "layer_types", None) is not None and getattr(enc_cfg, "num_hidden_layers", None) is not None:
            enc_cfg.layer_types = list(enc_cfg.layer_types)[: enc_cfg.num_hidden_layers]
        enc_cfg.save_pretrained(os.path.join(output_dir, "encoder"))
    tok.save_pretrained(os.path.join(output_dir, "tokenizer"))

    # Save post-hoc calibrator state if provided
    if calibrator is not None and hasattr(calibrator, "state_dict"):
        import torch
        cal_payload = {
            "strategy": calibration_info.get("selected_strategy") if calibration_info else type(calibrator).__name__,
            "class_name": type(calibrator).__name__,
            "state_dict": {k: v.detach().cpu() for k, v in calibrator.state_dict().items()},
        }
        torch.save(cal_payload, os.path.join(output_dir, "post_hoc_calibrator.pt"))

    # Update and save agent config
    cfg_copy = dict(cfg)
    cfg_copy["fine_tuned"] = True
    cfg_copy["model_name"] = os.path.basename(os.path.normpath(output_dir))
    cfg_copy["temperature"] = fitted_temperatures
    cfg_copy.pop("temperature_by_options", None)
    if calibration_info is not None:
        cfg_copy["post_hoc_calibration"] = calibration_info

    with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg_copy, f, indent=2)

    return output_dir


# ---------------------------------------------------------------------------
# Post-Hoc Calibration Modules & Helpers
# ---------------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    """Temperature Scaling (Guo et al., 2017).
    Divides logits by a single learned positive scalar T in [min_temp, max_temp].
    Parameterized smoothly in unconstrained log/logit space so L-BFGS never hits
    a zero-gradient hard clamp boundary.
    Preserves 100% of top-1 prediction rankings and classification accuracy.
    """

    def __init__(self, init_temp: float = 1.2, min_temp: float = 0.5, max_temp: float = 5.0):
        super().__init__()
        self.min_temp = float(min_temp)
        self.max_temp = float(max_temp)
        clamped_init = min(max(float(init_temp), self.min_temp + 1e-3), self.max_temp - 1e-3)
        ratio = (clamped_init - self.min_temp) / (self.max_temp - self.min_temp)
        init_raw = torch.logit(torch.tensor([ratio], dtype=torch.float32))
        self.raw_temp = nn.Parameter(init_raw)

    def _compute_temp(self) -> torch.Tensor:
        return self.min_temp + (self.max_temp - self.min_temp) * torch.sigmoid(self.raw_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = self._compute_temp().to(device=logits.device, dtype=logits.dtype)
        return logits / t

    def get_temperature(self) -> float:
        with torch.no_grad():
            return float(self._compute_temp().item())


class VectorScaling(nn.Module):
    """Platt / Vector Scaling (Diagonal Affine Transformation).
    Learns class-specific scaling factors and biases: z_cal = a * z + b.
    Addresses class frequency imbalance while adding only 2*K parameters.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        k = logits.size(-1)
        if k == self.num_classes:
            return logits * self.scale + self.bias
        elif k < self.num_classes:
            return logits * self.scale[:k] + self.bias[:k]
        else:
            pad_len = k - self.num_classes
            scale = torch.cat([self.scale, torch.ones(pad_len, device=logits.device, dtype=logits.dtype)])
            bias = torch.cat([self.bias, torch.zeros(pad_len, device=logits.device, dtype=logits.dtype)])
            return logits * scale + bias


class MatrixScaling(nn.Module):
    """Full Matrix Scaling (Linear Transformation).
    Learns an inter-class weight matrix and bias vector: z_cal = W * z + b.
    Learns correlations across ambiguous classes (K^2 + K parameters).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(num_classes, num_classes)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(num_classes))
            self.linear.bias.zero_()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.size(-1) == self.num_classes:
            return self.linear(logits)
        return logits


def _fit_calibrator_lbfgs(
    calibrator: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.05,
    l2_reg: float = 1e-2,
) -> nn.Module:
    """Fit a post-hoc calibrator module on soft/one-hot target distributions using L-BFGS.

    Optimizes a proper scoring composite (NLL + Brier score) with sample-size-aware
    L2 regularization toward the identity transform (T=1.0 or W=I, b=0) so small
    calibration splits cannot collapse into extreme overconfidence.
    """
    if logits.numel() == 0 or len(logits) == 0:
        return calibrator

    n_samples = max(1, len(logits))
    # Scale regularization inversely with calibration sample count on small splits
    adaptive_reg = l2_reg * max(1.0, 50.0 / float(n_samples))

    optimizer = torch.optim.LBFGS(
        calibrator.parameters(),
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        cal_logits = calibrator(logits)
        masked_logits = cal_logits.masked_fill(~mask, -1e4)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        probs = torch.softmax(masked_logits, dim=-1)

        nll_loss = -(targets * log_probs * mask.float()).sum(dim=-1).mean()
        brier_loss = (((probs - targets) ** 2) * mask.float()).sum(dim=-1).mean()
        loss = nll_loss + brier_loss

        if isinstance(calibrator, TemperatureScaling):
            t_val = calibrator._compute_temp()
            reg = ((t_val - 1.0) ** 2).mean()
            loss = loss + adaptive_reg * reg
        elif isinstance(calibrator, VectorScaling):
            reg = ((calibrator.scale - 1.0) ** 2).mean() + (calibrator.bias ** 2).mean()
            loss = loss + adaptive_reg * reg
        elif isinstance(calibrator, MatrixScaling):
            eye = torch.eye(calibrator.num_classes, device=logits.device, dtype=logits.dtype)
            reg = ((calibrator.linear.weight - eye) ** 2).mean() + (calibrator.linear.bias ** 2).mean()
            loss = loss + adaptive_reg * reg
        loss.backward()
        return loss

    optimizer.step(closure)
    return calibrator


def _evaluate_calibrated_logits(
    cal_logits: torch.Tensor,
    targets: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    reward_strategy: str = "brier",
    custom_reward_fn: Optional[Callable] = None,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute Accuracy, 10-bin ECE, Mean Brier Score, and NLL on calibrated logits."""
    total = len(targets)
    if total == 0:
        return {"accuracy": 0.0, "ece": 0.0, "mean_brier": 0.0, "nll": 0.0, "avg_confidence": 0.0}

    with torch.no_grad():
        masked_logits = cal_logits.masked_fill(~mask, -1e4)
        probs = torch.softmax(masked_logits, dim=-1)
        log_probs = torch.log_softmax(masked_logits, dim=-1)

        pred_labels = probs.argmax(dim=-1)
        gold_labels = targets.argmax(dim=-1)
        acc_mask = (pred_labels == gold_labels).float()
        acc = float(acc_mask.mean().item())

        confs = probs.max(dim=-1).values
        avg_conf = float(confs.mean().item())

        if custom_reward_fn is not None:
            r = custom_reward_fn(probs, targets, qtype, mask)
        elif reward_strategy == "brier":
            r = Brier.brier_reward(probs, targets, mask)
        else:
            raise ValueError(f"Unknown reward strategy: {reward_strategy}")

        mean_brier = float(r.mean().item())
        nll = float((-(targets * log_probs * mask.float()).sum(dim=-1)).mean().item())

        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
            bin_size = in_bin.sum().item()
            if bin_size > 0:
                bin_acc = acc_mask[in_bin].mean().item()
                bin_conf = confs[in_bin].mean().item()
                ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return {
        "accuracy": acc,
        "ece": float(ece),
        "mean_brier": mean_brier,
        "nll": nll,
        "avg_confidence": avg_conf,
    }