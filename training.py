"""Reinforcement Learning (RL) fine-tuning loop for decision models.

Implements the RL training pipeline based on the Laya 2xT4 Kaggle fine-tuning
methodology (https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb):
    - Group Relative Policy Optimization (GRPO) logit exploration with zero-mean projected Gaussian noise.
    - Soft cross-entropy guidance loss + activation regularization.
    - Decoupled parameter optimization (AdamW with separate learning rates for
      the pretrained encoder and the decision head).
    - Cosine annealing learning rate scheduling and gradient accumulation.
    - Held-out post-training temperature calibration (L-BFGS optimization per
      decision type: choice, score, noul).
    - Rolling checkpoints and final agent artifact export.
    - Test split benchmark evaluation (Accuracy, Soft Acc, Brier, ECE, MAE).
    - Distributed Data Parallel (DDP via torchrun) and single-device (GPU/CPU)
      execution.
"""
# Common libraries
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# RL libraries
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

# Relative functions
from models import DecisionModel
from reward_strategies.brier import Brier
from helper import (
    save_rolling_checkpoint,
    save_final_model,
    TemperatureScaling,
    VectorScaling,
    MatrixScaling,
    _fit_calibrator_lbfgs,
    _evaluate_calibrated_logits,
)
from processing import collate_train_batch


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RLTrainingConfig:
    """Hyperparameters and runtime settings for RL fine-tuning."""

    # Model & Checkpoint Paths
    model_name_or_path: str = os.getenv("DEFAULT_LAYA_MODEL", "convaiinnovations/laya")
    output_dir: str = "./laya_finetuned_rl"
    cache_dir: Optional[str] = os.getenv("DEFAULT_CACHE_DIR", "./.cache")

    # Sequence lengths
    max_len: int = 1024
    head_max_len: int = 256
    max_tokens_per_batch: int = 4096
    gradient_checkpointing: bool = True

    # RL & Optimization Hyperparameters
    epochs: int = 4
    micro_batch_size: int = 8
    grad_accum_steps: int = 4
    group_size: int = 4  # G noisy logit distribution samples for GRPO baseline
    lr_encoder: float = 2.5e-5
    lr_head: float = 1.0e-4
    min_lr: float = 1.0e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Exploration Noise Schedule (sigma)
    sigma_start: float = 0.4
    sigma_end: float = 0.1

    # Loss Weights
    ce_weight: float = 1.0  # Weight for soft cross-entropy guidance
    act_reg_weight: float = 0.0  # Penalty on head act summary activations

    # Reward Strategy
    reward_strategy: str = "brier" 

    # Calibration Holdout Settings
    calib_max: int = 400
    calib_ratio: float = 0.1
    calib_seed: int = 20260922
    calib_batch_size: int = 16

    # Environment
    device: Optional[str] = None  # None for auto-detect (cuda / cpu)
    fp16: bool = True  # Mixed precision (CUDA only)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLTrainingConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)





# ---------------------------------------------------------------------------
# Reward Functions & Loss Computation
# ---------------------------------------------------------------------------

def compute_rewards(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    reward_strategy: str = "brier",
    custom_reward_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """Compute rewards for reported probability distributions q vs target.

    Supports:
        - "brier": Negative Brier score sum -((q - target)**2) over options
        - custom_reward_fn: user supplied callable(q, target, qtype, mask)
    """
    if custom_reward_fn is not None:
        return custom_reward_fn(q, target, qtype, mask)

    if reward_strategy == "brier":
        return Brier.brier_reward(q, target, mask)
    else:
        raise ValueError(f"Unknown reward strategy: {reward_strategy}")


def compute_rl_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    group_size: int = 4,
    sigma: float = 0.4,
    ce_weight: float = 1.0,
    act_reg_weight: float = 0.0,
    reward_strategy: str = "brier",
    use_amp: bool = False,
    custom_reward_fn: Optional[Callable] = None,
) -> Tuple[torch.Tensor, float, float, float]:
    """Execute forward pass and compute GRPO policy gradient loss + soft CE.

    Returns:
        (total_loss, rl_loss_val, ce_loss_val, mean_reward_val)
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    marker_pos = batch["marker_pos"].to(device)
    marker_mask = batch["marker_mask"].to(device)
    qtype = batch["qtype"].to(device)
    target = batch["target"].to(device)

    # 1. Forward pass
    if use_amp and device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            logits, act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    else:
        logits, act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)

    logits = logits.float()
    mask = marker_mask
    k = mask.sum(-1, keepdim=True).float()

    # 2. GRPO logit perturbation with zero-mean projection
    # Noise shape: (group_size, batch_size, kmax)
    eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    # 3. Evaluate rewards and advantages
    with torch.no_grad():
        target_expanded = target.unsqueeze(0)
        r = compute_rewards(
            q,
            target_expanded,
            qtype,
            mask,
            reward_strategy=reward_strategy,
            custom_reward_fn=custom_reward_fn,
        )
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    # 4. Policy gradient log-probability and loss
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * (sigma ** 2))
    loss_rl = -(adv * logp).mean()

    # 5. Soft cross-entropy guidance loss
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()

    # 6. Activation penalty
    loss_act = act_reg_weight * act.sum() if act_reg_weight > 0 else 0.0

    total_loss = loss_rl + (ce_weight * loss_ce) + loss_act
    mean_reward = float(r.mean().item())

    return total_loss, float(loss_rl.item()), float(loss_ce.item()), mean_reward


# ---------------------------------------------------------------------------
# High-Level RL Trainer Class
# ---------------------------------------------------------------------------

class RLTrainer:
    """End-to-end Reinforcement Learning Trainer for Decision models."""

    def __init__(
        self,
        config: Optional[RLTrainingConfig] = None,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        custom_reward_fn: Optional[Callable] = None,
    ):
        """Initialize the RL Trainer.

        Args:
            config: Hyperparameters and paths (uses defaults if None).
            model: An optional pre-loaded model (skips checkpoint loading if provided).
            tokenizer: An optional pre-loaded tokenizer.
            agent_config: An optional pre-loaded configuration dictionary.
            custom_reward_fn: A custom callable to compute rewards, overriding default strategies.
        """
        self.config = config or RLTrainingConfig()
        self.custom_reward_fn = custom_reward_fn
        self.calibrator: Optional[nn.Module] = None
        self.calibration_info: Optional[Dict[str, Any]] = None

        # Setup distributed environment
        self._init_distributed()

        # Load or use injected model & tokenizer
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.agent_config = agent_config or {}
            self.model_dir = None
            self.marker_token_id = self.agent_config.get("marker_token_id")
        else:
            self._load_checkpoint()

    def _init_distributed(self):
        """Initialize distributed process group if running under torchrun."""
        is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.is_distributed = is_distributed

        if is_distributed:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cpu")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            if self.config.device:
                self.device = torch.device(self.config.device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_checkpoint(self):
        """Download or load model checkpoint, tokenizer, and config safely across DDP ranks."""
        if self.config.cache_dir:
            os.environ["DEFAULT_CACHE_DIR"] = self.config.cache_dir

        if self.is_distributed:
            if self.rank == 0:
                loaded = DecisionModel.load_model(self.config.model_name_or_path, device=self.device)
            dist.barrier()
            if self.rank != 0:
                loaded = DecisionModel.load_model(self.config.model_name_or_path, device=self.device)
        else:
            loaded = DecisionModel.load_model(self.config.model_name_or_path, device=self.device)

        self.model_dir = loaded["model_dir"]
        self.tokenizer = loaded["tokenizer"]
        self.marker_token_id = loaded.get("marker_token_id")
        cfg = loaded["config"]
        model = loaded["model"]

        cfg["gradient_checkpointing"] = self.config.gradient_checkpointing
        cfg["max_tokens_per_batch"] = self.config.max_tokens_per_batch
        cfg["max_len"] = self.config.max_len
        cfg["head_max_len"] = self.config.head_max_len
        self.agent_config = cfg

        if self.config.gradient_checkpointing and hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = self.config.gradient_checkpointing

        self.model = model

    def hyperparameter_sweep(
        self,
        train_items: List[Dict[str, Any]],
        calib_items: List[Dict[str, Any]],
        test_items: List[Dict[str, Any]],
        param_grid: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run a hyperparameter search across candidate configurations.
        
        Evaluates each candidate on held-out test_items and returns the best configuration.
        """
        print(f"\n========================================================")
        print(f"=== Starting Hyperparameter Sweep ({len(param_grid)} trials) ===")
        print(f"========================================================")

        best_score = -float("inf")
        best_config = None
        best_metrics = None
        results = []

        base_config = self.config

        for trial_idx, params in enumerate(param_grid, start=1):
            print(f"\n--- Running Trial {trial_idx}/{len(param_grid)} with params: {params} ---")
            trial_cfg_dict = base_config.to_dict()
            trial_cfg_dict.update(params)
            trial_cfg_dict["output_dir"] = os.path.join(base_config.output_dir, f"trial_{trial_idx}")
            trial_config = RLTrainingConfig.from_dict(trial_cfg_dict)

            trainer = RLTrainer(config=trial_config)
            trial_result = trainer.train(
                train_items=train_items,
                calib_items=calib_items,
                test_items=test_items,
            )

            final_metrics = trial_result.get("metrics", {}).get("final", {})
            mean_brier = final_metrics.get("mean_brier", -float("inf"))
            ece = final_metrics.get("ece", 1.0)
            acc = final_metrics.get("accuracy", 0.0)

            # Optimization criterion: Higher Brier score (closer to 0) + lower ECE
            # Score = Mean Brier - ECE
            combined_score = mean_brier - (0.5 * ece)

            record = {
                "trial": trial_idx,
                "params": params,
                "accuracy": acc,
                "mean_brier": mean_brier,
                "ece": ece,
                "combined_score": combined_score,
                "output_dir": trial_config.output_dir,
            }
            results.append(record)
            print(f"Trial {trial_idx} Results: Acc: {acc:.2%} | Brier: {mean_brier:.4f} | ECE: {ece:.4f} | Combined: {combined_score:.4f}")

            if combined_score > best_score:
                best_score = combined_score
                best_config = trial_config
                best_metrics = record

        print("\n========================================================")
        print("=== Hyperparameter Sweep Complete: Ranked Results ===")
        print("========================================================")
        results.sort(key=lambda r: r["combined_score"], reverse=True)
        for r in results:
            print(f"Rank {r['trial']}: Score={r['combined_score']:.4f} | Acc={r['accuracy']:.2%} | Brier={r['mean_brier']:.4f} | ECE={r['ece']:.4f} | Params={r['params']}")

        print(f"\nBest Configuration Found in Trial {best_metrics['trial']} with Score {best_score:.4f}:")
        print(json.dumps(best_metrics, indent=2))

        return {"best_config": best_config, "best_metrics": best_metrics, "all_results": results}

    def train(
        self,
        train_items: Optional[List[Dict[str, Any]]] = None,
        calib_items: Optional[List[Dict[str, Any]]] = None,
        test_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the complete RL fine-tuning loop.

        Args:
            train_items: Pre-built training items.
            calib_items: Pre-built calibration items.
            test_items: Optional held-out test items for before/after benchmark evaluation.

        Returns:
            Dict containing output directory and evaluation metrics.
        """
        cfg = self.config

        # 1. Preprocess items check
        if train_items is None or calib_items is None:
            raise ValueError("Must provide 'train_items' and 'calib_items'")

        metrics = {}
        if test_items and self.rank == 0:
            print("\n=== Running Baseline Evaluation on Held-Out Test Set (Pre-Training) ===")
            baseline_eval = self.evaluate(test_items)
            metrics["baseline"] = baseline_eval
            print(
                f"Baseline Test Results -> Accuracy: {baseline_eval['accuracy']:.2%} | "
                f"Mean Brier: {baseline_eval['mean_brier']:.4f} | "
                f"ECE: {baseline_eval['ece']:.4f} (on {baseline_eval['n_items']} test items)"
            )

        # In DDP, trim training items to multiple of world_size and shard
        if self.is_distributed:
            trimmed_len = (len(train_items) // self.world_size) * self.world_size
            train_items = train_items[:trimmed_len]
            my_items = train_items[self.rank :: self.world_size]
        else:
            my_items = train_items

        if self.rank == 0:
            print(
                f"Training set: {len(train_items)} items ({len(calib_items)} held out for calibration) | "
                f"{len(my_items)} items per worker | {cfg.epochs} epochs"
            )

        # 2. Setup DDP wrapper if running multi-gpu
        train_model = self.model
        train_model.train()
        if self.is_distributed and self.device.type == "cuda":
            ddp_model = DDP(train_model, device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            ddp_model = train_model

        # 3. Setup Optimizer with decoupled encoder vs head learning rates
        enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n and p.requires_grad]
        head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": cfg.lr_encoder},
                {"params": head_params, "lr": cfg.lr_head},
            ],
            weight_decay=cfg.weight_decay,
        )

        total_updates = max(1, (len(my_items) // (cfg.micro_batch_size * cfg.grad_accum_steps)) * cfg.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_updates,
            eta_min=cfg.min_lr,
        )

        use_amp = cfg.fp16 and (self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        t_start = time.time()

        # 4. RL Training Loop
        for epoch in range(cfg.epochs):
            random.seed(cfg.seed + epoch + self.rank)
            random.shuffle(my_items)

            epoch_loss = 0.0
            n_batches = 0
            accum_step = 0
            optimizer.zero_grad(set_to_none=True)

            # Linear sigma decay schedule
            progress = epoch / max(1, cfg.epochs - 1)
            sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * progress

            for b_idx in range(0, len(my_items), cfg.micro_batch_size):
                chunk = my_items[b_idx : b_idx + cfg.micro_batch_size]
                if not chunk:
                    continue

                batch = collate_train_batch(chunk, self.tokenizer.pad_token_id)

                loss, rl_loss_val, ce_loss_val, mean_reward = compute_rl_step(
                    model=ddp_model,
                    batch=batch,
                    device=self.device,
                    group_size=cfg.group_size,
                    sigma=sigma,
                    ce_weight=cfg.ce_weight,
                    act_reg_weight=cfg.act_reg_weight,
                    reward_strategy=cfg.reward_strategy,
                    use_amp=use_amp,
                    custom_reward_fn=self.custom_reward_fn,
                )

                loss_scaled = loss / cfg.grad_accum_steps

                if use_amp:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                accum_step += 1

                if (accum_step % cfg.grad_accum_steps == 0) or ((b_idx + cfg.micro_batch_size) >= len(my_items)):
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.max_grad_norm)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                n_batches += 1

                if self.rank == 0 and (n_batches % 25 == 0 or n_batches == 1):
                    cur_lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch + 1}/{cfg.epochs} | Step {n_batches} | "
                        f"Loss: {loss.item():.4f} (RL: {rl_loss_val:.4f}, CE: {ce_loss_val:.4f}) | "
                        f"Reward: {mean_reward:.3f} | sigma: {sigma:.2f} | lr: {cur_lr:.2e}"
                    )

            avg_loss = epoch_loss / max(1, n_batches)
            if self.rank == 0:
                print(
                    f"=== Epoch {epoch + 1}/{cfg.epochs} Completed in {time.time() - t_start:.1f}s | "
                    f"Avg Loss: {avg_loss:.4f} ==="
                )

            if self.is_distributed:
                dist.barrier()

            # Save rolling checkpoint after each epoch on rank 0
            if self.rank == 0:
                save_rolling_checkpoint(
                    model=self.model,
                    tok=self.tokenizer,
                    output_dir=cfg.output_dir,
                    epoch=epoch + 1,
                    total_epochs=cfg.epochs,
                    avg_loss=avg_loss,
                )

        if self.is_distributed:
            dist.barrier()

        # 5. Post-Hoc Calibration & Final Model Package Export (rank 0 only)
        if self.rank == 0:
            del optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if calib_items:
                print(f"\n=== Running Post-Hoc Calibration on {len(calib_items)} Held-Out Calibration Items ===")
                calib_result = self.calibrate(calib_items)
                fitted_temps = calib_result["fitted_temperatures"]
                metrics["post_hoc_calibration"] = self.calibration_info
            else:
                fitted_temps = [1.2, 1.2, 1.2]

            print("\nSaving final model package...")
            final_path = save_final_model(
                model=self.model,
                tok=self.tokenizer,
                cfg=self.agent_config,
                output_dir=cfg.output_dir,
                fitted_temperatures=fitted_temps,
                calibrator=self.calibrator,
                calibration_info=self.calibration_info,
            )
            print(f"\nFine-tuned model successfully saved to: {final_path}")

            # Clean up intermediate rolling checkpoint after final export to save disk space
            rolling_dir = os.path.join(cfg.output_dir, "checkpoint_latest")
            if os.path.isdir(rolling_dir):
                import shutil
                shutil.rmtree(rolling_dir, ignore_errors=True)

        if self.is_distributed:
            dist.destroy_process_group()

        if test_items and self.rank == 0:
            print("\n=== Running Final Evaluation on Held-Out Test Set (Post-Training) ===")
            final_eval = self.evaluate(test_items)
            metrics["final"] = final_eval
            print(
                f"Final Test Results -> Accuracy: {final_eval['accuracy']:.2%} | "
                f"Mean Brier: {final_eval['mean_brier']:.4f} | "
                f"ECE: {final_eval['ece']:.4f} (on {final_eval['n_items']} test items)"
            )
            if "baseline" in metrics:
                b_acc, f_acc = metrics["baseline"]["accuracy"], final_eval["accuracy"]
                b_br, f_br = metrics["baseline"]["mean_brier"], final_eval["mean_brier"]
                b_ece, f_ece = metrics["baseline"]["ece"], final_eval["ece"]
                print(f"Delta Improvement -> Acc: {f_acc - b_acc:+.2%} | Brier: {f_br - b_br:+.4f} | ECE: {f_ece - b_ece:+.4f}")

            # Save metrics to disk
            metrics_path = os.path.join(cfg.output_dir, "test_evaluation_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Test evaluation metrics saved to: {metrics_path}")

        return {"output_dir": cfg.output_dir, "metrics": metrics}

    def calibrate(
        self,
        calib_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run holdout-validated post-hoc calibration on calib_items.

        Evaluates the three calibration strategies:
            1. Temperature Scaling (1 scalar parameter T)
            2. Platt / Vector Scaling (2*K diagonal affine parameters)
            3. Matrix Scaling (K^2 + K linear parameters)

        Uses an internal 70/30 fit/validation split of `calib_items` to select
        the best calibration strategy without overfitting selection bias (never
        selecting an uncalibrated baseline), refits the winning strategy on 100%
        of `calib_items`, and computes per-primitive temperatures for
        `save_final_model` compatibility.
        """
        if not calib_items:
            default_temps = [1.2, 1.2, 1.2]
            self.calibrator = None
            self.calibration_info = None
            return {"fitted_temperatures": default_temps, "calibrator": None, "summary": None}

        model = self.model
        model.eval()
        calib_bs = max(1, min(self.config.calib_batch_size, len(calib_items)))
        use_amp = self.config.fp16 and (self.device.type == "cuda")

        batch_logits_list: List[torch.Tensor] = []
        batch_targets_list: List[torch.Tensor] = []
        batch_masks_list: List[torch.Tensor] = []
        batch_qtypes_list: List[torch.Tensor] = []
        max_k = 0

        with torch.no_grad():
            for b_idx in range(0, len(calib_items), calib_bs):
                chunk = calib_items[b_idx : b_idx + calib_bs]
                if not chunk:
                    continue
                batch = collate_train_batch(chunk, self.tokenizer.pad_token_id)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                marker_pos = batch["marker_pos"].to(self.device)
                marker_mask = batch["marker_mask"].to(self.device)
                qtype = batch["qtype"].to(self.device)
                target = batch["target"].to(self.device)

                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
                else:
                    logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)

                logits = logits.float().detach().cpu()
                target_cpu = target.float().detach().cpu()
                mask_cpu = marker_mask.bool().detach().cpu()
                qtype_cpu = qtype.long().detach().cpu()

                max_k = max(max_k, logits.size(-1))
                batch_logits_list.append(logits)
                batch_targets_list.append(target_cpu)
                batch_masks_list.append(mask_cpu)
                batch_qtypes_list.append(qtype_cpu)

        # Pad across batches to uniform max_k
        padded_logits, padded_targets, padded_masks = [], [], []
        for l_t, t_t, m_t in zip(batch_logits_list, batch_targets_list, batch_masks_list):
            k_cur = l_t.size(-1)
            if k_cur < max_k:
                pad_k = max_k - k_cur
                b_sz = l_t.size(0)
                l_t = torch.cat([l_t, torch.full((b_sz, pad_k), -1e4, dtype=l_t.dtype)], dim=-1)
                t_t = torch.cat([t_t, torch.zeros((b_sz, pad_k), dtype=t_t.dtype)], dim=-1)
                m_t = torch.cat([m_t, torch.zeros((b_sz, pad_k), dtype=torch.bool)], dim=-1)
            padded_logits.append(l_t)
            padded_targets.append(t_t)
            padded_masks.append(m_t)

        all_logits = torch.cat(padded_logits, dim=0)
        all_targets = torch.cat(padded_targets, dim=0)
        all_masks = torch.cat(padded_masks, dim=0)
        all_qtypes = torch.cat(batch_qtypes_list, dim=0)

        n_total = len(all_logits)
        num_classes = max_k

        # Check whether option count K is uniform across all calibration items
        active_counts = all_masks.sum(dim=-1)
        uniform_k = bool((active_counts == num_classes).all().item())

        # Internal Holdout Split (70% fit / 30% validation) to prevent overfitting selection bias
        g = torch.Generator()
        g.manual_seed(self.config.calib_seed)
        perm = torch.randperm(n_total, generator=g)

        if n_total >= 4:
            n_fit = max(2, int(round(n_total * 0.70)))
            n_fit = min(n_fit, n_total - 1)
            fit_idx = perm[:n_fit]
            val_idx = perm[n_fit:]
        else:
            fit_idx = perm
            val_idx = perm

        fit_logits, fit_targets, fit_masks = all_logits[fit_idx], all_targets[fit_idx], all_masks[fit_idx]
        val_logits, val_targets, val_masks, val_qtypes = (
            all_logits[val_idx],
            all_targets[val_idx],
            all_masks[val_idx],
            all_qtypes[val_idx],
        )

        # Candidate builders & eligibility guards (only calibrated strategies compete)
        candidates_spec = [
            {
                "name": "Temperature Scaling",
                "params": 1,
                "builder": lambda: TemperatureScaling(init_temp=1.5),
                "eligible": True,
                "reason": "Universal 1-parameter scalar scaling (preserves 100% of top-1 rankings)",
            },
            {
                "name": "Platt / Vector Scaling",
                "params": 2 * num_classes,
                "builder": lambda: VectorScaling(num_classes=num_classes),
                "eligible": uniform_k and (len(fit_idx) >= 2 * num_classes),
                "reason": (
                    "Diagonal affine scaling (corrects class imbalance)"
                    if (uniform_k and len(fit_idx) >= 2 * num_classes)
                    else f"Skipped: requires uniform option count and >= {2 * num_classes} fit samples (got {len(fit_idx)})"
                ),
            },
            {
                "name": "Matrix Scaling",
                "params": (num_classes ** 2) + num_classes,
                "builder": lambda: MatrixScaling(num_classes=num_classes),
                "eligible": uniform_k and (len(fit_idx) >= 2 * ((num_classes ** 2) + num_classes)),
                "reason": (
                    "Full linear matrix scaling (captures inter-class correlations)"
                    if (uniform_k and len(fit_idx) >= 2 * ((num_classes ** 2) + num_classes))
                    else f"Skipped: requires uniform option count and >= {2 * ((num_classes ** 2) + num_classes)} fit samples to prevent overfitting (got {len(fit_idx)})"
                ),
            },
        ]

        evaluated_candidates = []
        best_spec = None
        best_val_score = -float("inf")
        temp_val_acc = 0.0

        for spec in candidates_spec:
            if not spec["eligible"]:
                evaluated_candidates.append(
                    {
                        "strategy": spec["name"],
                        "params": spec["params"],
                        "eligible": False,
                        "notes": spec["reason"],
                    }
                )
                continue

            t0 = time.time()
            cand_module = spec["builder"]()
            _fit_calibrator_lbfgs(cand_module, fit_logits, fit_targets, fit_masks)
            fit_time = time.time() - t0

            with torch.no_grad():
                val_cal_logits = cand_module(val_logits)
            val_metrics = _evaluate_calibrated_logits(
                val_cal_logits,
                val_targets,
                val_qtypes,
                val_masks,
                reward_strategy=self.config.reward_strategy,
                custom_reward_fn=self.custom_reward_fn,
            )

            if spec["name"] == "Temperature Scaling":
                temp_val_acc = val_metrics["accuracy"]

            # Selection objective on held-out calibration validation split:
            # Maximize Mean Brier (which is negative, closer to 0 is better) and minimize ECE,
            # while guarding against validation accuracy degradation relative to Temperature Scaling.
            acc_penalty = max(0.0, temp_val_acc - val_metrics["accuracy"]) * 2.0
            val_score = val_metrics["mean_brier"] - val_metrics["ece"] - acc_penalty

            record = {
                "strategy": spec["name"],
                "params": spec["params"],
                "eligible": True,
                "val_accuracy": val_metrics["accuracy"],
                "val_ece": val_metrics["ece"],
                "val_mean_brier": val_metrics["mean_brier"],
                "val_nll": val_metrics["nll"],
                "val_score": float(val_score),
                "fit_time_s": float(fit_time),
                "notes": spec["reason"],
            }
            evaluated_candidates.append(record)

            if val_score > best_val_score:
                best_val_score = val_score
                best_spec = spec

        # Refit the winning calibration strategy on 100% of calib_items
        t_refit = time.time()
        winning_calibrator = best_spec["builder"]()
        _fit_calibrator_lbfgs(winning_calibrator, all_logits, all_targets, all_masks)
        refit_time = time.time() - t_refit

        with torch.no_grad():
            full_cal_logits = winning_calibrator(all_logits)
        full_calib_metrics = _evaluate_calibrated_logits(
            full_cal_logits,
            all_targets,
            all_qtypes,
            all_masks,
            reward_strategy=self.config.reward_strategy,
            custom_reward_fn=self.custom_reward_fn,
        )

        # Fit per-primitive scalar temperatures [T_choice, T_score, T_noul] for save_final_model compatibility
        global_temp_module = TemperatureScaling(init_temp=1.5)
        _fit_calibrator_lbfgs(global_temp_module, all_logits, all_targets, all_masks)
        global_temp = round(global_temp_module.get_temperature(), 4)

        fitted_temperatures: List[float] = []
        for q_id in (0, 1, 2):
            q_mask = all_qtypes == q_id
            if q_mask.sum().item() >= 2:
                q_temp_mod = TemperatureScaling(init_temp=global_temp)
                _fit_calibrator_lbfgs(q_temp_mod, all_logits[q_mask], all_targets[q_mask], all_masks[q_mask])
                fitted_temperatures.append(round(q_temp_mod.get_temperature(), 4))
            else:
                fitted_temperatures.append(global_temp)

        self.calibrator = winning_calibrator.to(self.device)
        self.calibrator.eval()

        self.calibration_info = {
            "selected_strategy": best_spec["name"],
            "selected_params": best_spec["params"],
            "n_calib_total": n_total,
            "n_calib_fit": int(len(fit_idx)),
            "n_calib_val": int(len(val_idx)),
            "fitted_temperatures": fitted_temperatures,
            "refit_time_s": round(refit_time, 4),
            "full_calib_metrics": full_calib_metrics,
            "candidates": evaluated_candidates,
        }

        # Print calibration summary table
        print("-" * 98)
        print(f"{'Post-Hoc Calibration Strategy':<30} | {'Params':<6} | {'Val Acc':<9} | {'Val ECE':<9} | {'Val Brier':<10} | {'Status'}")
        print("-" * 98)
        for c in evaluated_candidates:
            if not c["eligible"]:
                print(f"{c['strategy']:<30} | {c['params']:<6} | {'N/A':<9} | {'N/A':<9} | {'N/A':<10} | {c['notes']}")
            else:
                is_sel = "SELECTED" if c["strategy"] == best_spec["name"] else "Evaluated"
                print(
                    f"{c['strategy']:<30} | {c['params']:<6} | "
                    f"{c['val_accuracy']:>7.2%}  | {c['val_ece']:>7.2%}  | "
                    f"{c['val_mean_brier']:>10.4f} | {is_sel} ({c['fit_time_s']:.2f}s)"
                )
        print("-" * 98)
        print(
            f"Selected Post-Hoc Calibrator: {best_spec['name']} "
            f"(fitted_temperatures={fitted_temperatures})"
        )

        return {
            "fitted_temperatures": fitted_temperatures,
            "calibrator": self.calibrator,
            "summary": self.calibration_info,
        }

    def evaluate(
        self,
        eval_items: List[Dict[str, Any]],
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Evaluate the model on held-out test items.
        Computes Accuracy, Mean Brier Score, and Expected Calibration Error (ECE).
        Automatically applies `self.calibrator` when post-hoc calibration has been fitted.
        """
        if not eval_items:
            return {"accuracy": 0.0, "mean_brier": 0.0, "ece": 0.0, "n_items": 0}

        model = self.model
        model.eval()
        if self.calibrator is not None:
            self.calibrator.eval()
        eval_bs = min(batch_size, max(4, self.config.micro_batch_size * 2))
        use_amp = self.config.fp16 and (self.device.type == "cuda")

        correct = 0
        total = 0
        brier_sum = 0.0
        confidences = []
        accuracies = []
        all_pred_labels = []
        all_gold_labels = []

        with torch.no_grad():
            for b_idx in range(0, len(eval_items), eval_bs):
                chunk = eval_items[b_idx : b_idx + eval_bs]
                if not chunk:
                    continue
                batch = collate_train_batch(chunk, self.tokenizer.pad_token_id)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                marker_pos = batch["marker_pos"].to(self.device)
                marker_mask = batch["marker_mask"].to(self.device)
                qtype = batch["qtype"].to(self.device)
                target = batch["target"].to(self.device)

                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
                else:
                    logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
                logits = logits.float()
                if self.calibrator is not None:
                    logits = self.calibrator(logits)
                probs = torch.softmax(logits.masked_fill(~marker_mask, -1e4), -1)

                r = compute_rewards(
                    probs,
                    target,
                    qtype,
                    marker_mask,
                    reward_strategy=self.config.reward_strategy,
                    custom_reward_fn=self.custom_reward_fn,
                )
                brier_sum += r.sum().item()

                pred_labels = probs.argmax(dim=-1)
                gold_labels = target.argmax(dim=-1)
                batch_correct = (pred_labels == gold_labels)
                correct += batch_correct.sum().item()
                total += len(chunk)

                all_pred_labels.extend(pred_labels.cpu().tolist())
                all_gold_labels.extend(gold_labels.cpu().tolist())

                batch_conf = probs.max(dim=-1).values
                confidences.extend(batch_conf.cpu().tolist())
                accuracies.extend(batch_correct.cpu().tolist())

        acc = (correct / total) if total > 0 else 0.0
        mean_brier = (brier_sum / total) if total > 0 else 0.0

        # Precision, Recall, F1 calculation
        precision, recall, f1 = 0.0, 0.0, 0.0
        if total > 0 and len(set(all_gold_labels)) > 0:
            from sklearn.metrics import precision_recall_fscore_support
            p_val, r_val, f1_val, _ = precision_recall_fscore_support(
                all_gold_labels, all_pred_labels, average="weighted", zero_division=0
            )
            precision, recall, f1 = float(p_val), float(r_val), float(f1_val)

        # ECE calculation (10 bins)
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        if total > 0:
            conf_t = torch.tensor(confidences)
            acc_t = torch.tensor(accuracies, dtype=torch.float32)
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                in_bin = (conf_t > bin_lower) & (conf_t <= bin_upper) if i > 0 else (conf_t >= bin_lower) & (conf_t <= bin_upper)
                bin_size = in_bin.sum().item()
                if bin_size > 0:
                    bin_acc = acc_t[in_bin].mean().item()
                    bin_conf = conf_t[in_bin].mean().item()
                    ece += (bin_size / total) * abs(bin_acc - bin_conf)

        model.train()
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_brier": mean_brier,
            "ece": ece,
            "n_items": total,
        }