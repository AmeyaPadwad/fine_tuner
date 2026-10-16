"""Decision model loading and adaptation module for the RLCD pipeline.

Provides dedicated loader classes for the open-weight decision models listed in
`docs/test_models_datasets.md`:
    1. `Laya` (`convaiinnovations/laya`)
    2. `Von` (`wfzyx/von`)
    3. `OpenDecision` (`open-decision-ai` / `open-decision-150m`)
    4. `Typical` (`Typical` by Oz Labs)
    5. `Decision1` (`vllm-sr/Decision-1.0-Lex-0.6B`)
    6. `DecisionModel`: Unified router that automatically dispatches to the
       appropriate model class based on `model_name`.

Every model class exposes a `load_model(model_name, device="cpu", marker_token_id=None)`
method returning a standardized dictionary:
    `{"model": nn.Module, "tokenizer": AutoTokenizer, "config": dict, "model_dir": str, "marker_token_id": Optional[int]}`
where `model(input_ids, attention_mask, marker_pos, marker_mask, qtype)` returns
`(logits, act_logits)` compatible with `RLTrainer`.
"""

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

VALID_DECISION_TYPES = ("choice", "score", "noul")
DEFAULT_CACHE_DIR = os.getenv("DEFAULT_CACHE_DIR", "./.cache")


# ===========================================================================
# Unified Marker-Readout Decision Wrapper for Encoder Backbones
# ===========================================================================

class MarkerDecisionModelWrapper(nn.Module):
    """Wraps a Hugging Face encoder (or custom backbone) with RLCD marker-readout heads.

    Ensures any encoder-based decision model (Von, OpenDecision, Typical, Decision 1.0,
    or a generic ModernBERT/BERT encoder) conforms to the forward signature expected
    by `training.py`:
        `forward(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits, act_logits)`
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        option_marker_vec: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.type_emb = nn.Embedding(3, hidden_size)
        self.scorer = nn.Linear(hidden_size, 1)
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )
        self.head_checkpointing = False
        if option_marker_vec is not None:
            self.register_buffer("option_marker_vec", option_marker_vec.float())
        else:
            self.option_marker_vec = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ):
        try:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except TypeError:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = h.float()
        h = h + self.type_emb(qtype.clamp(0, 2))[:, None, :]

        idx = marker_pos.clamp(min=0, max=max(0, h.size(1) - 1))[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        if self.option_marker_vec is not None and m.size(-1) == self.option_marker_vec.numel():
            m = m + self.option_marker_vec.to(m.device, dtype=m.dtype)

        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_input = torch.cat([pooled, feats], -1).to(self.act_head[0].weight.dtype)
        act_logits = self.act_head(act_input)
        return logits, act_logits


def _download_or_locate(
    model_name: str,
    family_subdir: str,
    required_file: str = "config.json",
    allow_patterns: Optional[list] = None,
) -> str:
    """Resolve a local path or download/cache a Hugging Face repo snapshot."""
    if os.path.isdir(model_name):
        return model_name

    cache_root = os.getenv("DEFAULT_CACHE_DIR", DEFAULT_CACHE_DIR)
    cache_dir = f"{cache_root}/{family_subdir}/{model_name}/"
    already_cached = os.path.isdir(cache_dir) and os.path.exists(os.path.join(cache_dir, required_file))

    if already_cached:
        print(f"Found '{model_name}' already cached at {cache_dir}, skipping download.")
        return cache_dir

    print(f"Downloading checkpoint '{model_name}'...")
    return snapshot_download(model_name, local_dir=cache_dir, allow_patterns=allow_patterns)


def _load_json_if_exists(path: str) -> Dict[str, Any]:
    """Load a JSON file if it exists, otherwise return an empty dict."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ===========================================================================
# 1. Laya Decision Model (`convaiinnovations/laya`)
# ===========================================================================

class Laya:
    """Loader for the `convaiinnovations/laya` decision model family."""

    DEFAULT_MODEL = "convaiinnovations/laya"

    @staticmethod
    def load_model(model_name=None, device="cpu", marker_token_id=None):
        """Download and build a Laya checkpoint's trainable pieces, for fine-tuning.

        Args:
            model_name (str, optional): Which checkpoint in the Laya
                family to download, e.g. "convaiinnovations/laya".
                Falls back to the DEFAULT_LAYA_MODEL environment variable.
            device (str): Device to move the model onto, e.g. "cpu" or
                "cuda". Defaults to "cpu".
            marker_token_id (int, optional): Optional custom option marker token ID.
                If None, checks the checkpoint config (`marker_token_id` / `option_marker_id`)
                or leaves as None so `processing.py` falls back to `tokenizer.mask_token_id`.

        Returns:
            dict: {"model": the built, weight-loaded model (torch.nn.Module),
            "tokenizer": its AutoTokenizer, "config": the parsed
            rl_agent_config.json dict, "model_dir": the local directory
            the checkpoint was downloaded into, "marker_token_id": custom marker
            token ID or None}.
        """
        if model_name is None:
            model_name = os.getenv("DEFAULT_LAYA_MODEL", Laya.DEFAULT_MODEL)

        model_dir = _download_or_locate(model_name, "laya", required_file="rl_agent_config.json")
        _fix_tokenizer_config(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        config = _load_json_if_exists(os.path.join(model_dir, "rl_agent_config.json"))

        print(f"Building model for '{model_name}'...")
        model = build_model(config, encoder_dir=os.path.join(model_dir, "encoder"))

        weights_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(weights_path):
            model.load_state_dict(load_file(weights_path), strict=True)

        resolved_marker_id = (
            marker_token_id
            if marker_token_id is not None
            else config.get("marker_token_id", config.get("option_marker_id"))
        )

        model.to(device)
        print(f"Loaded '{model_name}' for fine-tuning (device='{device}').")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "model_dir": model_dir,
            "marker_token_id": resolved_marker_id,
        }


# ===========================================================================
# 2. Von Decision Model (`wfzyx/von`)
# ===========================================================================

class Von:
    """Loader for the `wfzyx/von` open-source System One decision model."""

    DEFAULT_MODEL = "wfzyx/von"

    @staticmethod
    def load_model(model_name=None, device="cpu", marker_token_id=None):
        """Download and build a Von checkpoint wrapped for RLCD fine-tuning.

        Args:
            model_name (str, optional): Hugging Face repo ID or local path.
                Defaults to `"wfzyx/von"`.
            device (str): Target device (`"cpu"` or `"cuda"`). Defaults to `"cpu"`.
            marker_token_id (int, optional): Custom option marker token ID override.

        Returns:
            dict: Standardized dict with keys `("model", "tokenizer", "config", "model_dir", "marker_token_id")`.
        """
        model_name = model_name or Von.DEFAULT_MODEL
        model_dir = _download_or_locate(model_name, "von", required_file="config.json")

        tok_dir = os.path.join(model_dir, "tokenizer") if os.path.isdir(os.path.join(model_dir, "tokenizer")) else model_dir
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        config = _load_json_if_exists(os.path.join(model_dir, "config.json"))
        calib = _load_json_if_exists(os.path.join(model_dir, "marker_calibration.json"))
        config.update({"calibration": calib, "model_family": "von"})

        print(f"Building Von model for '{model_name}'...")
        encoder = AutoModel.from_pretrained(model_dir)
        hidden_size = int(getattr(encoder.config, "hidden_size", config.get("hidden_size", 1024)))

        marker_pt_path = os.path.join(model_dir, "option_marker.pt")
        marker_vec = torch.load(marker_pt_path, map_location="cpu", weights_only=True) if os.path.exists(marker_pt_path) else None
        if isinstance(marker_vec, dict):
            marker_vec = next((v for v in marker_vec.values() if isinstance(v, torch.Tensor)), None)

        model = MarkerDecisionModelWrapper(encoder=encoder, hidden_size=hidden_size, option_marker_vec=marker_vec)
        resolved_marker_id = (
            marker_token_id
            if marker_token_id is not None
            else config.get("marker_token_id", config.get("option_marker_id"))
        )

        model.to(device)
        print(f"Loaded '{model_name}' for fine-tuning (device='{device}').")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "model_dir": model_dir,
            "marker_token_id": resolved_marker_id,
        }


# ===========================================================================
# 3. OpenDecision AI (`open-decision-ai` / `open-decision-150m`)
# ===========================================================================

class OpenDecision:
    """Loader for the `open-decision-ai` (`open-decision-150m` / `Tokz-labs/OpenDecision-*`) decision model."""

    DEFAULT_MODEL = "open-decision-150m"
    DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"

    @staticmethod
    def load_model(model_name=None, device="cpu", marker_token_id=None):
        """Load an OpenDecision checkpoint directory, HF repo, or `open-decision-150m` (ModernBERT-base).

        Supports:
            1. `"open-decision-150m"` / `"open-decision-ai"`: Builds the 150M ModernBERT-base
               decision architecture specified by the `open-decision-ai` package when no local
               archive directory is present.
            2. `"Tokz-labs/OpenDecision-Small"` / `"Tokz-labs/OpenDecision-Large"`: Loads the
               Hugging Face OpenDecision checkpoint (`backbone_config.json` + `model.safetensors`).
            3. Local directory containing `open_decision.json` + `config.json` + `model.safetensors`.

        Args:
            model_name (str, optional): Model identifier or local directory.
                Defaults to `"open-decision-150m"`.
            device (str): Target device (`"cpu"` or `"cuda"`). Defaults to `"cpu"`.
            marker_token_id (int, optional): Custom option marker token ID override.

        Returns:
            dict: Standardized dict with keys `("model", "tokenizer", "config", "model_dir", "marker_token_id")`.
        """
        model_name = model_name or OpenDecision.DEFAULT_MODEL

        # If "open-decision-150m" or "open-decision-ai" is requested and isn't a local folder,
        # resolve to its underlying 150M ModernBERT-base encoder on Hugging Face Hub
        hub_target = (
            OpenDecision.DEFAULT_BACKBONE
            if (not os.path.isdir(model_name) and model_name.lower() in ("open-decision-150m", "open-decision-ai", "open-decision"))
            else model_name
        )

        model_dir = _download_or_locate(hub_target, "open_decision", required_file="config.json")

        # Tokz-labs/OpenDecision-* compatibility: Hugging Face AutoTokenizer and AutoConfig require
        # a `model_type` in config.json. Tokz-labs places model_type inside backbone_config.json.
        # Patch config.json locally in cache if model_type is missing so HF utilities succeed.
        cfg_file_path = os.path.join(model_dir, "config.json")
        backbone_cfg_path = os.path.join(model_dir, "backbone_config.json")
        if os.path.exists(cfg_file_path):
            try:
                with open(cfg_file_path, "r", encoding="utf-8") as f:
                    raw_cfg = json.load(f)
                if "model_type" not in raw_cfg:
                    model_type_val = "deberta-v2"
                    if os.path.exists(backbone_cfg_path):
                        with open(backbone_cfg_path, "r", encoding="utf-8") as bf:
                            raw_bcfg = json.load(bf)
                            model_type_val = raw_bcfg.get("model_type", model_type_val)
                    raw_cfg["model_type"] = model_type_val
                    with open(cfg_file_path, "w", encoding="utf-8") as f:
                        json.dump(raw_cfg, f, indent=2)
            except Exception:
                pass

        tok_dir = os.path.join(model_dir, "tokenizer") if os.path.isdir(os.path.join(model_dir, "tokenizer")) else model_dir
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")

        config = _load_json_if_exists(os.path.join(model_dir, "open_decision.json"))
        if not config:
            config = _load_json_if_exists(os.path.join(model_dir, "config.json"))
        config.setdefault("model_family", "open_decision")
        config.setdefault("model_id", model_name)

        print(f"Building OpenDecision model for '{model_name}'...")
        backbone_cfg_path = os.path.join(model_dir, "backbone_config.json")
        enc_dir = os.path.join(model_dir, "encoder") if os.path.isdir(os.path.join(model_dir, "encoder")) else model_dir

        # Ensure backbone_config.json is fetched if missing from cache
        if not os.path.exists(backbone_cfg_path) and not os.path.isdir(hub_target):
            try:
                from huggingface_hub import hf_hub_download
                backbone_cfg_path = hf_hub_download(hub_target, "backbone_config.json", local_dir=model_dir)
            except Exception:
                pass

        if os.path.exists(backbone_cfg_path):
            # Tokz-labs/OpenDecision-* layout: encoder config lives in backbone_config.json
            enc_cfg = AutoConfig.from_pretrained(backbone_cfg_path)
            encoder = AutoModel.from_config(enc_cfg)
            weights_path = os.path.join(model_dir, "model.safetensors")
            if os.path.exists(weights_path):
                sd = load_file(weights_path)
                enc_sd = {
                    (k[len("backbone.") :] if k.startswith("backbone.") else (k[len("encoder.") :] if k.startswith("encoder.") else k)): v
                    for k, v in sd.items()
                }
                encoder.load_state_dict(enc_sd, strict=False)
        elif config.get("backbone"):
            # Fallback to config.json's specified backbone (e.g. microsoft/deberta-v3-xsmall)
            enc_cfg = AutoConfig.from_pretrained(config["backbone"])
            encoder = AutoModel.from_config(enc_cfg)
            weights_path = os.path.join(model_dir, "model.safetensors")
            if os.path.exists(weights_path):
                sd = load_file(weights_path)
                enc_sd = {
                    (k[len("backbone.") :] if k.startswith("backbone.") else (k[len("encoder.") :] if k.startswith("encoder.") else k)): v
                    for k, v in sd.items()
                }
                encoder.load_state_dict(enc_sd, strict=False)
        else:
            try:
                encoder = AutoModel.from_pretrained(enc_dir)
            except Exception:
                # Ultimate fallback to ModernBERT-base
                encoder = AutoModel.from_pretrained(OpenDecision.DEFAULT_BACKBONE)

        hidden_size = int(getattr(encoder.config, "hidden_size", config.get("hidden_size", 768)))
        model = MarkerDecisionModelWrapper(encoder=encoder, hidden_size=hidden_size)

        resolved_marker_id = (
            marker_token_id
            if marker_token_id is not None
            else config.get("marker_token_id", config.get("option_marker_id"))
        )
        if resolved_marker_id is None and getattr(tokenizer, "mask_token_id", None) is None:
            resolved_marker_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 0

        model.to(device)
        print(f"Loaded '{model_name}' for fine-tuning (device='{device}').")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "model_dir": model_dir,
            "marker_token_id": resolved_marker_id,
        }


# ===========================================================================
# 4. Typical Decision Model (`Typical` by Oz Labs)
# ===========================================================================

class Typical:
    """Loader for the `Typical` (Oz Labs, e.g. `OzLabs/typical-small`) open decision model."""

    DEFAULT_MODEL = "OzLabs/typical-small"

    @staticmethod
    def load_model(model_name=None, device="cpu", marker_token_id=None):
        """Load a Typical checkpoint directory or Hugging Face repo for RLCD fine-tuning.

        Supports both `OzLabs/typical-small` (`best.pt` + backbone in `ckpt["args"]`)
        and standard Hugging Face / Laya directory layouts.

        Args:
            model_name (str, optional): Local directory or repo ID for Typical.
                Defaults to `"OzLabs/typical-small"`.
            device (str): Target device (`"cpu"` or `"cuda"`). Defaults to `"cpu"`.
            marker_token_id (int, optional): Custom option marker token ID override.

        Returns:
            dict: Standardized dict with keys `("model", "tokenizer", "config", "model_dir", "marker_token_id")`.
        """
        model_name = model_name or Typical.DEFAULT_MODEL
        model_dir = _download_or_locate(
            model_name,
            "typical",
            required_file="best.pt",
            allow_patterns=["MANIFEST.json", "best.pt", "*.json", "tokenizer/*", "encoder/*"],
        )

        best_pt_path = os.path.join(model_dir, "best.pt")
        if os.path.exists(best_pt_path):
            print(f"Building Typical model from '{best_pt_path}'...")
            ckpt = torch.load(best_pt_path, map_location="cpu", weights_only=True)
            args = ckpt.get("args", {})
            backbone_id = args.get("backbone", "Qwen/Qwen3-1.7B")
            tokenizer = AutoTokenizer.from_pretrained(backbone_id, padding_side="right")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            encoder = AutoModel.from_pretrained(backbone_id, torch_dtype=torch.float32)
            if hasattr(encoder, "language_model") and hasattr(encoder, "visual"):
                encoder = encoder.language_model
            if hasattr(encoder, "config"):
                encoder.config.use_cache = False
            tap_layer = int(args.get("tap_layer", 0))
            if 0 < tap_layer < getattr(encoder.config, "num_hidden_layers", 0) and hasattr(encoder, "layers"):
                encoder.layers = encoder.layers[:tap_layer]
                encoder.config.num_hidden_layers = tap_layer
                if getattr(encoder.config, "layer_types", None) is not None:
                    encoder.config.layer_types = list(encoder.config.layer_types)[:tap_layer]
            # Freeze lower layers of the 1.7B trunk so 16GB T4 GPUs train fast without OOM
            if hasattr(encoder, "layers") and len(encoder.layers) > 4:
                for param in encoder.parameters():
                    param.requires_grad = False
                for layer in encoder.layers[-4:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            hidden_size = int(getattr(encoder.config, "hidden_size", 2048))
            model = MarkerDecisionModelWrapper(encoder=encoder, hidden_size=hidden_size)
            config = {"model_family": "typical", "backbone": backbone_id, **args}
        else:
            tok_dir = os.path.join(model_dir, "tokenizer") if os.path.isdir(os.path.join(model_dir, "tokenizer")) else model_dir
            tokenizer = AutoTokenizer.from_pretrained(tok_dir)
            config = _load_json_if_exists(os.path.join(model_dir, "rl_agent_config.json"))
            if not config:
                config = _load_json_if_exists(os.path.join(model_dir, "config.json"))
            config.setdefault("model_family", "typical")

            print(f"Building Typical model for '{model_name}'...")
            if os.path.exists(os.path.join(model_dir, "rl_agent_config.json")) and os.path.isdir(os.path.join(model_dir, "encoder")):
                model = build_model(config, encoder_dir=os.path.join(model_dir, "encoder"))
                weights_path = os.path.join(model_dir, "model.safetensors")
                if os.path.exists(weights_path):
                    model.load_state_dict(load_file(weights_path), strict=True)
            else:
                enc_dir = os.path.join(model_dir, "encoder") if os.path.isdir(os.path.join(model_dir, "encoder")) else model_dir
                encoder = AutoModel.from_pretrained(enc_dir)
                hidden_size = int(getattr(encoder.config, "hidden_size", config.get("hidden_size", 768)))
                model = MarkerDecisionModelWrapper(encoder=encoder, hidden_size=hidden_size)

        # Fallback marker_token_id if the backbone tokenizer (e.g. Qwen) has no mask_token_id
        resolved_marker_id = (
            marker_token_id
            if marker_token_id is not None
            else config.get("marker_token_id", config.get("option_marker_id"))
        )
        if resolved_marker_id is None and getattr(tokenizer, "mask_token_id", None) is None:
            resolved_marker_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 0

        model.to(device)
        print(f"Loaded '{model_name}' for fine-tuning (device='{device}').")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "model_dir": model_dir,
            "marker_token_id": resolved_marker_id,
        }


# ===========================================================================
# 5. Decision 1.0 (`vllm-sr/Decision-1.0-Lex-0.6B`)
# ===========================================================================

class Decision1:
    """Loader for the `vllm-sr/Decision-1.0-Lex-0.6B` (and Kai) decision model family."""

    DEFAULT_MODEL = "vllm-sr/Decision-1.0-Lex-0.6B"

    @staticmethod
    def load_model(model_name=None, device="cpu", marker_token_id=None):
        """Download and build a Decision 1.0 checkpoint for RLCD fine-tuning.

        Supports both native Decision 1.0 directory layout (`native/encoder`,
        `native/tokenizer`, `native/decision_config.json`) and standard Hugging Face
        `AutoModel` loading.

        Args:
            model_name (str, optional): Repo ID or local path.
                Defaults to `"vllm-sr/Decision-1.0-Lex-0.6B"`.
            device (str): Target device (`"cpu"` or `"cuda"`). Defaults to `"cpu"`.
            marker_token_id (int, optional): Custom option marker token ID override.

        Returns:
            dict: Standardized dict with keys `("model", "tokenizer", "config", "model_dir", "marker_token_id")`.
        """
        model_name = model_name or Decision1.DEFAULT_MODEL
        model_dir = _download_or_locate(
            model_name,
            "decision1",
            required_file=os.path.join("native", "encoder", "model.safetensors"),
            allow_patterns=[
                "config.json",
                "native/decision_config.json",
                "native/tokenizer/*",
                "native/encoder/*",
                " native/decision_heads.safetensors",
            ],
        )

        native_tok_dir = os.path.join(model_dir, "native", "tokenizer")
        std_tok_dir = os.path.join(model_dir, "tokenizer")
        if os.path.isdir(native_tok_dir):
            tok_dir = native_tok_dir
        elif os.path.isdir(std_tok_dir):
            tok_dir = std_tok_dir
        else:
            tok_dir = model_dir

        tokenizer = AutoTokenizer.from_pretrained(tok_dir)

        decision_cfg_path = os.path.join(model_dir, "native", "decision_config.json")
        config = _load_json_if_exists(decision_cfg_path) or _load_json_if_exists(os.path.join(model_dir, "config.json"))
        config.setdefault("model_family", "decision1")

        print(f"Building Decision 1.0 model for '{model_name}'...")
        native_enc_dir = os.path.join(model_dir, "native", "encoder")
        enc_dir = native_enc_dir if os.path.isdir(native_enc_dir) else model_dir
        enc_cfg = AutoConfig.from_pretrained(enc_dir)
        # Ensure compatibility across Transformers versions where ModernBertConfig expects standard rope_scaling
        if getattr(enc_cfg, "rope_scaling", None) and isinstance(enc_cfg.rope_scaling, dict):
            if enc_cfg.rope_scaling.get("rope_type") == "yarn":
                enc_cfg.rope_scaling = None
        encoder = AutoModel.from_pretrained(enc_dir, config=enc_cfg, ignore_mismatched_sizes=True)
        hidden_size = int(getattr(encoder.config, "hidden_size", config.get("hidden_size", 768)))

        model = MarkerDecisionModelWrapper(encoder=encoder, hidden_size=hidden_size)
        resolved_marker_id = (
            marker_token_id
            if marker_token_id is not None
            else config.get("marker_token_id", config.get("option_marker_id"))
        )

        model.to(device)
        print(f"Loaded '{model_name}' for fine-tuning (device='{device}').")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "model_dir": model_dir,
            "marker_token_id": resolved_marker_id,
        }


# ===========================================================================
# 6. Unified DecisionModel Router
# ===========================================================================

class DecisionModel:
    """Unified factory that routes `load_model` to Laya, Von, OpenDecision, Typical, or Decision1."""

    REGISTRY = {
        "laya": Laya,
        "von": Von,
        "open-decision": OpenDecision,
        "opendecision": OpenDecision,
        "typical": Typical,
        "decision-1.0": Decision1,
        "decision1": Decision1,
    }

    @staticmethod
    def resolve_loader_class(model_name: Optional[str]):
        """Select the model loader class matching `model_name`."""
        if not model_name:
            return Laya
        low = model_name.lower()
        if "von" in low:
            return Von
        if "open-decision" in low or "opendecision" in low:
            return OpenDecision
        if "typical" in low:
            return Typical
        if "decision-1.0" in low or "decision1" in low:
            return Decision1
        return Laya

    @staticmethod
    def load_model(model_name=None, device="cpu", marker_token_id=None):
        """Dispatch `load_model` to the appropriate decision model class."""
        loader_cls = DecisionModel.resolve_loader_class(model_name)
        return loader_cls.load_model(
            model_name=model_name,
            device=device,
            marker_token_id=marker_token_id,
        )