"""Laya model loading and adaptation module.

Handles downloading, caching, and building the `convaiinnovations/laya`
model architecture. Translates raw dataset configurations into format-agnostic
Laya inputs.
"""

import json
import os
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_model
from safetensors.torch import load_file
from transformers import AutoTokenizer


VALID_DECISION_TYPES = ("choice", "score", "noul")
DEFAULT_CACHE_DIR = os.getenv("DEFAULT_CACHE_DIR", "./.cache") 


class Laya:
    """Namespace for loading Laya and converting raw dataset rows into Laya's input shape.
    """

    @staticmethod
    def load_model(model_name=None, device="cpu"):
        """Download and build a Laya checkpoint's trainable pieces, for fine-tuning.

        Args:
            model_name (str, optional): Which checkpoint in the Laya
                family to download, e.g. "convaiinnovations/laya".
                Falls back to the DEFAULT_LAYA_MODEL environment variable.
            device (str): Device to move the model onto, e.g. "cpu" or
                "cuda". Defaults to "cpu".

        Returns:
            dict: {"model": the built, weight-loaded model (torch.nn.Module),
            "tokenizer": its AutoTokenizer, "config": the parsed
            rl_agent_config.json dict, "model_dir": the local directory
            the checkpoint was downloaded into}.
        """


        if model_name is None:
            model_name = os.getenv("DEFAULT_LAYA_MODEL")

        DEFAULT_LAYA_CACHE_DIR = DEFAULT_CACHE_DIR+f"/laya/{model_name}/"

        already_cached = os.path.isdir(DEFAULT_LAYA_CACHE_DIR) and os.path.exists(
            os.path.join(DEFAULT_LAYA_CACHE_DIR, "rl_agent_config.json")
        )

        if already_cached:
            print(f"Found '{model_name}' already cached at {DEFAULT_LAYA_CACHE_DIR}, skipping download.")
            model_dir = DEFAULT_LAYA_CACHE_DIR
        else:
            print(f"Downloading Laya checkpoint '{model_name}'...")
            model_dir = snapshot_download(model_name, local_dir=DEFAULT_LAYA_CACHE_DIR)

        _fix_tokenizer_config(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
            config = json.load(f)

        print(f"Building model for '{model_name}'...")
        model = build_model(config, encoder_dir=os.path.join(model_dir, "encoder"))

        weights_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(weights_path):
            model.load_state_dict(load_file(weights_path), strict=True)

        model.to(device)
        print(f"Loaded '{model_name}' for fine-tuning (device='{device}').")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "model_dir": model_dir,
        }
