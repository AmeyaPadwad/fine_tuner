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
from typing import Any, Dict, List
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoTokenizer

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

    try:
        dataset = datasets.load_dataset(dataset_name)
    except Exception as e:
        if dataset_name == "Salesforce/xlam-function-calling-60k":
            print(f"Notice: '{dataset_name}' is gated on HF Hub ({e}).")
            print("Falling back to public verified mirror 'lockon/xlam-function-calling-60k'...")
            dataset = datasets.load_dataset("lockon/xlam-function-calling-60k")
        else:
            raise e

    print(f"Loaded {dataset_name}...")

    return dataset

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
        base_model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))

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
) -> str:
    """Export the fully fine-tuned model package in Laya Agent format.

    Creates the required artifacts for a Laya agent to be instantly loaded for inference:
        - model.safetensors
        - encoder/
        - tokenizer/
        - rl_agent_config.json (with updated temperatures, fine_tuned=True)

    Args:
        model (nn.Module): The fully fine-tuned model.
        tok (AutoTokenizer): The tokenizer.
        cfg (Dict[str, Any]): The original `rl_agent_config.json` content as a dictionary.
        output_dir (str): Directory to save the final agent package into.
        fitted_temperatures (List[float]): The temperatures calibrated post-training for each decision type.

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
        base_model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
    tok.save_pretrained(os.path.join(output_dir, "tokenizer"))

    # Update and save agent config
    cfg_copy = dict(cfg)
    cfg_copy["fine_tuned"] = True
    cfg_copy["model_name"] = os.path.basename(os.path.normpath(output_dir))
    cfg_copy["temperature"] = fitted_temperatures
    cfg_copy.pop("temperature_by_options", None)

    with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg_copy, f, indent=2)

    return output_dir
