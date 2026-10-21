"""RLCD Python SDK for fine-tuning and calibrating open-weight System One decision models."""

from helper import hf_login, load_dataset, save_final_model, save_rolling_checkpoint
from main import build_arg_parser, cli_entry, fine_tune, main
from models import Decision1, DecisionModel, Laya, MarkerDecisionModelWrapper, OpenDecision, Typical, Von
from processing import (
    auto_prepare_dataset,
    build_item_from_row,
    build_sequence,
    build_training_item,
    collate_train_batch,
    infer_decision_primitive,
    prepare_dataset_items,
    render_options,
    resolve_schema_and_columns,
    split_dataset_items,
)
from reward_strategies.brier import Brier
from training import RLTrainer, RLTrainingConfig, compute_rewards, compute_rl_step

__all__ = [
    "fine_tune",
    "main",
    "cli_entry",
    "build_arg_parser",
    "RLTrainingConfig",
    "RLTrainer",
    "compute_rewards",
    "compute_rl_step",
    "DecisionModel",
    "Laya",
    "Von",
    "OpenDecision",
    "Typical",
    "Decision1",
    "MarkerDecisionModelWrapper",
    "auto_prepare_dataset",
    "prepare_dataset_items",
    "split_dataset_items",
    "collate_train_batch",
    "build_training_item",
    "build_item_from_row",
    "build_sequence",
    "render_options",
    "resolve_schema_and_columns",
    "infer_decision_primitive",
    "hf_login",
    "load_dataset",
    "save_rolling_checkpoint",
    "save_final_model",
    "Brier",
]
