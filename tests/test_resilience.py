"""Resilience and fault-tolerance tests for the fine-tuning pipeline."""

import json
from pathlib import Path
import pytest

from src.train.dataset import DatasetError, load_raw_rows
from src.train.schema import DatasetSpec, JobSpec, TrainingSpec


def test_empty_dataset_handling(tmp_path: Path):
    """Empty JSONL files should be handled cleanly without cryptic crashes."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")

    rows = load_raw_rows(str(empty_file))
    assert rows == []


def test_corrupted_jsonl_line_handling(tmp_path: Path):
    """Corrupted JSON line raises DatasetError with line number."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"context": "valid", "target": "urgent"}\n{bad json}\n')

    with pytest.raises(DatasetError, match=r":2: invalid JSON"):
        load_raw_rows(str(bad_file))


def test_training_spec_sigma_guards():
    """Validates that TrainingSpec prevents invalid sigma schedules."""
    with pytest.raises(ValueError, match="sigma_final should anneal down"):
        TrainingSpec(sigma_init=0.2, sigma_final=0.8)


def test_missing_dataset_path_rejected():
    """Non-existent dataset path should fail pre-flight validation immediately."""
    with pytest.raises(ValueError, match="dataset source_path does not exist"):
        DatasetSpec(source_path="/path/to/nonexistent/data.jsonl")


def test_json_array_loading(tmp_path: Path):
    """Handles standard JSON array format in addition to JSONL."""
    json_file = tmp_path / "data.json"
    data = [
        {"context": "case 1", "decision": "urgent", "outcome": 1},
        {"context": "case 2", "decision": "not urgent", "outcome": 0},
    ]
    json_file.write_text(json.dumps(data))

    rows = load_raw_rows(str(json_file))
    assert len(rows) == 2
    assert rows[0]["context"] == "case 1"
