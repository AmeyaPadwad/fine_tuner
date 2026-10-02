import json

import pytest
from pydantic import ValidationError

from src.train.schema import (
    DEFAULT_BASE_MODEL,
    DatasetSpec,
    DecisionType,
    JobSpec,
    TrainingSpec,
)


def _minimal_job_dict(tmp_data_path: str) -> dict:
    return {
        "job_id": "job_test_01",
        "dataset": {"source_path": tmp_data_path},
    }


def test_job_spec_defaults_to_laya(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}")
    job = JobSpec.model_validate(_minimal_job_dict(str(data_path)))
    assert job.model.base_model_id == DEFAULT_BASE_MODEL
    assert job.dataset.decision_type == DecisionType.CHOICE


def test_model_family_stays_free_form_for_other_laya_checkpoints(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}")
    payload = _minimal_job_dict(str(data_path))
    payload["model"] = {"base_model_id": "laya-multilingual"}
    job = JobSpec.model_validate(payload)
    assert job.model.base_model_id == "laya-multilingual"


def test_dataset_source_path_must_exist(tmp_path):
    missing = tmp_path / "does-not-exist.jsonl"
    with pytest.raises(ValidationError, match="does not exist"):
        DatasetSpec(source_path=str(missing))


def test_custom_format_requires_column_map(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}")
    with pytest.raises(ValidationError, match="column_map"):
        DatasetSpec(source_path=str(data_path), format="custom")


def test_eval_split_ratio_bounds(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}")
    with pytest.raises(ValidationError):
        DatasetSpec(source_path=str(data_path), eval_split_ratio=1.0)
    with pytest.raises(ValidationError):
        DatasetSpec(source_path=str(data_path), eval_split_ratio=-0.1)


def test_sigma_final_must_not_exceed_sigma_init():
    with pytest.raises(ValidationError, match="anneal down"):
        TrainingSpec(sigma_init=0.1, sigma_final=1.0)


def test_sigma_annealing_down_is_fine():
    spec = TrainingSpec(sigma_init=1.0, sigma_final=0.1)
    assert spec.sigma_final < spec.sigma_init


def test_rl_ce_fraction_bounds():
    with pytest.raises(ValidationError):
        TrainingSpec(rl_ce_fraction=1.5)
    with pytest.raises(ValidationError):
        TrainingSpec(rl_ce_fraction=-0.1)


def test_from_json_file_round_trip(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}")
    config_path = tmp_path / "job_config.json"
    config_path.write_text(json.dumps(_minimal_job_dict(str(data_path))))

    job = JobSpec.from_json_file(config_path)
    assert job.job_id == "job_test_01"


def test_invalid_job_id_type_rejected(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}")
    payload = _minimal_job_dict(str(data_path))
    payload["job_id"] = None
    with pytest.raises(ValidationError):
        JobSpec.model_validate(payload)


def test_decision_type_can_be_score_or_noul(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}")
    payload = _minimal_job_dict(str(data_path))
    payload["dataset"]["decision_type"] = "score"
    job = JobSpec.model_validate(payload)
    assert job.dataset.decision_type == DecisionType.SCORE
