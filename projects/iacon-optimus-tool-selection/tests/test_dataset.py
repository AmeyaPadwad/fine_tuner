import json

import pytest

from src.train.dataset import (
    DatasetError,
    abstain_rate,
    load_raw_rows,
    option_cardinality_histogram,
    prepare_dataset,
    row_to_example,
)
from src.train.schema import DatasetFormat, DatasetSpec, DecisionType


def test_load_raw_rows_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n\n')
    rows = load_raw_rows(str(p))
    assert rows == [{"a": 1}, {"a": 2}]


def test_load_raw_rows_bad_json_raises_with_line_number(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(DatasetError, match=r"data\.jsonl:2"):
        load_raw_rows(str(p))


def test_row_to_example_choice_with_call():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.CHOICE)
    row = {
        "context": "user wants weather in SF",
        "options": ["get_weather", "set_reminder"],
        "target": "get_weather",
    }
    example = row_to_example(row, spec)
    assert example.decision_type == DecisionType.CHOICE
    assert example.options == ["get_weather", "set_reminder"]
    assert example.target == "get_weather"


def test_row_to_example_choice_abstain():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.CHOICE)
    row = {"context": "user says thanks", "options": ["get_weather"], "target": None}
    example = row_to_example(row, spec)
    assert example.target is None


def test_row_to_example_choice_target_must_be_in_options():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.CHOICE)
    row = {"context": "x", "options": ["a", "b"], "target": "c"}
    with pytest.raises(DatasetError, match="not one of options"):
        row_to_example(row, spec)


def test_row_to_example_choice_missing_target_key_is_an_error():
    # Distinguish "target key absent" (malformed row) from "target: null" (abstain).
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.CHOICE)
    row = {"context": "x", "options": ["a", "b"]}
    with pytest.raises(DatasetError, match="missing 'target' key"):
        row_to_example(row, spec)


def test_row_to_example_score_in_range():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.SCORE)
    row = {"context": "rate this response", "scale": [0, 5], "target": 3.5}
    example = row_to_example(row, spec)
    assert example.scale == (0.0, 5.0)
    assert example.target == 3.5


def test_row_to_example_score_out_of_range_rejected():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.SCORE)
    row = {"context": "x", "scale": [0, 5], "target": 9}
    with pytest.raises(DatasetError, match="outside scale"):
        row_to_example(row, spec)


def test_row_to_example_score_invalid_scale_rejected():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.SCORE)
    row = {"context": "x", "scale": [5, 0], "target": 3}
    with pytest.raises(DatasetError, match="min must be"):
        row_to_example(row, spec)


def test_row_to_example_noul():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.NOUL)
    row = {"context": "is this spam?", "target": True}
    example = row_to_example(row, spec)
    assert example.target is True


def test_row_to_example_noul_requires_bool():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.NOUL)
    row = {"context": "x", "target": "yes"}  # string, not bool
    with pytest.raises(DatasetError, match="boolean"):
        row_to_example(row, spec)


def test_row_to_example_missing_context():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.NOUL)
    with pytest.raises(DatasetError, match="context"):
        row_to_example({"target": True}, spec)


def test_row_to_example_custom_column_map():
    spec = DatasetSpec(
        source_path=__file__,
        format=DatasetFormat.CUSTOM,
        decision_type=DecisionType.CHOICE,
        column_map={"context": "state", "options": "tools", "target": "picked"},
    )
    row = {"state": "user wants X", "tools": ["a", "b"], "picked": "a"}
    example = row_to_example(row, spec)
    assert example.context == "user wants X"
    assert example.target == "a"


def test_prepare_dataset_skips_malformed_rows_and_splits(tmp_path):
    rows = [
        {"context": f"row {i}", "options": ["a", "b"], "target": "a"} for i in range(9)
    ]
    rows.append({"context": "broken"})  # missing options/target -> skipped
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))

    spec = DatasetSpec(source_path=str(p), decision_type=DecisionType.CHOICE, eval_split_ratio=0.2)
    train, eval_ = prepare_dataset(spec)

    assert len(train) + len(eval_) == 9
    assert len(eval_) >= 1


def test_prepare_dataset_raises_when_every_row_fails(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps({"nonsense": True}))
    spec = DatasetSpec(source_path=str(p), decision_type=DecisionType.CHOICE)
    with pytest.raises(DatasetError, match="every row failed"):
        prepare_dataset(spec)


def test_option_cardinality_histogram():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.CHOICE)
    rows = [
        {"context": "a", "options": ["x", "y"], "target": "x"},
        {"context": "b", "options": ["x", "y"], "target": "y"},
        {"context": "c", "options": ["x", "y", "z"], "target": "z"},
    ]
    examples = [row_to_example(r, spec) for r in rows]
    assert option_cardinality_histogram(examples) == {2: 2, 3: 1}


def test_abstain_rate():
    spec = DatasetSpec(source_path=__file__, decision_type=DecisionType.CHOICE)
    rows = [
        {"context": "a", "options": ["x"], "target": "x"},
        {"context": "b", "options": ["x"], "target": None},
        {"context": "c", "options": ["x"], "target": None},
        {"context": "d", "options": ["x"], "target": "x"},
    ]
    examples = [row_to_example(r, spec) for r in rows]
    assert abstain_rate(examples) == 0.5


def test_abstain_rate_none_when_no_choice_examples():
    assert abstain_rate([]) is None
