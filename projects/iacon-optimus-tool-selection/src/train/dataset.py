"""Data ingestion for Laya's decision primitives (choice / score / noul).

REWRITTEN Sept 28 for the Laya/RLCD scope correction (see schema.py's
module docstring and docs/decisions.md). Laya doesn't take chat-formatted
turns, it takes a context plus a fixed decision shape and returns a
distribution over that shape, so there's no chat template and no
response-masking step — those were SFT/causal-LM concepts that don't apply
here. What replaces them is per-decision-type row validation and building
a target distribution (or scalar/boolean target) the reward function in
trainer.py can score against.

Deliberately still has no dependency on `laya` or `torch` at this stage —
this module only normalizes raw rows into a DecisionExample and doesn't
tokenize. Tokenization happens in model.py/trainer.py against Laya's own
encoder, once that's wired up (see trainer.py's caveats). Keeping this
split means the row-parsing and validation logic here stays unit-testable
with no heavy deps, same as before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .schema import DatasetSpec, DecisionType


class DatasetError(ValueError):
    """Raised on a malformed or empty row. Callers should catch this per-row
    and skip with a warning rather than crash the whole ingestion run.
    """


@dataclass
class DecisionExample:
    """One normalized training example, shape depending on decision_type:

      - CHOICE: `options` is the list of candidate keys, `target` is the
        index into `options` of the correct choice (or None for abstain —
        tool selection's explicit no-tool case represented as an extra
        implicit "none" option, see row_to_example).
      - SCORE: `scale` is (min, max), `target` is a float in that range.
      - NOUL: `options`/`scale` unused, `target` is a bool.
    """

    context: str
    decision_type: DecisionType
    target: Any
    options: Optional[list[str]] = None
    scale: Optional[tuple[float, float]] = None
    metadata: dict = field(default_factory=dict)


def load_raw_rows(source_path: str) -> list[dict]:
    """Read JSONL (one JSON object per line) or a single JSON array file."""
    path = Path(source_path)
    text = path.read_text()

    if path.suffix == ".jsonl":
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DatasetError(f"{source_path}:{lineno}: invalid JSON — {e}") from e
        return rows

    data = json.loads(text)
    if not isinstance(data, list):
        raise DatasetError(f"{source_path}: expected a JSON array of rows")
    return data


def row_to_example(row: dict, spec: DatasetSpec) -> DecisionExample:
    """Normalize one raw row into a DecisionExample, per spec.decision_type.

    Expected row shape (provisional — same status as the old tool_selection
    format was: owned jointly with src/data_gen/ and src/validate/, update
    here if the generator's shape changes):

      CHOICE: {"context": "...", "options": ["get_weather", "set_reminder"],
               "target": "get_weather" | null}
        `target: null` is the explicit abstain / no-tool-call case —
        represented internally as target=None, distinct from any option.

      SCORE:  {"context": "...", "scale": [0, 5], "target": 3.5}

      NOUL:   {"context": "...", "target": true}
    """
    if spec.format.value == "custom":
        mapping = spec.column_map or {}
        row = {
            "context": row.get(mapping.get("context", "context")),
            "options": row.get(mapping.get("options", "options")),
            "scale": row.get(mapping.get("scale", "scale")),
            "target": row.get(mapping.get("target", "target")),
        }

    context = row.get("context")
    if not context or not isinstance(context, str):
        raise DatasetError("row missing non-empty string 'context'")

    dtype = spec.decision_type

    if dtype == DecisionType.CHOICE:
        options = row.get("options")
        if not options or not isinstance(options, list):
            raise DatasetError("choice row missing non-empty 'options' list")
        target = row.get("target", "__missing__")
        if target == "__missing__":
            raise DatasetError("choice row missing 'target' key")
        if target is not None and target not in options:
            raise DatasetError(f"target {target!r} is not one of options {options!r}")
        return DecisionExample(context=context, decision_type=dtype, target=target, options=options)

    if dtype == DecisionType.SCORE:
        scale = row.get("scale")
        if not scale or len(scale) != 2:
            raise DatasetError("score row missing 'scale' as [min, max]")
        lo, hi = float(scale[0]), float(scale[1])
        if lo >= hi:
            raise DatasetError(f"scale min must be < max, got {scale}")
        target = row.get("target")
        if target is None:
            raise DatasetError("score row missing 'target'")
        target = float(target)
        if not (lo <= target <= hi):
            raise DatasetError(f"target {target} outside scale {scale}")
        return DecisionExample(context=context, decision_type=dtype, target=target, scale=(lo, hi))

    if dtype == DecisionType.NOUL:
        if "target" not in row or not isinstance(row["target"], bool):
            raise DatasetError("noul row missing boolean 'target'")
        return DecisionExample(context=context, decision_type=dtype, target=row["target"])

    raise DatasetError(f"unsupported decision_type: {dtype}")


def prepare_dataset(spec: DatasetSpec) -> tuple[list[DecisionExample], list[DecisionExample]]:
    """End-to-end: load rows -> normalize -> validate -> train/eval split.

    Malformed rows are skipped with a count, never raised past this
    function — matches the "skip with warning, not crash" requirement
    carried over from the original fault-injection testing plan.
    """
    raw_rows = load_raw_rows(spec.source_path)
    if not raw_rows:
        raise DatasetError(f"{spec.source_path}: no rows found")

    examples: list[DecisionExample] = []
    skipped = 0
    for row in raw_rows:
        try:
            examples.append(row_to_example(row, spec))
        except DatasetError:
            skipped += 1

    if not examples:
        raise DatasetError(f"{spec.source_path}: every row failed formatting ({skipped} skipped)")

    split_idx = max(1, int(len(examples) * (1 - spec.eval_split_ratio)))
    train, eval_ = examples[:split_idx], examples[split_idx:]
    if not eval_:
        train, eval_ = examples[:-1], examples[-1:]

    return train, eval_


def option_cardinality_histogram(examples: list[DecisionExample]) -> dict[int, int]:
    """Data quality profiling for CHOICE examples: how many options do rows
    actually offer? A wildly uneven distribution is worth knowing about
    before training, same spirit as the old token-length histogram.
    """
    hist: dict[int, int] = {}
    for ex in examples:
        if ex.options is None:
            continue
        n = len(ex.options)
        hist[n] = hist.get(n, 0) + 1
    return dict(sorted(hist.items()))


def abstain_rate(examples: list[DecisionExample]) -> Optional[float]:
    """Fraction of CHOICE examples where target is None (the no-tool /
    abstain case). Returns None if there are no CHOICE examples to measure.
    """
    choice_examples = [ex for ex in examples if ex.decision_type == DecisionType.CHOICE]
    if not choice_examples:
        return None
    abstains = sum(1 for ex in choice_examples if ex.target is None)
    return abstains / len(choice_examples)
