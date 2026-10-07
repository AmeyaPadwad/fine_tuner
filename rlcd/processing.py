"""Data preprocessing and tensor collation for Laya RL training.

Handles standardizing datasets, extracting answers and criteria,
building [MASK] token sequences via `build_sequence`, and padding
them into uniform PyTorch tensors for the GPU.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer
from laya.common import QTYPES, build_sequence, render_options

VALID_DECISION_TYPES = ("choice", "score", "noul")

def row_to_laya_input(
    row,
    context_fn,
    decision_type="choice",
    options_fn=None,
    target_fn=None,
    scale=None,
):
    """Convert one raw dataset row into Laya's input shape.

    Args:
        row (dict): One record
        context_fn (Callable[[dict], str]): Builds the textual context Laya conditions on from the raw row
        decision_type (str): One of "choice", "score", "noul". Defaults to "choice".
        options_fn (Callable[[dict], list[str]], optional): Required when decision_type == "choice". Builds the list of candidate option strings from the raw row.
        target_fn (Callable[[dict], Any], optional): Builds the ground-truth answer from the raw row.
        scale (tuple[float, float], optional): Required when decision_type == "score", as (min, max).

    Returns:
        dict: {"context": str, "decision_type": str, "target": Any, "options": list[str] | None, "scale": tuple | None}, ready to feed into Laya fine-tuning or evaluation.

    Raises:
        ValueError: If decision_type isn't one of the three supported primitives, if decision_type == "choice" and options_fn is missing, or if decision_type == "score" and scale is missing.
    """
    if decision_type not in VALID_DECISION_TYPES:
        raise ValueError(
            f"unsupported decision_type: {decision_type!r} "
            f"(expected one of {VALID_DECISION_TYPES})"
        )

    if decision_type == "choice" and options_fn is None:
        raise ValueError("decision_type='choice' requires options_fn")

    if decision_type == "score" and scale is None:
        raise ValueError("decision_type='score' requires a scale=(min, max)")

    context = context_fn(row)
    options = options_fn(row) if options_fn else None
    target = target_fn(row) if target_fn else None

    return {
        "context": context,
        "decision_type": decision_type,
        "target": target,
        "options": options,
        "scale": scale,
    }

def prepare_dataset_for_laya(
    dataset,
    context_fn,
    decision_type="choice",
    options_fn=None,
    target_fn=None,
    scale=None,
):
    """Map an entire dataset into Laya-ready rows.

    Applies row_to_laya_input across every row of dataset; falls back to a plain list comprehension for any other iterable of row dicts.

    The original columns are left untouched.

    Args:
        dataset: A datasets.Dataset, datasets.DatasetDict, or any iterable of row dicts to convert.
        context_fn, decision_type, options_fn, target_fn, scale: See row_to_laya_input -- forwarded unchanged to every row.

    Returns:
        The same container type as dataset (with the new Laya fields added to each row) if dataset supports .map, otherwise a plain ``list[dict]`` of Laya-ready rows.
    """
    def _convert(row):
        return row_to_laya_input(
            row,
            context_fn=context_fn,
            decision_type=decision_type,
            options_fn=options_fn,
            target_fn=target_fn,
            scale=scale,
        )

    if hasattr(dataset, "map"):
        print(f"Preparing dataset for Laya (decision_type='{decision_type}')...")
        prepared = dataset.map(_convert)
        print("Dataset is now Laya-input-ready.")
        return prepared

    return [_convert(row) for row in dataset]


def prepare_xlam_for_laya(dataset):
    """Prepare Salesforce/xlam-function-calling-60k as Laya "choice" inputs.

    Args:
        dataset: The dataset object returned by helper.load_dataset.

    Returns:
        Same container type as dataset, with Laya-ready
        context/decision_type/target/options/
        scale fields added to each row.
    """

    def _as_list(value):
        """Return value as a parsed list, decoding it first if it's a JSON string.

        Args:
            value (Union[str, list, Any]): The value to decode or return.

        Returns:
            list: The decoded list, or an empty list if the input is None.
        """
        if isinstance(value, str):
            value = json.loads(value)
        return value or []

    def context_fn(row):
        return row["query"]

    def options_fn(row):
        return [tool["name"] for tool in _as_list(row["tools"])]

    def target_fn(row):
        answers = _as_list(row["answers"])
        if not answers:
            return None
        first = answers[0]
        return first.get("name") or first.get("tool_name")

    return prepare_dataset_for_laya(
        dataset,
        context_fn=context_fn,
        decision_type="choice",
        options_fn=options_fn,
        target_fn=target_fn,
    )


def auto_prepare_dataset_for_laya(dataset, column_map: Optional[Dict[str, str]] = None):
    """Automatically adapt any dataset into Laya's input format.

    Prevents crashes by detecting dataset schema:
      1. Native Laya schema: context, decision_type, options, target, scale.
      2. xLAM schema: query, tools, answers.
      3. Medical/Triage rollout schema: context, decision, confidence, outcome.
      4. Generic classification/NLP schema: prompt/text/input and label/output/target.
      5. Arbitrary custom schema via column_map.

    Serializes dict or list contexts into string JSON automatically.
    """
    if len(dataset) == 0:
        return dataset

    sample = dataset[0] if hasattr(dataset, "__getitem__") else next(iter(dataset))
    mapping = column_map or {}

    # Case 1: Already has Laya fields
    if "context" in sample and "target" in sample and ("options" in sample or "scale" in sample or "decision_type" in sample):
        def _clean_laya(row):
            ctx = row["context"]
            if isinstance(ctx, (dict, list)):
                ctx = json.dumps(ctx)
            return {
                "context": str(ctx),
                "decision_type": row.get("decision_type", "choice"),
                "target": row.get("target"),
                "options": row.get("options"),
                "scale": row.get("scale"),
            }
        return dataset.map(_clean_laya) if hasattr(dataset, "map") else [_clean_laya(r) for r in dataset]

    # Case 2: xLAM schema (query, tools, answers)
    if "query" in sample and "tools" in sample and "answers" in sample:
        return prepare_xlam_for_laya(dataset)

    # Case 3: Rollout schema (decision, outcome, context)
    if "decision" in sample and "outcome" in sample:
        def _rollout_convert(row):
            ctx = row.get("context", "")
            if isinstance(ctx, (dict, list)):
                ctx = json.dumps(ctx)
            dec = row.get("decision")
            outcome = row.get("outcome", 1)
            target = dec if outcome == 1 else ("not urgent" if dec == "urgent" else "urgent")
            return {
                "context": str(ctx),
                "decision_type": "choice",
                "target": target,
                "options": ["urgent", "not urgent"],
                "scale": None,
            }
        return dataset.map(_rollout_convert) if hasattr(dataset, "map") else [_rollout_convert(r) for r in dataset]

    # Case 4: Generic mapped columns
    ctx_col = mapping.get("context") or next((c for c in ("prompt", "instruction", "input", "text", "query", "context") if c in sample), None)
    target_col = mapping.get("target") or next((c for c in ("label", "target", "output", "response", "answer") if c in sample), None)
    options_col = mapping.get("options") or next((c for c in ("options", "choices", "candidates") if c in sample), None)

    if not ctx_col or not target_col:
        raise ValueError(
            f"Could not auto-detect context and target columns from keys: {list(sample.keys())}. "
            "Please provide a column_map, e.g., {'text': 'context', 'label': 'target'}."
        )

    def _generic_convert(row):
        ctx = row[ctx_col]
        if isinstance(ctx, (dict, list)):
            ctx = json.dumps(ctx)
        target = row[target_col]
        options = row.get(options_col) if options_col else None
        dtype = "choice" if options else "noul"
        return {
            "context": str(ctx),
            "decision_type": dtype,
            "target": target,
            "options": options,
            "scale": None,
        }

    return dataset.map(_generic_convert) if hasattr(dataset, "map") else [_generic_convert(r) for r in dataset]



def build_training_item(
    state: Union[str, dict, list],
    q: Dict[str, Any],
    gold_q: Dict[str, Any],
    tok: AutoTokenizer,
    max_len: int = 1024,
    head_max_len: int = 256,
) -> Optional[Dict[str, Any]]:
    """Build a single tokenized training item from a question and its gold answer.

    Supports typed-decision questions: choice, score, noul.

    Args:
        state: Context/state dict or serialized string.
        q: Question dict containing 'type', 'instructions', 'criteria'.
        gold_q: Ground-truth dict containing 'probabilities' and 'label'.
        tok: Hugging Face AutoTokenizer.
        max_len: Maximum total token sequence length.
        head_max_len: Maximum budget for question head & option tokens.

    Returns:
        Dict with keys (ids, markers, qtype, target, label) or None if
        the option markers do not fit the token budget.
    """
    t = q["type"]
    crit = q.get("criteria", {})
    probs = gold_q.get("probabilities", {})

    if t == "choice":
        keys = list(crit.keys())
        target = [probs.get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [probs.get("false", 0.5), probs.get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [probs.get(str(i), 0.0) for i in range(n_levels)]
    else:
        raise ValueError(f"Unsupported decision type: {t}")

    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))

    opts = render_options({"t": t, "crit": crit})
    k = len(opts)

    seq, markers = build_sequence(
        tok,
        state,
        {"t": t, "ins": q.get("instructions", ""), "crit": crit},
        max_len=max_len,
        head_max_len=head_max_len,
    )

    if len(markers) != k:
        return None

    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[t],
        "target": target,
        "label": label,
    }


def build_item_from_laya_row(
    row: Dict[str, Any],
    tok: AutoTokenizer,
    max_len: int = 1024,
    head_max_len: int = 256,
) -> Optional[Dict[str, Any]]:
    """Convert a row formatted by Laya.prepare_dataset_for_laya into a training item.

    Handles rows with 'context', 'decision_type', 'options', 'target', 'scale'.
    """
    decision_type = row.get("decision_type", "choice")
    context = row.get("context", "")
    options = row.get("options") or []
    target_val = row.get("target")

    if decision_type == "choice":
        criteria = {opt: "" for opt in options}
        target_probs = {}
        for opt in options:
            target_probs[opt] = 1.0 if (opt == target_val) else 0.0
        s = sum(target_probs.values())
        if s > 0:
            target_probs = {k: v / s for k, v in target_probs.items()}
        else:
            uniform = 1.0 / max(1, len(options))
            target_probs = {k: uniform for k in target_probs}

        q = {"type": "choice", "instructions": "Select the best option.", "criteria": criteria}
        gold = {"probabilities": target_probs, "label": target_val}
        return build_training_item(context, q, gold, tok, max_len=max_len, head_max_len=head_max_len)

    elif decision_type == "noul":
        target_bool = bool(target_val) if target_val is not None else False
        probs = {"false": 0.0 if target_bool else 1.0, "true": 1.0 if target_bool else 0.0}
        q = {"type": "noul", "instructions": "Does the statement hold?", "criteria": None}
        gold = {"probabilities": probs, "label": "true" if target_bool else "false"}
        return build_training_item(context, q, gold, tok, max_len=max_len, head_max_len=head_max_len)

    elif decision_type == "score":
        scale = row.get("scale") or (0, 3)
        n_levels = int(scale[1] - scale[0] + 1)
        target_idx = int(round(float(target_val))) if target_val is not None else 0
        target_idx = max(0, min(n_levels - 1, target_idx))
        probs = {str(i): (1.0 if i == target_idx else 0.0) for i in range(n_levels)}
        criteria = [f"level {i}" for i in range(n_levels)]
        q = {"type": "score", "instructions": "Assign a score rating.", "criteria": criteria}
        gold = {"probabilities": probs, "label": target_idx}
        return build_training_item(context, q, gold, tok, max_len=max_len, head_max_len=head_max_len)

    return None


def prepare_dataset_items(
    dataset: Any,
    tok: AutoTokenizer,
    max_len: int = 1024,
    head_max_len: int = 256,
) -> List[Dict[str, Any]]:
    """Convert an arbitrary dataset into a list of model-ready training sequences.

    Supports:
        1. LocalLLaMA/typed-decisions benchmark (state, questions, gold JSON strings)
        2. Laya-prepared dataset rows (context, decision_type, options, target)
        3. Pre-tokenized items list (ids, markers, qtype, target, label)
    """
    items = []
    if isinstance(dataset, list) and dataset and "ids" in dataset[0] and "markers" in dataset[0]:
        return dataset

    for row in dataset:
        if "state" in row and "questions" in row and "gold" in row:
            state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
            questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
            gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
            for qid, q in questions.items():
                if qid in gold:
                    it = build_training_item(state, q, gold[qid], tok, max_len=max_len, head_max_len=head_max_len)
                    if it:
                        items.append(it)
        elif "context" in row and "decision_type" in row:
            it = build_item_from_laya_row(row, tok, max_len=max_len, head_max_len=head_max_len)
            if it:
                items.append(it)

    return items


def collate_train_batch(items: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Collate variable-length tokenized items and markers into padded tensors.

    Args:
        items: List of item dictionaries (ids, markers, target, qtype, label).
        pad_id: Padding token ID from the tokenizer.

    Returns:
        Dict of padded torch.Tensors:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            marker_pos: (batch_size, max_options)
            marker_mask: (batch_size, max_options)
            target: (batch_size, max_options)
            qtype: (batch_size,)
            label: (batch_size,)
    """
    n = len(items)
    max_l = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)

    ids = torch.full((n, max_l), pad_id, dtype=torch.long)
    att = torch.zeros((n, max_l), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)

    for i, it in enumerate(items):
        seq_len = len(it["ids"])
        ids[i, :seq_len] = torch.tensor(it["ids"], dtype=torch.long)
        att[i, :seq_len] = 1

        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"], dtype=torch.long)
        mmask[i, :k] = True

        t_len = len(it["target"])
        target[i, :t_len] = torch.tensor(it["target"], dtype=torch.float32)

    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items], dtype=torch.long),
        "label": torch.tensor([it["label"] for it in items], dtype=torch.long),
    }
