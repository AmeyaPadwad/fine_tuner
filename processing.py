"""Universal data preprocessing, schema inference, and tensor collation for decision-model RLCD training.

This module adapts arbitrary datasets (Hugging Face `Dataset` objects or Python
lists of dicts) into the tokenized `[MASK]` / marker sequence representation
required by System-One decision models (Laya, Von, Decision 1.0, OpenDecision, etc.)
across all three decision primitives:
    - `choice`: Categorical or tool-calling selection over candidate options.
    - `noul`: Binary / boolean (True/False, Yes/No, Safe/Unsafe) decisions.
    - `score`: Ordinal rating decisions over a bounded numeric scale `(min, max)`.

The pipeline is organized into five logical groups:
    1. Field Extraction & Target Normalization Helpers
    2. Schema Resolution & Decision Primitive Inference
    3. Default Sequence Packing & Option Rendering (`render_options`, `build_sequence`)
    4. Dataset Preparation & Sequence Tokenization (`[MASK]` / Marker Alignment)
    5. Batch Collation & Dataset Splitting
"""

import ast
import json
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

VALID_DECISION_TYPES = ("choice", "score", "noul")
QTYPES = {"choice": 0, "score": 1, "noul": 2}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}


# ===========================================================================
# 1. Field Extraction & Target Normalization Helpers
# ===========================================================================

def _take_sample(dataset: Any, n: int = 100) -> List[Dict[str, Any]]:
    """Extract up to `n` rows from a Hugging Face Dataset or Python iterable.

    Args:
        dataset (Any): A `datasets.Dataset`, list of dicts, or iterable of row dicts.
        n (int): Maximum number of sample rows to inspect. Defaults to 100.

    Returns:
        List[Dict[str, Any]]: A list of up to `n` row dictionaries.
    """
    if hasattr(dataset, "select") and hasattr(dataset, "__len__"):
        k = min(n, len(dataset))
        return [dataset[i] for i in range(k)]
    if isinstance(dataset, list):
        return dataset[:n]
    out = []
    for row in dataset:
        out.append(row)
        if len(out) >= n:
            break
    return out


def _build_context_string(
    row: Dict[str, Any],
    context_cols: Union[Callable, List[str], str, None],
) -> str:
    """Serialize one or more context columns (or a custom callable) into a single string.

    Args:
        row (Dict[str, Any]): A single dataset record.
        context_cols (Union[Callable, List[str], str, None]): Column name, list of
            column names to combine as JSON, or callable `fn(row) -> str`.

    Returns:
        str: Formatted context string ready for sequence construction.
    """
    if callable(context_cols):
        val = context_cols(row)
        return json.dumps(val) if isinstance(val, (dict, list)) else str(val)
    if not context_cols:
        return ""
    if isinstance(context_cols, str):
        context_cols = [context_cols]
    if len(context_cols) == 1:
        val = row.get(context_cols[0], "")
        return json.dumps(val) if isinstance(val, (dict, list)) else str(val)
    return json.dumps({col: row.get(col) for col in context_cols})


def _extract_functions_from_text(text: str) -> List[str]:
    """Extract Python function names from `def func_name(` blocks inside a prompt string.

    Supports tool-calling datasets such as `Nexusflow/NexusRaven-V2` that embed
    candidate tool declarations directly in the prompt text.

    Args:
        text (str): Raw prompt or context string.

    Returns:
        List[str]: Ordered, deduplicated list of discovered function names.
    """
    if not isinstance(text, str):
        return []
    matches = re.findall(r"(?:Function:\s*)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    seen = []
    for m in matches:
        if m not in seen:
            seen.append(m)
    return seen


def _parse_row_options(raw_options: Any, context_text: str = "") -> List[str]:
    """Parse candidate option names from JSON strings, lists of dicts/strings, or prompt text.

    Args:
        raw_options (Any): Raw value from an options/tools column (JSON string, list, or dict).
        context_text (str, optional): Fallback context string to scan for embedded function headers.

    Returns:
        List[str]: List of candidate option strings for a `choice` decision.
    """
    parsed = raw_options
    if isinstance(raw_options, str):
        raw_str = raw_options.strip()
        if raw_str.startswith("[") or raw_str.startswith("{"):
            try:
                parsed = json.loads(raw_str)
            except Exception:
                try:
                    parsed = ast.literal_eval(raw_str)
                except Exception:
                    parsed = None
        else:
            fn_names = _extract_functions_from_text(raw_str)
            if fn_names:
                return fn_names
            parsed = [s.strip() for s in raw_str.split(",") if s.strip()]

    options: List[str] = []
    if isinstance(parsed, dict):
        options = [str(k) for k in parsed.keys()]
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                name = (
                    item.get("name")
                    or item.get("tool_name")
                    or (item.get("function", {}) if isinstance(item.get("function"), dict) else {}).get("name")
                    or item.get("id")
                    or item.get("label")
                )
                if name is not None:
                    options.append(str(name))
            elif item is not None:
                options.append(str(item))

    if not options and context_text:
        options = _extract_functions_from_text(context_text)

    return options


def _parse_choice_target(
    raw_target: Any,
    options: List[str],
    label_names: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve the ground-truth option string from a raw target value.

    Handles JSON tool-call arrays (e.g. xLAM), function call strings
    (e.g. `Call: get_weather(...)` in NexusRaven-V2), integer class indices,
    and plain label strings.

    Args:
        raw_target (Any): Raw ground-truth value from the dataset row.
        options (List[str]): Candidate option strings for the row.
        label_names (Optional[List[str]]): Optional class label names for integer index mapping.

    Returns:
        Optional[str]: Resolved target option string, or `None` if `raw_target` is missing.
    """
    if raw_target is None:
        return None

    parsed = raw_target
    if isinstance(raw_target, str):
        s = raw_target.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = s

    if isinstance(parsed, list):
        if not parsed:
            return "none" if "none" in options else (options[0] if options else None)
        first = parsed[0]
        if isinstance(first, dict):
            name = (
                first.get("name")
                or first.get("tool_name")
                or (first.get("function", {}) if isinstance(first.get("function"), dict) else {}).get("name")
            )
            if name is not None:
                return str(name)
        return str(first)

    if isinstance(parsed, dict):
        name = (
            parsed.get("name")
            or parsed.get("tool_name")
            or (parsed.get("function", {}) if isinstance(parsed.get("function"), dict) else {}).get("name")
            or parsed.get("label")
        )
        if name is not None:
            return str(name)

    if isinstance(parsed, int) and not isinstance(parsed, bool):
        if label_names and 0 <= parsed < len(label_names):
            return str(label_names[parsed])
        if str(parsed) in options:
            return str(parsed)
        if 0 <= parsed < len(options):
            return str(options[parsed])

    target_str = str(parsed).strip()
    if options and target_str not in options:
        for opt in options:
            if re.search(rf"\b{re.escape(opt)}\s*\(", target_str):
                return opt
    return target_str


def _normalize_to_bool(val: Any) -> bool:
    """Convert a boolean, binary number, or binary label string into a Python `bool`.

    Args:
        val (Any): Raw target value (e.g. `1`, `0`, `"yes"`, `"no"`, `"safe"`, `"unsafe"`).

    Returns:
        bool: Normalized boolean target for `noul` decisions.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val) > 0.5
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "true", "yes", "y", "safe", "helpful", "harmless", "positive", "urgent"):
            return True
        if s in ("0", "false", "no", "n", "unsafe", "unhelpful", "harmful", "negative", "not urgent"):
            return False
    return bool(val)


def _infer_numeric_scale(values: List[Any]) -> Tuple[int, int]:
    """Infer an ordinal integer scale `(min_int, max_int)` from sample target values.

    Args:
        values (List[Any]): Sample target values from the dataset.

    Returns:
        Tuple[int, int]: Inferred `(min_val, max_val)` scale tuple.
    """
    nums: List[float] = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return (0, 3)
    lo, hi = min(nums), max(nums)
    if 0.0 <= lo and hi <= 1.0 and any(isinstance(v, float) and not float(v).is_integer() for v in values):
        return (1, 5)
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i <= lo_i:
        return (0, max(3, hi_i))
    if (hi_i - lo_i) > 10:
        return (1, 5)
    return (lo_i, hi_i)


def _normalize_to_scale(
    val: Any,
    scale: Tuple[float, float],
    value_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Map a raw numeric or JSON moderation score onto the target scale `[scale[0], scale[1]]`.

    Supports nested moderation arrays such as `lmsys/toxic-chat`'s `openai_moderation`
    column (`[[category, score], ...]`) as well as standard floats and integers.

    Args:
        val (Any): Raw numeric or string target value.
        scale (Tuple[float, float]): Target `(min, max)` ordinal scale.
        value_range (Optional[Tuple[float, float]]): Optional source `(min, max)` range for linear rescaling.

    Returns:
        float: Score clamped/projected onto `[scale[0], scale[1]]`.
    """
    lo, hi = float(scale[0]), float(scale[1])
    if val is None:
        return lo
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list) and parsed:
                    scores = [
                        float(item[1])
                        for item in parsed
                        if isinstance(item, (list, tuple)) and len(item) >= 2
                    ]
                    if scores:
                        raw_f = max(scores)
                        return lo + raw_f * (hi - lo)
            except Exception:
                pass
    try:
        raw_f = float(val)
    except (TypeError, ValueError):
        return lo

    if 0.0 <= raw_f <= 1.0 and (lo >= 1.0 or hi > 1.0) and isinstance(val, float) and not raw_f.is_integer():
        return lo + raw_f * (hi - lo)

    if value_range is not None:
        v_lo, v_hi = value_range
        if v_hi > v_lo and (raw_f < lo or raw_f > hi):
            norm = (raw_f - v_lo) / (v_hi - v_lo)
            return max(lo, min(hi, lo + norm * (hi - lo)))

    return max(lo, min(hi, raw_f))


def _collect_unique_labels(
    dataset: Any,
    samples: List[Dict[str, Any]],
    target_col: str,
    hf_features: Any = None,
) -> Tuple[List[str], Optional[List[str]]]:
    """Collect global unique categorical options from Hugging Face `ClassLabel` or dataset rows.

    Used when a classification dataset (e.g. `mteb/tweet_sentiment_extraction`)
    does not store candidate options in every row.

    Args:
        dataset (Any): The full dataset or iterable.
        samples (List[Dict[str, Any]]): Pre-sampled rows from the dataset.
        target_col (str): Name of the target label column.
        hf_features (Any, optional): Hugging Face `dataset.features` dictionary if available.

    Returns:
        Tuple[List[str], Optional[List[str]]]: `(global_options, class_label_names)`.
    """
    if hf_features and target_col in hf_features:
        feat = hf_features[target_col]
        names = getattr(feat, "names", None)
        if names:
            str_names = [str(n) for n in names]
            return str_names, str_names

    scan_rows = _take_sample(dataset, n=1000) if len(samples) < 100 else samples
    seen = []
    for r in scan_rows:
        v = r.get(target_col)
        if v is not None and not isinstance(v, (list, dict)):
            s = str(v).strip()
            if s and s not in seen:
                seen.append(s)
    return sorted(seen), None


# ===========================================================================
# 2. Schema Resolution & Decision Primitive Inference
# ===========================================================================

def infer_decision_primitive(
    dataset: Any,
    samples: List[Dict[str, Any]],
    target_col: str,
    options_source: Union[str, List[str], None],
    explicit_type: Optional[str] = None,
    explicit_scale: Optional[Tuple[int, int]] = None,
    hf_features: Any = None,
) -> Tuple[str, Union[str, List[str], None], Optional[Tuple[int, int]], Optional[List[str]]]:
    """Infer the decision primitive (`choice`, `noul`, or `score`), options source, and scale.

    Respects explicit user overrides (`explicit_type`, `explicit_scale`, `options_source`)
    first, and otherwise inspects `hf_features` and the empirical distribution of `target_col`.

    Args:
        dataset (Any): Input dataset.
        samples (List[Dict[str, Any]]): Sampled records from the dataset.
        target_col (str): Resolved target column name.
        options_source (Union[str, List[str], None]): Column name, static list of options, or `None`.
        explicit_type (Optional[str]): User-specified decision primitive (`"choice"`, `"noul"`, `"score"`).
        explicit_scale (Optional[Tuple[int, int]]): User-specified numeric scale `(min, max)` for `"score"`.
        hf_features (Any, optional): Hugging Face `dataset.features` metadata if available.

    Returns:
        Tuple[str, Union[str, List[str], None], Optional[Tuple[int, int]], Optional[List[str]]]:
            `(decision_type, resolved_options_source, scale, label_names)`.

    Raises:
        ValueError: If `explicit_type` is not one of `VALID_DECISION_TYPES`.
    """
    if explicit_type is not None and explicit_type not in VALID_DECISION_TYPES:
        raise ValueError(
            f"unsupported decision_type: {explicit_type!r} (expected one of {VALID_DECISION_TYPES})"
        )

    target_vals = [r.get(target_col) for r in samples if r.get(target_col) is not None]

    # 1. Explicit 'score' or explicit scale provided
    if explicit_type == "score" or explicit_scale is not None:
        scale = tuple(explicit_scale) if explicit_scale is not None else _infer_numeric_scale(target_vals)
        return "score", None, scale, None

    # 2. Explicit 'noul'
    if explicit_type == "noul":
        return "noul", None, None, None

    # 3. Per-row options column or static options list -> 'choice'
    if options_source is not None:
        return "choice", options_source, None, None

    # 4. Hugging Face ClassLabel feature inspection
    if hf_features and target_col in hf_features:
        feat = hf_features[target_col]
        names = getattr(feat, "names", None)
        if names:
            if len(names) == 2 and explicit_type is None and set(n.lower() for n in names).issubset(
                {"true", "false", "yes", "no", "safe", "unsafe", "0", "1"}
            ):
                return "noul", None, None, list(names)
            return "choice", [str(n) for n in names], None, [str(n) for n in names]

    # 5. Explicit 'choice' without an options column -> collect global unique labels
    if explicit_type == "choice":
        global_opts, label_names = _collect_unique_labels(dataset, samples, target_col, hf_features)
        return "choice", global_opts, None, label_names

    # 6. Auto-infer from target values when explicit_type is None:
    if target_vals:
        first_non_null = target_vals[0]
        if isinstance(first_non_null, str) and first_non_null.strip().startswith("[["):
            return "score", None, (1, 5), None
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in target_vals) and any(
            isinstance(v, float) and not float(v).is_integer() for v in target_vals
        ):
            return "score", None, _infer_numeric_scale(target_vals), None

    scalar_vals = [v for v in target_vals if not isinstance(v, (list, dict))]
    if scalar_vals:
        unique_lower = {str(v).strip().lower() for v in scalar_vals}
        if unique_lower.issubset({"0", "1", "true", "false", "yes", "no", "safe", "unsafe"}):
            return "noul", None, None, None

    if scalar_vals and all(isinstance(v, int) and not isinstance(v, bool) for v in scalar_vals):
        int_vals = [int(v) for v in scalar_vals]
        if min(int_vals) >= 1 and max(int_vals) >= 4 and len(set(int_vals)) >= 4:
            return "score", None, (min(int_vals), max(int_vals)), None

    global_opts, label_names = _collect_unique_labels(dataset, samples, target_col, hf_features)
    if len(global_opts) >= 2:
        return "choice", global_opts, None, label_names

    return "noul", None, None, None


def resolve_schema_and_columns(
    dataset: Any,
    column_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve dataset schema mode, column mappings, and decision primitive.

    Combines user-supplied overrides in `column_map` with automatic structural
    detection across native decision datasets, medical/triage rollouts, contrastive
    preference pairs (e.g. `Anthropic/hh-rlhf`), tool-calling datasets
    (e.g. `xLAM`, `NexusRaven-V2`), classification benchmarks
    (e.g. `mteb/tweet_sentiment_extraction`), and ordinal/moderation datasets
    (e.g. `lmsys/toxic-chat`).

    Args:
        dataset (Any): A Hugging Face `Dataset` or iterable of row dicts.
        column_map (Optional[Dict[str, Any]]): Optional dictionary of user overrides:
            - `"context"`: `str | List[str] | Callable`
            - `"target"`: `str`
            - `"options"`: `str | List[str]`
            - `"chosen"` / `"rejected"`: `str`
            - `"decision_type"`: `"choice" | "noul" | "score"`
            - `"scale"`: `Tuple[int, int]`
            - `"instructions"`: `str`

    Returns:
        Dict[str, Any]: Resolved schema specification dictionary.

    Raises:
        ValueError: If the dataset is empty or required context/target columns cannot be resolved.
    """
    mapping = dict(column_map or {})
    samples = _take_sample(dataset, n=100)
    if not samples:
        raise ValueError("Cannot resolve schema on an empty dataset.")

    sample0 = samples[0]
    keys = set(sample0.keys())
    hf_features = getattr(dataset, "features", None)

    # Case 1: Native Decision schema
    if (
        not mapping
        and "context" in keys
        and "target" in keys
        and ("options" in keys or "scale" in keys or "decision_type" in keys)
    ):
        return {
            "mode": "native_decision",
            "context_cols": ["context"],
            "target_col": "target",
            "options_source": "options" if "options" in keys else None,
            "decision_type": sample0.get("decision_type", "choice"),
            "scale": tuple(sample0["scale"]) if sample0.get("scale") is not None else None,
            "label_names": None,
            "instructions": None,
        }

    # Case 2: Medical/Triage Rollout schema (context, decision, outcome)
    if not mapping and "decision" in keys and "outcome" in keys:
        return {
            "mode": "rollout",
            "context_cols": ["context"] if "context" in keys else [],
            "target_col": "outcome",
            "options_source": ["urgent", "not urgent"],
            "decision_type": "choice",
            "scale": None,
            "label_names": None,
            "instructions": None,
        }

    # Case 3: Contrastive / Preference Pair schema (e.g. Anthropic/hh-rlhf: "chosen" & "rejected")
    chosen_col = mapping.get("chosen") or ("chosen" if "chosen" in keys else None)
    rejected_col = mapping.get("rejected") or ("rejected" if "rejected" in keys else None)
    if chosen_col and rejected_col and "target" not in mapping:
        ctx_override = mapping.get("context")
        if isinstance(ctx_override, str):
            ctx_override = [ctx_override]
        elif ctx_override is None and "prompt" in keys:
            ctx_override = ["prompt"]
        return {
            "mode": "contrastive_pair",
            "context_cols": ctx_override,
            "chosen_col": chosen_col,
            "rejected_col": rejected_col,
            "target_col": f"{chosen_col}/{rejected_col}",
            "options_source": None,
            "decision_type": mapping.get("decision_type", "noul"),
            "scale": None,
            "label_names": None,
            "instructions": mapping.get("instructions"),
        }

    # Case 4: Standard / Tool-Calling / Classification / Score schema
    if "context" in mapping and mapping["context"] is not None:
        ctx_cols = mapping["context"]
        if isinstance(ctx_cols, str):
            ctx_cols = [ctx_cols]
        elif isinstance(ctx_cols, (list, tuple)):
            ctx_cols = list(ctx_cols)
    elif "user_input" in keys and "model_output" in keys:
        ctx_cols = ["user_input", "model_output"]
    else:
        ctx_candidates = (
            "query",
            "prompt",
            "instruction",
            "input",
            "text",
            "user_input",
            "question",
            "sentence",
            "context",
            "messages",
            "conversation",
        )
        found_ctx = next((c for c in ctx_candidates if c in keys), None)
        if not found_ctx:
            raise ValueError(
                f"Could not auto-detect context column from keys: {sorted(keys)}. "
                "Please provide column_map={'context': '<col_name>', 'target': '<col_name>'}."
            )
        ctx_cols = [found_ctx]

    used_ctx_set = set(ctx_cols) if isinstance(ctx_cols, list) else set()

    if "target" in mapping and mapping["target"] is not None:
        target_col = mapping["target"]
        if target_col not in keys:
            raise ValueError(f"Specified target column {target_col!r} not found in dataset keys: {sorted(keys)}.")
    else:
        requested_dtype = mapping.get("decision_type")
        if requested_dtype == "score":
            score_candidates = ("score", "rating", "toxicity", "openai_moderation", "label", "target", "output")
            target_col = next((c for c in score_candidates if c in keys and c not in used_ctx_set), None)
        else:
            target_candidates = (
                "label_text",
                "python_function_name",
                "answers",
                "answer",
                "tool_calls",
                "target",
                "label",
                "toxicity",
                "score",
                "rating",
                "outcome",
                "output",
                "response",
                "category",
                "class",
            )
            target_col = next((c for c in target_candidates if c in keys and c not in used_ctx_set), None)

        if not target_col:
            raise ValueError(
                f"Could not auto-detect target column from keys: {sorted(keys)}. "
                "Please provide column_map={'context': '<col_name>', 'target': '<col_name>'}."
            )

    if "options" in mapping and mapping["options"] is not None:
        options_source = mapping["options"]
    else:
        opt_candidates = ("tools", "functions", "context_functions", "options", "choices", "candidates")
        options_source = next((c for c in opt_candidates if c in keys and c != target_col), None)
        if options_source is None and isinstance(ctx_cols, list) and len(ctx_cols) == 1:
            sample_ctx = str(sample0.get(ctx_cols[0], ""))
            if _extract_functions_from_text(sample_ctx):
                options_source = "__from_context_functions__"

    decision_type, resolved_options, scale, label_names = infer_decision_primitive(
        dataset=dataset,
        samples=samples,
        target_col=target_col,
        options_source=options_source,
        explicit_type=mapping.get("decision_type"),
        explicit_scale=mapping.get("scale"),
        hf_features=hf_features,
    )

    return {
        "mode": "standard",
        "context_cols": ctx_cols,
        "target_col": target_col,
        "options_source": resolved_options,
        "decision_type": decision_type,
        "scale": scale,
        "label_names": label_names,
        "instructions": mapping.get("instructions"),
    }


def _expand_contrastive_pairs(dataset: Any, spec: Dict[str, Any]):
    """Expand contrastive preference rows (`chosen` / `rejected`) into decision-formatted rows.

    For `decision_type == "noul"`, each input row produces two examples:
    `(context=chosen, target=True)` and `(context=rejected, target=False)`.
    For `decision_type == "choice"`, each row becomes a 2-option choice between
    `chosen` and `rejected`.

    Args:
        dataset (Any): Input dataset containing `chosen` and `rejected` columns.
        spec (Dict[str, Any]): Resolved schema specification from `resolve_schema_and_columns`.

    Returns:
        Dataset or List[Dict[str, Any]]: Expanded decision-ready records.
    """
    chosen_col = spec["chosen_col"]
    rejected_col = spec["rejected_col"]
    ctx_cols = spec.get("context_cols")
    dtype = spec["decision_type"]

    rows_out: List[Dict[str, Any]] = []
    for row in dataset:
        prefix = (_build_context_string(row, ctx_cols) + "\n\n") if ctx_cols else ""
        chosen_txt = prefix + str(row.get(chosen_col, ""))
        rejected_txt = prefix + str(row.get(rejected_col, ""))

        if dtype == "choice":
            rows_out.append({
                "context": prefix.strip() or "Select the safer and more helpful response.",
                "decision_type": "choice",
                "options": [chosen_txt, rejected_txt],
                "target": chosen_txt,
                "scale": None,
            })
        else:
            rows_out.append({
                "context": chosen_txt,
                "decision_type": "noul",
                "options": None,
                "target": True,
                "scale": None,
            })
            rows_out.append({
                "context": rejected_txt,
                "decision_type": "noul",
                "options": None,
                "target": False,
                "scale": None,
            })

    if hasattr(dataset, "from_list"):
        try:
            return type(dataset).from_list(rows_out)
        except Exception:
            return rows_out
    return rows_out


# ===========================================================================
# 3. Default Sequence Packing & Option Rendering
# ===========================================================================

def _render_criterion(value: Any) -> str:
    """Render a single criterion value into a clean single-line string.

    Args:
        value (Any): Criterion description (string, dict, list, or number).

    Returns:
        str: Formatted criterion string.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _resolve_noul_labels(labels: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """Validate and return the `(false_label, true_label)` pair for `noul` questions.

    Args:
        labels (Optional[Dict[str, str]]): Optional custom mapping for `'false'` and `'true'`.

    Returns:
        Tuple[str, str]: `(false_label, true_label)`.

    Raises:
        ValueError: If `labels` does not map `'false'` and `'true'` to distinct non-empty strings.
    """
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return false_label, true_label


def render_options(q: Dict[str, Any]) -> List[str]:
    """Render option strings in label-index order for `choice`, `score`, or `noul` questions.

    Built-in model-agnostic implementation.
    `noul` semantic order is always `[false, true]`.

    Args:
        q (Dict[str, Any]): Question dictionary with keys `'t'` (`"choice" | "score" | "noul"`),
            optional `'crit'`, and optional `'labels'` (for `'noul'`).

    Returns:
        List[str]: Rendered option strings in label-index order.
    """
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        return [
            str(k) if v is None or v == "" else f"{k}: {_render_criterion(v)}"
            for k, v in (crit or {}).items()
        ]
    if t == "score":
        return [f"level {i}: {_render_criterion(c)}" for i, c in enumerate(crit or [])]

    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label + ": " + (_render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label + ": " + (_render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(
    tok: AutoTokenizer,
    state: Union[str, dict, list],
    q: Dict[str, Any],
    max_len: int = 1024,
    head_max_len: int = 256,
    marker_token_id: Optional[int] = None,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
) -> Tuple[List[int], List[int]]:
    """Pack question instructions, option markers, and state into a single token sequence.

    Built-in default packing rule that works across encoder-based decision models:
        `[CLS] <type> question: <instructions> [SEP] <MARKER> opt_0 <MARKER> opt_1 ... [SEP] <state> [SEP]`

    If `marker_token_id` is passed (e.g. from `models.py` for a model with a custom
    candidate marker token), it is placed before every option; otherwise this function
    falls back to `tok.mask_token_id`.

    Args:
        tok (AutoTokenizer): Hugging Face tokenizer for the target decision model.
        state (Union[str, dict, list]): Context/state string, dict, or list.
        q (Dict[str, Any]): Question spec with keys `'t'` (`"choice" | "score" | "noul"`),
            `'ins'` (instructions), and `'crit'` (criteria).
        max_len (int): Maximum total token sequence length. Defaults to 1024.
        head_max_len (int): Maximum token budget for the question header and options. Defaults to 256.
        marker_token_id (Optional[int]): Custom option marker token ID from `models.py`.
            Falls back to `tok.mask_token_id` when `None`.
        option_order (Optional[List[int]]): Optional permutation of option indices.
        truncate_left (bool): If `True`, keep the tail of long states instead of the head.

    Returns:
        Tuple[List[int], List[int]]: `(input_ids, marker_positions)`.

    Raises:
        ValueError: If neither `marker_token_id` nor `tok.mask_token_id` is defined.
    """
    active_marker_id = marker_token_id if marker_token_id is not None else tok.mask_token_id
    if active_marker_id is None:
        raise ValueError(
            f"Tokenizer {getattr(tok, 'name_or_path', tok)} has no mask_token_id and no marker_token_id was provided."
        )

    cls_id = tok.cls_token_id if tok.cls_token_id is not None else (tok.bos_token_id or 0)
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else (tok.eos_token_id or 0)
    mask_str = tok.mask_token or "[MASK]"
    marker_str = tok.convert_ids_to_tokens(active_marker_id) if hasattr(tok, "convert_ids_to_tokens") else mask_str

    def _sanitize(text: str) -> str:
        out = text.replace(mask_str, " ")
        if isinstance(marker_str, str) and marker_str and marker_str != mask_str:
            out = out.replace(marker_str, " ")
        return out

    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = _sanitize(str(q.get("ins", "")))

    head_ids = tok(
        f"{q['t']} question: {ins}",
        add_special_tokens=False,
    )["input_ids"]

    opt_ids: List[List[int]] = []
    for i in order:
        opt_tokens = tok(
            " " + _sanitize(opts[i]),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )["input_ids"]
        opt_ids.append([active_marker_id] + opt_tokens)

    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)

    head_ids = head_ids[: max(8, opt_budget)]
    ids = [cls_id] + head_ids + [sep_id]
    markers: List[int] = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(sep_id)

    room = max(0, max_len - len(ids) - 1)
    state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    state_ids = tok(_sanitize(state_str), add_special_tokens=False)["input_ids"]
    st = state_ids[max(0, len(state_ids) - room) :] if truncate_left else state_ids[:room]

    ids = (ids + st + [sep_id])[:max_len]
    markers = [m for m in markers if m < max_len]
    return ids, markers


# ===========================================================================
# 4. Dataset Preparation & Sequence Tokenization ([MASK] / Marker Alignment)
# ===========================================================================

def auto_prepare_dataset(
    dataset: Any,
    column_map: Optional[Dict[str, Any]] = None,
):
    """Automatically adapt any dataset into the standard decision-model input format.

    Supports both zero-configuration automatic detection and partial or complete
    user column mapping via `column_map`:
        - `"context"`: `str | List[str] | Callable` (e.g. `"text"` or `["user_input", "model_output"]`)
        - `"target"`: `str` (e.g. `"label_text"`, `"answers"`, `"toxicity"`, `"label"`)
        - `"options"`: `str | List[str]` (column name like `"tools"`, or static list `["neg", "neu", "pos"]`)
        - `"chosen"` / `"rejected"`: `str` (contrastive preference columns like in `Anthropic/hh-rlhf`)
        - `"decision_type"`: `"choice"` | `"noul"` | `"score"` (or omitted for auto-inference)
        - `"scale"`: `Tuple[int, int]` (for `"score"`, e.g. `(1, 5)`)

    Args:
        dataset (Any): A Hugging Face `Dataset` or Python list of row dictionaries.
        column_map (Optional[Dict[str, Any]]): Optional user overrides for column roles and primitive type.

    Returns:
        Dataset or List[Dict[str, Any]]: Dataset mapped into the standard decision row schema:
            `{"context": str, "decision_type": str, "target": Any, "options": list | None, "scale": list | None}`.
    """
    if len(dataset) == 0:
        return dataset

    spec = resolve_schema_and_columns(dataset, column_map=column_map)
    print(
        f"Preparing dataset (mode='{spec['mode']}', decision_type='{spec['decision_type']}', "
        f"context={spec['context_cols']}, target='{spec['target_col']}')..."
    )

    # Mode 1: Native decision format
    if spec["mode"] == "native_decision":
        def _clean_decision_row(row):
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
        return dataset.map(_clean_decision_row) if hasattr(dataset, "map") else [_clean_decision_row(r) for r in dataset]

    # Mode 2: Rollout format (context, decision, outcome)
    if spec["mode"] == "rollout":
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

    # Mode 3: Contrastive preference pairs (e.g. Anthropic/hh-rlhf)
    if spec["mode"] == "contrastive_pair":
        prepared = _expand_contrastive_pairs(dataset, spec)
        print("Dataset is now model-input-ready.")
        return prepared

    # Mode 4: Standard / Tool-Calling / Classification / Score
    dtype = spec["decision_type"]
    ctx_cols = spec["context_cols"]
    target_col = spec["target_col"]
    opt_src = spec["options_source"]
    scale = spec["scale"]
    label_names = spec.get("label_names")

    def _standard_convert(row):
        ctx_str = _build_context_string(row, ctx_cols)
        raw_target = row.get(target_col)

        if dtype == "choice":
            if isinstance(opt_src, list):
                options = [str(o) for o in opt_src]
            elif opt_src == "__from_context_functions__":
                options = _extract_functions_from_text(ctx_str)
            elif isinstance(opt_src, str):
                options = _parse_row_options(row.get(opt_src), context_text=ctx_str)
            else:
                options = []

            target = _parse_choice_target(raw_target, options, label_names=label_names)
            return {
                "context": ctx_str,
                "decision_type": "choice",
                "target": target,
                "options": options,
                "scale": None,
            }

        elif dtype == "noul":
            return {
                "context": ctx_str,
                "decision_type": "noul",
                "target": _normalize_to_bool(raw_target),
                "options": None,
                "scale": None,
            }

        elif dtype == "score":
            use_scale = list(scale) if scale is not None else [0, 3]
            return {
                "context": ctx_str,
                "decision_type": "score",
                "target": _normalize_to_scale(raw_target, tuple(use_scale)),
                "options": None,
                "scale": use_scale,
            }

    if hasattr(dataset, "map"):
        prepared = dataset.map(_standard_convert)
        print("Dataset is now model-input-ready.")
        return prepared

    return [_standard_convert(r) for r in dataset]


def build_training_item(
    state: Union[str, dict, list],
    q: Dict[str, Any],
    gold_q: Dict[str, Any],
    tok: AutoTokenizer,
    max_len: int = 1024,
    head_max_len: int = 256,
    marker_token_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build a single tokenized training item from a question definition and gold answer.

    Uses the built-in `render_options` and `build_sequence` packing rules. If
    `marker_token_id` is passed (from `models.py`), uses that custom marker token ID;
    otherwise falls back to `tok.mask_token_id`.

    Args:
        state (Union[str, dict, list]): Context/state dict or serialized string.
        q (Dict[str, Any]): Question dict containing `'type'`, `'instructions'`, and `'criteria'`.
        gold_q (Dict[str, Any]): Ground-truth dict containing `'probabilities'` and `'label'`.
        tok (AutoTokenizer): Hugging Face `AutoTokenizer`.
        max_len (int): Maximum total token sequence length. Defaults to 1024.
        head_max_len (int): Maximum token budget for the question header and options. Defaults to 256.
        marker_token_id (Optional[int]): Optional custom option marker token ID from `models.py`.

    Returns:
        Optional[Dict[str, Any]]: Dictionary with keys `("ids", "markers", "qtype", "target", "label")`,
            or `None` if the option markers do not fit within the token budget.

    Raises:
        ValueError: If `q["type"]` is not a supported decision primitive.
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
        marker_token_id=marker_token_id,
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


def build_item_from_row(
    row: Dict[str, Any],
    tok: AutoTokenizer,
    max_len: int = 1024,
    head_max_len: int = 256,
    marker_token_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Convert a standardized decision row dictionary into a tokenized training item.

    Args:
        row (Dict[str, Any]): Row produced by `auto_prepare_dataset` containing
            `'context'`, `'decision_type'`, `'options'`, `'target'`, and `'scale'`.
        tok (AutoTokenizer): Hugging Face `AutoTokenizer`.
        max_len (int): Maximum total token sequence length. Defaults to 1024.
        head_max_len (int): Maximum token budget for the question header and options. Defaults to 256.
        marker_token_id (Optional[int]): Optional custom option marker token ID from `models.py`.

    Returns:
        Optional[Dict[str, Any]]: Tokenized training item dict, or `None` if invalid/truncated.
    """
    decision_type = row.get("decision_type", "choice")
    context = row.get("context", "")
    options = row.get("options") or []
    target_val = row.get("target")

    if decision_type == "choice":
        if not options:
            return None
        criteria = {str(opt): "" for opt in options}
        target_str = str(target_val) if target_val is not None else None
        target_probs = {}
        for opt in criteria.keys():
            target_probs[opt] = 1.0 if (opt == target_str) else 0.0
        s = sum(target_probs.values())
        if s > 0:
            target_probs = {k: v / s for k, v in target_probs.items()}
        else:
            uniform = 1.0 / max(1, len(criteria))
            target_probs = {k: uniform for k in target_probs}

        q = {"type": "choice", "instructions": "Select the best option.", "criteria": criteria}
        gold = {"probabilities": target_probs, "label": target_str}
        return build_training_item(
            context, q, gold, tok, max_len=max_len, head_max_len=head_max_len, marker_token_id=marker_token_id
        )

    elif decision_type == "noul":
        target_bool = _normalize_to_bool(target_val) if target_val is not None else False
        probs = {"false": 0.0 if target_bool else 1.0, "true": 1.0 if target_bool else 0.0}
        q = {"type": "noul", "instructions": "Does the statement hold?", "criteria": None}
        gold = {"probabilities": probs, "label": "true" if target_bool else "false"}
        return build_training_item(
            context, q, gold, tok, max_len=max_len, head_max_len=head_max_len, marker_token_id=marker_token_id
        )

    elif decision_type == "score":
        scale = row.get("scale") or (0, 3)
        scale_min, scale_max = float(scale[0]), float(scale[1])
        n_levels = max(2, int(round(scale_max - scale_min + 1)))
        if target_val is not None:
            raw_f = float(target_val)
            target_idx = int(round(raw_f - scale_min))
        else:
            target_idx = 0
        target_idx = max(0, min(n_levels - 1, target_idx))
        probs = {str(i): (1.0 if i == target_idx else 0.0) for i in range(n_levels)}
        criteria = [f"level {int(scale_min) + i}" for i in range(n_levels)]
        q = {"type": "score", "instructions": "Assign a score rating.", "criteria": criteria}
        gold = {"probabilities": probs, "label": target_idx}
        return build_training_item(
            context, q, gold, tok, max_len=max_len, head_max_len=head_max_len, marker_token_id=marker_token_id
        )

    return None


def prepare_dataset_items(
    dataset: Any,
    tok: AutoTokenizer,
    max_len: int = 1024,
    head_max_len: int = 256,
    marker_token_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Tokenize an entire prepared dataset into a list of model-ready training items.

    Supports:
        1. `LocalLLaMA/typed-decisions` benchmark rows (`state`, `questions`, `gold` JSON strings).
        2. Standardized decision rows (`context`, `decision_type`, `options`, `target`, `scale`).
        3. Pre-tokenized item lists (`ids`, `markers`, `qtype`, `target`, `label`).

    Args:
        dataset (Any): Prepared dataset or iterable of rows.
        tok (AutoTokenizer): Hugging Face `AutoTokenizer`.
        max_len (int): Maximum total token sequence length. Defaults to 1024.
        head_max_len (int): Maximum token budget for the question header and options. Defaults to 256.
        marker_token_id (Optional[int]): Optional custom option marker token ID from `models.py`.
            If `None`, falls back to the default packing rule (`tok.mask_token_id`).

    Returns:
        List[Dict[str, Any]]: List of tokenized training items ready for splitting and batching.
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
                    it = build_training_item(
                        state,
                        q,
                        gold[qid],
                        tok,
                        max_len=max_len,
                        head_max_len=head_max_len,
                        marker_token_id=marker_token_id,
                    )
                    if it:
                        items.append(it)
        elif "context" in row and "decision_type" in row:
            it = build_item_from_row(
                row,
                tok,
                max_len=max_len,
                head_max_len=head_max_len,
                marker_token_id=marker_token_id,
            )
            if it:
                items.append(it)

    return items


# ===========================================================================
# 5. Batch Collation & Dataset Splitting
# ===========================================================================

def collate_train_batch(
    items: List[Dict[str, Any]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """Collate variable-length tokenized items and option markers into padded PyTorch tensors.

    Args:
        items (List[Dict[str, Any]]): List of tokenized item dicts (`ids`, `markers`, `target`, `qtype`, `label`).
        pad_id (int): Padding token ID from the tokenizer.

    Returns:
        Dict[str, torch.Tensor]: Padded batch dictionary with keys:
            - `input_ids`: `(batch_size, max_seq_len)`
            - `attention_mask`: `(batch_size, max_seq_len)`
            - `marker_pos`: `(batch_size, max_options)`
            - `marker_mask`: `(batch_size, max_options)`
            - `target`: `(batch_size, max_options)`
            - `qtype`: `(batch_size,)`
            - `label`: `(batch_size,)`
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


def split_dataset_items(
    all_items: List[Dict[str, Any]],
    test_ratio: float = 0.20,
    calib_ratio: float = 0.10,
    calib_max: int = 20,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition tokenized items into non-overlapping Train, Calibration, and Test splits.

    Args:
        all_items (List[Dict[str, Any]]): Full list of tokenized training items.
        test_ratio (float): Fraction of items held out for evaluation. Defaults to 0.20.
        calib_ratio (float): Fraction of items held out for temperature calibration. Defaults to 0.10.
        calib_max (int): Maximum number of items held out for calibration. Defaults to 20.
        seed (int): Random seed for deterministic shuffling. Defaults to 42.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
            `(train_items, calib_items, test_items)`.

    Raises:
        ValueError: If `all_items` contains fewer than 3 items.
    """
    n_total = len(all_items)
    if n_total < 3:
        raise ValueError(f"Need at least 3 items to split, got {n_total}")

    order = list(range(n_total))
    random.Random(seed).shuffle(order)

    n_test = max(1, int(n_total * test_ratio))
    n_calib = min(calib_max, max(1, int(n_total * calib_ratio)))
    n_train = n_total - n_test - n_calib

    if n_train < 1:
        n_train = 1
        n_test = max(1, n_total - n_train - n_calib)

    test_indices = order[:n_test]
    calib_indices = order[n_test : n_test + n_calib]
    train_indices = order[n_test + n_calib :]

    test_items = [all_items[i] for i in test_indices]
    calib_items = [all_items[i] for i in calib_indices]
    train_items = [all_items[i] for i in train_indices]

    print(
        f"\nDataset Split Summary (Seed {seed}):\n"
        f"  Total:       {n_total} items\n"
        f"  Train:       {len(train_items)} items ({len(train_items)/n_total:.1%})\n"
        f"  Calibration: {len(calib_items)} items ({len(calib_items)/n_total:.1%})\n"
        f"  Test:        {len(test_items)} items ({len(test_items)/n_total:.1%})"
    )

    return train_items, calib_items, test_items
