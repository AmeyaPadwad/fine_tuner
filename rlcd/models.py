import json
import os
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_model
from safetensors.torch import load_file
from transformers import AutoTokenizer


VALID_DECISION_TYPES = ("choice", "score", "noul")
DEFAULT_CACHE_DIR = os.getenv("DEFAULT_CACHE_DIR") 


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

        DEFAULT_LAYA_CACHE_DIR = DEFAULT_CACHE_DIR+"/laya/model_name/"

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

    @staticmethod
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

    @staticmethod
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
            return Laya.row_to_laya_input(
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

    @staticmethod
    def _as_list(value):
        """Return value as a parsed list, decoding it first if it's a JSON string.

        """
        if isinstance(value, str):
            value = json.loads(value)
        return value or []

    @staticmethod
    def prepare_xlam_for_laya(dataset):
        """Prepare Salesforce/xlam-function-calling-60k as Laya "choice" inputs.

        Args:
            dataset: The dataset object returned by helper.load_dataset.

        Returns:
            Same container type as dataset, with Laya-ready
            context/decision_type/target/options/
            scale fields added to each row.
        """

        def context_fn(row):
            return row["query"]

        def options_fn(row):
            return [tool["name"] for tool in Laya._as_list(row["tools"])]

        def target_fn(row):
            answers = Laya._as_list(row["answers"])
            if not answers:
                return None
            first = answers[0]
            return first.get("name") or first.get("tool_name")

        return Laya.prepare_dataset_for_laya(
            dataset,
            context_fn=context_fn,
            decision_type="choice",
            options_fn=options_fn,
            target_fn=target_fn,
        )
