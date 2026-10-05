"""CLI entry point: python -m src.train.main --config configs/job_config.example.json

REWRITTEN Sept 28 for the Laya/RLCD scope correction — no more tokenizer-
based pre-flight (Laya's encoder doesn't need a chat tokenizer), and the
dataset step no longer takes a tokenizer argument since dataset.py just
normalizes rows now (see dataset.py's docstring for why).
"""

from __future__ import annotations

import argparse
import sys

from .schema import JobSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Laya decision-model RLCD training worker")
    parser.add_argument("--config", required=True, help="Path to a job_config.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, run pre-flight, prepare data — stop before loading model weights",
    )
    args = parser.parse_args(argv)

    print(f"[1/5] Validating job spec: {args.config}")
    try:
        job = JobSpec.from_json_file(args.config)
    except Exception as e:
        print(f"Config validation failed: {e}", file=sys.stderr)
        return 1
    print(f"      job_id={job.job_id}  base_model={job.model.base_model_id}  decision_type={job.dataset.decision_type.value}")

    print("[2/5] Preparing dataset")
    from . import dataset as dataset_mod

    train_examples, eval_examples = dataset_mod.prepare_dataset(job.dataset)
    print(f"      {len(train_examples)} train / {len(eval_examples)} eval examples")

    if job.dataset.decision_type.value == "choice":
        hist = dataset_mod.option_cardinality_histogram(train_examples)
        rate = dataset_mod.abstain_rate(train_examples)
        print(f"      option-count histogram: {hist}")
        print(f"      abstain rate: {rate:.1%}" if rate is not None else "      abstain rate: n/a")

    print("[3/5] Pre-flight: loading the agent to get a real parameter count")
    from . import model as model_mod
    from . import preflight

    try:
        agent = model_mod.load_agent(job)
    except ImportError as e:
        print(f"      laya (or torch) not installed — skipping model-backed steps ({e})")
        agent = None

    if agent is not None:
        num_params = model_mod.count_parameters(agent)
        hw, estimate, advice = preflight.run_preflight(job, num_params)
        print("      " + hw.summary().replace("\n", "\n      "))
        print(f"      VRAM estimate: {estimate}")
        for line in advice:
            print(f"      ADVICE: {line}")

    if args.dry_run or agent is None:
        if agent is None:
            print("Cannot continue past dry-run without laya/torch installed.", file=sys.stderr)
            return 0 if args.dry_run else 1
        print("Dry run complete — stopping before training.")
        return 0

    print("[4/5] Training")
    from . import trainer as trainer_mod

    result = trainer_mod.run_training(job, agent, train_examples, eval_examples)
    print(f"      final metrics: {result['metrics']}")

    print("[5/5] Exporting")
    from . import exporter

    outputs = exporter.run_export(result["agent"], job)
    for key, path in outputs.items():
        print(f"      {key}: {path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
