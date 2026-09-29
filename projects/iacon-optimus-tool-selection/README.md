# Fine-tune FunctionGemma 270M for Tool Selection

**Iacon Optimus project — Gemma track**
**Org:** Humanitarians AI (Nik Bear Brown, Northeastern University)
**Model reference:** [FunctionGemma model card](https://ai.google.dev/gemma/docs/functiongemma/model_card)

## Research question

Can a fine-tuned ~270M-parameter model select tools reliably enough to remove
tool-routing work from the main agent?

## Approach

- Start from pretrained FunctionGemma 270M. Fine-tune existing weights; no
  pretraining or training from scratch.
- **Input:** task state, recent observations, available tool
  descriptions/schemas. **Output:** a tool decision, including an explicit
  no-tool / abstain option.
- Generate supervised examples from recorded or synthetic tool-use tasks.
  Validate teacher-produced labels through sandbox execution before using them.
- **Stage 1 — supervised fine-tuning** on validated labels.
- **Stage 2 — execution-reward optimization.** Reward task completion and
  valid, useful calls; penalize unnecessary calls and invalid selections.
  Report gains separately from Stage 1.
- Hold out whole tool families and task domains, not random examples. Include
  distractor tools, renamed tools, changed schemas, and situations requiring
  no call.
- **Baselines:** untouched pretrained model, embedding-based routing, and the
  main agent choosing its own tools. Argument generation held fixed to isolate
  selection quality.

## Evaluation

Routing accuracy, downstream task success, unnecessary calls, abstention
quality, unseen-tool generalization, latency, and total cost. Reward-stage
gains reported separately from supervised gains, across multiple seeds.

## Deliverables

- [ ] Data-generation and validation pipeline (`src/data_gen/`, `src/validate/`)
- [ ] Licensed / provenanced dataset manifest (`data/manifest.json`)
- [ ] Fine-tuned checkpoint or adapter (`src/train/`)
- [ ] Inference component (`src/infer/`)
- [ ] Reproducible configs (`configs/`)
- [ ] Held-out results across multiple seeds (`results/`)

No human feedback is required inside the optimization loop.

## Repo layout

```
configs/      training, eval, and seed settings
data/         manifest.json (sources, licenses, provenance) — raw data is gitignored
src/data_gen/ task generation and teacher labeling
src/validate/ sandbox execution checks on labels
src/train/    supervised stage, reward stage
src/eval/     metrics and baseline comparisons
src/infer/    inference component
results/      one results table per seed
docs/         decisions, failures, weekly notes
```

## Status

Backlog → in progress. See [docs/decisions.md](docs/decisions.md) for why this
project was chosen over the other six options, and [docs/failures.md](docs/failures.md)
for a running log of what didn't work.

## Setup

```bash
pip install -r requirements.txt
```

(Fill in as the pipeline takes shape.)
