# Decisions

## Why this project

Chose the Gemma track, building around FunctionGemma 270M, for three reasons:

1. It's a 270M model meant to be fine-tuned for one narrow task, so it can be
   trained and tested on modest hardware without a large GPU setup.
2. Google publishes a dataset and fine-tuning recipe as a working baseline;
   the model card reports 58% before fine-tuning and 85% after on its Mobile
   Actions task, so improvement is easy to measure.
3. Success can be checked automatically by running the selected tool in a
   sandbox, and the project includes an explicit no-tool option, which fits
   the Optimus loop of environment, reward, and evaluation.

(Add: which of the other six projects were seriously considered, and why they
were set aside.)

## Scope correction: Laya/RLCD, not FunctionGemma/SFT (Sept 28, evening)

Dev's feedback on the initial plan: "generalized for any model" meant any
model *in the Laya/Jev family*, fine-tuned with *RLCD*, not any Hugging
Face causal LM fine-tuned with SFT. This invalidated the FunctionGemma/SFT
pipeline built earlier that day, not just its defaults, its whole approach
(SFTTrainer, chat templates, completion masking, LoRA/QLoRA) assumed an
autoregressive text-generation model, and Laya is a bidirectional encoder
(ModernBERT-based) that never generates text, it returns a calibrated
decision (`choice` / `score` / `noul`).

**The actual good news:** tool selection (this project's research
question, including the explicit no-tool/abstain case) maps directly onto
Laya's `choice` primitive — arguably a more natural fit than forcing a
causal LM to generate JSON for what's fundamentally a discrete decision.

**Rebuilt same day**, all in `src/train/`:
- `schema.py` — `JobSpec` now targets any Laya-family checkpoint
  (default `"laya"`, the base English 421M ModernBERT-large checkpoint),
  with an RLCD-shaped `TrainingSpec` (reward_samples, sigma annealing,
  proper-scoring-rule weights, rl/ce loss blend) replacing the old
  LoRA/QLoRA fields entirely.
- `dataset.py` — rows now normalize to `{context, options?, target}` per
  decision type, no chat turns, no response masking (neither concept
  applies to a non-generative encoder).
- `preflight.py` — VRAM estimate now assumes full fine-tuning (no LoRA
  fraction), matching how the one reference implementation found actually
  trained Laya.
- `model.py`, `trainer.py`, `exporter.py` — rewritten for `laya` (real
  PyPI package, confirmed installable) instead of `transformers`/`peft`/`trl`.

**What's confirmed vs. still speculative, be careful here:**
- Confirmed: Laya is a real, open-source, pip-installable (`laya`,
  Apache 2.0), non-autoregressive decision model with the three primitives
  above; Jev is its closed-source, non-fine-tunable counterpart.
- NOT confirmed: the exact RLCD training algorithm in `trainer.py`
  (perturbed-logit sampling, the proper-scoring-rule reward, the
  policy-gradient/CE loss blend) is reconstructed from a third-party
  open-source repo (github.com/shyamsridhar123/LAYA-RLCD), itself only
  read via an AI-summarized fetch rather than literal source. Same for the
  `agent.encoder`/`agent.scorer` attribute names in `model.py`. Every
  place this matters is marked "NEEDS CONFIRMATION" in the file itself.
- **Before running this for real**, pull the official Laya fine-tuning
  reference (the convaiinnovations site mentions Kaggle notebooks for
  this) and diff `trainer.py`/`model.py` against it. Tests currently
  passing (33/33) only exercise `schema.py`/`dataset.py`/`preflight.py`,
  the parts with no real uncertainty left, they say nothing about whether
  the RLCD loop itself is correct.

## Other decisions

(Log format decisions, tool-family choices, reward design choices, etc. here
as they're made, with the reasoning behind each.)

### SFT worker pipeline — initial build (Sept 28)

Started from a teammate's SFT Training Runtime Worker design (job spec
schema, five-stage architecture: config/pre-flight → data ingestion →
model/PEFT setup → SFT loop → export). Built into `src/train/` rather than
a separate top-level `worker/` package, since this project's repo layout
already splits `data_gen/validate/train/eval/infer`, and the worker's scope
lines up with `train/`.

- **Base model:** used `google/functiongemma-270m-it` as `JobSpec`'s
  default, not the `Qwen/Qwen2.5-0.5B` placeholder from the teammate's
  README example — that's the model actually scoped for this project (see
  "Why this project" above). Still open whether the runtime worker needs to
  prove itself against a second model family first; see the planning doc's
  open questions.
- **Dataset format:** added a fourth `DatasetFormat`, `tool_selection`,
  alongside the README's `messages`/`alpaca`/`custom`, matching this
  project's actual label shape (task state + observations + available
  tools → a tool decision, with an explicit `tool: null` abstain case).
  This is provisional and owned jointly with whoever builds out
  `src/data_gen/` and `src/validate/` — update `dataset.py`'s
  `_tool_selection_row_to_turns` if the generator's row shape changes.
- **Response masking:** implemented completion-only loss masking
  (`dataset.build_masked_example`) — loss is computed only on the final
  assistant turn, never on the system prompt or user turn. Tested against a
  fake tokenizer since this sandbox has no GPU or `transformers` installed;
  needs re-validating against the real FunctionGemma tokenizer's chat
  template once running on a GPU host.
- **Environment constraint:** this build sandbox has no GPU and no
  `torch`/`transformers`/`peft`/`trl` installed. Every module that needs
  those defers the import into function bodies (not module level), so
  `schema.py`, `dataset.py`, and `preflight.py` are fully unit-tested here
  (26 passing tests, `pytest tests/`), while `model.py`, `trainer.py`, and
  `exporter.py` are written and import cleanly but are untested against a
  real model — that has to happen on a GPU host per the planning doc's
  compute allocation strategy (Week 1-2 local dev on a small model, Week 3
  cloud GPU).
- **Pre-flight VRAM estimate** (`preflight.estimate_vram_gb`) uses rough
  constants (bytes/param by quantization method, a flat LoRA-trainable-
  fraction, a coarse activation-memory multiplier) good enough to catch an
  obviously bad config, not to size a real run precisely. Calibrate the
  constants against observed runs once there's a GPU to test on — log what's
  off in `failures.md`.
- **LoRA target-module auto-detection:** the first pass hardcoded
  `q_proj`/`k_proj`/etc. as LoRA target modules, which only holds for
  Llama/Mistral/Gemma/Qwen-family naming and would silently do nothing
  useful on an architecture that names its projections differently (GPT-2,
  OPT, Falcon). Replaced with `model.find_lora_target_modules()`, which
  inspects the loaded model's actual `nn.Linear` leaf module names and
  targets whatever it finds (excluding `lm_head`/embedding layers), falling
  back to the hardcoded list only if detection finds nothing. This is the
  piece that actually makes `base_model_id` swappable in practice, not just
  in the schema.
- **Not built yet:** RL/reward-loop integration (workstream 4 in the
  planning doc) — intentionally excluded from this pass since it's still an
  open question whether it's in scope for this team at all (planning doc
  Section 8). GGUF export in `exporter.py` is written but requires
  `llama.cpp`'s `convert_hf_to_gguf.py` on PATH; it raises a clear error
  rather than silently skipping if that's missing.
