# Agent Verification Plan: RLCD SDK & `rlcd-fine-tune` Skill

This document defines a **3-Phase Agent Verification Plan** and step-by-step instructions to verify that an AI agent (with zero prior knowledge of the codebase) can autonomously read [`skills/rlcd-fine-tune/SKILL.md`](../skills/rlcd-fine-tune/SKILL.md), invoke the `rlcd` Python SDK or `rlcd-fine-tune` CLI, and fine-tune arbitrary decision models and datasets end to end.

---

## Pre-Test Setup

Before starting an agent evaluation session, complete these two setup steps once in the workspace root:

### Step 1: Install the `rlcd` SDK and `rlcd-fine-tune` CLI in `.venv`

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate
# Or on macOS / Linux:
# source .venv/bin/activate

# Install editable package so `rlcd-fine-tune` and `import rlcd` are registered
pip install -e . --no-deps
```

### Step 2: Register the Skill in Your Agent Framework's Skill Directory

Copy `skills/rlcd-fine-tune/SKILL.md` into the skill discovery folder of the agent platform you are testing:

```bash
# For arkenOS / Optimus agents:
mkdir -p .optimus/skills/rlcd-fine-tune
cp skills/rlcd-fine-tune/SKILL.md .optimus/skills/rlcd-fine-tune/SKILL.md

# For Claude Code agents:
mkdir -p .claude/skills/rlcd-fine-tune
cp skills/rlcd-fine-tune/SKILL.md .claude/skills/rlcd-fine-tune/SKILL.md
```

---

## 3-Phase Agent Verification Plan

Always run each phase in a **fresh agent session** (or via an isolated specialist agent run) so the agent has no memory of previous sessions and must rely solely on `SKILL.md` and the `rlcd` SDK/CLI.

---

### Phase 1: Skill Usability, SDK Integrity & Smoke Tests

**Objective:** Verify that the agent discovers and loads `SKILL.md`, confirms that the `rlcd` Python SDK and `rlcd-fine-tune` CLI are callable, and executes the repository's smoke test suites.

#### 1. Prompt to Give the Agent
> *"Use the `rlcd-fine-tune` skill to verify that the `rlcd` Python SDK and `rlcd-fine-tune` CLI command are installed and working in `.venv`, then run the project's smoke test suites (`smoke_tests.py` and `reward_strategies/smoke_tests.py`) and report the results."*

#### 2. Expected Agent Actions
1. Reads `skills/rlcd-fine-tune/SKILL.md` (or `.optimus/skills/rlcd-fine-tune/SKILL.md`).
2. Checks `rlcd-fine-tune --help` and `import rlcd` inside `.venv`.
3. Runs:
   ```bash
   python smoke_tests.py --all
   python reward_strategies/smoke_tests.py
   ```

#### 3. Pass Criteria
- [ ] Agent locates and reads `SKILL.md` without human guidance.
- [ ] `rlcd-fine-tune --help` and `import rlcd` succeed with exit code `0`.
- [ ] All smoke test suites (`HelperSmokeTests`, `ModelsSmokeTests`, `ProcessingSmokeTests`, `TrainingSmokeTests`, and `BrierSmokeTests`) report `FINAL RESULT: ALL PASS`.

---

### Phase 2: End-to-End Pipeline Execution (Default Model & Dataset)

**Objective:** Verify that the agent can run the complete RLCD pipeline end to end using the **default model** (`convaiinnovations/laya`), **default dataset** (`Salesforce/xlam-function-calling-60k`), and **default hyperparameters**, overriding only the subset size and output directory.

#### 1. Prompt to Give the Agent
> *"Using the `rlcd-fine-tune` skill, run the pipeline with the default model, default dataset, and default hyperparameters on a small subset of 15 examples, saving the outputs to `./outputs/test_phase2_defaults`. Report the baseline and final Accuracy, Mean Brier Score, and ECE."*

#### 2. Expected Agent Actions
1. Invokes the single CLI command (or `rlcd.fine_tune(...)` in Python):
   ```bash
   rlcd-fine-tune --subset 15 --output-dir ./outputs/test_phase2_defaults
   ```
2. Waits for the pipeline to complete all stages (data loading $\to$ schema auto-detection $\to$ `[MASK]` tokenization $\to$ train/calibration/test split $\to$ baseline evaluation $\to$ GRPO training $\to$ post-hoc calibration $\to$ model export $\to$ final evaluation).
3. Reads `./outputs/test_phase2_defaults/test_evaluation_metrics.json`.

#### 3. Pass Criteria
- [ ] Agent does **not** modify any source code files (`main.py`, `processing.py`, `training.py`, `models.py`).
- [ ] `./outputs/test_phase2_defaults/model.safetensors`, `rl_agent_config.json`, and `test_evaluation_metrics.json` are created on disk.
- [ ] Agent accurately reports the `baseline` and `final` Accuracy, Mean Brier Score, ECE, and selected `post_hoc_calibration` strategy from `test_evaluation_metrics.json`.

---

### Phase 3: Blind Multi-Run Reliability Test (Unseen Models & Datasets)

**Objective:** Verify that the agent can **reliably and repeatedly** use the skill and SDK across multiple unseen `(model, dataset)` pairs spanning all three decision primitives (`choice`, `noul`, `score`) without being told the dataset column names or schema structure in advance.

Run the following three blind tasks in separate agent turns or fresh agent sessions:

#### Blind Run 3A — Multi-Class Text Classification (`choice` Primitive)
- **Prompt to Give the Agent:**
  > *"Fine-tune `convaiinnovations/laya` on 20 examples of `mteb/tweet_sentiment_extraction` for 1 epoch and save the model to `./outputs/test_phase3_runA`. Report the before and after evaluation metrics."*
- **What This Tests:**
  - Whether the agent and `auto_prepare_dataset` automatically detect `text` $\to$ `label_text` and auto-collect the global sentiment options (`["negative", "neutral", "positive"]`) without requiring an options column in each row.
- **Expected Command:**
  ```bash
  rlcd-fine-tune --model convaiinnovations/laya --dataset mteb/tweet_sentiment_extraction --subset 20 --epochs 1 --output-dir ./outputs/test_phase3_runA
  ```

#### Blind Run 3B — Contrastive Preference Pairs (`noul` Primitive)
- **Prompt to Give the Agent:**
  > *"Fine-tune `convaiinnovations/laya` on 15 examples of `Anthropic/hh-rlhf` for 1 epoch and save the model to `./outputs/test_phase3_runB`. Report the baseline and final metrics."*
- **What This Tests:**
  - Whether the pipeline automatically detects the contrastive `chosen`/`rejected` preference schema and expands it into calibrated binary (`noul`) examples without user intervention.
- **Expected Command:**
  ```bash
  rlcd-fine-tune --model convaiinnovations/laya --dataset Anthropic/hh-rlhf --subset 15 --epochs 1 --output-dir ./outputs/test_phase3_runB
  ```

#### Blind Run 3C — Multi-Column Ordinal Rating (`score` Primitive + Custom Model)
- **Prompt to Give the Agent:**
  > *"Fine-tune `wfzyx/von` on 20 examples of `lmsys/toxic-chat` to score `openai_moderation` on a 1 to 5 scale for 1 epoch, saving the output to `./outputs/test_phase3_runC`. Report the baseline and final metrics."*
- **What This Tests:**
  - Whether the agent follows `SKILL.md` to pass `--column-map` for a multi-column dataset with an explicit ordinal `score` target and loads a second model family (`wfzyx/von`).
- **Expected Command:**
  ```bash
  rlcd-fine-tune \
    --model wfzyx/von \
    --dataset lmsys/toxic-chat \
    --subset 20 \
    --epochs 1 \
    --column-map '{"context": ["user_input", "model_output"], "target": "openai_moderation", "decision_type": "score", "scale": [1, 5]}' \
    --output-dir ./outputs/test_phase3_runC
  ```

#### Pass Criteria for Phase 3
- [ ] All three blind runs (`3A`, `3B`, `3C`) finish with exit code `0` and zero manual debugging or code changes.
- [ ] Each output directory (`test_phase3_runA`, `test_phase3_runB`, `test_phase3_runC`) contains `model.safetensors`, `rl_agent_config.json`, and `test_evaluation_metrics.json`.
- [ ] Agent reports verified metrics read directly from each `test_evaluation_metrics.json` artifact.

---

## Summary Verification Checklist Table

Use this table to record the agent test outcomes:

| Phase | Test Case | Target Model | Target Dataset | Primitive | Exit Code `0`? | `test_evaluation_metrics.json` Verified? |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Phase 1** | SDK, CLI & Smoke Tests | `convaiinnovations/laya` | Unit/Smoke Fixtures | All | [ ] | N/A (`ALL PASS`) |
| **Phase 2** | Default End-to-End Run | `convaiinnovations/laya` | `Salesforce/xlam-function-calling-60k` | `choice` | [ ] | [ ] |
| **Phase 3A** | Blind Classification | `convaiinnovations/laya` | `mteb/tweet_sentiment_extraction` | `choice` | [ ] | [ ] |
| **Phase 3B** | Blind Contrastive Pairs | `convaiinnovations/laya` | `Anthropic/hh-rlhf` | `noul` | [ ] | [ ] |
| **Phase 3C** | Blind Ordinal Scoring | `wfzyx/von` | `lmsys/toxic-chat` | `score` | [ ] | [ ] |
