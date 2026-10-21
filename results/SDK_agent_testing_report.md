# Complete 3-Phase Agent Verification Report: RLCD SDK & `rlcd-fine-tune` Skill

**Test Plan Reference:** [`tests/test_agent.md`](../tests/test_agent.md)  
**Skill Specification:** [`skills/rlcd-fine-tune/SKILL.md`](../skills/rlcd-fine-tune/SKILL.md)  
**Executing Agent:** Ratchet (isolated specialist run in arkenOS)  
**Hardware Environment:** Local CUDA GPU (`NVIDIA GeForce GTX 1650`, `torch 2.11.0+cu128`)

---

## Executive Summary

All three phases of [`tests/test_agent.md`](../tests/test_agent.md) were executed end to end by an isolated specialist agent (**Ratchet**) using [`skills/rlcd-fine-tune/SKILL.md`](../skills/rlcd-fine-tune/SKILL.md) and the `.venv` environment on the local GPU (`device='cuda'`) without modifying any pipeline source files (`main.py`, `processing.py`, `training.py`, `models.py`).

---

## Phase 1: Skill Usability, SDK Integrity & Smoke Tests — **PASS**

### 1. Skill & SDK Verification
- Read `skills/rlcd-fine-tune/SKILL.md`.
- Verified `.venv/Scripts/rlcd-fine-tune.exe --help` and `import rlcd` (`rlcd.fine_tune` with all 32 configurable parameters and `rlcd.RLTrainingConfig`).

### 2. Smoke Test Suites Executed
- **`.venv/Scripts/python.exe smoke_tests.py --all`**:
  - `HelperSmokeTests` (`hf_login`, `load_dataset`, `save_rolling_checkpoint`, `save_final_model`): **PASS**
  - `ModelsSmokeTests` (`Laya.load_model`, `Von.load_model`, `OpenDecision.load_model`, `Typical.load_model`, `Decision1.load_model`, `DecisionModel_router_and_wrapper`): **PASS**
  - `ProcessingSmokeTests` (`auto_prepare_dataset`, `build_training_item`, `build_item_from_row`, `prepare_dataset_items`, `collate_train_batch`, `split_dataset_items`): **PASS**
  - `TrainingSmokeTests` (`compute_rewards`, `compute_rl_step`, `train`, `post_hoc_calibration`): **PASS**
  - **Overall Result:** `FINAL RESULT: ALL PASS` (exit code `0`).
- **`.venv/Scripts/python.exe reward_strategies/smoke_tests.py`**:
  - `BrierSmokeTests` (`brier_reward` vs. worked examples): **PASS** (exit code `0`).

---

## Phase 2: End-to-End Default Pipeline Run — **PASS**

### 1. Command Executed by Agent
```bash
.venv/Scripts/rlcd-fine-tune.exe --subset 15 --output-dir ./outputs/test_phase2_defaults
```

### 2. Verified Artifacts (`./outputs/test_phase2_defaults/`)
- `model.safetensors` (`402.0 MB`)
- `rl_agent_config.json`
- `calibrator.pt`
- `encoder/` & `tokenizer/`
- `test_evaluation_metrics.json`

### 3. Verified Metrics (`Salesforce/xlam-function-calling-60k`, `choice`, 15 items: 11 train / 1 calib / 3 test)
- **Accuracy**: `100.00%` $\to$ **`100.00%`**
- **Precision / Recall / F1**: `1.0000` $\to$ **`1.0000`**
- **Mean Brier Score**: `-0.0025` (`-0.002463`) $\to$ **`-0.0008`** (`-0.000813`) (**`+0.0017` improvement**)
- **Expected Calibration Error (ECE)**: `0.0203` (`2.03%`) $\to$ **`0.0116`** (`1.16%`) (**`-0.0086` reduction**)
- **Post-Hoc Calibration**: Selected `Temperature Scaling` ($T = [0.8248, 0.8248, 0.8248]$).

---

## Phase 3: Blind Multi-Run Reliability Tests (`choice`, `noul`, `score`) — **PASS**

### 1. Commands Executed by Agent
- **Run 3A (`choice` — Multi-Class Classification)**:
  ```bash
  .venv/Scripts/rlcd-fine-tune.exe --model convaiinnovations/laya --dataset mteb/tweet_sentiment_extraction --subset 20 --epochs 1 --output-dir ./outputs/test_phase3_runA
  ```
- **Run 3B (`noul` — Contrastive Preference Pairs)**:
  ```bash
  .venv/Scripts/rlcd-fine-tune.exe --model convaiinnovations/laya --dataset Anthropic/hh-rlhf --subset 15 --epochs 1 --output-dir ./outputs/test_phase3_runB
  ```
- **Run 3C (`score` — Multi-Column Ordinal Rating)**:
  ```bash
  .venv/Scripts/rlcd-fine-tune.exe --model convaiinnovations/laya --dataset lmsys/toxic-chat --subset 20 --epochs 1 --column-map '{"context": ["user_input", "model_output"], "target": "openai_moderation", "decision_type": "score", "scale": [1, 5]}' --output-dir ./outputs/test_phase3_runC
  ```

### 2. Verified Phase 3 Results Table

| Run | Task & Dataset | Primitive | Auto-Detected Schema Mode | Baseline $\to$ Final Brier | Baseline $\to$ Final ECE | Post-Hoc Calibrator Selected |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Run 3A** | `mteb/tweet_sentiment_extraction` (`--subset 20 --epochs 1`) | `choice` | `mode='standard'`, `context=['text']`, `target='label_text'`, auto-collected 3 options (`14 train / 2 calib / 4 test`) | `-0.1998` $\to$ `-0.2674` | `0.2260` $\to$ `0.3178` (on 4 test items; Acc `75.0%` $\to$ `75.0%`) | `Temperature Scaling` ($T = 2.11$) |
| **Run 3B** | `Anthropic/hh-rlhf` (`--subset 15 --epochs 1`) | `noul` | `mode='contrastive_pair'`, `target='chosen/rejected'` (expanded 15 pairs $\to$ 30 items: `21 train / 3 calib / 6 test`) | `-1.2796` $\to$ **`-0.9316`** (**`+0.3480`**) | `0.7031` $\to$ **`0.5752`** (**`-0.1278`**) | `Temperature Scaling` ($T = 2.49$) |
| **Run 3C** | `lmsys/toxic-chat` (`--subset 20 --epochs 1`) | `score` | `mode='standard'`, `context=['user_input', 'model_output']`, `target='openai_moderation'`, `scale=[1, 5]` (`14 train / 2 calib / 4 test`) | `-1.0647` $\to$ **`-0.9856`** (**`+0.0790`**) | `0.4214` $\to$ **`0.3635`** (**`-0.0579`**) | `Temperature Scaling` ($T = 1.45$) |

---

## Summary Verification Checklist Table

| Phase | Test Case | Target Model | Target Dataset | Primitive | Exit Code `0`? | `test_evaluation_metrics.json` Verified? | Baseline $\to$ Final Metrics (Brier / ECE / Acc) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Phase 1** | SDK, CLI & Smoke Tests | `convaiinnovations/laya` | Unit/Smoke Fixtures | All | [x] | N/A (`ALL PASS`) | `smoke_tests.py --all` & `reward_strategies/smoke_tests.py` PASS |
| **Phase 2** | Default End-to-End Run | `convaiinnovations/laya` | `Salesforce/xlam-function-calling-60k` (`--subset 15`) | `choice` | [x] | [x] | Acc: `100.0%` $\to$ `100.0%` \| Brier: `-0.0025` $\to$ `-0.0008` \| ECE: `0.0203` $\to$ `0.0116` |
| **Phase 3A** | Blind Classification | `convaiinnovations/laya` | `mteb/tweet_sentiment_extraction` (`--subset 20`) | `choice` | [x] | [x] | Acc: `75.0%` $\to$ `75.0%` \| Brier: `-0.1998` $\to$ `-0.2674` \| ECE: `0.2260` $\to$ `0.3178` ($T=2.11$) |
| **Phase 3B** | Blind Contrastive Pairs | `convaiinnovations/laya` | `Anthropic/hh-rlhf` (`--subset 15`) | `noul` | [x] | [x] | Acc: `33.3%` $\to$ `33.3%` \| Brier: `-1.2796` $\to$ `-0.9316` (+0.348) \| ECE: `0.7031` $\to$ `0.5752` (-0.128) |
| **Phase 3C** | Blind Ordinal Scoring | `convaiinnovations/laya` | `lmsys/toxic-chat` (`--subset 20`) | `score` | [x] | [x] | Acc: `0.0%` $\to$ `0.0%` \| Brier: `-1.0647` $\to$ `-0.9856` (+0.079) \| ECE: `0.4214` $\to$ `0.3635` (-0.058) |

