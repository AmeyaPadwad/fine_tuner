# RLCD Reward Strategies

This README documents two approaches for fine-tuning a decision model (Jev/Laya family) using Reinforcement Learning with Calibrated Decisions (RLCD) as the objective. Both are illustrated using the same worked example: a medical triage model deciding whether a patient case is "urgent" or "not urgent," with a confidence score attached to each decision.

---

## Shared Setup

The model outputs, for each case:

- **context** [dict]: details about patient case
- **decision** [string]: "urgent" / "not urgent"
- **confidence** [float]: a value between 0 and 1
- **outcome** [int]: whether the decision was actually correct (1) or wrong (0), determined from ground truth in the dataset

Example logged rollouts:

| context    | decision     | confidence | outcome     |
| ---------- | ------------ | ---------- | ----------- |
| Case #1023 | "urgent"     | 0.92       | correct (1) |
| Case #1024 | "not urgent" | 0.60       | correct (1) |
| Case #1025 | "urgent"     | 0.55       | wrong (0)   |
| Case #1026 | "urgent"     | 0.90       | correct (1) |
| Case #1027 | "not urgent" | 0.85       | wrong (0)   |

Example Context:

```py
Case #1024: {
    "age": 67,
    "heart_rate": 118,
    "blood_pressure_systolic": 88,
    "symptom_category": "chest_pain",
    "num_prior_visits": 3,
    "triage_nurse_initial_flag": "yellow"
}
```

---

## Idea 1: Proper Scoring Rule as Reward

**Core concept:** Use a strictly proper scoring rule (e.g., negative Brier score) directly as the RL reward. The reward is computed instantly from a single decision's confidence and its own outcome.

**Formula:**

```
reward = -(confidence - outcome)^2
```

### Worked example

| context    | decision     | confidence | outcome     | reward = -(confidence - outcome)² |
| ---------- | ------------ | ---------- | ----------- | --------------------------------- |
| Case #1023 | "urgent"     | 0.92       | correct (1) | -(0.92-1)² = **-0.0064**          |
| Case #1024 | "not urgent" | 0.60       | correct (1) | -(0.60-1)² = **-0.16**            |
| Case #1025 | "urgent"     | 0.55       | wrong (0)   | -(0.55-0)² = **-0.3025**          |
| Case #1026 | "urgent"     | 0.90       | correct (1) | -(0.90-1)² = **-0.01**            |
| Case #1027 | "not urgent" | 0.85       | wrong (0)   | -(0.85-0)² = **-0.7225**          |

### Why it works

The Brier score is a **strictly proper scoring rule** — its expected value is maximized only when the reported confidence equals the true probability of being correct. So gradient descent, applied across many such samples, statistically pushes the model toward calibration, even though each individual reward is computed in isolation.

### Tradeoffs

- ✅ Simple to implement
- ✅ Reward available immediately, per sample
- ⚠️ Noisy — a single outlier swings the reward heavily
- Best suited to single-shot / bandit-style decisions

---

## Idea 2: Learned Auxiliary Reward Model

**Core concept:** Calibration error is fundamentally a _population-level_ statistic. Train a small auxiliary model to learn this mapping from logged data, then use it to produce a smooth, per-sample reward for RL.

### Step 1 — Collect rollout logs

Run the model on the dataset and log `(context, decision, confidence, outcome)` for many samples (same table as above, at scale — e.g., 5,000 cases).

### Step 2 — Bin by confidence, compute empirical accuracy

| confidence bin | # samples | # correct | empirical accuracy |
| -------------- | --------- | --------- | ------------------ |
| 0.5–0.6        | 420       | 210       | 0.50               |
| 0.6–0.7        | 380       | 228       | 0.60               |
| 0.8–0.9        | 500       | 300       | 0.60               |
| 0.9–1.0        | 600       | 570       | 0.95               |

### Step 3 — Compute per-sample training targets

For each row, look up its bin's empirical accuracy and compute a target reward:

```
target_reward = -(confidence - empirical_accuracy_of_bin)^2
```

| confidence bin | # samples | # correct | Empirical accuracy | target reward² |
| -------------- | --------- | --------- | ------------------ | -------------- |
| 0.5–0.6        | 420       | 210       | 0.50               | -0.0025        |
| 0.6–0.7        | 380       | 228       | 0.60               | -0.0025        |
| 0.8–0.9        | 500       | 300       | 0.60               | -0.0625        |
| 0.9–1.0        | 600       | 570       | 0.95               | 0.0000         |

### Step 4 — Train the auxiliary model

Train a small regression model (e.g., shallow MLP over embeddings) to predict `target_reward` from `(context, decision, confidence)`.
This generalizes across contexts and confidence levels, smoothing out the blocky bin boundaries.

The context can be converted to text embeddings of the raw input for input to the auxillary model.

### Step 5 — Use it in the RL loop

For any new decision sampled during policy fine-tuning:

```python
reward = auxiliary_model(context, decision, confidence)
```

### Important: periodic refresh

As the policy updates, its confidence distribution shifts, so the auxiliary model's learned calibration surface can go stale.

Mitigations:

- Re-roll the current policy periodically and retrain/fine-tune the auxiliary model (EM-style loop)
- Add a KL penalty vs. a reference policy to prevent drifting into regions the auxiliary model wasn't trained on
- Use an ensemble of auxiliary models (mean or pessimistic aggregation) to reduce reward-hacking risk
- Track true ECE on a held-out set, independent of the auxiliary model, as a sanity check

### Tradeoffs

- ✅ Smooth, lower-variance reward - reflects _typical_ accuracy at a confidence level rather than one noisy binary outcome
- ✅ Can condition on context, correcting for some cases being inherently harder than others
- ⚠️ More complex - requires data collection, binning/fitting, model training, and periodic retraining
- ⚠️ Risk of reward hacking if the auxiliary model goes stale relative to the current policy
