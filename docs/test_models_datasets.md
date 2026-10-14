# RL Pipeline Models and Datasets

This document outlines the models and datasets that can be used or adapted for the Reinforcement Learning with Calibrated Decisions (RLCD) pipeline.

---

## Models (Open-Weight Decision Models)

### 1. Von

- **Link:** [wfzyx/von (Hugging Face)](https://huggingface.co/wfzyx/von)
- **Description:** An open-source, non-autoregressive System One decision model. It operates similarly to Laya, packing the premise and options into one sequence and using `[MASK]` markers to score options jointly.

### 2. OpenDecision AI

- **Link:** [open-decision-ai (PyPI)](https://pypi.org/project/open-decision-ai/)
- **Description:** Distributed as a Python package, this model provides local, open-weight probabilistic decisions derived from arbitrary state (JSON, logs, conversations).

### 3. Typical

- **Link:** [Typical by Oz Labs](https://typical.ozlabs.ai/)
- **Description:** An open decision model designed to natively handle state, questions, and option probabilities, including a built-in "none of the above" abstention feature.

### 4. Decision 1.0

- **Link:** [vllm-sr/Decision-1.0-Lex-0.6B (Hugging Face)](https://huggingface.co/vllm-sr/Decision-1.0-Lex-0.6B)
- **Paper:** [Decision 1.0: Towards Open Decision Foundation Models](https://vllm-sr.ai/decision-paper.pdf)
- **Description:** An open-weight family designed specifically for software control flow. It natively supports Choice, Score, and Noul (Yes/No) interfaces.

---

## Datasets for Pipeline Testing

### Choice / Classification Datasets

Requires mapping a context (state + question) and a list of options to a specific target choice.

- **[Nexusflow/NexusRaven-V2 Data](https://huggingface.co/Nexusflow/NexusRaven-V2-13B)**
  - **Primitive:** `choice`
  - **Use Case:** Function calling and tool selection. Excellent for mapping user queries (context) and available tools (options) to the correct tool (target).
- **[mteb/tweet_sentiment_extraction](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction)**
  - **Primitive:** `choice`
  - **Use Case:** A standard classification benchmark. Perfect for testing the pipeline's basic convergence by mapping text to fixed sentiment choices (e.g., positive, neutral, negative).

### Noul (Yes/No) Datasets

Requires mapping a context to a boolean target. Excellent for training guardrails and moderation models.

- **[Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)**
  - **Primitive:** `noul`
  - **Use Case:** The classic Helpful/Harmless dataset. You can adapt the "Harmless" prompts to train a yes/no guardrail model (_"Is this response safe?"_). RLCD trains the model to output calibrated 50% probabilities when prompts are highly ambiguous.

### Score (Ordinal) Datasets

Requires mapping a context to a numeric scale.

- **[lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat)**
  - **Primitive:** `score`
  - **Use Case:** Real-world user interactions with AI, annotated for toxicity. Ideal for testing the `score` primitive (e.g., scoring toxicity on a 1-5 ordinal scale) and validating uncertainty calibration.
