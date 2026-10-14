# LLM Fine-Tuner: Training Runtime Worker (SFT)

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Job Specification Contract](#job-specification-contract)
3. [Proposed Project Directory Structure](#proposed-project-directory-structure)
4. [Technical Implementation Plan (Core Components)](#technical-implementation-plan-core-components)
5. [1-Month (20-Day) Day-by-Day Execution Plan for 2 Engineers](#1-month-20-day-day-by-day-execution-plan-for-2-engineers)
   - [Team Roles & Ownership](#team-roles--ownership)
   - [Week 1: Core Plumbing & First Smoke Test (Days 1–5)](#week-1-core-plumbing--first-smoke-test-days-15)
   - [Week 2: Full SFT Pipeline & Telemetry (Days 6–10)](#week-2-full-sft-pipeline--telemetry-days-610)
   - [Week 3: Multi-Model Testing & Evaluation Matrix (Days 11–15)](#week-3-multi-model-testing--evaluation-matrix-days-1115)
   - [Week 4: Final Polishing, Packaging & Delivery (Days 16–20)](#week-4-final-polishing-packaging--delivery-days-1620)
6. [Daily Operating Cadence & Best Practices](#daily-operating-cadence--best-practices)
7. [Recommended Technology Stack](#recommended-technology-stack)

---

## System Architecture

The Training Runtime Worker is a self-contained execution unit. It can run locally via CLI, in a cloud VM/container (RunPod, Modal, Lambda, AWS), or be invoked as a task runner by an orchestration queue.

```
                      ┌────────────────────────────┐
                      │    Job Specification JSON   │
                      └─────────────┬──────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        TRAINING RUNTIME WORKER                         │
│                                                                        │
│  [ 1. Config & Pre-flight Guard ]                                      │
│    ├── Pydantic validation (model, dataset, hyperparameters)           │
│    └── GPU hardware & VRAM feasibility check (prevent CUDA OOMs)       │
│                                                                        │
│  [ 2. Data Ingestion & Formatting Engine ]                             │
│    ├── Format parsing: OpenAI/ShareGPT messages, Alpaca, or custom     │
│    ├── Tokenization & Model-specific Chat Template application         │
│    ├── Response Masking (loss calculated ONLY on assistant response)   │
│    └── Train / Eval split creation (e.g., 90/10)                       │
│                                                                        │
│  [ 3. Model & PEFT Setup ]                                             │
│    ├── Base model loader (4-bit NF4 QLoRA / 8-bit / 16-bit LoRA)       │
│    ├── PEFT configuration (rank, alpha, target modules)                │
│    └── Gradient checkpointing & memory optimizations                   │
│                                                                        │
│  [ 4. Core SFT Training Engine ]                                       │
│    ├── Hugging Face TRL (SFTTrainer) + Accelerate                      │
│    └── Structured Event Callback (streams JSON lines / live metrics)   │
│                                                                        │
│  [ 5. Post-Training, Evaluation & Export ]                             │
│    ├── Validation loss & perplexity calculation                        │
│    ├── LoRA adapter export (adapter_model.safetensors)                 │
│    ├── Optional: Merge adapter with base model into 16-bit safetensors │
│    └── Optional: Quantize to GGUF (for Ollama/llama.cpp)               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
                      ┌────────────────────────────┐
                      │     Output Artifacts       │
                      │  ├── adapter/              │
                      │  ├── merged_model/         │
                      │  ├── metrics.json          │
                      │  └── events.jsonl          │
                      └────────────────────────────┘
```

---

## Job Specification Contract

The worker is driven by a single, declarative JSON/YAML specification.

### Example `job_config.json`:

```json
{
  "job_id": "job_sft_20260928_01",
  "model": {
    "base_model_id": "Qwen/Qwen2.5-0.5B",
    "auth_token": null,
    "trust_remote_code": true
  },
  "dataset": {
    "source_path": "data/training_data.jsonl",
    "format": "messages",
    "max_seq_length": 2048,
    "eval_split_ratio": 0.1
  },
  "training": {
    "method": "qlora_4bit",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "optimizer": "paged_adamw_8bit",
    "gradient_checkpointing": true
  },
  "output": {
    "output_dir": "outputs/run_01",
    "save_adapter": true,
    "merge_weights": true,
    "export_gguf": false,
    "logging_steps": 5
  }
}
```

---

## Proposed Project Directory Structure

```
fine-tuner/
├── worker/
│   ├── __init__.py
│   ├── schema.py          # Pydantic models for JobSpec & Event schemas
│   ├── preflight.py       # Hardware, VRAM & environment checks
│   ├── dataset.py         # Multi-format parsing, chat template, masking
│   ├── model.py           # Model loader (4-bit QLoRA, 16-bit LoRA, PEFT)
│   ├── trainer.py         # SFTTrainer loop & training orchestration
│   ├── callbacks.py       # Structured JSON progress & metrics callback
│   ├── exporter.py        # Adapter saving, weight merging, GGUF export
│   └── main.py            # CLI entry point (python -m worker.main)
├── tests/
│   ├── test_schema.py     # Schema validation tests
│   ├── test_dataset.py    # Data formatting & masking tests
│   ├── test_preflight.py  # VRAM estimation unit tests
│   └── test_integration.py# Fast end-to-end 5-step mock training run
├── examples/
│   ├── sample_conversations.jsonl
│   ├── sample_alpaca.jsonl
│   └── job_config.json
├── requirements.txt       # Dependencies
├── Dockerfile             # Production container definition
└── README.md
```

---

## Technical Implementation Plan (Core Components)

### 1. Job Specification & Validation Layer

- Implements strict validation using Pydantic.
- Validates hyperparameter ranges, file existence, and model IDs.
- Provides informative error messages on misconfigurations before initializing any heavy dependencies.

### 2. Data Ingestion & Formatting Engine

- **Multi-format Support:**
  - Standard conversational messages (`[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`).
  - Instruction format / Alpaca (`instruction`, `input`, `output`).
  - Custom column mapping for CSV/JSON files.
- **Tokenizer & Chat Template Integration:**
  - Uses the base model's native Jinja chat template (`tokenizer.apply_chat_template`).
- **Response Masking (Critical for SFT):**
  - Implements completion-only loss masking so that the gradient updates apply strictly to the model's responses, not to the user prompts or system instructions.
- **Data Quality Profiling:**
  - Generates token-length distributions.
  - Warns on truncated rows and filters corrupt or empty rows.

### 3. Hardware Pre-flight & Memory Guard

- Checks CUDA availability, GPU device count, compute capability, and total VRAM.
- Estimates peak memory usage before starting:
  $$\text{Memory} \approx \text{Model Weights} + \text{LoRA Weights} + \text{Activation Memory} + \text{Optimizer State}$$
- Automatically warns or recommends parameter adjustments (e.g., lowering batch size, enabling gradient accumulation, or switching to 4-bit QLoRA) to prevent runtime OOM crashes.

### 4. Core SFT Trainer & Telemetry Streamer

- Integrates Hugging Face `TRL` (`SFTTrainer`) with `PEFT`.
- Supports 4-bit NF4 QLoRA (via `bitsandbytes`) and standard 16-bit LoRA.
- Automatically selects linear projection targets (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Emits real-time JSONL telemetry (`events.jsonl`) for seamless dashboard integration:
  ```json
  {
    "event": "step_completed",
    "step": 50,
    "total_steps": 500,
    "train_loss": 1.42,
    "lr": 0.0002,
    "vram_used_gb": 6.8,
    "eta_seconds": 320
  }
  ```

### 5. Evaluation, Checkpointing & Export Engine

- Calculates final evaluation loss and perplexity on the holdout validation split.
- Exports standard Hugging Face PEFT LoRA adapter files.
- Supports fusing the adapter into the base model (`merge_and_unload`) to output standard 16-bit safetensors.
- Optional GGUF export for local execution with Ollama or llama.cpp.

---

## 1-Month (20-Day) Day-by-Day Execution Plan for 2 Engineers

### Team Roles & Ownership

| Responsibility       | Engineer 1 (ML & Runtime Lead)                                                          | Engineer 2 (Data & Systems Lead)                                                               |
| -------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Primary Domain**   | Model loading, PEFT/QLoRA, Hardware guard, Trainer loop, Checkpointing, Weight exporter | Job schema, Data parsing, Chat templates, Response masking, Telemetry/Callbacks, CLI/Packaging |
| **Integration Goal** | Ensure memory efficiency, numerical stability, and correct adapter generation           | Ensure clean data flow, reliable job configuration, telemetry streaming, and user DX           |

---

### Week 1: Core Plumbing & First Smoke Test (Days 1–5)

**Milestone:** End-to-end smoke test on a toy model (data in $\rightarrow$ model trains 10 steps $\rightarrow$ adapter saved).

- **Day 1: Project Setup & Schema Agreement (Joint)**
  - Joint: Agree on repo structure, Git branch workflow, Python dependencies, and virtual environment.
  - Joint: Write `worker/schema.py` defining the complete `JobSpec` Pydantic model.
- **Day 2: Data Ingestion vs. Model Loader**
  - **Eng 2:** Build `worker/dataset.py` to read JSONL/CSV files with conversational role/content structures.
  - **Eng 1:** Build `worker/model.py` to load base models and tokenizers in FP16/BF16.
- **Day 3: Tokenization & Masking vs. LoRA Integration**
  - **Eng 2:** Implement chat template application and completion-only masking for assistant tokens.
  - **Eng 1:** Implement PEFT `LoraConfig` and model adapter wrapping with automatic target module detection.
- **Day 4: Integration into SFTTrainer**
  - **Joint:** Assemble `worker/trainer.py` using Hugging Face `TRL` `SFTTrainer`. Connect dataset preprocessing with the model training loop.
- **Day 5: Week 1 Integration Smoke Test**
  - Run a 10-step smoke test using a micro-model (`SmolLM-135M` or `Qwen2.5-0.5B`) and sample dataset.
  - Verify that the training step executes and outputs adapter files (`adapter_model.safetensors`).

---

### Week 2: Full SFT Pipeline & Telemetry (Days 6–10)

**Milestone:** Fully functional, standalone SFT worker with QLoRA, telemetry streaming, memory safety, and export.

- **Day 6: QLoRA (4-bit NF4) & Memory Optimization**
  - **Eng 1:** Implement `BitsAndBytesConfig` for 4-bit NF4 quantization, double quantization, and gradient checkpointing.
  - **Eng 2:** Add support for Alpaca format (`instruction`, `input`, `output`) and arbitrary column mapping.
- **Day 7: Pre-flight Memory Guard & Data Profiling**
  - **Eng 1:** Build `worker/preflight.py` to check GPU VRAM and estimate memory usage to prevent OOMs.
  - **Eng 2:** Add data quality validation: token-length histograms, empty row filtering, and truncation warnings.
- **Day 8: Structured Telemetry & Metrics Callback**
  - **Eng 2:** Build `worker/callbacks.py` emitting structured JSON lines (`events.jsonl`) with step, loss, lr, ETA, and VRAM.
  - **Eng 1:** Integrate the validation loop (`eval_dataset`), computing eval loss and perplexity.
- **Day 9: Checkpointing & Weight Exporter**
  - **Eng 1:** Build `worker/exporter.py` for saving adapters and merging adapter weights into full 16-bit safetensors.
  - **Eng 2:** Implement the CLI entrypoint (`python -m worker.main --config job_config.json`).
- **Day 10: Week 2 Milestone Review**
  - Execute a full 1-epoch fine-tuning job via CLI using a real dataset.
  - Verify pre-flight checks, QLoRA execution, real-time JSONL logging, metrics generation, and model export.

---

### Week 3: Multi-Model Testing & Evaluation Matrix (Days 11–15)

**Milestone:** Stress-testing across models, formats, and hardware; edge-case hardening.

- **Day 11: Multi-Model Architecture Matrix**
  - Test fine-tuning across major open-weight architectures:
    1. **Qwen 2.5** (0.5B, 3B, 7B)
    2. **Llama 3.1 / 3.2** (1B, 3B, 8B)
    3. **Mistral / Gemma 2**
  - Verify chat template compatibility, special token handling, and LoRA projection targeting across all families.
- **Day 12: Diverse Dataset & Scale Testing**
  - Test diverse datasets:
    - Multi-turn conversations (5+ turns).
    - Long-context samples (testing truncation at `max_seq_length`).
    - Datasets with 1,000 to 10,000 samples to verify training stability and memory leak absence over time.
- **Day 13: Edge Case & Fault Injection Testing**
  - Intentionally trigger potential failure points:
    - Corrupt or empty JSONL rows (worker must skip with warning, not crash).
    - Oversized batch size (pre-flight must warn or handle gracefully).
    - Missing Hugging Face tokens for gated models.
- **Day 14: Quality & Learning Verification**
  - Verify that the models are actually learning:
    - Inspect training loss curves for monotonic convergence.
    - Measure validation perplexity improvements.
    - Run qualitative inference comparisons: compare base model vs. fine-tuned model outputs on test prompts.
- **Day 15: Week 3 Bug Triage & Hardening**
  - Fix any regressions, memory bottlenecks, or configuration bugs discovered during matrix testing.
  - Finalize the list of polish items for Week 4.

---

### Week 4: Final Polishing, Packaging & Delivery (Days 16–20)

**Milestone:** Production-ready containerized release with automated tests, GGUF export, and documentation.

- **Day 16: Automated Test Suite (Pytest)**
  - **Eng 2:** Write comprehensive unit tests for schema validation, dataset parsing, and event logging.
  - **Eng 1:** Write automated integration tests running a 5-step mock training loop with assertions on generated artifacts.
- **Day 17: Containerization (Docker)**
  - **Eng 1:** Create an optimized `Dockerfile` with CUDA, PyTorch, bitsandbytes, and FlashAttention-2.
  - Test container build and GPU execution locally or on a cloud GPU instance (e.g. RunPod / Modal).
- **Day 18: High-Value Value-Add Features**
  - **Eng 1: GGUF Quantization Export:** Add an export option that converts trained models to GGUF format (`Q4_K_M` / `Q8_0`) for instant use in Ollama / llama.cpp.
  - **Eng 2: Hugging Face Hub Direct Push:** Add optional parameters (`hub_model_id`, `hub_token`) to upload the trained adapter directly to the Hugging Face Hub upon completion.
- **Day 19: Documentation & Developer Experience**
  - **Eng 2:** Write complete documentation, including configuration reference, troubleshooting guide, and example datasets.
  - **Eng 1:** Calibrate smart defaults for hyperparameters (auto-calculating learning rate, batch size, and warmup based on dataset size).
- **Day 20: Final Verification & Release Tagging**
  - Run the full end-to-end verification demo:
    - Start container $\rightarrow$ Submit job $\rightarrow$ Train $\rightarrow$ Export to GGUF $\rightarrow$ Test in Ollama.
  - Tag release `v1.0.0` and freeze worker code for orchestration integration.

---

## Daily Operating Cadence & Best Practices

1. **Daily 10-Minute Standup:**
   - What did I complete yesterday?
   - What am I shipping today?
   - Any blockers or schema/contract changes?
2. **Contract-First Discipline:**
   - Both engineers build strictly against `worker/schema.py`. Any schema change requires mutual agreement.
3. **Compute Allocation Strategy:**
   - **Weeks 1 & 2 (Local Dev):** Use lightweight models (`Qwen2.5-0.5B`, `SmolLM-135M`) that run quickly on consumer GPUs or CPU.
   - **Week 3 (Testing):** Leverage on-demand cloud GPUs (e.g., RunPod, Lambda, or Modal A10G/A100 instances at ~$1-2/hr) for full-scale 7B/8B QLoRA testing.

---

## Recommended Technology Stack

| Component                   | Library / Tool                      | Purpose                                                               |
| --------------------------- | ----------------------------------- | --------------------------------------------------------------------- |
| **Language & Schema**       | Python 3.10+, Pydantic v2           | Type-safe configuration, schema validation, and event models          |
| **Deep Learning Core**      | PyTorch, Hugging Face Transformers  | Base model loading, tokenization, and forward/backward passes         |
| **SFT Training Loop**       | Hugging Face TRL (`SFTTrainer`)     | Built-in chat template formatting, response masking, and SFT routines |
| **Efficient Fine-Tuning**   | PEFT + bitsandbytes                 | LoRA adapter injection and 4-bit NF4 quantization (QLoRA)             |
| **Hardware & Acceleration** | Accelerate, FlashAttention-2 / SDPA | Hardware abstraction, multi-GPU scaling, and memory efficiency        |
| **Testing & Packaging**     | Pytest, Docker (CUDA base)          | Automated verification and reproducible containerized deployments     |
