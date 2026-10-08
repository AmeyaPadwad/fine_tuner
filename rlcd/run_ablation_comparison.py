import json
import os
import random
import time
import torch
from sklearn.metrics import precision_recall_fscore_support
from helper import load_dataset
from processing import auto_prepare_dataset_for_laya, prepare_dataset_items, collate_train_batch
from models import Laya
from training import RLTrainingConfig, RLTrainer, compute_rewards
from main import split_dataset_items

torch.set_num_threads(6)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===================================================================")
print("=== RLCD ABLATION STUDY: WITH VS. WITHOUT RLCD FINE-TUNING ===")
print("===================================================================")

# 1. Load exact same 150-example dataset slice
print("\n1. Loading 150-example dataset slice...")
raw = load_dataset("lockon/xlam-function-calling-60k")["train"].select(range(150))
prepared = auto_prepare_dataset_for_laya(raw)

# Base model setup
base = Laya.load_model("convaiinnovations/laya", device="cpu")
tok = base["tokenizer"]
items = prepare_dataset_items(prepared, tok)

# Split with seed 42
train_items, calib_items, test_items = split_dataset_items(
    items, test_ratio=0.20, calib_ratio=0.10, calib_max=20, seed=42
)
print(f"Data partitioned -> Train: {len(train_items)} | Calib: {len(calib_items)} | Test: {len(test_items)}")

batch = collate_train_batch(test_items, tok.pad_token_id)
input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]
marker_pos = batch["marker_pos"]
marker_mask = batch["marker_mask"]
qtype = batch["qtype"]
target = batch["target"]
gold_labels = target.argmax(dim=-1).tolist()


def evaluate_model_on_test(model, name="Model"):
    """Evaluate a model checkpoint on the 30 held-out test items."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        logits = logits.float()
        probs = torch.softmax(logits.masked_fill(~marker_mask, -1e4), -1)

        # Accuracy, Precision, Recall, F1
        preds = probs.argmax(dim=-1).tolist()
        correct = sum(1 for p, g in zip(preds, gold_labels) if p == g)
        acc = correct / len(gold_labels)

        p, r, f1, _ = precision_recall_fscore_support(
            gold_labels, preds, average="weighted", zero_division=0
        )

        # Brier Score
        rewards = compute_rewards(probs, target, qtype, marker_mask, reward_strategy="brier")
        mean_brier = rewards.mean().item()

        # ECE (10 bins)
        confidences = probs.max(dim=-1).values
        accuracies = torch.tensor([1.0 if p == g else 0.0 for p, g in zip(preds, gold_labels)])
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(gold_labels)
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
            bin_size = in_bin.sum().item()
            if bin_size > 0:
                bin_acc = accuracies[in_bin].mean().item()
                bin_conf = confidences[in_bin].mean().item()
                ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return {
        "name": name,
        "accuracy": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "mean_brier": mean_brier,
        "ece": ece,
    }


# =========================================================================
# EVALUATION 1: Untouched Pretrained Base Model (Zero Fine-Tuning)
# =========================================================================
print("\n=== Evaluating Model 1: Untouched Base Model (No Fine-Tuning) ===")
metrics_base = evaluate_model_on_test(base["model"], name="Untouched Base Model")
print(f"Accuracy: {metrics_base['accuracy']:.2%} | Precision: {metrics_base['precision']:.4f} | Recall: {metrics_base['recall']:.4f} | F1: {metrics_base['f1']:.4f} | Brier: {metrics_base['mean_brier']:.4f} | ECE: {metrics_base['ece']:.4f}")


# =========================================================================
# EVALUATION 2: Standard Supervised Fine-Tuning (SFT / CE-Only, NO RLCD)
# =========================================================================
print("\n=== Training Model 2: Standard SFT (Pure Cross-Entropy, NO RLCD) ===")
sft_model = Laya.load_model("convaiinnovations/laya", device="cpu")["model"]
sft_model.train()

enc_params = [p for n, p in sft_model.named_parameters() if "encoder." in n and p.requires_grad]
head_params = [p for n, p in sft_model.named_parameters() if "encoder." not in n and p.requires_grad]
optimizer = torch.optim.AdamW(
    [{"params": enc_params, "lr": 2.5e-5}, {"params": head_params, "lr": 1.0e-4}],
    weight_decay=0.01,
)

t0 = time.time()
micro_batch_size = 8
grad_accum_steps = 2
my_items = train_items.copy()

for epoch in range(2):
    random.seed(42 + epoch)
    random.shuffle(my_items)
    accum_step = 0
    optimizer.zero_grad(set_to_none=True)

    for b_idx in range(0, len(my_items), micro_batch_size):
        chunk = my_items[b_idx : b_idx + micro_batch_size]
        if not chunk:
            continue
        tr_batch = collate_train_batch(chunk, tok.pad_token_id)
        tr_in = tr_batch["input_ids"]
        tr_att = tr_batch["attention_mask"]
        tr_pos = tr_batch["marker_pos"]
        tr_mask = tr_batch["marker_mask"]
        tr_qtype = tr_batch["qtype"]
        tr_target = tr_batch["target"]

        logits, _ = sft_model(tr_in, tr_att, tr_pos, tr_mask, tr_qtype)
        logits = logits.float()

        # Pure Cross-Entropy loss WITHOUT RLCD (no perturbed logits, no Brier reward, no policy gradient)
        loss_ce = -(tr_target * torch.log_softmax(logits.masked_fill(~tr_mask, -1e4), -1)).sum(-1).mean()
        loss = loss_ce / grad_accum_steps
        loss.backward()

        accum_step += 1
        if accum_step % grad_accum_steps == 0 or (b_idx + micro_batch_size) >= len(my_items):
            torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

print(f"Standard SFT training completed in {time.time() - t0:.1f}s.")
metrics_sft = evaluate_model_on_test(sft_model, name="Standard SFT (No RLCD)")
print(f"Accuracy: {metrics_sft['accuracy']:.2%} | Precision: {metrics_sft['precision']:.4f} | Recall: {metrics_sft['recall']:.4f} | F1: {metrics_sft['f1']:.4f} | Brier: {metrics_sft['mean_brier']:.4f} | ECE: {metrics_sft['ece']:.4f}")


# =========================================================================
# EVALUATION 3: Full RLCD Strategy 1 (Brier Reward + Policy Gradient + CE)
# =========================================================================
print("\n=== Evaluating Model 3: Full RLCD Strategy 1 (Brier Proper Scoring Rule) ===")
from safetensors.torch import load_file

ft_model = Laya.load_model("convaiinnovations/laya", device="cpu")["model"]
ft_weights = load_file("./laya_finetuned_xlam_rl/model.safetensors")
ft_model.load_state_dict(ft_weights, strict=False)

metrics_rlcd = evaluate_model_on_test(ft_model, name="Full RLCD Strategy 1")
print(f"Accuracy: {metrics_rlcd['accuracy']:.2%} | Precision: {metrics_rlcd['precision']:.4f} | Recall: {metrics_rlcd['recall']:.4f} | F1: {metrics_rlcd['f1']:.4f} | Brier: {metrics_rlcd['mean_brier']:.4f} | ECE: {metrics_rlcd['ece']:.4f}")


# =========================================================================
# FINAL COMPARISON TABLE
# =========================================================================
print("\n====================================================================================================")
print("=== FINAL ABLATION COMPARISON: BASE MODEL vs. STANDARD SFT (NO RLCD) vs. FULL RLCD STRATEGY 1 ===")
print("====================================================================================================")
print(f"{'Metric':<25} | {'1. Untouched Base':<18} | {'2. Standard SFT (No RLCD)':<25} | {'3. Full RLCD Strategy 1':<22}")
print("-" * 100)
print(f"{'Accuracy':<25} | {metrics_base['accuracy']:<18.2%} | {metrics_sft['accuracy']:<25.2%} | {metrics_rlcd['accuracy']:<22.2%}")
print(f"{'Precision (Weighted)':<25} | {metrics_base['precision']:<18.4f} | {metrics_sft['precision']:<25.4f} | {metrics_rlcd['precision']:<22.4f}")
print(f"{'Recall (Weighted)':<25} | {metrics_base['recall']:<18.4f} | {metrics_sft['recall']:<25.4f} | {metrics_rlcd['recall']:<22.4f}")
print(f"{'F1-Score (Weighted)':<25} | {metrics_base['f1']:<18.4f} | {metrics_sft['f1']:<25.4f} | {metrics_rlcd['f1']:<22.4f}")
print(f"{'Mean Brier Score':<25} | {metrics_base['mean_brier']:<18.4f} | {metrics_sft['mean_brier']:<25.4f} | {metrics_rlcd['mean_brier']:<22.4f}")
print(f"{'Expected Calib Error (ECE)':<25} | {metrics_base['ece']:<18.4f} | {metrics_sft['ece']:<25.4f} | {metrics_rlcd['ece']:<22.4f}")
print("====================================================================================================")

# Save results
out_file = "./ablation_no_rlcd_results.json"
with open(out_file, "w") as f:
    json.dump({"base": metrics_base, "sft_no_rlcd": metrics_sft, "rlcd_strategy_1": metrics_rlcd}, f, indent=2)
print(f"\nAblation results saved to: {out_file}")
