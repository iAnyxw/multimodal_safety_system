# scripts/eval_ch_text.py
"""快速评估中文模型，输出 per-class F1"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_PATH = "/root/autodl-tmp/multimodal_safety_system/data/ch_text/ch_data.csv"
MODEL_PATH = "/root/autodl-tmp/multimodal_safety_system/checkpoints/xlmr_ch"
LABELS = ["insult", "obscene", "identity_hate"]
MAX_LENGTH = 256
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载模型 ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, local_files_only=True
).to(DEVICE)
model.eval()

# === 读取数据 & 用和训练时同样的划分 ===
df = pd.read_csv(DATA_PATH)
texts = df["comment"].fillna("").tolist()
labels = df[LABELS].values.astype(float)

_, val_texts, _, val_labels = train_test_split(
    texts, labels,
    test_size=0.15,
    random_state=42,
    stratify=labels.sum(axis=1) > 0
)

# === 逐 batch 推理 ===
all_preds = []
with torch.no_grad():
    for i in range(0, len(val_texts), BATCH_SIZE):
        batch_texts = val_texts[i:i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts, truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        ).to(DEVICE)
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)

probs = np.concatenate(all_preds)
preds = (probs > 0.5).astype(int)

# === 结果 ===
print("\n" + "=" * 60)
print("📊 Per-Class F1 & Classification Report")
print("=" * 60)
print(classification_report(val_labels, preds, target_names=LABELS, zero_division=0))

# === Safe 指标（三标签全 0 = safe） ===
safe_true  = (val_labels.sum(axis=1) == 0).astype(int)
safe_pred  = (preds.sum(axis=1) == 0).astype(int)
n_safe_true = int(safe_true.sum())
n_safe_pred = int(safe_pred.sum())
n_safe_correct = int(((safe_true == 1) & (safe_pred == 1)).sum())
safe_precision = n_safe_correct / n_safe_pred if n_safe_pred > 0 else 0
safe_recall    = n_safe_correct / n_safe_true if n_safe_true > 0 else 0
safe_f1        = 2 * safe_precision * safe_recall / (safe_precision + safe_recall) if (safe_precision + safe_recall) > 0 else 0
print("-" * 65)
print(f"       safe       {safe_precision:.2f}      {safe_recall:.2f}      {safe_f1:.2f}       {n_safe_true}")
print(f"                                   (三标签全 0 视为 safe)")

print("Micro F1:", f1_score(val_labels, preds, average="micro"))
print("Macro F1:", f1_score(val_labels, preds, average="macro"))

# === 阈值调优扫描 ===
print("\n" + "=" * 60)
print("🎯 Threshold 扫描 (0.3 ~ 0.7)")
print("=" * 60)
best_t, best_f1 = 0.5, 0
for t in np.arange(0.3, 0.71, 0.05):
    preds_t = (probs > t).astype(int)
    f1_m = f1_score(val_labels, preds_t, average="micro")
    marker = " ⭐" if f1_m > best_f1 else ""
    if f1_m > best_f1:
        best_f1, best_t = f1_m, t
    print(f"  threshold={t:.2f}  micro_f1={f1_m:.4f}{marker}")

print(f"\n✅ 最佳阈值: {best_t:.2f} (micro_f1={best_f1:.4f})")
