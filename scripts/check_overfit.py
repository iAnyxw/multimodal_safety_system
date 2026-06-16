# scripts/check_overfit.py
"""验证是否过拟合：交叉验证 + Train/Val Gap"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

DATA_PATH = "/root/autodl-tmp/multimodal_safety_system/data/ch_text/ch_data.csv"
MODEL_NAME = "xlm-roberta-base"
LABELS = ["insult", "obscene", "identity_hate"]
MAX_LENGTH = 256
N_FOLDS = 3

df = pd.read_csv(DATA_PATH)
texts = df["comment"].fillna("").tolist()
labels = df[LABELS].values.astype(float)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# 只取有毒样本 + 等量无毒样本，避免全零稀释
toxic_idx = np.where(labels.sum(axis=1) > 0)[0]
clean_idx = np.where(labels.sum(axis=1) == 0)[0]
clean_sample = np.random.RandomState(42).choice(clean_idx, size=min(len(clean_idx), len(toxic_idx) * 2), replace=False)
all_idx = np.concatenate([toxic_idx, clean_sample])
np.random.RandomState(42).shuffle(all_idx)

print(f"子集: {len(all_idx)} 条 (有毒={len(toxic_idx)}, 无毒={len(clean_sample)})")
print(f"标签分布: insult={labels[all_idx][:,0].sum():.0f}, obscene={labels[all_idx][:,1].sum():.0f}, identity_hate={labels[all_idx][:,2].sum():.0f}\n")

class TinyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

scores = []
for fold, (train_i, val_i) in enumerate(kf.split(all_idx)):
    tr_idx, vl_idx = all_idx[train_i], all_idx[val_i]
    tr_texts = [texts[i] for i in tr_idx]
    tr_labels = labels[tr_idx]
    vl_texts = [texts[i] for i in vl_idx]
    vl_labels = labels[vl_idx]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tr_enc = tokenizer(tr_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    vl_enc = tokenizer(vl_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        problem_type="multi_label_classification",
        local_files_only=True
    )

    args = TrainingArguments(
        output_dir="/tmp/cv_check", learning_rate=2e-5,
        per_device_train_batch_size=16, num_train_epochs=3,
        eval_strategy="epoch", logging_strategy="no",
        save_strategy="no", fp16=True, report_to="none",
    )

    trainer = Trainer(model=model, args=args,
        train_dataset=TinyDataset(tr_enc, tr_labels),
        eval_dataset=TinyDataset(vl_enc, vl_labels),
        compute_metrics=compute_metrics)

    trainer.train()
    result = trainer.evaluate()
    scores.append({
        "fold": fold,
        "val_f1_micro": result["eval_f1_micro"],
        "val_f1_macro": result["eval_f1_macro"],
    })
    print(f"  Fold {fold+1}: micro={result['eval_f1_micro']:.4f}  macro={result['eval_f1_macro']:.4f}")

micros = [s["val_f1_micro"] for s in scores]
macros = [s["val_f1_macro"] for s in scores]

print(f"\n{'='*50}")
print(f"📊 {N_FOLDS}-Fold CV 结果:")
print(f"  micro F1: {np.mean(micros):.4f} ± {np.std(micros):.4f}")
print(f"  macro F1: {np.mean(macros):.4f} ± {np.std(macros):.4f}")
print(f"\n🔍 判断:")
if np.std(micros) < 0.03:
    print(f"  ✅ 各 fold 分数稳定 (std={np.std(micros):.4f})，不是过拟合")
else:
    print(f"  ⚠️ 各 fold 波动大 (std={np.std(micros):.4f})，可能过拟合")
print(f"  分数水平: {'真实性能强' if np.mean(micros) > 0.9 else '还需提升'}")
