import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================
# 1️⃣ 配置
# ======================
MODEL_PATH = "/root/autodl-tmp/multimodal_safety_system/checkpoints/xlmr"
VAL_PATH   = "../data/text/val_split.csv"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 2️⃣ 数据
# ======================
val_df = pd.read_csv(VAL_PATH)
texts = val_df["comment_text"].fillna("").tolist()
labels = val_df[LABELS].values

# ======================
# 3️⃣ 模型
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ======================
# 4️⃣ batch inference
# ======================
from torch.utils.data import DataLoader, TensorDataset

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
loader = DataLoader(dataset, batch_size=16)

all_logits = []

print("🚀 Running inference...")

with torch.no_grad():
    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        all_logits.append(outputs.logits.cpu())

logits = torch.cat(all_logits, dim=0).numpy()
probs = 1 / (1 + np.exp(-logits))

labels = labels

# ======================
# 5️⃣ GLOBAL search（核心）
# ======================

def evaluate_thresholds(thresholds, probs, labels):
    preds = (probs > thresholds).astype(int)
    return f1_score(labels, preds, average="micro")


def global_threshold_search(probs, labels):
    print("\n🔍 Global threshold search...")

    best_thresholds = np.array([0.5] * probs.shape[1])
    best_f1 = 0

    # 🔥 coarse grid search（先粗后细）
    grid = np.arange(0.1, 0.9, 0.1)

    for t1 in grid:
        for t2 in grid:
            for t3 in grid:
                for t4 in grid:
                    for t5 in grid:
                        for t6 in grid:

                            thresholds = np.array([t1, t2, t3, t4, t5, t6])

                            f1 = evaluate_thresholds(thresholds, probs, labels)

                            if f1 > best_f1:
                                best_f1 = f1
                                best_thresholds = thresholds

    return best_thresholds, best_f1


best_thresholds, best_f1 = global_threshold_search(probs, labels)

# ======================
# 6️⃣ 输出结果
# ======================
print("\n📊 BEST RESULT:")
print("Micro F1:", best_f1)

for l, t in zip(LABELS, best_thresholds):
    print(f"{l:>15}: {t:.2f}")

# ======================
# 7️⃣ 保存
# ======================
save_path = os.path.join(MODEL_PATH, "thresholds.json")

json.dump(dict(zip(LABELS, best_thresholds.tolist())), open(save_path, "w"), indent=2)

print(f"\n✅ Saved to {save_path}")