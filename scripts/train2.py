# scripts/train_text_xlmr.py

import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# ======================
# 1️⃣ 配置
# ======================
DATA_PATH = "/root/autodl-tmp/multimodal_safety_system/data/text/train.csv"
MODEL_SAVE_PATH = "/root/autodl-tmp/multimodal_safety_system/checkpoints/xlmr"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

MODEL_NAME = "xlm-roberta-base"

# ======================
# 2️⃣ 读取数据
# ======================
train_df = pd.read_csv("../data/text/train_split.csv")
val_df   = pd.read_csv("../data/text/val_split.csv")

train_texts = train_df["comment_text"].fillna("").tolist()
val_texts   = val_df["comment_text"].fillna("").tolist()

train_labels = train_df[LABELS].values
val_labels   = val_df[LABELS].values

# ======================
# 3️⃣ Tokenizer（重点替换）
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=256
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=256
)

# ======================
# 4️⃣ Dataset
# ======================
class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToxicDataset(train_encodings, train_labels)
val_dataset   = ToxicDataset(val_encodings, val_labels)

# ======================
# 5️⃣ 模型（XLM-R）
# ======================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
)

# ⚠️ XLM-R 没有 token_type_ids（不用管 Trainer 会自动处理）

# ======================
# 6️⃣ F1 metric
# ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs > 0.5).astype(int)

    f1 = f1_score(labels, preds, average="micro")
    return {"f1": f1}

# ======================
# 7️⃣ Training Args（冲分版）
# ======================
training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/multimodal_safety_system/results/xlmr",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,

    evaluation_strategy="epoch",
    save_strategy="epoch",

    logging_steps=100,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    fp16=True,   # 🚀 直接加速（有GPU就开）
)

# ======================
# 8️⃣ Trainer
# ======================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ======================
# 9️⃣ train
# ======================
trainer.train()

# ======================
# 🔟 save
# ======================
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("✅ XLM-R训练完成:", MODEL_SAVE_PATH)