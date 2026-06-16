# scripts/train_text.py

import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ======================
# 1️⃣ 配置
# ======================
DATA_PATH = "/root/autodl-tmp/multimodal_safety_system/data/text/train.csv"
MODEL_SAVE_PATH = "/root/autodl-tmp/multimodal_safety_system/checkpoints/bert"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# ======================
# ======================
# 2️⃣ 读取数据
# ======================
train_df = pd.read_csv("../data/text/train_split.csv")
val_df = pd.read_csv("../data/text/val_split.csv")

train_texts = train_df["comment_text"].fillna("").tolist()
train_labels = train_df[LABELS].values
val_texts = val_df["comment_text"].fillna("").tolist()
val_labels = val_df[LABELS].values

# ======================
# 3️⃣ Tokenizer
# ======================
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
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
val_dataset = ToxicDataset(val_encodings, val_labels)

# ======================
# 5️⃣ 模型
# ======================
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
)

# ======================
# 6️⃣ 训练参数
# ======================
training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/multimodal_safety_system/results/bert",
    learning_rate=3e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="/root/autodl-tmp/multimodal_safety_system/logs/bertlogs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# ======================
# 7️⃣ 评估指标（F1）
# ======================
from sklearn.metrics import f1_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    preds = (probs > 0.5).astype(int)

    f1 = f1_score(labels, preds, average="micro")
    return {"f1": f1}

# ======================
# 8️⃣ Trainer
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ======================
# 9️⃣ 开始训练
# ======================
trainer.train()

# ======================
# 🔟 保存模型
# ======================
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("✅ 模型训练完成并保存到:", MODEL_SAVE_PATH)
