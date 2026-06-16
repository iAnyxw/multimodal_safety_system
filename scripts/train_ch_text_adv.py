# scripts/train_ch_text_adv.py
# 对抗增强版中文 XLM-R 训练
# 与 train_ch_text.py 结构相同，仅改数据源和输出路径

import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

# ======================
# 配置
# ======================
DATA_PATH = "/root/autodl-tmp/multimodal_safety_system/data/ch_text/ch_data_augmented.csv"
MODEL_SAVE_PATH = "/root/autodl-tmp/multimodal_safety_system/checkpoints/xlmr_ch_adv"
RESULTS_PATH = "/root/autodl-tmp/multimodal_safety_system/results/xlmr_ch_adv"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

LABELS = ["insult", "obscene", "identity_hate"]
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 256
TEST_SIZE = 0.15
SEED = 42

# ======================
# 读取数据 & 划分
# ======================
df = pd.read_csv(DATA_PATH)
texts = df["comment"].fillna("").tolist()
labels = df[LABELS].values.astype(float)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=SEED,
    stratify=labels.sum(axis=1) > 0
)

print(f"训练集: {len(train_texts)} 条")
print(f"验证集: {len(val_texts)} 条")
for i, name in enumerate(LABELS):
    print(f"  {name}: train={train_labels[:, i].sum():.0f}  val={val_labels[:, i].sum():.0f}")

# ======================
# Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

# ======================
# Dataset
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
# Training Arguments
# ======================
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=MODEL_SAVE_PATH + "/logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),
    dataloader_drop_last=False,
    report_to="none",
)

# ======================
# Metrics
# ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    results = {}
    for i, name in enumerate(LABELS):
        results[f"{name}_f1"] = f1_score(labels[:, i], preds[:, i], zero_division=0)

    results["micro_f1"] = f1_score(labels, preds, average="micro", zero_division=0)
    results["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)
    return results

# ======================
# Trainer
# ======================
trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        problem_type="multi_label_classification",
        local_files_only=True
    ),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# ======================
# 训练
# ======================
print("\n🚀 开始对抗增强训练...")
trainer.train()

# 保存
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"\n✅ 模型保存至: {MODEL_SAVE_PATH}")
