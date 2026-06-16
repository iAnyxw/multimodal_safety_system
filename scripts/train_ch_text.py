# scripts/train_ch_text.py
# 中文文本安全分类 — 基于 XLM-RoBERTa

import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
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
DATA_PATH = "/root/autodl-tmp/multimodal_safety_system/data/ch_text/ch_data.csv"
MODEL_SAVE_PATH = "/root/autodl-tmp/multimodal_safety_system/checkpoints/xlmr_ch"
RESULTS_PATH = "/root/autodl-tmp/multimodal_safety_system/results/xlmr_ch"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# 中文数据集只有 3 个标签
LABELS = [
    "insult",         # 辱骂
    "obscene",        # 淫秽
    "identity_hate"   # 身份仇恨
]

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 256
TEST_SIZE = 0.15
SEED = 42

# ======================
# 2️⃣ 读取数据 & 划分
# ======================
df = pd.read_csv(DATA_PATH)

# 文本列
texts = df["comment"].fillna("").tolist()
labels = df[LABELS].values.astype(float)

# 划分训练 / 验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=labels.sum(axis=1) > 0  # 按是否有毒分层
)

print(f"训练集: {len(train_texts)} 条")
print(f"验证集: {len(val_texts)} 条")
for i, name in enumerate(LABELS):
    print(f"  {name}: train={train_labels[:, i].sum():.0f}  val={val_labels[:, i].sum():.0f}")

# ======================
# 3️⃣ Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
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
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
    local_files_only=True
)

# ======================
# 6️⃣ 评估指标
# ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs > 0.5).astype(int)

    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}

# ======================
# 7️⃣ 训练参数
# ======================
training_args = TrainingArguments(
    output_dir=RESULTS_PATH,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    warmup_steps=100,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=20,

    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    save_total_limit=1,

    fp16=True,
    dataloader_drop_last=False,
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
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ======================
# 9️⃣ 训练
# ======================
trainer.train()

# ======================
# 🔟 保存
# ======================
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("✅ 中文 XLM-R 训练完成:", MODEL_SAVE_PATH)
