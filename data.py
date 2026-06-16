# split_dataset_80_10_10.py

import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ======================
# 1️⃣ 配置
# ======================
DATA_PATH = "/root/autodl-tmp/multimodal_safety_system/data/text/train.csv"
SAVE_DIR = "/root/autodl-tmp/multimodal_safety_system/data/text/"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

RANDOM_STATE = 42

# ======================
# 2️⃣ 读取数据
# ======================
df = pd.read_csv(DATA_PATH)

texts = df["comment_text"].fillna("").values
labels = df[LABELS].values

print(f"📊 原始数据量: {len(df)}")

# ======================
# 3️⃣ 第一次划分：10% test
# ======================
msss_1 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.1,
    random_state=RANDOM_STATE
)

for train_val_idx, test_idx in msss_1.split(texts, labels):
    X_train_val = texts[train_val_idx]
    y_train_val = labels[train_val_idx]

    X_test = texts[test_idx]
    y_test = labels[test_idx]

print(f"✅ train+val: {len(X_train_val)}")
print(f"✅ test: {len(X_test)}")

# ======================
# 4️⃣ 第二次划分：从90%里再切10% → 实际就是总数据10%（val）
# ======================
msss_2 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.111111,  # 10% / 90% ≈ 0.111111
    random_state=RANDOM_STATE
)

for train_idx, val_idx in msss_2.split(X_train_val, y_train_val):
    X_train = X_train_val[train_idx]
    y_train = y_train_val[train_idx]

    X_val = X_train_val[val_idx]
    y_val = y_train_val[val_idx]

print(f"✅ train: {len(X_train)}")
print(f"✅ val: {len(X_val)}")

# ======================
# 5️⃣ 保存函数
# ======================
def save_split(texts, labels, path):
    df_out = pd.DataFrame({
        "comment_text": texts
    })
    for i, label in enumerate(LABELS):
        df_out[label] = labels[:, i]

    df_out.to_csv(path, index=False)
    print(f"💾 已保存: {path}")

# ======================
# 6️⃣ 保存
# ======================
save_split(X_train, y_train, SAVE_DIR + "train_split.csv")
save_split(X_val, y_val, SAVE_DIR + "val_split.csv")
save_split(X_test, y_test, SAVE_DIR + "test_split.csv")

# ======================
# 7️⃣ 分布检查
# ======================
def show_distribution(name, y):
    print(f"\n📊 {name} 分布：")
    counts = y.sum(axis=0)
    for i, label in enumerate(LABELS):
        print(f"{label:15s}: {int(counts[i])}")

show_distribution("Train", y_train)
show_distribution("Val", y_val)
show_distribution("Test", y_test)

print("\n🎯 80/10/10 划分完成！")