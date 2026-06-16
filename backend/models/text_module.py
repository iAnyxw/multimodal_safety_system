# text_module.py
# 支持英文 (6 labels) 和中文 (3 labels) 双模型，自动语言检测分流

import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

from ..config import settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = settings.TEXT_MAX_LENGTH

# ======================
# 1️⃣ 语言检测工具
# ======================
def _contains_chinese(text: str) -> bool:
    """检测文本中是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]+', text))


# ======================
# 2️⃣ 加载英文模型 (6 labels)
# ======================
print("🚀 Loading EN Text Model (XLM-R, 6 labels)...")
_en_tokenizer = AutoTokenizer.from_pretrained(
    settings.TEXT_MODEL_PATH, local_files_only=True
)
_en_model = AutoModelForSequenceClassification.from_pretrained(
    settings.TEXT_MODEL_PATH, local_files_only=True
).to(DEVICE).eval()

_en_labels = list(settings.TEXT_LABELS)
_en_thresholds_path = os.path.join(settings.TEXT_MODEL_PATH, "thresholds.json")
if os.path.exists(_en_thresholds_path):
    _en_thresholds = np.array([
        json.load(open(_en_thresholds_path))[l] for l in _en_labels
    ])
else:
    _en_thresholds = np.array([settings.TEXT_THRESHOLDS[l] for l in _en_labels])
_en_unsafe_exceed = settings.TEXT_UNSAFE_EXCEED

# ======================
# 3️⃣ 加载中文模型 (3 labels)
# ======================
print("🚀 Loading ZH Text Model (XLM-R, 3 labels)...")
_zh_tokenizer = AutoTokenizer.from_pretrained(
    settings.TEXT_ZH_MODEL_PATH, local_files_only=True
)
_zh_model = AutoModelForSequenceClassification.from_pretrained(
    settings.TEXT_ZH_MODEL_PATH, local_files_only=True
).to(DEVICE).eval()

_zh_labels = list(settings.TEXT_ZH_LABELS)
_zh_thresholds_path = os.path.join(settings.TEXT_ZH_MODEL_PATH, "thresholds.json")
if os.path.exists(_zh_thresholds_path):
    _zh_thresholds = np.array([
        json.load(open(_zh_thresholds_path))[l] for l in _zh_labels
    ])
else:
    _zh_thresholds = np.array([settings.TEXT_ZH_THRESHOLDS[l] for l in _zh_labels])
_zh_unsafe_exceed = settings.TEXT_ZH_UNSAFE_EXCEED


# ======================
# 4️⃣ 核心预测（通用）
# ======================
def _predict(text: str, tokenizer, model, labels, thresholds, unsafe_exceed):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs > thresholds).astype(int)
    hit_labels = [labels[i] for i in range(len(labels)) if preds[i] == 1]

    exceed = np.maximum(probs - thresholds, 0)
    score = float(np.max(exceed))

    if len(hit_labels) > 0 and score > unsafe_exceed:
        decision = "UNSAFE"
    elif len(hit_labels) > 0:
        decision = "REVIEW"
    else:
        decision = "SAFE"

    return {
        "text": text,
        "score": score,
        "labels": hit_labels,
        "probs": dict(zip(labels, probs.tolist())),
        "decision": decision,
    }


# ======================
# 5️⃣ 对外接口
# ======================

def predict_text_en(text: str) -> dict:
    """英文模型预测 (6 labels: toxic, severe_toxic, obscene, threat, insult, identity_hate)"""
    return _predict(text, _en_tokenizer, _en_model,
                    _en_labels, _en_thresholds, _en_unsafe_exceed)


def predict_text_zh(text: str) -> dict:
    """中文模型预测 (3 labels: insult, obscene, identity_hate)"""
    return _predict(text, _zh_tokenizer, _zh_model,
                    _zh_labels, _zh_thresholds, _zh_unsafe_exceed)


def predict_text(text: str) -> dict:
    """
    自动语言检测分流
    - 包含中文 → 中文模型 (3 labels)
    - 纯英文/其他 → 英文模型 (6 labels)
    """
    if _contains_chinese(text):
        return predict_text_zh(text)
    else:
        return predict_text_en(text)


# ======================
# 6️⃣ 测试
# ======================
if __name__ == "__main__":
    tests = [
        ("英文", "I will kill you"),
        ("中文", "你这个傻逼"),
        ("中英混合", "你这个 idiot 真是够了"),
        ("正常中文", "今天天气真好"),
        ("正常英文", "Hello, how are you?"),
    ]

    for name, text in tests:
        result = predict_text(text)
        print(f"\n[{name}] {text}")
        print(f"  decision={result['decision']}  score={result['score']:.4f}  labels={result['labels']}")