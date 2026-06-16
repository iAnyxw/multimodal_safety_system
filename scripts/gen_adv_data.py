# scripts/gen_adv_data.py
# 对抗样本数据增强 —— 针对性生成 safe 对抗变体
# 核心目标：让模型学会"变形后的 safe 文本仍然是 safe"

import random
import pandas as pd

# ======================
# 配置
# ======================
DATA_PATH = "/root/autodl-tmp/multimodal_safety_system/data/ch_text/ch_data.csv"
OUT_PATH = "/root/autodl-tmp/multimodal_safety_system/data/ch_text/ch_data_augmented.csv"

LABELS = ["insult", "obscene", "identity_hate"]
SEED = 42
random.seed(SEED)

# ======================
# 对抗变换函数（保留空格和 emoji，不 normalize）
# ======================

def space_separate(text: str) -> str:
    """中文字间插空格：你好吗 → 你 好 吗"""
    return " ".join(list(text))

def partial_space(text: str, prob: float = 0.5) -> str:
    """随机部分插入空格"""
    return "".join(c + (" " if random.random() < prob else "") for c in text)

def insert_emoji(text: str) -> str:
    """随机位置插入 emoji"""
    all_emojis = ["😊", "😄", "😍", "💪", "🎉", "✨", "👍", "❤️", "🌸",
                  "🔪", "💀", "😡", "👊", "🔫"]
    emoji = random.choice(all_emojis)
    idx = random.randint(0, len(text)) if text else 0
    return text[:idx] + emoji + text[idx:]

def emoji_replace(text: str) -> str:
    """用 emoji 替换对应字符：杀 → 🔪"""
    emoji_map = {"杀": "🔪", "死": "💀", "打": "👊", "爱": "😍",
                 "笑": "😄", "好": "👍"}
    replaceable = [(i, c) for i, c in enumerate(text) if c in emoji_map]
    if not replaceable:
        return text
    i, c = random.choice(replaceable)
    return text[:i] + emoji_map[c] + text[i+1:]

def pinyin_mix(text: str) -> str:
    """随机将中文字替换为拼音"""
    pinyin_map = {
        "你": "ni", "好": "hao", "吗": "ma", "我": "wo", "他": "ta",
        "是": "shi", "不": "bu", "了": "le", "的": "de", "在": "zai",
        "有": "you", "人": "ren", "大": "da", "天": "tian", "气": "qi",
        "去": "qu", "来": "lai", "说": "shuo", "看": "kan", "想": "xiang",
        "爱": "ai", "恨": "hen",  "杀": "sha", "死": "si", "滚": "gun",
        "操": "cao", "妈": "ma",  "逼": "bi", "傻": "sha", "贱": "jian",
        "谢": "xie", "开": "kai", "心": "xin", "今": "jin", "明": "ming",
    }
    chars = list(text)
    for i in range(len(chars)):
        if chars[i] in pinyin_map and random.random() < 0.5:
            chars[i] = pinyin_map[chars[i]]
    return " ".join(chars)

def homophone_replace(text: str) -> str:
    """谐音替换"""
    homophone_map = {
        "傻": "煞", "逼": "笔", "操": "草", "你": "尼",
        "吗": "嘛", "滚": "棍", "死": "屎", "贱": "剑",
        "杀": "沙", "我": "窝", "了": "啦", "的": "滴",
    }
    chars = list(text)
    for i in range(len(chars)):
        if chars[i] in homophone_map and random.random() < 0.4:
            chars[i] = homophone_map[chars[i]]
    return "".join(chars)


# ======================
# 主生成逻辑
# ======================

def main():
    df = pd.read_csv(DATA_PATH)
    print(f"✅ 原始数据: {len(df)} 条")

    # === A：从原始 safe 样本抽样做对抗增强 ===
    safe_df = df[(df[LABELS].sum(axis=1) == 0)].copy()
    safe_sample = safe_df.sample(n=min(200, len(safe_df)), random_state=SEED)
    safe_texts = safe_sample["comment"].fillna("").tolist()

    aug_rows = []
    safe_transforms = [space_separate, partial_space, insert_emoji]
    # 注意：不对 safe 文本做 pinyin_mix / homophone_replace
    # 否则模型会学到"拼音=安全"，导致漏放 unsafe 拼音

    for text in safe_texts:
        if len(text) < 2:
            continue
        for _ in range(random.randint(2, 3)):
            func = random.choice(safe_transforms)
            aug_text = func(text)
            if aug_text and aug_text != text and len(aug_text) >= 2:
                aug_rows.append({
                    "comment": aug_text,
                    "insult": 0, "obscene": 0, "identity_hate": 0
                })

    # === B：针对实验 D 弱点手工补充 safe 对抗样本 ===
    handcrafted_safe = [
        # 拼音 safe（原实验 D 中 nihaoma / xiexie ni 被误拦）
        ("zaijian", 0, 0, 0), ("dui bu qi", 0, 0, 0),
        ("mei guan xi", 0, 0, 0), ("zhen bang", 0, 0, 0),
        ("hao de", 0, 0, 0), ("wan an", 0, 0, 0),
        ("zhu ni hao yun", 0, 0, 0), ("fei chang gan xie", 0, 0, 0),
        ("hen kai xin", 0, 0, 0), ("tai hao le", 0, 0, 0),
        # 空格 safe
        ("你 好", 0, 0, 0), ("谢 谢", 0, 0, 0),
        ("没 关 系", 0, 0, 0), ("很 开 心", 0, 0, 0),
        ("天 气 真 好", 0, 0, 0), ("明 天 见", 0, 0, 0),
        ("加 油", 0, 0, 0), ("太 棒 了", 0, 0, 0),
        # 谐音 safe
        ("好哒", 0, 0, 0), ("嗯呐", 0, 0, 0),
        ("欧克", 0, 0, 0), ("奈斯", 0, 0, 0),
        # emoji safe
        ("加油💪", 0, 0, 0), ("太棒了🎉", 0, 0, 0),
        ("晚安😴", 0, 0, 0), ("👍很好", 0, 0, 0),
        ("❤️谢谢", 0, 0, 0), ("开心✨", 0, 0, 0),
        # 语境暧昧 safe（情绪表达，非攻击）
        ("我服了", 0, 0, 0), ("真服了", 0, 0, 0),
        ("烦死了", 0, 0, 0), ("气死我了", 0, 0, 0),
        ("无语了", 0, 0, 0), ("我也是醉了", 0, 0, 0),
        ("真是够了", 0, 0, 0), ("牛啊", 0, 0, 0),
    ]
    for text, ins, obs, hate in handcrafted_safe:
        aug_rows.append({"comment": text,
                         "insult": ins, "obscene": obs, "identity_hate": hate})

    # === C：unsafe 样本少量增强（保持召回率） ===
    unsafe_df = df[(df[LABELS].sum(axis=1) > 0)]
    unsafe_sample = unsafe_df.sample(n=min(100, len(unsafe_df)), random_state=SEED)
    unsafe_transforms = [space_separate, partial_space, insert_emoji,
                         pinyin_mix, homophone_replace]

    for _, row in unsafe_sample.iterrows():
        text = row["comment"]
        labels = [row[l] for l in LABELS]
        if len(text) < 2:
            continue
        for _ in range(random.randint(1, 2)):
            func = random.choice(unsafe_transforms)
            aug_text = func(text)
            if aug_text and aug_text != text and len(aug_text) >= 2:
                aug_rows.append({
                    "comment": aug_text,
                    "insult": int(labels[0]),
                    "obscene": int(labels[1]),
                    "identity_hate": int(labels[2]),
                })

    # === 合并 & 去重 & 保存 ===
    aug_df = pd.DataFrame(aug_rows)
    aug_df = aug_df.drop_duplicates(subset=["comment"])

    orig_texts = set(df["comment"].fillna("").tolist())
    aug_df = aug_df[~aug_df["comment"].isin(orig_texts)]
    aug_df = aug_df[["comment"] + LABELS]

    print(f"📊 生成增强样本: {len(aug_df)} 条")
    safe_n = (aug_df[LABELS].sum(axis=1) == 0).sum()
    print(f"   其中 safe: {safe_n} / unsafe: {len(aug_df) - safe_n}")

    df_combined = pd.concat([df, aug_df], ignore_index=True)
    df_combined.to_csv(OUT_PATH, index=False)
    print(f"\n✅ 保存至: {OUT_PATH}")
    print(f"   原始 {len(df)} + 增强 {len(aug_df)} = 总计 {len(df_combined)} 条")


if __name__ == "__main__":
    main()
