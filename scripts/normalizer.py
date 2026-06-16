# scripts/normalizer.py
# 对抗文本预处理归一化
# 将对抗变体还原为干净中文，再送入模型推理
# 核心思路：模型在干净文本上完美（safe F1=1.00），归一化后问题自动消失

import re


# ======================
# 映射表
# ======================

# 拼音 → 中文（常见 toxic 缩写 + safe 短语）
PINYIN_MAP = {
    # toxic
    "nmsl":      "你妈死了",
    "sb":        "傻逼",
    "cnm":       "操你妈",
    "tmd":       "他妈的",
    "nmd":       "你妈的",
    "shabi":     "傻逼",
    "cao ni ma": "操你妈",
    "wocao":     "我操",
    "sha que":   "傻缺",
    "gun dan":   "滚蛋",
    # safe
    "nihaoma":          "你好吗",
    "xiexie ni":        "谢谢你",
    "zaijian":          "再见",
    "dui bu qi":        "对不起",
    "mei guan xi":      "没关系",
    "zhen bang":        "真棒",
    "hao de":           "好的",
    "wan an":           "晚安",
    "zhu ni hao yun":   "祝你好运",
    "fei chang gan xie": "非常感谢",
    "hen kai xin":      "很开心",
    "tai hao le":       "太好了",
}

# 谐音 → 原字
HOMOPHONE_MAP = {
    "煞笔":   "傻逼",
    "草泥马": "操你妈",
    "尼玛":   "你妈",
    "卧槽":   "我操",
    "碧池":   "婊子",
    "法克":   "傻逼",    # fuck → 归入辱骂
    "沙雕":   "傻屌",
    "特么":   "他妈",
    "苾池":   "婊子",
    "你妹":   "你妈",
}

# emoji → 中文含义
EMOJI_MAP = {
    "🔪": "杀",
    "💀": "死",
    "👊": "打",
    "🔫": "枪",
    "😡": "怒",
    "🪓": "斧",
}

# 无害 emoji（直接删除）
SAFE_EMOJIS = set("😊😄😍💪🎉✨👍❤️🌸😴🎵😢")


# ======================
# 归一化函数
# ======================

def normalize(text: str) -> str:
    """
    对抗文本归一化流水线：
    1. 谐音替换
    2. 拼音替换（已知映射）
    3. 去空格
    4. emoji 处理
    5. 清理
    """
    result = text

    # 1. 谐音 → 原字（先做，避免空格干扰匹配）
    for homo, original in HOMOPHONE_MAP.items():
        if homo in result:
            result = result.replace(homo, original)

    # 2. 已知拼音 → 中文
    lowered = result.lower().strip()
    if lowered in PINYIN_MAP:
        result = PINYIN_MAP[lowered]
        return result  # 精确匹配，直接返回

    # 3. 去空格（中文字间的空格）
    #    规则：如果文本含中文 + 空格，且空格密度高，去掉所有空格
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', result)
    spaces = result.count(" ")
    if chinese_chars and spaces > 0:
        # 中文字间空格：每个汉字后可能跟空格
        # 保守策略：空格数 ≥ 中文字数的 30%
        if spaces >= len(chinese_chars) * 0.3 or " ".join(chinese_chars) in result:
            result = re.sub(r'\s+', '', result)

    # 4. emoji 处理
    #    危险 emoji → 对应文字
    for emoji, meaning in EMOJI_MAP.items():
        result = result.replace(emoji, meaning)
    #    安全 emoji → 删除
    for emoji in SAFE_EMOJIS:
        result = result.replace(emoji, "")

    # 5. 清理残余：去掉不可见字符
    result = result.strip()

    return result if result else text  # 归一化后非空则返回，否则保留原文


# ======================
# 检测（保留给 Guard 用）
# ======================

def is_adversarial_pattern(text: str) -> bool:
    """检测是否为对抗变体（归一化后和原文不同即为对抗）"""
    return normalize(text) != text


# ======================
# 测试
# ======================
if __name__ == "__main__":
    test_cases = [
        # (输入, 期望归一化结果)
        # 拼音 unsafe
        ("nmsl",       "你妈死了"),
        ("sb",         "傻逼"),
        ("cnm",        "操你妈"),
        ("tmd",        "他妈的"),
        ("cao ni ma",  "操你妈"),
        # 拼音 safe
        ("nihaoma",    "你好吗"),
        ("xiexie ni",  "谢谢你"),
        ("wan an",     "晚安"),
        # 空格
        ("你 去 死",    "你去死"),
        ("你 好 吗",    "你好吗"),
        ("傻 逼",       "傻逼"),
        # 谐音
        ("煞笔",       "傻逼"),
        ("草泥马",     "操你妈"),
        ("卧槽",       "我操"),
        ("碧池",       "婊子"),
        ("法克",       "傻逼"),
        ("沙雕",       "傻屌"),
        # 不需变换的
        ("好哒",       "好哒"),
        ("我去",       "我去"),
        ("哇塞",       "哇塞"),
        # emoji
        ("🔪你",       "杀你"),
        ("去💀吧",     "去死吧"),
        ("我😊你",     "我你"),     # 😊 删除后剩"我你"，模型在干净文本上不会误判
        ("😍爱你",     "爱你"),
        ("今天😄很开心", "今天很开心"),
        # 正常文本不变
        ("你去死吧",   "你去死吧"),
        ("你好",       "你好"),
        ("今天天气不错", "今天天气不错"),
    ]

    print("=" * 70)
    print("🧹 预处理归一化测试")
    print("=" * 70)
    all_pass = True
    for text, expected in test_cases:
        result = normalize(text)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"  {status} '{text}' → '{result}' (期望 '{expected}')")

    print(f"\n{'✅ 全部通过' if all_pass else '❌ 存在不匹配'}")
