# scripts/adversarial_guard.py
# 对抗文本检测 Guard
# 核心逻辑：检测输入是否为对抗变体（拼音/空格/谐音/emoji干扰）
#          若是 + 模型判 unsafe → 降级为 REVIEW（不直接拦截）

import re


def is_adversarial_pattern(text: str) -> bool:
    """
    检测文本是否为对抗变体。
    返回 True 表示文本可能经过了对抗变换，模型结果应该降级处理。
    """

    # 1. 纯拼音模式：连续拼音音节（如 "nmsl", "cao ni ma"）
    #    中文文本通常不会出现大段纯拼音，除非对抗
    pinyin_words = re.findall(r'[a-z]+', text.lower())
    pinyin_chars = sum(len(w) for w in pinyin_words)
    total_chars = len(text.replace(" ", ""))
    if total_chars > 0 and pinyin_chars / total_chars > 0.5:
        return True

    # 2. 中文字间空格分隔：每个汉字之间都有空格
    #    正常中文不会这样写
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(chinese_chars) >= 2:
        # 检查是否每个汉字之间都有空格
        spaced_pattern = " ".join(chinese_chars)
        if spaced_pattern in text:
            return True
        # 检查空格密度：中文之间空格比例异常
        spaces = text.count(" ")
        if spaces >= len(chinese_chars) * 0.5 and spaces >= 2:
            return True

    # 3. 谐音检测：包含常见对抗谐音字
    homophone_chars = set("煞笔草泥马尼玛卧槽碧池法克沙雕特么苾池")
    found_homophones = sum(1 for c in text if c in homophone_chars)
    if found_homophones >= 2:
        return True
    # 单个强谐音词检测
    homophone_words = ["煞笔", "草泥马", "尼玛", "卧槽", "碧池", "法克",
                       "沙雕", "特么", "你妹", "苾池"]
    for w in homophone_words:
        if w in text:
            return True

    # 4. emoji 干扰：文本中出现可能改意的 emoji
    dangerous_emojis = set("🔪💀😡👊🔫🪓")
    safe_emojis = set("😊😄😍💪🎉✨👍❤️🌸")
    # 危险 emoji + 短文本 = 高风险
    if any(e in text for e in dangerous_emojis) and len(text) <= 5:
        return True
    # emoji 夹在中文中间（不是纯 emoji 表达）
    emoji_pattern = re.compile(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]')
    emojis = emoji_pattern.findall(text)
    if emojis and chinese_chars and len(text) <= 15:
        return True

    # 5. 纯短拼音（无空格）：如 "nmsl", "sb", "cnm"
    #    特征：纯字母、全小写、无元音间隔或极短
    if re.match(r'^[a-z]{2,10}$', text.strip()):
        return True

    return False


def apply_guard(decision: str, score: float, text: str) -> tuple:
    """
    对抗 Guard：如果文本是 adversarial + 模型判 UNSAFE → 降为 REVIEW

    Args:
        decision: 原始决策 (SAFE / REVIEW / UNSAFE)
        score: 原始风险分数
        text: 输入文本

    Returns:
        (new_decision, new_score, guarded: bool)
    """
    if decision == "UNSAFE" and is_adversarial_pattern(text):
        return "REVIEW", min(score, 0.7), True
    return decision, score, False


# ======================
# 测试
# ======================
if __name__ == "__main__":
    test_cases = [
        ("nmsl", True),
        ("nihaoma", True),
        ("cao ni ma", True),
        ("你 去 死", True),
        ("你 好 吗", True),
        ("煞笔", True),
        ("好哒", False),       # 单谐音不够
        ("🔪你", True),
        ("我😊你", True),
        ("今天😄很开心", True),
        ("你去死吧", False),    # 正常中文
        ("你好", False),
        ("今天天气不错", False),
        ("sb", True),
    ]

    print("对抗模式检测测试：")
    for text, expected in test_cases:
        result = is_adversarial_pattern(text)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{text}' → adversarial={result} (expected={expected})")
