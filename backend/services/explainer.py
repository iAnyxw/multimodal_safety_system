"""
自然语言解释器
将审核结果翻译成人类可读的中文描述
"""

from typing import Any

# ======================
# 风险类型中文映射
# ======================
RISK_TYPE_ZH = {
    # 文本
    "toxic": "辱骂",
    "severe_toxic": "严重辱骂",
    "obscene": "色情低俗",
    "threat": "威胁言论",
    "insult": "人身攻击",
    "identity_hate": "仇恨言论",
    # 图像
    "porn": "色情内容",
    "violence": "暴力/血腥/威胁",
    "safe": "安全",
    # 通用
    "safe": "安全",
}

MODALITY_ICON = {"text": "📝 文本", "image": "🖼️ 图片", "audio": "🎵 音频"}

DECISION_ZH = {"SAFE": "安全 ✅", "REVIEW": "需人工复审 ⚠️", "UNSAFE": "不安全 ❌"}

FUSION_LOGIC_ZH = {
    "multi_modal_agreement": "多个模态均检测到风险，互相印证，可信度高",
    "modality_conflict": "不同模态检测结果不一致，为保险起见采取保守策略",
    "single_modal_high_conf": "仅有一个模态可用，但风险置信度较高",
    "single_modal_low_conf": "仅有一个模态可用，且风险置信度较低，需要人工确认",
    "single_modal_safe": "仅有一个模态可用，未检测到风险",
    "all_safe": "所有模态均未检测到风险",
    "no_modality": "未提供任何审核内容",
}


def explain(result: dict) -> str:
    """
    将审核结果生成为自然语言描述

    Args:
        result: predict_multimodal() 的返回结果

    Returns:
        一段中文自然语言描述
    """
    parts = []

    decision = result.get("decision", "SAFE")
    score = result.get("score", 0.0)
    details = result.get("details", {})
    modalities = details.get("modalities", [])
    fusion_logic = details.get("fusion_logic", "")
    policy = details.get("policy", {})

    # ======================
    # 1️⃣ 开场 —— 一句话结论
    # ======================
    if decision == "UNSAFE":
        parts.append(f"🚫 系统判定为【{DECISION_ZH[decision]}】，综合风险评分 {score:.2f}（满分 1.0），{policy.get('rule_applied', '需要拦截')}。")
    elif decision == "REVIEW":
        parts.append(f"⚠️ 系统判定为【{DECISION_ZH[decision]}】，综合风险评分 {score:.2f}（满分 1.0），{policy.get('rule_applied', '建议人工复核')}。")
    else:
        parts.append(f"✅ 系统判定为【{DECISION_ZH[decision]}】，综合风险评分 {score:.2f}（满分 1.0），{policy.get('rule_applied', '可以放行')}。")

    # ======================
    # 2️⃣ 各模态检测结果
    # ======================
    if modalities:
        parts.append("")
        parts.append("📋 各模态检测明细：")

        for m in modalities:
            mod = m.get("modality", "")
            mod_decision = m.get("decision", "SAFE")
            mod_score = m.get("score", 0.0)
            risk_type = m.get("risk_type", "safe")
            labels = m.get("labels", [])

            icon = MODALITY_ICON.get(mod, mod)
            risk_cn = RISK_TYPE_ZH.get(risk_type, risk_type)
            decision_cn = DECISION_ZH.get(mod_decision, mod_decision)
            score_desc = _score_to_word(mod_score)

            detail = f"  {icon}：{decision_cn}，{score_desc}（评分 {mod_score:.2f}）"
            if risk_type != "safe" and risk_cn:
                detail += f"，主要风险类型为「{risk_cn}」"
            if labels:
                labels_str = "、".join(labels[:3])
                detail += f"，涉及标签：{labels_str}"
            parts.append(detail)

    # ======================
    # 3️⃣ 融合逻辑说明
    # ======================
    if fusion_logic:
        parts.append("")
        fusion_desc = FUSION_LOGIC_ZH.get(fusion_logic, f"融合规则：{fusion_logic}")
        boost = details.get("fusion_boost", 1.0)
        if boost != 1.0:
            if boost > 1.0:
                parts.append(f"🔗 融合分析：{fusion_desc}，因此风险分数上浮 {((boost - 1.0) * 100):.0f}%。")
            else:
                parts.append(f"🔗 融合分析：{fusion_desc}，因此风险分数下调 {((1.0 - boost) * 100):.0f}%。")
        else:
            parts.append(f"🔗 融合分析：{fusion_desc}。")

    # ======================
    # 4️⃣ 建议
    # ======================
    parts.append("")
    if decision == "UNSAFE":
        parts.append("💡 建议：该内容违反安全规则，建议拦截处理。")
    elif decision == "REVIEW":
        parts.append("💡 建议：该内容存在一定风险，建议人工复核后决定是否放行。")
    else:
        parts.append("💡 建议：该内容未检测到风险，可以正常展示。")

    return "\n".join(parts)


def _score_to_word(score: float) -> str:
    """将分数转为程度描述"""
    if score < 0.2:
        return "风险极低"
    elif score < 0.4:
        return "风险较低"
    elif score < 0.6:
        return "存在一定风险"
    elif score < 0.8:
        return "风险较高"
    else:
        return "风险极高"


def explain_short(result: dict) -> str:
    """简短版解释，适合在消息推送等场景使用"""
    decision = result.get("decision", "SAFE")
    score = result.get("score", 0.0)
    details = result.get("details", {})
    modalities = details.get("modalities", [])

    parts = [f"{DECISION_ZH.get(decision, decision)}（{score:.2f}）"]

    for m in modalities:
        mod = m.get("modality", "")
        mod_score = m.get("score", 0.0)
        risk_type = m.get("risk_type", "")
        icon = MODALITY_ICON.get(mod, mod)
        risk_cn = RISK_TYPE_ZH.get(risk_type, "")
        if risk_cn and risk_type != "safe":
            parts.append(f"{icon} {mod_score:.2f}（{risk_cn}）")
        else:
            parts.append(f"{icon} {mod_score:.2f}")

    return " | ".join(parts)
