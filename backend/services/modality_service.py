# modality_service.py
"""
统一多模态安全审核服务层
对上层隐藏 text / image / audio 三个模块的差异
"""

from ..models.text_module import predict_text
from ..models.image_model import predict_image, predict_image_dual
from ..models.audio_module import predict_audio
from .fusion_engine import FusionEngine
from .policy_engine import PolicyEngine
from .explainer import explain
from ..config import settings


# ======================
# 🔧 全局引擎实例（单例）
# ======================
_fusion_engine = FusionEngine(config={
    "agreement_boost": settings.FUSION_AGREEMENT_BOOST,
    "all_safe_penalty": settings.FUSION_ALL_SAFE_PENALTY,
    "modality_weights": settings.FUSION_MODALITY_WEIGHTS,
})
_policy_multi = PolicyEngine(rules=settings.POLICY_RULES_MULTI)
_policy_single_text = PolicyEngine(rules=settings.POLICY_RULES_SINGLE_TEXT)
_policy_single_image = PolicyEngine(rules=settings.POLICY_RULES_SINGLE_IMAGE)
_policy_single_audio = PolicyEngine(rules=settings.POLICY_RULES_SINGLE_AUDIO)


# ======================
# 1️⃣ 统一输出 schema
# ======================
def _normalize(modality: str, raw: dict) -> dict:
    """将各模态输出统一成相同结构（所有模态使用同一组字段名）"""

    if modality == "text":
        return {
            "modality": "text",
            "decision": raw["decision"],
            "risk_type": raw["labels"][0] if raw["labels"] else "safe",
            "score": raw["score"],                    # ← 统一为 score
            "labels": raw["labels"],
            "probs": raw.get("probs", {}),
            "raw": raw,
        }

    elif modality == "image":
        return {
            "modality": "image",
            "decision": raw["decision"],
            "risk_type": raw["risk_type"],
            "score": raw["max_risk"],                 # ← 统一为 score
            "labels": raw.get("labels", []),
            "clip_scores": {
                k: v for k, v in raw.get("clip_scores", {}).items()
                if k != "top_idx"
            },
            "raw": raw,
        }

    elif modality == "audio":
        # 转换 numpy 类型为 Python 原生类型（避免 FastAPI 序列化报错）
        import numpy as np
        def _to_native(v):
            if isinstance(v, (np.floating, np.integer)): return v.item()
            if isinstance(v, np.ndarray): return v.tolist()
            if isinstance(v, dict): return {k2: _to_native(v2) for k2, v2 in v.items()}
            if isinstance(v, (list, tuple)): return [_to_native(v2) for v2 in v]
            return v
        return {
            "modality": "audio",
            "decision": raw["decision"],
            "risk_type": raw["labels"][0] if raw["labels"] else "safe",
            "score": _to_native(raw["audio_risk"]),
            "labels": raw["labels"],
            "probs": {k: _to_native(v) for k, v in raw.get("probs", {}).items()},
            "transcription": raw.get("transcription", ""),
            "raw": {k: _to_native(v) if k == "audio_stats" else v for k, v in raw.items()},
        }

    else:
        raise ValueError(f"Unknown modality: {modality}")


# ======================
# 2️⃣ 统一入口
# ======================
def predict(modality: str, content: str, image_deep: bool = False):
    """
    统一预测接口

    Args:
        modality:   "text" | "image" | "audio"
        content:    文本内容 / 图片路径 / 音频路径
        image_deep: 图片是否启用 CLIP+Qwen2-VL 双轨审核（默认 False）

    Returns:
        统一格式的 dict
    """
    modality = modality.lower()

    if modality == "text":
        raw = predict_text(content)
    elif modality == "image":
        if image_deep:
            raw = predict_image_dual(content)
        else:
            raw = predict_image(content)
    elif modality == "audio":
        raw = predict_audio(content)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return _normalize(modality, raw)


# ======================
# 3️⃣ 多模态融合预测 🔥（系统核心入口）
# ======================
def predict_multimodal(
    text: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
    image_deep: bool = False,
) -> dict:
    """
    多模态融合预测 —— 这才是系统的核心入口

    1. 并行调用所有可用模态
    2. FusionEngine 融合
    3. PolicyEngine 落地策略
    4. 统一格式返回

    Args:
        text:       文本内容（可选）
        image_path: 图片文件路径（可选）
        audio_path: 音频文件路径（可选）
        image_deep: 图片是否启用 Qwen2-VL 双轨审核（默认 False）

    Returns:
        {
            "decision": "SAFE" | "REVIEW" | "UNSAFE",
            "score": float,
            "details": {
                "fusion_logic": str,
                "policy": {...},
                "modalities": [{...}, ...],
            }
        }
    """
    # ---- 1. 收集可用模态 ----
    verdicts = []

    if text:
        verdicts.append(_normalize("text", predict_text(text)))
    if image_path:
        if image_deep:
            verdicts.append(_normalize("image", predict_image_dual(image_path)))
        else:
            verdicts.append(_normalize("image", predict_image(image_path)))
    if audio_path:
        verdicts.append(_normalize("audio", predict_audio(audio_path)))

    if not verdicts:
        output = {
            "decision": "SAFE",
            "score": 0.0,
            "details": {
                "fusion_logic": "no_modality",
                "policy": _policy_multi.apply(0.0),
                "modalities": [],
            },
        }
        output["explanation"] = explain(output)
        return output

    # ---- 统一用策略引擎改写各模态决策（明细表显示策略判定）----
    for v in verdicts:
        mod = v["modality"]
        if mod == "text":
            v["decision"] = _policy_single_text.apply(v["score"])["decision"]
        elif mod == "image":
            v["decision"] = _policy_single_image.apply(v["score"])["decision"]
        elif mod == "audio":
            v["decision"] = _policy_single_audio.apply(v["score"])["decision"]

    # ---- 2. 融合 + 策略 ----
    if len(verdicts) >= 2:
        # 多模态：FusionEngine 算分 → 多模态策略决策
        fused = _fusion_engine.fuse(verdicts)
        final_score = fused["score"]
        policy_result = _policy_multi.apply(final_score)
    else:
        # 单模态：模型分数直送对应模态策略
        final_score = verdicts[0]["score"]
        modality = verdicts[0]["modality"]
        if modality == "text":
            policy_result = _policy_single_text.apply(final_score)
        elif modality == "image":
            policy_result = _policy_single_image.apply(final_score)
        elif modality == "audio":
            policy_result = _policy_single_audio.apply(final_score)
        else:
            policy_result = _policy_single_text.apply(final_score)
        fused = {"logic": "single_modal_direct", "details": {}}

    final_decision = policy_result["decision"]

    # ---- 融合系数（前端展示用）----
    _boost_map = {
        "multi_modal_agreement": settings.FUSION_AGREEMENT_BOOST,
        "all_safe": settings.FUSION_ALL_SAFE_PENALTY,
    }
    fusion_boost = _boost_map.get(fused["logic"], 1.0)

    # ---- 3. 返回 ----
    output = {
        "decision": final_decision,
        "score": final_score,
        "details": {
            "fusion_logic": fused["logic"],
            "fusion_details": fused.get("details", {}),
            "fusion_boost": fusion_boost,
            "policy": policy_result,
            "modalities": verdicts,
        },
    }

    # ---- 5. 自然语言解释 ----
    output["explanation"] = explain(output)

    return output


# ======================
# 4️⃣ 批量预测
# ======================
def batch_predict(items: list[dict]) -> list[dict]:
    """
    批量预测

    Args:
        items: [{"modality": "text", "content": "..."}, ...]

    Returns:
        [统一格式的 dict, ...]
    """
    return [predict(item["modality"], item["content"]) for item in items]


# ======================
# 4️⃣ 测试
# ======================
if __name__ == "__main__":
    # 测试文本
    print("=" * 50)
    print("TEXT TEST")
    r = predict("text", "I will kill you")
    for k, v in r.items():
        if k == "raw":
            continue
        print(f"  {k}: {v}")

    # 测试图片
    print("=" * 50)
    print("IMAGE TEST")
    r = predict("image", "/root/autodl-tmp/multimodal_safety_system/data/image/test.jpg")
    for k, v in r.items():
        if k in ("raw", "clip_scores"):
            continue
        print(f"  {k}: {v}")
