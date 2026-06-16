# image_module.py

import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from ..config import settings

# ======================
# 1️⃣ 初始化 CLIP 模型（全局只加载一次）
# ======================
CLIP_MODEL_PATH = settings.CLIP_MODEL_PATH
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, use_safetensors=False)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)
clip_model.eval()

# ======================
# 1.5️⃣ CLIP 多维度文本提示词
# ======================
CLIP_TEXTS = [
    # 0-3: 色情
    "a pornographic image",
    "sexual content",
    "nude image",
    "anime or cartoon pornographic content",
    # 4-12: 暴力/威胁/血腥/自残/虐待
    "a violent scene",
    "people fighting",
    "blood or gore",
    "a person with a weapon attacking someone, causing blood",
    "a person shooting or stabbing someone with blood",
    "a threatening scene",
    "a horror movie scene, ghost or supernatural creature",
    "self-harm, suicide or cutting wrists imagery",
    "animal abuse or cruelty",
    # 13-17: 安全/正常
    "a normal daily photo",
    "safe content",
    "a person showing their physique in a fitness or fashion context",
    "a medical anatomy diagram",
    "a fitness or gym workout photo",

]

# ======================
# 2️⃣ CLIP 多维度语义分数
# ======================
def clip_multi_score(image_path):
    """
    用 CLIP 计算多维度安全分数：porn / violence / weapon / threat / safe
    """
    try:
        image = Image.open(image_path).convert("RGB")

        inputs = clip_processor(
            text=CLIP_TEXTS,
            images=image,
            return_tensors="pt",
            padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))

        porn_score     = float(max(probs[0], probs[1], probs[2], probs[3]))
        violence_score = float(max(probs[4], probs[5], probs[6],
                                   probs[7], probs[8], probs[9],
                                   probs[10], probs[11], probs[12]))
        safe_score     = float(max(probs[13], probs[14], probs[15], probs[16], probs[17]))

        return {
            "porn": porn_score,
            "violence": violence_score,
            "safe": safe_score,
            "top_idx": top_idx,
        }

    except Exception:
        return {
            "porn": 0.0, "violence": 0.0, "safe": 1.0,
            "top_idx": 13,
        }


# ======================
# 3️⃣ 主函数：CLIP 图像审核
# ======================
def predict_image(image_path):
    """
    输入：图片路径
    输出：CLIP 多维度审核结果
    """

    # CLIP 多维度语义检测
    clip_scores = clip_multi_score(image_path)

    # 综合风险
    max_risk = max(
        clip_scores["porn"],
        clip_scores["violence"],
    )

    risks = {
        "porn":     clip_scores["porn"],
        "violence": clip_scores["violence"],
    }
    risk_type = max(risks, key=risks.get)

    if max_risk > settings.IMAGE_UNSAFE_THRESHOLD:
        decision = "UNSAFE"
    elif max_risk > settings.IMAGE_REVIEW_THRESHOLD:
        decision = "REVIEW"
    else:
        decision = "SAFE"

    return {
        "image_path": image_path,
        "clip_scores": clip_scores,
        "risk_type": risk_type,
        "max_risk": float(max_risk),
        "decision": decision,
    }


# ======================
# 4️⃣ CLIP + Qwen2-VL 双轨检测 🔥
# ======================
def predict_image_dual(image_path: str) -> dict:
    """
    CLIP 快速筛查 + Qwen2-VL 深度确认

    流程:
        CLIP score < 0.3  → SAFE（快速路径，CLIP 确信安全）
        CLIP score > 0.6  → UNSAFE（快速路径，CLIP 确信危险）
        CLIP 0.3~0.6     → Qwen2-VL 二次确认（模糊路径）

    Args:
        image_path: 图片路径

    Returns:
        与 predict_image 格式一致，增加 qwen_scores 字段
    """
    from ..config import settings as _s
    from .vlm_module import qwen2vl_classify  # 延迟导入，避免启动时加载

    # ---- Step 1: CLIP 快速筛查 ----
    clip_result = predict_image(image_path)
    clip_max_risk = clip_result["max_risk"]

    # 快速路径：CLIP 确信安全
    if clip_max_risk < _s.IMAGE_DEEP_LOW:
        clip_result["model_used"] = "clip_fast_safe"
        clip_result["qwen_scores"] = None
        return clip_result

    # 快速路径：CLIP 确信危险
    if clip_max_risk > _s.IMAGE_DEEP_HIGH:
        clip_result["model_used"] = "clip_fast_unsafe"
        clip_result["qwen_scores"] = None
        return clip_result

    # ---- Step 2: Qwen2-VL 深度确认（仅 0.4~0.6 模糊区）----
    print(f"🔍 CLIP={clip_max_risk:.2f}（不确定），调用 Qwen2-VL 深度审核...")
    qwen_scores = qwen2vl_classify(image_path)

    # Qwen 推理失败 → 回退到 CLIP 结果
    if qwen_scores is None:
        clip_result["model_used"] = "clip_fallback"
        clip_result["qwen_scores"] = None
        return clip_result

    # Qwen 各维度最高风险
    qwen_max_risk = max(
        qwen_scores["porn"],
        qwen_scores["violence"],
    )

    # ---- Step 3: 双模型决策 ----
    if qwen_max_risk < 0.4:
        if clip_max_risk > 0.5:
            # CLIP 确信危险，Qwen 说安全 → 保守 REVIEW，不降级为 SAFE
            fused_score = clip_max_risk
            final_decision = "REVIEW"
        else:
            fused_score = qwen_max_risk
            final_decision = "REVIEW" if qwen_max_risk > 0.2 else "SAFE"
    elif qwen_max_risk > 0.6:
        # Qwen 确认危险 → 信任大模型
        fused_score = qwen_max_risk
        final_decision = "UNSAFE"
    else:
        # Qwen 也不确定 → 保守 REVIEW
        fused_score = qwen_max_risk
        final_decision = "REVIEW"

    # 风险类型优先用 Qwen
    risks = {
        "porn":     qwen_scores["porn"],
        "violence": qwen_scores["violence"],
    }
    risk_type = max(risks, key=risks.get)

    return {
        "image_path": image_path,
        "clip_scores": clip_result["clip_scores"],
        "qwen_scores": {
            k: v for k, v in qwen_scores.items() if k != "reasoning"
        },
        "qwen_reasoning": qwen_scores.get("reasoning", ""),
        "risk_type": risk_type,
        "max_risk": float(fused_score),
        "decision": final_decision,
        "model_used": "clip+qwen2vl",
    }


# ======================
# 6️⃣ 测试
# ======================
if __name__ == "__main__":
    test_img = "/root/autodl-tmp/multimodal_safety_system/data/image/test.jpg"

    print("\n" + "=" * 50)
    print("📸 CLIP 模式（默认）")
    result = predict_image(test_img)
    for k, v in result.items():
        if k == "clip_scores":
            print(f"  clip_scores:")
            for dim, score in v.items():
                if dim != "top_idx":
                    print(f"    {dim}: {score:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 50)
    print("🔬 CLIP + Qwen2-VL 双轨模式")
    result = predict_image_dual(test_img)
    for k, v in result.items():
        if k in ("clip_scores", "qwen_scores"):
            continue
        print(f"  {k}: {v}")
    if result.get("qwen_scores"):
        print(f"  qwen_scores:")
        for dim, score in result["qwen_scores"].items():
            if dim != "top_idx":
                print(f"    {dim}: {score:.4f}")