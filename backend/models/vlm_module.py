"""
VLM 深度审核模块（Qwen2-VL-2B）
用于对 CLIP 筛查出的疑图（风险分数 0.4~0.6）进行二次确认
对外接口与 clip_multi_score() 一致，可即插即用
"""

import json
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from ..config import settings

# ======================
# 配置
# ======================
MODEL_PATH = settings.VLM_MODEL_PATH
MAX_NEW_TOKENS = settings.VLM_MAX_NEW_TOKENS
TEMPERATURE = settings.VLM_TEMPERATURE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 加载模型（全局单例）
# ======================
print("🚀 Loading Qwen2-VL-2B...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ======================
# 系统 prompt（可调）
# ======================
SYSTEM_PROMPT = """你是一个严格的内容安全审核模型。

请判断图片是否违规（色情、暴力、威胁、血腥、自残、虐待、成人用品、性行为、暗示性行为）。

只允许输出：YES 或 NO

禁止：输出解释、句子、JSON、标点、其他字符

不确定时默认输出 YES"""


# ======================
# 核心函数
# ======================
def qwen2vl_classify(image_path: str) -> dict | None:
    """
    用 Qwen2-VL-2B 进行深度安全分类

    Args:
        image_path: 图片路径

    Returns:
        与 clip_multi_score 格式一致:
        {"porn":float,"violence":float,"weapon":float,
         "threat":float,"safe":float,"top_idx":int,"reasoning":str}
        失败时返回 None，由调用方回退到 CLIP 结果
    """
    try:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "YES or NO?",
                    },
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,
            )

        response = processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        scores = _parse_response(response)
        if scores is None:
            return None

        # 计算 top_idx（与 CLIP 对齐）
        risk_dims = [
            scores["porn"],
            scores["violence"],
        ]
        if scores["safe"] > max(risk_dims):
            scores["top_idx"] = 2  # safe 维度
        else:
            scores["top_idx"] = max(range(2), key=lambda i: risk_dims[i])

        return scores

    except Exception as e:
        print(f"⚠️ Qwen2-VL 推理失败: {e}，回退到 CLIP 结果")
        return None


def _parse_response(response: str) -> dict | None:
    """从模型回复中提取结果，优先匹配 YES/NO"""
    cleaned = response.strip().upper()

    # 直接匹配 YES/NO（优先、最快）
    if cleaned == "YES" or (cleaned.startswith("YES") and "NO" not in cleaned):
        return _extract_scores("YES")
    if cleaned == "NO" or cleaned.startswith("NO"):
        return _extract_scores("NO")

    # 回退：从 ```json 块中提取
    if "```" in response:
        for block in response.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                obj = json.loads(block)
                return _extract_scores(obj)
            except json.JSONDecodeError:
                continue

    # 回退：直接 JSON 解析
    try:
        obj = json.loads(response)
        return _extract_scores(obj)
    except json.JSONDecodeError:
        pass

    # 回退：{ 到 } 截取
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        obj = json.loads(response[start:end])
        return _extract_scores(obj)
    except (ValueError, json.JSONDecodeError):
        print(f"⚠️ 无法解析 Qwen 输出: {response[:200]}")
        return None


def _extract_scores(obj: dict | str) -> dict:
    """从 YES/NO 或 dict 中提取分数"""
    if isinstance(obj, str):
        text = obj.strip().upper()
        if text.startswith("YES"):
            return {
                "porn": 0.8, "violence": 0.8, "safe": 0.1,
                "reasoning": "Qwen: YES",
            }
        elif text.startswith("NO"):
            return {
                "porn": 0.0, "violence": 0.0, "safe": 1.0,
                "reasoning": "Qwen: NO",
            }
        else:
            print(f"⚠️ 无法解析 Qwen 输出: {text[:200]}")
            return None

    return {
        "porn": float(obj.get("porn", 0)),
        "violence": float(obj.get("violence", 0)),
        "safe": float(obj.get("safe", 1)),
        "reasoning": obj.get("reasoning", ""),
    }


# ======================
# 测试
# ======================
if __name__ == "__main__":
    import sys

    img = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/root/autodl-tmp/multimodal_safety_system/data/image/test.jpg"
    )

    print(f"📸 测试图片: {img}")
    result = qwen2vl_classify(img)

    if result:
        print("\n📊 Qwen2-VL 结果:")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print("❌ 推理失败")
