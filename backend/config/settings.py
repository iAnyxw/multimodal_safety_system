"""
集中配置中心
所有模块从此读取配置，不再硬编码
支持环境变量覆盖（.env 文件）
"""

from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict

# ======================
# 项目根目录
# ======================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ======================
    # 通用
    # ======================
    DEVICE: str = "cuda"  # 自动检测在模块层做，这里给默认偏好

    # ======================
    # 文本模块 - 英文 (6 标签)
    # ======================
    TEXT_MODEL_PATH: str = str(CHECKPOINTS_DIR / "xlmr")
    TEXT_LABELS: list[str] = [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate",
    ]
    TEXT_MAX_LENGTH: int = 256
    TEXT_THRESHOLDS: dict[str, float] = {
        "toxic": 0.5, "severe_toxic": 0.6, "obscene": 0.6,
        "threat": 0.5, "insult": 0.5, "identity_hate": 0.4,
    }
    TEXT_UNSAFE_EXCEED: float = 0.15

    # ======================
    # 文本模块 - 中文 (3 标签)
    # ======================
    TEXT_ZH_MODEL_PATH: str = str(CHECKPOINTS_DIR / "xlmr_ch")
    TEXT_ZH_LABELS: list[str] = [
        "insult",         # 辱骂
        "obscene",        # 淫秽
        "identity_hate",  # 身份仇恨
    ]
    TEXT_ZH_THRESHOLDS: dict[str, float] = {
        "insult": 0.5,
        "obscene": 0.5,
        "identity_hate": 0.5,
    }
    TEXT_ZH_UNSAFE_EXCEED: float = 0.15

    # ======================
    # 图像模块 - CLIP
    # ======================
    CLIP_MODEL_PATH: str = str(
        Path.home()
        / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32"
        / "snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    )
    # 决策阈值
    IMAGE_UNSAFE_THRESHOLD: float = 0.6
    IMAGE_REVIEW_THRESHOLD: float = 0.3

    # CLIP 双轨深度审核阈值
    IMAGE_DEEP_LOW: float = 0.3   # 低于此值 → SAFE 快速放行
    IMAGE_DEEP_HIGH: float = 0.6  # 高于此值 → UNSAFE 快速拦截

    # ======================
    # 图像模块 - Qwen2-VL
    # ======================
    VLM_MODEL_PATH: str = str(CHECKPOINTS_DIR / "Qwen2-VL-2B-Instruct")
    VLM_MAX_NEW_TOKENS: int = 256
    VLM_TEMPERATURE: float = 0.1

    # ======================
    # 音频模块
    # ======================
    WHISPER_MODEL_SIZE: str = "base"
    # 音频增强阈值
    AUDIO_ENERGY_THRESHOLD: float = 0.02
    AUDIO_SNR_THRESHOLD: float = 10.0
    AUDIO_SILENCE_RATIO_THRESHOLD: float = 0.6

    # ======================
    # 融合引擎（仅多模态用，只算分不决策）
    # ======================
    FUSION_AGREEMENT_BOOST: float = 1.3
    FUSION_ALL_SAFE_PENALTY: float = 0.9
    # 模态信任权重（冲突时加权平均）
    FUSION_MODALITY_WEIGHTS: dict[str, float] = {
        "text": 1.0,
        "image": 1.0,
        "audio": 1.0,
    }

    # ======================
    # 策略引擎（多模态用——FusionEngine 融合后的分数）
    # ======================
    POLICY_RULES_MULTI: list[dict] = [
        {"min_score": 0.50, "decision": "UNSAFE", "label": "high_risk",
         "description": "高风险，直接拦截"},
        {"min_score": 0.20, "decision": "REVIEW", "label": "medium_risk",
         "description": "中风险，人工复审"},
        {"min_score": 0.0, "decision": "SAFE", "label": "low_risk",
         "description": "低风险，放行"},
    ]

    # ======================
    # 策略引擎（单模态用——模型原始分数直接走这里）
    # ======================
    POLICY_RULES_SINGLE_TEXT: list[dict] = [
        {"min_score": 0.44, "decision": "UNSAFE", "label": "high_risk",
         "description": "高风险"},
        {"min_score": 0.20, "decision": "REVIEW", "label": "medium_risk",
         "description": "中风险"},
        {"min_score": 0.0, "decision": "SAFE", "label": "low_risk",
         "description": "低风险"},
    ]
    POLICY_RULES_SINGLE_IMAGE: list[dict] = [
        {"min_score": 0.60, "decision": "UNSAFE", "label": "high_risk",
         "description": "高风险"},
        {"min_score": 0.40, "decision": "REVIEW", "label": "medium_risk",
         "description": "中风险"},
        {"min_score": 0.0, "decision": "SAFE", "label": "low_risk",
         "description": "低风险"},
    ]
    POLICY_RULES_SINGLE_AUDIO: list[dict] = [
        {"min_score": 0.44, "decision": "UNSAFE", "label": "high_risk",
         "description": "高风险"},
        {"min_score": 0.20, "decision": "REVIEW", "label": "medium_risk",
         "description": "中风险"},
        {"min_score": 0.0, "decision": "SAFE", "label": "low_risk",
         "description": "低风险"},
    ]

    # ======================
    # API
    # ======================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Multimodal Safety System"
    API_VERSION: str = "1.0.0"
    CORS_ORIGINS: list[str] = ["*"]
