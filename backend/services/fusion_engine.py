# backend/services/fusion_engine.py
"""
融合引擎 —— 多模态结果 → 统一风险分数

职责:
  接收多个模态的判定结果，根据"信号一致性"计算融合分数
  不负责最终决策（decision），只输出 score

核心原则:
  - 多模态一致危险 → 强化（score ×boost）
  - 多模态冲突 → 加权平均
  - 全部安全 → 保守降权（score ×penalty）
"""

from typing import Any


class FusionEngine:
    """多模态融合引擎 —— 只算分，不决策"""

    def __init__(self, config: dict | None = None):
        self.cfg = config or {}
        self.agreement_boost = self.cfg.get("agreement_boost", 1.2)
        self.conflict_penalty = self.cfg.get("conflict_penalty", 0.8)
        self.all_safe_penalty = self.cfg.get("all_safe_penalty", 0.9)
        self.modality_weights = self.cfg.get("modality_weights", {})

    def fuse(self, verdicts: list[dict]) -> dict:
        """
        融合多个模态 → 输出统一分数（不做 decision）

        Returns:
            {
                "score": float,       # 融合分数 (0~1)
                "logic": str,         # 融合逻辑标签
                "details": {...}      # 调试信息
            }
        """
        if not verdicts:
            return {"score": 0.0, "logic": "no_modality", "details": {"n_modalities": 0}}

        n = len(verdicts)
        unsafe_cnt = sum(1 for v in verdicts if v["decision"] == "UNSAFE")
        max_score = max(v["score"] for v in verdicts)
        decisions = [v["decision"] for v in verdicts]

        # ----- 多模态一致危险（≥2 个 UNSAFE）-----
        if unsafe_cnt >= 2:
            return {
                "score": min(1.0, max_score * self.agreement_boost),
                "logic": "multi_modal_agreement",
                "details": {"n_modalities": n, "unsafe_count": unsafe_cnt, "decisions": decisions},
            }

        # ----- 多模态冲突（1 unsafe，其他 safe）-----
        if unsafe_cnt == 1 and n > 1:
            weights = [self.modality_weights.get(v["modality"], 1.0) for v in verdicts]
            scores = [v["score"] for v in verdicts]
            weighted_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
            return {
                "score": weighted_score,
                "logic": "modality_conflict_weighted",
                "details": {
                    "n_modalities": n, "unsafe_count": unsafe_cnt, "decisions": decisions,
                    "raw_scores": scores, "weights": weights, "weighted_score": round(weighted_score, 4),
                },
            }

        # ----- 全部安全 -----
        return {
            "score": max_score * self.all_safe_penalty,
            "logic": "all_safe",
            "details": {"n_modalities": n, "decisions": decisions},
        }


# ======================
# 测试
# ======================
if __name__ == "__main__":
    engine = FusionEngine()
    test_cases = [
        ("多模态一致危险", [
            {"modality": "text", "decision": "UNSAFE", "score": 0.82},
            {"modality": "image", "decision": "UNSAFE", "score": 0.91},
        ]),
        ("多模态冲突", [
            {"modality": "text", "decision": "UNSAFE", "score": 0.82},
            {"modality": "image", "decision": "SAFE", "score": 0.12},
        ]),
        ("全部安全", [
            {"modality": "text", "decision": "SAFE", "score": 0.15},
            {"modality": "image", "decision": "SAFE", "score": 0.08},
        ]),
    ]
    for name, verdicts in test_cases:
        result = engine.fuse(verdicts)
        print(f"\n📌 {name}: score={result['score']:.3f} | logic={result['logic']}")


# ======================
# 测试
# ======================
if __name__ == "__main__":
    engine = FusionEngine()

    test_cases = [
        # (name, verdicts)
        ("多模态一致危险", [
            {"modality": "text", "decision": "UNSAFE", "score": 0.82},
            {"modality": "image", "decision": "UNSAFE", "score": 0.91},
        ]),
        ("单模态低置信", [
            {"modality": "text", "decision": "UNSAFE", "score": 0.55},
        ]),
        ("单模态高置信", [
            {"modality": "text", "decision": "UNSAFE", "score": 0.85},
        ]),
        ("多模态冲突", [
            {"modality": "text", "decision": "UNSAFE", "score": 0.82},
            {"modality": "image", "decision": "SAFE", "score": 0.12},
        ]),
        ("全部安全", [
            {"modality": "text", "decision": "SAFE", "score": 0.15},
            {"modality": "image", "decision": "SAFE", "score": 0.08},
        ]),
    ]

    for name, verdicts in test_cases:
        print(f"\n{'='*50}")
        print(f"📌 {name}")
        for v in verdicts:
            print(f"   [{v['modality']}] decision={v['decision']}, score={v['score']}")
        result = engine.fuse(verdicts)
        print(f"  → fused: decision={result['decision']}, score={result['score']:.3f}")
        print(f"     logic: {result['logic']} (boost={result['boost']})")
