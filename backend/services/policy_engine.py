# backend/services/policy_engine.py
"""
策略引擎 —— 风险分数 → 最终业务决策

职责:
  将融合后的风险分数映射为业务决策
  纯业务规则层，与模型逻辑完全解耦

核心设计:
  - 规则可配置（支持从 YAML/JSON 加载）
  - 与 FusionEngine 独立变化
  - 一份分数 + 策略 = 最终 decision
"""

import json
import os
from typing import Any


class PolicyEngine:
    """业务策略引擎"""

    def __init__(self, rules: list[dict] | None = None, rules_path: str | None = None):
        """
        Args:
            rules: 规则列表，格式见 DEFAULT_RULES
            rules_path: 可选，从 JSON/YAML 文件加载规则
        """
        if rules:
            self.rules = rules
        elif rules_path and os.path.exists(rules_path):
            self.rules = self._load_rules(rules_path)
        else:
            self.rules = self._default_rules()

    # ======================
    # 默认规则（业务可根据需要调整）
    # ======================
    @staticmethod
    def _default_rules() -> list[dict]:
        return [
            {
                "min_score": 0.8,
                "decision": "UNSAFE",
                "label": "high_risk",
                "description": "高风险，直接拦截",
            },
            {
                "min_score": 0.4,
                "decision": "REVIEW",
                "label": "medium_risk",
                "description": "中风险，人工复审",
            },
            {
                "min_score": 0.0,
                "decision": "SAFE",
                "label": "low_risk",
                "description": "低风险，放行",
            },
        ]

    # ======================
    # 加载规则
    # ======================
    @staticmethod
    def _load_rules(path: str) -> list[dict]:
        """从 JSON 文件加载规则"""
        with open(path) as f:
            data = json.load(f)
        # 按 min_score 降序排列（高优先级在前）
        return sorted(data, key=lambda r: r["min_score"], reverse=True)

    # ======================
    # 核心接口
    # ======================
    def apply(self, score: float, context: dict | None = None) -> dict:
        """
        将分数映射为业务决策

        Args:
            score: 融合后的风险分数 (0~1)
            context: 可选上下文信息（如风险类型、用户等级等）

        Returns:
            {
                "decision": "SAFE" | "REVIEW" | "UNSAFE",
                "label": str,
                "score": float,
                "rule_applied": str,
            }
        """
        for rule in self.rules:
            if score >= rule["min_score"]:
                return {
                    "decision": rule["decision"],
                    "label": rule["label"],
                    "score": score,
                    "rule_applied": rule["description"],
                }

        # 兜底（理论上不会到这里）
        return {
            "decision": "SAFE",
            "label": "low_risk",
            "score": score,
            "rule_applied": "fallback",
        }

    # ======================
    # 规则管理
    # ======================
    def get_rules(self) -> list[dict]:
        """查看当前规则"""
        return self.rules

    def set_rules(self, rules: list[dict]) -> None:
        """动态更新规则"""
        self.rules = sorted(rules, key=lambda r: r["min_score"], reverse=True)

    def export_rules(self, path: str) -> None:
        """导出规则到 JSON 文件"""
        with open(path, "w") as f:
            json.dump(self.rules, f, indent=2, ensure_ascii=False)
        print(f"📄 规则已导出到 {path}")


# ======================
# 测试
# ======================
if __name__ == "__main__":
    engine = PolicyEngine()

    test_scores = [0.95, 0.82, 0.65, 0.40, 0.35, 0.10, 0.0]

    print(f"{'='*50}")
    print("📊 PolicyEngine 测试")

    for s in test_scores:
        result = engine.apply(s)
        print(f"  score={s:.2f} → {result['decision']:8s} | {result['label']:15s} | {result['rule_applied']}")
