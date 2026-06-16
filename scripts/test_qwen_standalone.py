# scripts/test_qwen_standalone.py
"""单独测试 Qwen2-VL 的审核能力"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.vlm_module import qwen2vl_classify

IMAGE_DIR = "/root/autodl-tmp/multimodal_safety_system/data/image"

start_time = time.time()

results = []
total = 0

for label in ["safe", "unsafe"]:
    folder = os.path.join(IMAGE_DIR, label)
    if not os.path.isdir(folder):
        continue

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])

    print(f"\n{'='*70}")
    print(f"📁 {label.upper()}  ({len(files)} 张)")
    print(f"{'='*70}")
    print(f"  {'file':10s} | verdict  | porn    | violence | safe    | decision |")

    for fname in files:
        path = os.path.join(folder, fname)
        scores = qwen2vl_classify(path)

        if scores is None:
            print(f"  {fname:10s} | {'FAILED':^60s} |")
            continue

        porn = scores["porn"]
        violence = scores["violence"]
        safe = scores["safe"]
        reasoning = scores.get("reasoning", "")

        # YES/NO → decision
        if reasoning.startswith("Qwen: YES"):
            verdict = "YES"
            max_risk = 0.8
            decision = "UNSAFE"
        elif reasoning.startswith("Qwen: NO"):
            verdict = "NO"
            max_risk = 0.0
            decision = "SAFE"
        else:
            verdict = "SCORE"
            max_risk = max(porn, violence)
            if max_risk > 0.6:
                decision = "UNSAFE"
            elif max_risk > 0.4:
                decision = "REVIEW"
            else:
                decision = "SAFE"

        # 判断对错
        if label == "safe":
            is_correct = (decision == "SAFE")
        else:
            is_correct = (decision != "SAFE")

        total += 1
        marker = "✅" if is_correct else "❌"
        top_label = scores["top_idx"]
        if isinstance(top_label, int):
            top_label = ["porn", "violence", "safe"][min(top_label, 2)]

        print(f"  {fname:10s} | {verdict:7s} | {porn:.4f}  | {violence:.4f}  | {safe:.4f}  | {decision:6s} | {marker}")

        results.append({
            "file": fname, "label": label,
            "porn": porn, "violence": violence, "safe": safe,
            "decision": decision, "correct": is_correct,
        })

# ======================
# 统计
# ======================
safe_results = [r for r in results if r["label"] == "safe"]
unsafe_results = [r for r in results if r["label"] == "unsafe"]

print(f"\n{'='*70}")
print(f"📊 Qwen 单独表现")
print(f"{'='*70}")
print(f"  Safe  准确率: {sum(1 for r in safe_results if r['correct']):2d}/{len(safe_results)} = {sum(1 for r in safe_results if r['correct'])/len(safe_results)*100:.1f}%")
print(f"  Unsafe 召回率: {sum(1 for r in unsafe_results if r['correct']):2d}/{len(unsafe_results)} = {sum(1 for r in unsafe_results if r['correct'])/len(unsafe_results)*100:.1f}%")
print(f"  总体正确率:     {sum(1 for r in results if r['correct'])}/{total} = {sum(1 for r in results if r['correct'])/total*100:.1f}%")
print(f"\n⏱️  总耗时: {time.time() - start_time:.1f}s")

# 错误明细
errors = [r for r in results if not r["correct"]]
if errors:
    print(f"\n❌ Qwen 误判明细 ({len(errors)} 张):")
    for e in errors:
        print(f"  {e['label']:6s} {e['file']:10s} → {e['decision']:6s} (porn={e['porn']:.3f} violence={e['violence']:.3f} safe={e['safe']:.3f})")
