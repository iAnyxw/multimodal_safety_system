# scripts/test_image_clip.py
"""批量测试 CLIP 图像审核，统计正确率"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.image_model import predict_image

IMAGE_DIR = "/root/autodl-tmp/multimodal_safety_system/data/image"

start_time = time.time()

results = []
correct = 0
total = 0

for label in ["safe", "unsafe"]:
    folder = os.path.join(IMAGE_DIR, label)
    if not os.path.isdir(folder):
        continue

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])

    print(f"\n{'='*60}")
    print(f"📁 {label.upper()}  ({len(files)} 张)")
    print(f"{'='*60}")

    for fname in files:
        path = os.path.join(folder, fname)
        result = predict_image(path)

        decision = result["decision"]
        risk_type = result["risk_type"]
        score = result["max_risk"]

        # 分三档统计
        if label == "safe":
            if decision == "SAFE":
                is_correct = True; is_review = False; is_error = False
            elif decision == "REVIEW":
                is_correct = False; is_review = True; is_error = False
            else:
                is_correct = False; is_review = False; is_error = True
        else:
            if decision == "UNSAFE":
                is_correct = True; is_review = False; is_error = False
            elif decision == "REVIEW":
                is_correct = False; is_review = True; is_error = False
            else:
                is_correct = False; is_review = False; is_error = True

        if is_correct:
            correct += 1
        total += 1

        if is_correct:
            marker = "✅"
        elif is_review:
            marker = "🔶"
        else:
            marker = "❌"

        print(f"  {fname:12s} | decision={decision:6s} | risk={risk_type:8s} | score={score:.4f} | {marker}")

        results.append({
            "file": fname,
            "label": label,
            "decision": decision,
            "risk_type": risk_type,
            "score": score,
            "correct": is_correct,
            "is_review": is_review,
            "is_error": is_error,
        })

# ======================
# 统计（区分真正错误 vs 复审）
# ======================
safe_total = sum(1 for r in results if r["label"] == "safe")
safe_ok   = sum(1 for r in results if r["label"] == "safe" and r["correct"])
safe_rev  = sum(1 for r in results if r["label"] == "safe" and r["is_review"])
safe_err  = sum(1 for r in results if r["label"] == "safe" and r["is_error"])

unsafe_total = sum(1 for r in results if r["label"] == "unsafe")
unsafe_ok   = sum(1 for r in results if r["label"] == "unsafe" and r["correct"])
unsafe_rev  = sum(1 for r in results if r["label"] == "unsafe" and r["is_review"])
unsafe_err  = sum(1 for r in results if r["label"] == "unsafe" and r["is_error"])

print(f"\n{'='*60}")
print(f"📊 汇总")
print(f"{'='*60}")
print(f"  ✅ 完全正确:    {correct}/{total} = {correct/total*100:.1f}%")
print(f"  🔶 复审兜底:    {safe_rev+unsafe_rev}/{total} = {(safe_rev+unsafe_rev)/total*100:.1f}%")
print(f"  ❌ 真正误判:    {safe_err+unsafe_err}/{total} = {(safe_err+unsafe_err)/total*100:.1f}%")
print(f"")
print(f"  Safe:  ✅{safe_ok}  🔶{safe_rev}(复审)  ❌{safe_err}(误拦)")
print(f"  Unsafe:✅{unsafe_ok}  🔶{unsafe_rev}(复审)  ❌{unsafe_err}(漏放)")
print(f"\n⏱️  总耗时: {time.time() - start_time:.1f}s")

# 真正错误明细
real_errors = [r for r in results if r["is_error"]]
reviewed = [r for r in results if r["is_review"]]
if real_errors:
    print(f"\n❌ 真正误判 ({len(real_errors)} 张):")
    for e in real_errors:
        print(f"  {e['label']:6s} {e['file']:12s} → {e['decision']:6s} ({e['risk_type']}, score={e['score']:.4f})")
