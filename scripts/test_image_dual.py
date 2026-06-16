# scripts/test_image_dual.py
"""批量测试 CLIP + Qwen2-VL 双轨审核，统计正确率"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.image_model import predict_image, predict_image_dual

IMAGE_DIR = "/root/autodl-tmp/multimodal_safety_system/data/image"

start_time = time.time()

clip_only_results = []
dual_results = []
correct_clip = 0
correct_dual = 0
qwen_triggered = 0
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
    print(f"  {'file':10s} | CLIP    | Qwen? | Dual   | risk      | score    |")

    for fname in files:
        path = os.path.join(folder, fname)

        # CLIP 单独
        r_clip = predict_image(path)
        # CLIP + Qwen 双轨
        r_dual = predict_image_dual(path)

        clip_decision = r_clip["decision"]
        dual_decision = r_dual["decision"]
        clip_score = r_clip["max_risk"]
        dual_score = r_dual["max_risk"]
        model_used = r_dual.get("model_used", "unknown")
        risk_type = r_dual["risk_type"]
        qwen_called = "qwen" in model_used.lower() or model_used == "clip_fallback"

        if qwen_called:
            qwen_triggered += 1

        # 三档统计：✅正确 / 🔶复审 / ❌真误判
        if label == "safe":
            clip_ok = (clip_decision == "SAFE")
            dual_ok = (dual_decision == "SAFE")
            clip_rev = (clip_decision == "REVIEW")
            dual_rev = (dual_decision == "REVIEW")
        else:
            clip_ok = (clip_decision == "UNSAFE")
            dual_ok = (dual_decision == "UNSAFE")
            clip_rev = (clip_decision == "REVIEW")
            dual_rev = (dual_decision == "REVIEW")

        if clip_ok:
            correct_clip += 1
        if dual_ok:
            correct_dual += 1
        total += 1

        clip_mark = "✅" if clip_ok else ("🔶" if clip_rev else "❌")
        dual_mark = "✅" if dual_ok else ("🔶" if dual_rev else "❌")
        qwen_tag = "QWEN" if qwen_called else "    "
        clip_mark = "✅" if clip_ok else "❌"
        dual_mark = "✅" if dual_ok else "❌"

        print(f"  {fname:10s} | {clip_decision:6s} {clip_mark} | {qwen_tag} | {dual_decision:5s} {dual_mark} | {risk_type:8s} | cl={clip_score:.3f} qw={dual_score:.3f} |")

        clip_only_results.append({
            "file": fname, "label": label,
            "decision": clip_decision, "score": clip_score, "correct": clip_ok,
        })
        dual_results.append({
            "file": fname, "label": label,
            "decision": dual_decision, "score": dual_score,
            "model_used": model_used, "risk_type": risk_type,
            "qwen_called": qwen_called, "correct": dual_ok,
        })

# ======================
# 统计
# ======================
safe_total = sum(1 for r in dual_results if r["label"] == "safe")
unsafe_total = sum(1 for r in dual_results if r["label"] == "unsafe")

print(f"\n{'='*70}")
print(f"📊 汇总")
print(f"{'='*70}")
print(f"  {'':20s} | CLIP Only | CLIP+Qwen |")
print(f"  {'Safe  正确率':20s} | {sum(1 for r in clip_only_results if r['label']=='safe' and r['correct']):2d}/{safe_total} = {sum(1 for r in clip_only_results if r['label']=='safe' and r['correct'])/safe_total*100:5.1f}%  | {sum(1 for r in dual_results if r['label']=='safe' and r['correct']):2d}/{safe_total} = {sum(1 for r in dual_results if r['label']=='safe' and r['correct'])/safe_total*100:5.1f}% |")
print(f"  {'Unsafe 召回率':20s} | {sum(1 for r in clip_only_results if r['label']=='unsafe' and r['correct']):2d}/{unsafe_total} = {sum(1 for r in clip_only_results if r['label']=='unsafe' and r['correct'])/unsafe_total*100:5.1f}%  | {sum(1 for r in dual_results if r['label']=='unsafe' and r['correct']):2d}/{unsafe_total} = {sum(1 for r in dual_results if r['label']=='unsafe' and r['correct'])/unsafe_total*100:5.1f}% |")
print(f"  {'总体正确率':20s} | {correct_clip}/{total} = {correct_clip/total*100:5.1f}%  | {correct_dual}/{total} = {correct_dual/total*100:5.1f}% |")
print(f"\n  Qwen 被触发: {qwen_triggered}/{total} 次 ({qwen_triggered/total*100:.1f}%)")
print(f"\n⏱️  总耗时: {time.time() - start_time:.1f}s")

# 错误明细
clip_errors = [r for r in clip_only_results if not r["correct"]]
dual_errors = [r for r in dual_results if not r["correct"]]

if clip_errors:
    print(f"\n❌ CLIP 误判 ({len(clip_errors)} 张):")
    for e in clip_errors:
        d = next(d for d in dual_results if d["file"] == e["file"])
        print(f"  {e['label']:6s} {e['file']:10s} → clip={e['decision']:6s} ({e['score']:.3f})  dual={d['decision']:5s} ({d['score']:.3f}) {'✅ 被Qwen纠正' if d['correct'] else '❌ 仍误判'}")

if dual_errors:
    print(f"\n❌ Dual 仍误判 ({len(dual_errors)} 张):")
    for d in dual_errors:
        print(f"  {d['label']:6s} {d['file']:10s} → {d['decision']:5s} ({d['risk_type']}, score={d['score']:.3f})")
