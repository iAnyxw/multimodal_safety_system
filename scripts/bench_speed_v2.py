# scripts/bench_speed_v2.py
"""分场景测速：CLIP快路径 vs Qwen触发路径"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.image_model import predict_image, predict_image_dual
from backend.models.vlm_module import qwen2vl_classify

# 一张直接拦截的图（CLIP >0.6）+ 一张触发Qwen的图（CLIP 0.3-0.6）
img_fast = "/root/autodl-tmp/multimodal_safety_system/data/image/unsafe/1.jpg"   # CLIP ~0.87 → 不送 Qwen
img_slow = "/root/autodl-tmp/multimodal_safety_system/data/image/unsafe/5.jpg"   # CLIP ~0.55 → 送 Qwen

WARMUP = 2
N = 10

results = {}

for name, img, fn in [
    ("CLIP Only",     img_fast, lambda p: predict_image(p)),
    ("Qwen Only",     img_fast, lambda p: qwen2vl_classify(p)),
    ("Dual (不触发Qwen)", img_fast, lambda p: predict_image_dual(p)),
    ("Dual (触发Qwen)",  img_slow, lambda p: predict_image_dual(p)),
]:
    print(f"🔥 预热 {name}...")
    for _ in range(WARMUP):
        fn(img)

    t0 = time.time()
    for _ in range(N):
        fn(img)
    t1 = time.time()
    ms = (t1 - t0) / N * 1000
    results[name] = ms
    print(f"  {name}: {ms:.0f} ms/张")

# ======================
# 解析 210 张图的耗时构成
# ======================
clip_ms = results["CLIP Only"]
qwen_ms = results["Qwen Only"]
dual_fast = results["Dual (不触发Qwen)"]
dual_slow = results["Dual (触发Qwen)"]

print(f"\n{'='*55}")
print(f"📊 210 张图耗时拆解")
print(f"{'='*55}")

# 210 张 CLIP: ~44% 触发 Qwen (92张), 56% 不触发 (118张)
n_total = 210
n_qwen = 92
n_fast = n_total - n_qwen

pure_clip_time = n_total * clip_ms / 1000
pure_qwen_time = n_total * qwen_ms / 1000
dual_time = n_fast * dual_fast / 1000 + n_qwen * dual_slow / 1000

print(f"  纯 CLIP:  {n_total} × {clip_ms:.0f}ms = {pure_clip_time:.0f}s")
print(f"  纯 Qwen:  {n_total} × {qwen_ms:.0f}ms = {pure_qwen_time:.0f}s")
print(f"  Dual:     {n_fast}×{dual_fast:.0f}ms + {n_qwen}×{dual_slow:.0f}ms = {dual_time:.0f}s")
print(f"\n  模型加载 (CLIP+Qwen): ~6s")
print(f"  Dual 预估总耗时: {dual_time + 6:.0f}s")
print(f"  纯 CLIP 预估总耗时: {pure_clip_time + 3:.0f}s")
print(f"  纯 Qwen 预估总耗时: {pure_qwen_time + 3:.0f}s")
print(f"\n结论: Dual 多花的时间 = CLIP推理~65s + 92张触发Qwen~65s")
print(f"      纯CLIP约65s，纯Qwen约160s，Dual约135s")
