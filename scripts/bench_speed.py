# scripts/bench_speed.py
"""对比 CLIP / Qwen / CLIP+Qwen 单张推理耗时"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.image_model import predict_image, predict_image_dual
from backend.models.vlm_module import qwen2vl_classify

img = "/root/autodl-tmp/multimodal_safety_system/data/image/unsafe/1.jpg"
WARMUP = 3
N = 20

# ======================
# CLIP Only
# ======================
print("🔥 预热中...")
for _ in range(WARMUP):
    predict_image(img)

t0 = time.time()
for _ in range(N):
    predict_image(img)
t1 = time.time()
clip_ms = (t1 - t0) / N * 1000
print(f"\n📊 CLIP Only:     {clip_ms:6.1f} ms/张  ({N} 次均值)")

# ======================
# Qwen Only
# ======================
print("🔥 预热中...")
qwen2vl_classify(img)  # warmup

t0 = time.time()
for _ in range(N):
    qwen2vl_classify(img)
t1 = time.time()
qwen_ms = (t1 - t0) / N * 1000
print(f"📊 Qwen Only:     {qwen_ms:6.1f} ms/张  ({N} 次均值)")

# ======================
# CLIP + Qwen (Dual)
# ======================
print("🔥 预热中...")
predict_image_dual(img)

t0 = time.time()
for i in range(N):
    predict_image_dual(img)
t1 = time.time()
dual_ms = (t1 - t0) / N * 1000
print(f"📊 CLIP + Qwen:   {dual_ms:6.1f} ms/张  ({N} 次均值)")

# ======================
# 汇总
# ======================
print(f"\n{'='*50}")
print(f"📊 速度对比")
print(f"{'='*50}")
print(f"  CLIP        : {clip_ms:.0f} ms")
print(f"  Qwen        : {qwen_ms:.0f} ms  (×{qwen_ms/clip_ms:.0f} 于 CLIP)")
print(f"  CLIP + Qwen : {dual_ms:.0f} ms  (×{dual_ms/clip_ms:.0f} 于 CLIP)")

print(f"\n📊 效率分析（假设日活 100 万张图）:")
print(f"  纯 Qwen 日耗: {qwen_ms/1000*1_000_000/3600:.1f} GPU·小时")
print(f"  CLIP 筛掉 56% 不触发 Qwen，Dual 日耗: {dual_ms/1000*1_000_000*0.44/3600:.1f} GPU·小时")
print(f"  CLIP 节省: {int((1 - dual_ms*0.44/qwen_ms)*100)}% Qwen 调用")
