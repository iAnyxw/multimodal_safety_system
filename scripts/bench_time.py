# scripts/bench_time.py
"""对比 CLIP / Qwen / Dual 单张推理耗时"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.image_model import predict_image, predict_image_dual
from backend.models.vlm_module import qwen2vl_classify

img = "/root/autodl-tmp/multimodal_safety_system/data/image/unsafe/5.jpg"
WARMUP, N = 2, 10

print("🔥 预热...")
for _ in range(WARMUP):
    predict_image(img)
    qwen2vl_classify(img)
    predict_image_dual(img)

print("\n📊 单张推理耗时 (10次均值):\n")

# CLIP
t0 = time.time()
for _ in range(N):
    predict_image(img)
t_clip = (time.time() - t0) / N * 1000
print(f"  CLIP Only    : {t_clip:6.0f} ms")

# Qwen
t0 = time.time()
for _ in range(N):
    qwen2vl_classify(img)
t_qwen = (time.time() - t0) / N * 1000
print(f"  Qwen Only    : {t_qwen:6.0f} ms  (×{t_qwen/t_clip:.0f} CLIP)")

# Dual (送 Qwen 的图)
t0 = time.time()
for _ in range(N):
    predict_image_dual(img)
t_dual = (time.time() - t0) / N * 1000
print(f"  Dual(送Qwen)  : {t_dual:6.0f} ms  (×{t_dual/t_clip:.0f} CLIP)")
