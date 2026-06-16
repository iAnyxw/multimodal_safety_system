# audio_asr_smart.py

import os
import torch
import whisper
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np

# ======================
# 1️⃣ 设备
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 2️⃣ Whisper模型
# ======================
print("🚀 Loading Whisper...")
model = whisper.load_model("base", device=DEVICE)

# ======================
# 3️⃣ 音频增强
# ======================
def enhance_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None)

    y_denoised = nr.reduce_noise(y=y, sr=sr)
    y_norm = librosa.util.normalize(y_denoised)

    sf.write(output_path, y_norm, sr)
    return output_path


# ======================
# 4️⃣ 质量评估（核心）
# ======================
def audio_quality_check(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # 1️⃣ 能量
    energy = np.mean(np.abs(y))

    # 2️⃣ 简易SNR（近似）
    noise = y[: int(0.1 * len(y))]  # 前10%当噪声
    signal_power = np.mean(y**2)
    noise_power = np.mean(noise**2) + 1e-9
    snr = 10 * np.log10(signal_power / noise_power)

    # 3️⃣ 静音比例
    silence = np.sum(np.abs(y) < 0.01) / len(y)

    return {
        "energy": energy,
        "snr": snr,
        "silence_ratio": silence
    }


# ======================
# 5️⃣ 是否需要增强（策略）
# ======================
def should_enhance(stats):
    """
    可调参数（你可以自己改）
    """

    if stats["energy"] < 0.02:
        return True

    if stats["snr"] < 10:
        return True

    if stats["silence_ratio"] > 0.6:
        return True

    return False


# ======================
# 6️⃣ 转写（带智能增强）
# ======================
def smart_transcribe(audio_path):

    stats = audio_quality_check(audio_path)

    print("\n📊 Audio Stats:")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    use_enhance = should_enhance(stats)

    print(f"\n🤖 Decision: {'Enhance' if use_enhance else 'Raw'}")

    input_path = audio_path

    if use_enhance:
        enhanced_path = audio_path.replace(".wav", "_enhanced.wav")

        if not os.path.exists(enhanced_path):
            print("🔊 Enhancing audio...")
            enhance_audio(audio_path, enhanced_path)

        input_path = enhanced_path

    result = model.transcribe(input_path)

    print("\n📝 Transcription:")
    print(result["text"])

    return {
        "text": result["text"],
        "used_enhancement": use_enhance,
        "stats": stats
    }


# ======================
# 7️⃣ 测试
# ======================
if __name__ == "__main__":

    test_audio = "/root/autodl-tmp/multimodal_safety_system/data/audio/audio1.wav"  # 🔥 换成你的音频路径

    smart_transcribe(test_audio)