# audio_module.py

import os
import torch
import whisper
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np

from .text_module import predict_text  # 👉 复用你的XLM-R
from ..config import settings

# ======================
# 1️⃣ 设备
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 2️⃣ Whisper模型
# ======================
whisper_model = whisper.load_model(settings.WHISPER_MODEL_SIZE, device=DEVICE)


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
# 4️⃣ 音频质量评估
# ======================
def audio_quality_check(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    energy = np.mean(np.abs(y))

    noise = y[: int(0.1 * len(y))]
    signal_power = np.mean(y**2)
    noise_power = np.mean(noise**2) + 1e-9
    snr = 10 * np.log10(signal_power / noise_power)

    silence = np.sum(np.abs(y) < 0.01) / len(y)

    return {
        "energy": energy,
        "snr": snr,
        "silence_ratio": silence
    }


# ======================
# 5️⃣ 是否增强
# ======================
def should_enhance(stats):

    if stats["energy"] < settings.AUDIO_ENERGY_THRESHOLD:
        return True

    if stats["snr"] < settings.AUDIO_SNR_THRESHOLD:
        return True

    if stats["silence_ratio"] > settings.AUDIO_SILENCE_RATIO_THRESHOLD:
        return True

    return False


# ======================
# 6️⃣ 语音转文本（带智能增强）
# ======================
def speech_to_text(audio_path):

    stats = audio_quality_check(audio_path)
    use_enhance = should_enhance(stats)

    input_path = audio_path

    if use_enhance:
        enhanced_path = audio_path.replace(".wav", "_enhanced.wav")

        if not os.path.exists(enhanced_path):
            enhance_audio(audio_path, enhanced_path)

        input_path = enhanced_path

    result = whisper_model.transcribe(input_path)

    return result["text"], use_enhance, stats


# ======================
# 7️⃣ 语音审核（对外接口）
# ======================
def predict_audio(audio_path):

    text, used_enhance, stats = speech_to_text(audio_path)

    text_result = predict_text(text)

    return {
        "audio_path": audio_path,

        "transcription": text,
        "audio_risk": text_result.get("score", 0.0),
        "labels": text_result.get("labels", []),

        "probs": text_result.get("probs", {}),

        "used_enhancement": used_enhance,
        "audio_stats": stats,

        "decision": text_result.get("decision", "SAFE"),
    }