# api/main.py
"""
多模态安全审核 API
"""
import os
import uuid
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..services.modality_service import predict as service_predict
from ..services.modality_service import predict_multimodal
from ..config import settings

# ======================
# 1️⃣ App
# ======================
app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================
# 2️⃣ 文本审核（JSON body）
# ======================
@app.post("/predict/text")
def predict_text(body: dict):
    """
    body: {"text": "..."}
    """
    text = body.get("text", "")
    if not text:
        raise HTTPException(400, "text is required")
    return service_predict("text", text)


# ======================
# 3️⃣ 图片审核（文件上传）
# ======================
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    上传图片文件，返回审核结果
    """
    ext = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}{ext}")

    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        return service_predict("image", tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ======================
# 4️⃣ 音频审核（文件上传）
# ======================
@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """
    上传音频文件，返回审核结果
    """
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}{ext}")

    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        return service_predict("audio", tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ======================
# 5️⃣ 通用接口（按路径）
# ======================
@app.post("/predict")
def predict_by_path(body: dict):
    """
    按文件路径审核

    body: {"modality": "text|image|audio", "content": "文本内容 或 文件路径"}
    """
    modality = body.get("modality", "")
    content = body.get("content", "")

    if modality not in ("text", "image", "audio"):
        raise HTTPException(400, "modality must be text / image / audio")

    if not content:
        raise HTTPException(400, "content is required")

    return service_predict(modality, content)


# ======================
# 6️⃣ 多模态融合审核 🔥（系统核心入口）
# ======================
@app.post("/predict/multimodal")
async def predict_multimodal_endpoint(
    text: str = Form(None, description="文本内容（可选）"),
    image: UploadFile = File(None, description="图片文件（可选）"),
    audio: UploadFile = File(None, description="音频文件（可选）"),
):
    """
    多模态融合审核 —— 同时提交文本 + 图片 + 音频，系统自动融合判定

    - 至少提供一个模态
    - 三个模态可任意组合（1~3个）
    - 返回融合后的统一决策
    """
    if not any([text, image, audio]):
        raise HTTPException(400, "至少提供一个模态（text / image / audio）")

    # ---- 处理上传文件到临时路径 ----
    image_path = None
    audio_path = None

    if image:
        ext = os.path.splitext(image.filename or "img.jpg")[1] or ".jpg"
        image_path = os.path.join(tempfile.gettempdir(), f"upload_img_{uuid.uuid4().hex}{ext}")
        with open(image_path, "wb") as f:
            f.write(await image.read())

    if audio:
        ext = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        audio_path = os.path.join(tempfile.gettempdir(), f"upload_audio_{uuid.uuid4().hex}{ext}")
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

    # ---- 调用融合引擎 ----
    try:
        result = predict_multimodal(
            text=text,
            image_path=image_path,
            audio_path=audio_path,
        )
        return result
    finally:
        # 清理临时文件
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


# ======================
# 7️⃣ 健康检查
# ======================
@app.get("/health")
def health():
    return {"status": "ok"}


# ======================
# 7️⃣ 启动
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
