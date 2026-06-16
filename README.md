# 多模态安全审核系统 — Multimodal Safety System

> 覆盖文本、图像、音频三种模态的统一安全审核引擎，支持多模态融合判定与可解释风险输出。

- **仓库地址**：https://github.com/yourname/multimodal_safety_system
- **作者**：XXX
- **邮箱**：xxx@example.com

---

## 项目简介

面向互联网内容安全审核场景，构建覆盖**文本、图像、音频**三种模态的统一安全审核系统。以 XLM-RoBERTa 为文本基座（Micro F1=0.954）、CLIP-ViT 为图像基座、Whisper-base 为语音基座，设计多模态融合引擎与可配置策略引擎，实现端到端风险判定与自然语言解释输出。后端基于 FastAPI 构建 RESTful API，前端基于 Streamlit 实现可视化交互界面。

### 核心功能

| 功能 | 说明 |
|------|------|
| **文本审核** | 中文/英文双模型自动分流，3 标签（辱骂/淫秽/身份仇恨）多分类 |
| **图像审核** | CLIP 快速初筛 + Qwen2-VL 深度审核双轨架构（准确率 91.7%） |
| **音频审核** | Whisper 语音转写 → XLM-R 文本审核流水线 |
| **多模态融合** | 一致强化 / 冲突降权 / 孤证降级三种智能融合策略 |
| **可配置策略** | JSON 规则驱动，SAFE / REVIEW / UNSAFE 三级业务决策 |
| **可解释输出** | 自动生成结构化中文解释，含各模态明细与融合逻辑 |
| **对抗防御** | 拼音/谐音/空格/emoji 对抗文本检测 Guard + 归一化预处理 |

---

## 环境与依赖

### 运行环境

| 项目 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Ubuntu 20.04 / 22.04 | 开发与测试所用系统 |
| Python | 3.10 ~ 3.12 | 核心语言版本 |
| GPU | NVIDIA RTX 3090 / 4090（24GB VRAM） | 推荐使用 GPU 加速推理 |
| CUDA | 11.8+ | GPU 驱动版本 |

### 开源程序与第三方依赖

| 依赖名称 | 使用版本 | 安装方式 | 说明 |
|----------|----------|----------|------|
| PyTorch | 2.1.0+ | `pip install torch` | 深度学习框架 |
| CUDA Toolkit | 11.8 | NVIDIA 官方 | GPU 加速（可选） |

### Python 依赖

依赖清单文件：`backend/requirements.txt`（已包含在本仓库中）

安装命令：

```bash
pip install -r backend/requirements.txt
```

**核心依赖清单**：

| 包名 | 用途 |
|------|------|
| `torch` | 深度学习框架 |
| `transformers` | HuggingFace 模型加载与推理 |
| `fastapi` | RESTful API 框架 |
| `uvicorn` | ASGI 服务器 |
| `streamlit` | 前端可视化界面 |
| `whisper-openai` | 语音识别（音频转写） |
| `librosa` | 音频处理 |
| `noisereduce` | 音频降噪增强 |
| `Pillow` | 图像处理 |
| `scikit-learn` | 评估指标计算 |
| `pandas` | 数据处理 |
| `pydantic-settings` | 配置管理 |

> **模型权重说明**：预训练模型权重（XLM-RoBERTa、CLIP、Qwen2-VL）不纳入 Git 仓库，需通过 HuggingFace 自动下载或手动放置。详见下方"模型权重下载与放置"章节。

---

## 配置说明

### 配置中心

所有配置集中在 `backend/config/settings.py`，使用 Pydantic BaseSettings 管理，支持环境变量覆盖（`.env` 文件）。

```python
# 关键配置项示例（backend/config/settings.py）
class Settings(BaseSettings):
    # 文本模块
    TEXT_ZH_THRESHOLDS: dict = {"insult": 0.5, "obscene": 0.5, "identity_hate": 0.5}
    TEXT_ZH_UNSAFE_EXCEED: float = 0.15

    # 图像模块
    IMAGE_UNSAFE_THRESHOLD: float = 0.6
    IMAGE_REVIEW_THRESHOLD: float = 0.3
    IMAGE_DEEP_LOW: float = 0.3    # CLIP 双轨：低于此值 SAFE 快速放行
    IMAGE_DEEP_HIGH: float = 0.6   # CLIP 双轨：高于此值 UNSAFE 快速拦截

    # 融合引擎
    FUSION_AGREEMENT_BOOST: float = 1.3       # 多模态一致强化系数
    FUSION_CONFLICT_PENALTY: float = 0.8      # 冲突降权系数
    FUSION_MODALITY_WEIGHTS: dict = {"text": 1.0, "image": 1.0, "audio": 1.0}

    # 策略引擎
    POLICY_RULES: list = [
        {"min_score": 0.8, "decision": "UNSAFE", "label": "high_risk"},
        {"min_score": 0.35, "decision": "REVIEW", "label": "medium_risk"},
        {"min_score": 0.0, "decision": "SAFE", "label": "low_risk"},
    ]

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
```

> **安全提示**：敏感配置（密钥、密码等）请使用环境变量或 `.env` 文件，`.env` 已加入 `.gitignore`。

### 其他关键配置

| 配置项 | 默认值 | 说明 | 配置文件路径 |
|--------|--------|------|-------------|
| `API_HOST` | 0.0.0.0 | 服务绑定 IP | `backend/config/settings.py` |
| `API_PORT` | 8000 | 服务端口 | `backend/config/settings.py` |
| `TEXT_ZH_UNSAFE_EXCEED` | 0.15 | 中文文本 UNSAFE 判定门槛 | `backend/config/settings.py` |
| `IMAGE_UNSAFE_THRESHOLD` | 0.6 | 图像 UNSAFE 判定阈值 | `backend/config/settings.py` |
| `IMAGE_REVIEW_THRESHOLD` | 0.3 | 图像 REVIEW 判定阈值 | `backend/config/settings.py` |
| `FUSION_AGREEMENT_BOOST` | 1.3 | 多模态一致强化系数 | `backend/config/settings.py` |
| `VLM_TEMPERATURE` | 0.1 | Qwen2-VL 推理温度 | `backend/config/settings.py` |

---

## 数据集

### 数据集说明

| 数据集名称 | 来源 | 大小 | 格式 | 说明 |
|-----------|------|------|------|------|
| 中文文本安全审核数据集 | 自建 | 5,396 条 | CSV | 训练数据，含 insult/obscene/identity_hate 三标签 |
| 图像测试集 | 自标注 | 206 张 | JPG/PNG | 图片审核评估（Safe 100 张 / Unsafe 106 张） |
| 对抗文本测试集 | 手工构造 | 52 条 | Python 内联 | 5 类对抗样本鲁棒性测试 |

> **中文文本数据集字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `comment` | str | 文本内容（中文评论） |
| `insult` | int | 是否辱骂（0/1） |
| `obscene` | int | 是否淫秽（0/1） |
| `identity_hate` | int | 是否身份仇恨（0/1） |

**标签分布**：

| 标签 | 数量 | 占比 |
|------|------|------|
| insult（辱骂） | 1,598 | 29.6% |
| obscene（淫秽） | 965 | 17.9% |
| identity_hate（身份仇恨） | 899 | 16.7% |
| 全零（无毒） | 2,751 | 51.0% |

> **体积较大的数据集不纳入 Git 仓库**，请通过以下方式准备后放置到指定目录。
>
> **小部分数据示例已提交到 Git 仓库中**（放置于 `data/ch_text/` 目录），用于：
> - 让其他开发者无需下载完整数据集即可快速了解数据格式与字段含义
> - 支撑单元测试和本地调试的最小可运行数据
> - 作为数据处理流程的输入示例，方便 Code Review 时对照理解逻辑

### 模型权重下载与放置

预训练模型和微调权重需手动下载并放置到 `checkpoints/` 目录：

| 模型 | 下载位置 | 目标路径 | 说明 |
|------|---------|----------|------|
| XLM-R（中文 3 标签） | 脚本 `scripts/train_ch_text.py` 训练产出 | `checkpoints/xlmr_ch/` | 中文文本审核模型 |
| XLM-R（英文 6 标签） | HuggingFace `unitary/toxic-bert` 或自行训练 | `checkpoints/xlmr/` | 英文文本审核模型 |
| CLIP-ViT-Base-Patch32 | HuggingFace 自动缓存 | `~/.cache/huggingface/` | 图像初筛模型 |
| Qwen2-VL-2B-Instruct | HuggingFace `Qwen/Qwen2-VL-2B-Instruct` | `checkpoints/Qwen2-VL-2B-Instruct/` | VLM 深度审核模型 |

> **中文文本模型训练**：运行 `python scripts/train_ch_text.py` 即可自动训练并保存到 `checkpoints/xlmr_ch/`。

### 数据集目录结构

```
data/
├── ch_text/               # ✅ 中文文本数据集（已提交到 Git 仓库）
│   └── ch_data.csv        # 5,396 条标注数据
├── image/                 # 图像测试集（不提交）
│   ├── safe/              # 安全图片
│   └── unsafe/            # 不安全图片
├── audio/                 # 音频测试集（不提交）
├── test_cases.json        # 融合引擎测试用例
└── test_cases_v4.json     # 融合引擎 v4 测试用例
```

<details>
<summary><b>📊 中文文本数据集样本示例（点击展开）</b></summary>

```csv
comment,insult,obscene,identity_hate
你这个傻逼,1,0,0
今晚来我家,0,1,0
XX地方的人都这样,0,0,1
今天天气真好,0,0,0
去死吧你,1,0,0
```

</details>

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/multimodal_safety_system.git
cd multimodal_safety_system

# 2. 安装 Python 依赖
pip install -r backend/requirements.txt

# 3. 训练中文文本模型（可选，或使用已有权重）
python scripts/train_ch_text.py

# 4. 启动后端服务
python -m backend.main --port 8000

# 5. （新终端）启动前端界面
streamlit run frontend/streamlit_app.py
```

启动后访问：
- **API 文档**：http://localhost:8000/docs
- **前端界面**：http://localhost:8501（Streamlit 默认端口）

---

## API 文档

### 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/predict/text` | 文本审核 |
| POST | `/predict/image` | 图片审核（文件上传） |
| POST | `/predict/audio` | 音频审核（文件上传） |
| POST | `/predict` | 通用审核（按文件路径） |
| POST | `/predict/multimodal` | 多模态融合审核 |

### 调用示例

```python
import requests

BASE = "http://localhost:8000"

# 文本审核
r = requests.post(f"{BASE}/predict/text", json={"text": "你这个傻逼"})
print(r.json())
# {
#   "decision": "UNSAFE",
#   "score": 0.82,
#   "labels": ["insult"],
#   "probs": {"insult": 0.95, "obscene": 0.02, "identity_hate": 0.01}
# }

# 多模态融合审核
r = requests.post(
    f"{BASE}/predict/multimodal",
    data={"text": "这张图片有问题"},
    files={"image": open("test.jpg", "rb")}
)
print(r.json())
# {
#   "decision": "UNSAFE",
#   "score": 0.91,
#   "explanation": "🚫 系统判定为【不安全 ❌】，...",
#   "details": { ... }
# }
```

### 返回格式

```json
{
  "decision": "SAFE | REVIEW | UNSAFE",
  "score": 0.0 ~ 1.0,
  "explanation": "自然语言解释文本",
  "details": {
    "fusion_logic": "融合策略标识",
    "policy": { "decision": "...", "label": "...", "rule_applied": "..." },
    "modalities": [
      {
        "modality": "text | image | audio",
        "decision": "SAFE | REVIEW | UNSAFE",
        "score": 0.0 ~ 1.0,
        "risk_type": "风险类型",
        "labels": ["标签1", "标签2"]
      }
    ]
  }
}
```

---

## 项目结构

```
multimodal_safety_system/
├── backend/                     # 后端
│   ├── main.py                  # 启动入口
│   ├── requirements.txt         # 依赖清单
│   ├── api/
│   │   └── main.py              # FastAPI 路由（6 个端点）
│   ├── config/
│   │   └── settings.py          # 集中配置中心（Pydantic）
│   ├── models/                  # 模型层
│   │   ├── text_module.py       # 文本审核（XLM-R，中/英双模型）
│   │   ├── image_model.py       # 图像审核（CLIP 多维度分类）
│   │   ├── vlm_module.py        # VLM 深度审核（Qwen2-VL-2B）
│   │   └── audio_module.py      # 音频审核（Whisper → XLM-R）
│   └── services/                # 服务层
│       ├── modality_service.py  # 统一入口与多模态融合
│       ├── fusion_engine.py     # 融合引擎
│       ├── policy_engine.py     # 策略引擎
│       └── explainer.py         # 自然语言解释器
├── checkpoints/                 # 模型权重（不提交）
│   ├── xlmr_ch/                 # 中文 XLM-R（3 标签）
│   ├── xlmr/                    # 英文 XLM-R（6 标签）
│   └── Qwen2-VL-2B-Instruct/   # VLM 模型
├── data/                        # 数据集（不提交完整数据）
│   ├── ch_text/ch_data.csv      # 中文文本数据集（示例）
│   ├── image/                   # 图像测试集
│   ├── audio/                   # 音频测试集
│   └── test_cases*.json         # 融合引擎测试用例
├── scripts/                     # 训练 / 评估脚本
│   ├── train_ch_text.py         # 中文 XLM-R 训练
│   ├── eval_ch_text.py          # 中文模型评估
│   ├── check_overfit.py         # 过拟合诊断（3-Fold CV）
│   ├── test_adversarial.py      # 对抗样本鲁棒性测试
│   ├── adversarial_guard.py     # 对抗文本检测 Guard
│   ├── normalizer.py            # 对抗文本归一化预处理
│   ├── train_image_vit.py       # 图像模型训练（预留）
│   └── threshold_search.py      # 阈值搜索（预留）
├── frontend/
│   └── streamlit_app.py         # Streamlit 可视化前端
├── results/                     # 训练结果（不提交）
│   └── xlmr_ch/                 # 中文模型训练日志
├── conclude.md                  # 完整项目总结文档
├── 实验报告.md                   # 课程设计实验报告
└── README.md                    # 本文件
```

---

## 实验结果摘要

### 中文文本审核

| 指标 | 数值 |
|------|------|
| Micro F1 | **0.954** |
| Macro F1 | **0.952** |
| 3-Fold CV 标准差 | ±0.011（无过拟合） |
| 阈值稳定性 | 0.3~0.7 范围波动 ±0.002 |

### 图像审核（CLIP + Qwen 双轨）

| 指标 | 数值 |
|------|------|
| 总体正确率 | **91.7%** |
| Unsafe 召回率 | **92.5%** |
| CLIP-Only 误判率 | 4.4% |
| Qwen 触发率 | 43.7%（仅疑图） |
| 总耗时（206 张） | 39.4s |

### 对抗防御

| 指标 | 原始模型 | + Guard |
|------|---------|---------|
| Unsafe 召回率 | 100% | **100%** ✅ |
| Safe 误拦→REVIEW | 0% | **50%** ✅ |

---

## 引用

如果本项目对您的研究或工作有帮助，请参考：

```bibtex
@software{multimodal_safety_system,
  author = {Your Name},
  title = {Multimodal Safety System},
  year = {2026},
  url = {https://github.com/yourname/multimodal_safety_system}
}
```

