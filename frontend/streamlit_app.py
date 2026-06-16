"""
多模态安全审核系统 — Streamlit 前端

用法:
    pip install streamlit requests
    streamlit run frontend/streamlit_app.py

确保后端已启动:
    python -m backend.main --port 8000
"""

import os

import requests
import streamlit as st

# ======================
# 配置
# ======================
API_URL = os.getenv("API_URL", "http://localhost:8000")
BACKEND_HEALTH_URL = f"{API_URL}/health"
PREDICT_URL = f"{API_URL}/predict/multimodal"

# ======================
# 页面配置
# ======================
st.set_page_config(
    page_title="多模态安全审核系统",
    page_icon="🛡️",
    layout="wide",
)

# ======================
# 自定义 CSS
# ======================
st.markdown("""
<style>
    .decision-card {
        padding: 2rem 1.5rem; border-radius: 16px; text-align: center;
        margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .decision-card h1 { font-size: 3rem; margin: 0; font-weight: 800; }
    .decision-card .sub { font-size: 1rem; opacity: 0.85; margin-top: 0.3rem; }
    .decision-safe    { background: linear-gradient(135deg, #22c55e, #16a34a); color: white; }
    .decision-review  { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
    .decision-unsafe  { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
    .decision-none    { background: linear-gradient(135deg, #6b7280, #4b5563); color: white; }
    .score-bar-container {
        background: #e5e7eb; border-radius: 999px; height: 20px;
        width: 100%; margin: 0.5rem 0; position: relative; overflow: hidden;
    }
    .score-bar-fill {
        height: 100%; border-radius: 999px; transition: width 0.6s ease;
    }
    .score-bar-text {
        position: absolute; right: 8px; top: 50%;
        transform: translateY(-50%); font-size: 0.75rem; font-weight: 700;
    }
    .metric-box {
        text-align: center; padding: 0.8rem; background: white;
        border-radius: 10px; border: 1px solid #e5e7eb;
    }
    .metric-box .label { font-size: 0.75rem; color: #6b7280; }
    .metric-box .value { font-size: 1.3rem; font-weight: 700; color: #111827; }
    .modal-tag {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600;
    }
    .tag-safe   { background: #dcfce7; color: #166534; }
    .tag-review { background: #fef3c7; color: #92400e; }
    .tag-unsafe { background: #fecaca; color: #991b1b; }
    hr { margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ======================
# 工具函数
# ======================
def check_backend() -> bool:
    try:
        r = requests.get(BACKEND_HEALTH_URL, timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def decision_emoji(d: str) -> str:
    return {"SAFE": "🟢", "REVIEW": "🟡", "UNSAFE": "🔴"}.get(d, "⚪")


def decision_css(d: str) -> str:
    return {"SAFE": "decision-safe", "REVIEW": "decision-review",
            "UNSAFE": "decision-unsafe"}.get(d, "decision-none")


def tag_html(d: str) -> str:
    cls = {"SAFE": "tag-safe", "REVIEW": "tag-review",
           "UNSAFE": "tag-unsafe"}.get(d, "tag-safe")
    return f'<span class="modal-tag {cls}">{decision_emoji(d)} {d}</span>'


def score_bar(score: float) -> str:
    pct = min(max(score * 100, 0), 100)
    color = "#22c55e" if score < 0.4 else "#f59e0b" if score < 0.8 else "#ef4444"
    text_color = "#fff" if score > 0.5 else "#374151"
    return f"""
    <div class="score-bar-container">
        <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
        <span class="score-bar-text" style="color:{text_color}">{score:.2f}</span>
    </div>"""


def logic_label(logic: str) -> str:
    return {
        "multi_modal_agreement": "多模态一致 → 强化",
        "modality_conflict": "多模态冲突 → 保守",
        "single_modal_high_conf": "单模态高置信",
        "single_modal_low_conf": "单模态低置信 → 降级",
        "single_modal_safe": "单模态安全",
        "all_safe": "全部安全",
        "no_modality": "无输入",
    }.get(logic, logic)


def icon_of(modality: str) -> str:
    return {"text": "📝", "image": "🖼️", "audio": "🎵"}.get(modality, "📄")


# ======================
# Header
# ======================
c1, c2, c3 = st.columns([0.08, 0.72, 0.2])
with c1:
    st.markdown("# 🛡️")
with c2:
    st.markdown("## 多模态安全审核系统")
    st.caption("文本 · 图像 · 音频 — 融合判定")
with c3:
    ok = check_backend()
    color, text = ("#22c55e", "● 在线") if ok else ("#ef4444", "● 离线")
    st.markdown(f'<div style="text-align:right;padding:0.5rem">'
                f'<span style="color:{color}">{text}</span></div>',
                unsafe_allow_html=True)

st.divider()

# ======================
# 输入区
# ======================
st.subheader("📥 输入内容（至少提供一个模态）")
col_t, col_i, col_a = st.columns(3)

with col_t:
    st.markdown("##### 📝 文本")
    text_val = st.text_area("文本内容", placeholder="输入或粘贴文字…",
                            height=140, label_visibility="collapsed")
    st.markdown(f"<small>{'✅ 已输入 · ' + str(len(text_val)) + ' 字' if text_val else '⏳ 未输入'}</small>",
                unsafe_allow_html=True)

with col_i:
    st.markdown("##### 🖼️ 图片")
    img_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
    st.markdown(f"<small>{'✅ ' + img_file.name if img_file else '⏳ 未上传'}</small>",
                unsafe_allow_html=True)

with col_a:
    st.markdown("##### 🎵 音频")
    aud_file = st.file_uploader("上传音频", type=["wav", "mp3", "m4a", "ogg"],
                                label_visibility="collapsed")
    st.markdown(f"<small>{'✅ ' + aud_file.name if aud_file else '⏳ 未上传'}</small>",
                unsafe_allow_html=True)

# ======================
# 审核按钮
# ======================
btn = st.button("🚀 开始审核", type="primary", use_container_width=True,
                disabled=not (text_val or img_file or aud_file))
st.divider()

# ======================
# 结果区
# ======================
if btn:
    if not check_backend():
        st.error("❌ 后端未启动，请先运行 `python -m backend.main --port 8000`")
        st.stop()

    with st.spinner("🔄 审核中，请稍候…"):
        files = {}
        if img_file:
            files["image"] = (img_file.name, img_file.read(), img_file.type)
        if aud_file:
            files["audio"] = (aud_file.name, aud_file.read(), aud_file.type)

        try:
            r = requests.post(PREDICT_URL,
                              data={"text": text_val or ""},
                              files=files or None, timeout=60)
            r.raise_for_status()
            res = r.json()
        except Exception as e:
            st.error(f"❌ API 请求失败: {e}")
            st.stop()

    # ---- 解析结果 ----
    decision = res["decision"]
    score = res["score"]
    details = res.get("details", {})
    fusion_logic = details.get("fusion_logic", "")
    fusion_boost = details.get("fusion_boost", 1.0)
    policy = details.get("policy", {})
    modalities = details.get("modalities", [])
    explanation = res.get("explanation", "")

    # ======================
    # 决策大卡片
    # ======================
    st.markdown(f'''
    <div class="decision-card {decision_css(decision)}">
        <h1>{decision_emoji(decision)} {decision}</h1>
        <div class="sub">{policy.get("rule_applied", "")}</div>
    </div>''', unsafe_allow_html=True)

    # ======================
    # 自然语言解释
    # ======================
    if explanation:
        with st.container(border=True):
            st.markdown("**💬 审核说明**")
            st.markdown(explanation)

    # ======================
    # 指标行
    # ======================
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="label">风险分数</div>'
                    f'<div class="value">{score:.2f}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="label">融合逻辑</div>'
                    f'<div class="value" style="font-size:0.9rem">{logic_label(fusion_logic)}</div></div>',
                    unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="label">融合系数</div>'
                    f'<div class="value">×{fusion_boost}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="label">模态数</div>'
                    f'<div class="value">{len(modalities)}</div></div>', unsafe_allow_html=True)

    # ======================
    # 分数进度条
    # ======================
    st.markdown(score_bar(score), unsafe_allow_html=True)
    #st.caption("🟢 安全 (0.0~0.4)   🟡 需复审 (0.4~0.8)   🔴 危险 (0.8~1.0)")
    st.divider()

    # ======================
    # 模态明细表
    # ======================
    if modalities:
        st.subheader("🔍 各模态明细")
        rows_html = ""
        for m in modalities:
            mod = m.get("modality", "")
            rows_html += f"""
            <tr style="border-bottom:1px solid #f3f4f6">
                <td style="padding:0.6rem">{icon_of(mod)} {mod}</td>
                <td style="padding:0.6rem">{tag_html(m.get("decision", ""))}</td>
                <td style="padding:0.6rem">{m.get("score", 0):.2f}</td>
                <td style="padding:0.6rem">{m.get("risk_type", "")}</td>
                <td style="padding:0.6rem">{", ".join(m.get("labels", []))[:60] or "—"}</td>
            </tr>"""
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
            <thead><tr style="background:#f3f4f6;border-bottom:2px solid #e5e7eb">
                <th style="padding:0.6rem;text-align:left">模态</th>
                <th style="padding:0.6rem;text-align:left">决策</th>
                <th style="padding:0.6rem;text-align:left">风险分数</th>
                <th style="padding:0.6rem;text-align:left">风险类型</th>
                <th style="padding:0.6rem;text-align:left">标签</th>
            </tr></thead>
            <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)

        st.divider()

        # ======================
        # 原始详情（可折叠）
        # ======================
        with st.expander("📋 查看原始详情"):
            for m in modalities:
                mod = m.get("modality", "")
                st.markdown(f"**{icon_of(mod)} {mod.upper()}**")
                ca, cb = st.columns(2)
                with ca:
                    st.json({
                        "decision": m.get("decision"),
                        "score": m.get("score"),
                        "risk_type": m.get("risk_type"),
                        "labels": m.get("labels"),
                    })
                with cb:
                    extra = {}
                    if mod == "text" and "probs" in m:
                        extra["probs"] = m["probs"]
                    if mod == "image" and "clip_scores" in m:
                        extra["clip_scores"] = m["clip_scores"]
                    if mod == "audio":
                        if "transcription" in m:
                            extra["transcription"] = m["transcription"]
                    st.json(extra) if extra else st.caption("无额外信息")

            st.markdown("**⚙️ 融合引擎**")
            st.json({
                "fusion_logic": fusion_logic,
                "fusion_boost": fusion_boost,
                "policy": policy,
            })

elif not check_backend():
    st.info("💡 后端尚未启动。请先在终端运行：\n\n```bash\npython -m backend.main --port 8000```")
else:
    st.info("👆 选择至少一个模态输入，然后点击「开始审核」")
