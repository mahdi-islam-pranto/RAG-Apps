import streamlit as st
import requests

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Endpoints ─────────────────────────────────────────────────────────────
UPLOAD_URL = "http://114.130.69.239:5020/upload"
ASK_URL    = "http://114.130.69.239:5021/ask"

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Force visibility - fix blank page */
.main .block-container {
    opacity: 1 !important;
    visibility: visible !important;
}

/* Dark theme colors */
:root {
    --accent:   #5b8cff;
    --accent2:  #a78bfa;
    --success:  #34d399;
    --danger:   #f87171;
    --muted:    #64748b;
    --border:   #1f2433;
    --surface:  #13161e;
    --user-bg:  #1a2236;
    --bot-bg:   #151924;
}

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #13161e;
    border-right: 1px solid #1f2433;
}

/* Hide default Streamlit footer */
footer { visibility: hidden; }

/* Buttons */
.stButton > button {
    font-weight: 700 !important;
    font-size: 0.87rem !important;
    background: linear-gradient(135deg, #3b6fd4, #6d4fc7) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
    width: 100%;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #1f2433 !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}

/* Text input */
.stTextInput input {
    border: 1.5px solid #1f2433 !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
}
.stTextInput input:focus {
    border-color: #5b8cff !important;
    box-shadow: 0 0 0 3px rgba(91,140,255,0.15) !important;
}

/* Chat bubbles */
.msg-user {
    background: #1a2236;
    border: 1px solid #1e3550;
    border-radius: 14px 14px 4px 14px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0 0.4rem 4rem;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #e2e8f0;
}
.msg-bot {
    background: #151924;
    border: 1px solid #1f2433;
    border-radius: 14px 14px 14px 4px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 4rem 0.4rem 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #e2e8f0;
}
.msg-label-user {
    text-align: right;
    font-size: 0.72rem;
    color: #64748b;
    margin: 0.15rem 0.25rem 0 0;
}
.msg-label-bot {
    font-size: 0.72rem;
    color: #64748b;
    margin: 0.15rem 0 0 0.25rem;
}
.token-meta {
    font-size: 0.72rem;
    color: #475569;
    margin: 0.2rem 0 0 0.25rem;
}

/* File tag */
.file-tag {
    display: inline-block;
    background: rgba(91,140,255,0.12);
    border: 1px solid rgba(91,140,255,0.3);
    color: #93c5fd;
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.8rem;
    margin: 0.2rem 0.15rem;
}

/* Status chips */
.chip-success {
    display: inline-block;
    background: rgba(52,211,153,0.12);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 99px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    font-weight: 500;
}
.chip-error {
    display: inline-block;
    background: rgba(248,113,113,0.12);
    color: #f87171;
    border: 1px solid rgba(248,113,113,0.3);
    border-radius: 99px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Divider */
.hl { border: none; border-top: 1px solid #1f2433; margin: 1rem 0; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #475569;
    border: 1.5px dashed #1f2433;
    border-radius: 14px;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Document Upload
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG Assistant")
    st.caption("Retrieval-Augmented Generation")
    st.markdown('<hr class="hl">', unsafe_allow_html=True)

    st.markdown("**📂 Upload Documents**")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⬆ Upload to Knowledge Base"):
        if not uploaded_files:
            st.warning("Please select at least one file first.", icon="⚠️")
        else:
            with st.spinner("Uploading…"):
                try:
                    files_payload = [
                        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                        for f in uploaded_files
                    ]
                    resp = requests.post(UPLOAD_URL, files=files_payload, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()

                    names = data.get("uploaded_files", [f.name for f in uploaded_files])
                    for n in names:
                        if n not in st.session_state.uploaded_docs:
                            st.session_state.uploaded_docs.append(n)

                    pages = data.get("Length of loaded document pages", "—")
                    st.markdown(
                        f'<div class="chip-success">✓ {len(names)} file(s) · {pages} pages indexed</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="chip-error">✗ Upload failed: {e}</div>',
                        unsafe_allow_html=True,
                    )

    # Knowledge base list
    if st.session_state.uploaded_docs:
        st.markdown('<hr class="hl">', unsafe_allow_html=True)
        st.markdown("**📚 Knowledge Base**")
        tags = "".join(
            f'<span class="file-tag">📄 {name}</span>'
            for name in st.session_state.uploaded_docs
        )
        st.markdown(tags, unsafe_allow_html=True)
        st.markdown("")
        if st.button("🗑 Clear All"):
            st.session_state.uploaded_docs = []
            st.session_state.messages = []
            st.rerun()

    st.markdown('<hr class="hl">', unsafe_allow_html=True)
    st.caption("Supported: PDF · DOCX · TXT\n\nUpload documents first, then ask questions in the chat.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Chat Interface
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 💬 Ask Your Documents")
st.caption("Upload files to the knowledge base, then ask anything — answers are grounded in your content.")
st.markdown('<hr class="hl">', unsafe_allow_html=True)

# Render chat messages
if st.session_state.messages:
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown('<div class="msg-label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-user">{m["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="msg-label-bot">🤖 Assistant</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-bot">{m["content"]}</div>', unsafe_allow_html=True)
            if m.get("tokens"):
                t = m["tokens"]
                st.markdown(
                    f'<div class="token-meta">📥 {t.get("input_tokens","—")} in &nbsp;·&nbsp; '
                    f'📤 {t.get("output_tokens","—")} out &nbsp;·&nbsp; '
                    f'Σ {t.get("total_tokens","—")} total</div>',
                    unsafe_allow_html=True,
                )
else:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:2.2rem; margin-bottom:0.5rem;">💬</div>
        <p style="margin:0; font-size:0.9rem;">No messages yet.<br>Upload documents and ask your first question below.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="hl">', unsafe_allow_html=True)

# ── Input Row ─────────────────────────────────────────────────────────────────
col_q, col_btn = st.columns([5, 1])
with col_q:
    question = st.text_input(
        "Question",
        placeholder="e.g. What is the return policy?",
        label_visibility="collapsed",
        key="question_input",
    )
with col_btn:
    send = st.button("Send ➤")

# ── Handle send ───────────────────────────────────────────────────────────────
if send:
    if not question.strip():
        st.warning("Please type a question first.", icon="⚠️")
    else:
        q = question.strip()
        st.session_state.messages.append({"role": "user", "content": q})

        with st.spinner("Thinking…"):
            try:
                resp = requests.post(ASK_URL, data={"text": q}, timeout=60)
                resp.raise_for_status()
                body = resp.json()
                answer = (
                    body.get("data", {}).get("response")
                    or body.get("answer")
                    or "No answer returned."
                )
                meta = body.get("metadata", {})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "tokens": meta,
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error contacting the API: {e}",
                    "tokens": {},
                })

        st.rerun()