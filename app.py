"""
app.py — Streamlit chatbot UI for the Compliance RAG Demo.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from rag_pipeline import RAGPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Compliance RAG Demo",
    page_icon="⚖️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { font-size: 2rem; margin: 0 0 0.4rem 0; font-weight: 700; }
.hero p  { font-size: 1rem; margin: 0; opacity: 0.8; }

.doc-card {
    background: #f8f9fa;
    border-left: 3px solid #0f3460;
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
}

.confidence-high   { color: #28a745; font-weight: 600; }
.confidence-medium { color: #fd7e14; font-weight: 600; }
.confidence-low    { color: #dc3545; font-weight: 600; }

.source-badge {
    display: inline-block;
    background: #e9ecef;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin-right: 4px;
    color: #495057;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Policy Documents")
    docs = {
        "Anti-Money Laundering": "Monitoring thresholds, SAR filing, CDD procedures",
        "Know Your Customer": "Identity verification, risk tiers, document refresh",
        "Trade Surveillance": "Pre/post-trade monitoring, restricted list, escalation",
        "Data Governance & AI": "Data classification, access control, AI model approval",
    }
    for name, desc in docs.items():
        st.markdown(f"""<div class="doc-card"><strong>{name}</strong><br><span style="color:#6c757d">{desc}</span></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**[View on GitHub](https://github.com/aylineuyar-arch/compliance-rag-demo)**")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("All documents are synthetic/fictional.")

# ── Load pipeline ─────────────────────────────────────────────────────────────
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")

@st.cache_resource(show_spinner="Loading policy documents...")
def load_pipeline():
    return RAGPipeline(DOCUMENTS_DIR)

try:
    pipeline = load_pipeline()
except ValueError as e:
    st.error(f"**Setup error:** {e}")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Hero + examples (shown only on empty state) ───────────────────────────────
EXAMPLES = [
    "What are the transaction monitoring thresholds for suspicious activity?",
    "What documents are required for enhanced due diligence KYC?",
    "How should a compliance officer handle a trade surveillance alert?",
    "What is the AI model approval process under the data governance framework?",
]

if not st.session_state.messages:
    st.markdown("""
    <div class="hero">
        <h1>⚖️ Compliance Policy Q&A</h1>
        <p>Ask any question about AML, KYC, trade surveillance, or data governance — answers are grounded in internal policy documents.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Try an example question:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()

# ── Render existing chat messages ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("$", r"\$"))
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            st.markdown(meta, unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a compliance question...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

# ── Confidence helper ─────────────────────────────────────────────────────────
def confidence_label(score: float) -> str:
    if score >= 0.70:
        return f'<span class="confidence-high">● High confidence</span>'
    elif score >= 0.50:
        return f'<span class="confidence-medium">● Medium confidence</span>'
    else:
        return f'<span class="confidence-low">● Low confidence</span>'

# ── Process new query ─────────────────────────────────────────────────────────
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.status("Gathering information...", expanded=True) as status:
            st.write("Searching policy documents...")
            chunks = pipeline.retrieve(query, top_k=3)

            st.write("Reading relevant sections...")
            for chunk in chunks:
                source = chunk["source"].replace(".txt", "").replace("_", " ").title()
                st.caption(f"— {source}")

            st.write("Preparing your answer...")
            answer = pipeline.generate(query, chunks)
            status.update(label="Here's what I found", state="complete", expanded=False)

        st.markdown(answer.replace("$", r"\$"))

        # Confidence + source badges
        top_score = chunks[0]["similarity"] if chunks else 0
        source_badges = " ".join(
            f'<span class="source-badge">{c["source"].replace(".txt","").replace("_"," ").title()}</span>'
            for c in chunks
        )
        meta = f'{confidence_label(top_score)} &nbsp;·&nbsp; {source_badges}'
        st.markdown(f"<div style='margin-top:0.5rem;'>{meta}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": f"<div style='margin-top:0.5rem;'>{meta}</div>",
    })
