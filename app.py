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
        st.markdown(f"**{name}**")
        st.caption(desc)

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
    st.title("⚖️ Compliance Policy Q&A")
    st.info("Ask any question about AML, KYC, trade surveillance, or data governance — answers are grounded in internal policy documents.")

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

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a compliance question...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

# ── Confidence helper ─────────────────────────────────────────────────────────
def confidence_label(score: float) -> str:
    if score >= 0.70:
        return "🟢 High confidence"
    elif score >= 0.50:
        return "🟡 Medium confidence"
    else:
        return "🔴 Low confidence"

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

        # Confidence + sources
        top_score = chunks[0]["similarity"] if chunks else 0
        sources = " · ".join(
            c["source"].replace(".txt", "").replace("_", " ").title()
            for c in chunks
        )
        st.caption(f"{confidence_label(top_score)}  |  Sources: {sources}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })
