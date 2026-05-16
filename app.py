"""
app.py — Streamlit chatbot UI for the Compliance RAG Demo.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Compliance Policy Q&A",
    page_icon="⚖️",
    layout="wide",
)

# Inject minimal CSS — tighten spacing, style chips
st.markdown("""
<style>
    /* Tighten top padding */
    .block-container { padding-top: 1.5rem !important; }

    /* Chip-style tags */
    .chip {
        display: inline-block;
        background: #1a1f2e;
        border: 1px solid #4f8ef7;
        color: #4f8ef7;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Compliance RAG")
    st.caption("AI-powered policy Q&A · Financial Services")
    st.divider()

    st.markdown("**Indexed documents**")
    docs = [
        ("🔍", "Anti-Money Laundering", "Thresholds, SARs, CDD"),
        ("🪪", "Know Your Customer", "Verification, risk tiers"),
        ("📈", "Trade Surveillance", "Monitoring, escalation"),
        ("🗄️", "Data Governance & AI", "Classification, model approval"),
    ]
    for icon, name, desc in docs:
        st.markdown(f"{icon} **{name}**")
        st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;{desc}")

    st.divider()
    st.markdown("[![GitHub](https://img.shields.io/badge/View_on-GitHub-181717?logo=github)](https://github.com/aylineuyar-arch/compliance-rag-demo)")
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Documents are synthetic/fictional.")

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

# ── Empty state ───────────────────────────────────────────────────────────────
EXAMPLES = [
    "What are the transaction monitoring thresholds for suspicious activity?",
    "What documents are required for enhanced due diligence KYC?",
    "How should a compliance officer handle a trade surveillance alert?",
    "What is the AI model approval process under the data governance framework?",
]

if not st.session_state.messages:
    st.markdown("# Compliance Policy Q&A")
    st.markdown("Ask any operational compliance question. Answers are retrieved directly from indexed policy documents — no hallucination, grounded in the exact text.")

    # Real stats from the pipeline
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", getattr(pipeline, "num_documents", 4))
    c2.metric("Chunks indexed", getattr(pipeline, "num_chunks", "—"))
    c3.metric("Embedding dims", getattr(pipeline, "embedding_dim", 384))
    c4.metric("Generation model", "Claude Haiku")

    st.divider()

    # How it works expander
    with st.expander("How this works", expanded=False):
        st.markdown("""
**Retrieval-Augmented Generation (RAG)** — not fine-tuning, not general LLM knowledge.

1. **Ingest** — each policy document is split into 200-word chunks with 50-word overlap to avoid losing context at boundaries.
2. **Embed** — every chunk is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`, a local embedding model. No API call, no data leaving your machine.
3. **Retrieve** — your question is embedded with the same model. Cosine similarity is computed against all chunk vectors in memory (numpy). The top 3 most relevant passages are returned.
4. **Generate** — retrieved passages are injected into a structured prompt. Claude answers strictly from that context. If the answer isn't in the documents, it says so.

**Why this matters for compliance:** Every answer is traceable to a specific policy passage. That's the audit trail regulators and internal controls require — something a fine-tuned model can't provide.
        """)

    st.markdown("**Example questions — click to ask:**")
    col1, col2 = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if (col1 if i % 2 == 0 else col2).button(example, use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("$", r"\$"))

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a compliance question...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

# ── Process query ─────────────────────────────────────────────────────────────
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
            # Pass prior turns so Claude has conversation memory
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # exclude current user turn
            ]
            answer = pipeline.generate(query, chunks, chat_history=chat_history)

            st.write("Generating follow-up questions...")
            followups = pipeline.suggest_followups(query, answer)
            status.update(label="Done", state="complete", expanded=False)

        st.markdown(answer.replace("$", r"\$"))

        # Confidence + source chips
        top_score = chunks[0]["similarity"] if chunks else 0
        if top_score >= 0.70:
            conf = "🟢 High relevance"
        elif top_score >= 0.50:
            conf = "🟡 Moderate relevance"
        else:
            conf = "🔴 Low relevance — answer may be incomplete"

        chips = "".join(
            f'<span class="chip">{c["source"].replace(".txt","").replace("_"," ").title()} {c["similarity"]:.0%}</span>'
            for c in chunks
        )
        st.markdown(f"<div style='margin-top:0.4rem'>{conf} &nbsp; {chips}</div>", unsafe_allow_html=True)

        # Follow-up question suggestions
        if followups:
            st.markdown("**You might also ask:**")
            fu_cols = st.columns(len(followups))
            for i, fq in enumerate(followups):
                if fu_cols[i].button(fq, key=f"fu_{len(st.session_state.messages)}_{i}", use_container_width=True):
                    st.session_state.pending_query = fq
                    st.rerun()

    st.session_state.messages.append({"role": "assistant", "content": answer})
