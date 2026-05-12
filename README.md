# Compliance RAG Demo — Retrieval-Augmented Generation for Financial Services

A working end-to-end RAG pipeline that answers compliance questions using synthetic financial services policy documents, with clean conversational answers powered by Claude.

---

## What This Is

Retrieval-Augmented Generation (RAG) is a technique for grounding large language model responses in a specific corpus of documents. Rather than relying on a model's parametric memory — which can be stale, hallucinated, or unverifiable — RAG retrieves the most relevant document excerpts at query time and passes them as context to the model. The model then generates an answer that is directly traceable to source material. This is fundamentally different from fine-tuning: fine-tuning bakes knowledge into model weights (expensive, slow, opaque), while RAG leaves the knowledge in documents (cheap, instant, auditable). For compliance use cases where accuracy and traceability are non-negotiable, RAG is the right architecture.

---

## Why Compliance Is the Highest-Value RAG Use Case in Financial Services

Compliance analysts at financial institutions spend significant time manually searching policy documents, regulatory circulars, and internal procedure manuals to answer operational questions — what's the SAR filing deadline, what documents are required for enhanced due diligence, what's the escalation path for a surveillance alert. These questions are high-stakes (wrong answers create regulatory exposure), highly repetitive, and perfectly structured for RAG: the answer exists verbatim somewhere in internal documentation, it just needs to be found and surfaced. RAG automates this with answers that cite the exact source document and passage, creating an audit trail that satisfies internal controls and regulatory examination requirements. This demo shows the core pattern.

---

## Architecture

```
User Query
    │
    ▼
Embedding Model (sentence-transformers / all-MiniLM-L6-v2)
    │  Encodes query into a 384-dimensional vector — runs locally, no API cost
    ▼
Cosine Similarity Search over Document Chunks
    │  Compares query vector against all pre-indexed chunk vectors (numpy)
    ▼
Top-K Relevant Chunks Retrieved
    │  Returns the 3 most semantically similar passages with source filenames
    ▼
Claude API (claude-haiku-4-5) — grounded generation
    │  Prompt instructs Claude to answer only from retrieved context
    ▼
Clean conversational answer
```

---

## Why RAG Over Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|---|---|---|
| **Document updates** | Instant — re-index the new document | Requires full retraining run |
| **Auditability** | Every answer cites its source chunk | No visibility into what the model "learned" |
| **Cost** | Embedding + inference only | GPU compute for training + inference |
| **Data requirements** | Works with any document corpus | Needs thousands of labelled Q&A pairs |
| **Latency** | Fast (haiku is ~1–2s) | Same inference latency, higher training overhead |
| **Failure mode** | Returns "not enough context" if answer isn't in docs | May confidently hallucinate if topic wasn't in training data |

Compliance documents change frequently — regulators update thresholds, internal policies are revised, new product lines require new procedures. RAG reflects those changes as soon as the updated document is ingested. Fine-tuning would require a new training run every time a threshold changes, which is operationally impractical.

RAG answers are also auditable by design: the retrieved chunks are part of the system output, so a compliance officer or internal auditor can verify exactly which document passage generated the answer. This satisfies the kind of model explainability requirements that regulators increasingly impose under SR 11-7 (model risk management) and equivalent guidance.

---

## How to Run

**Prerequisites:** Python 3.9+

**1. Clone and navigate to the project**
```bash
git clone https://github.com/aylineuyar-arch/compliance-rag-demo.git
cd compliance-rag-demo
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

The sentence-transformers model (~90MB) will download automatically on first run.

**3. Set your Anthropic API key**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

**4a. Run the web UI (recommended)**
```bash
streamlit run app.py
```

Opens a chatbot interface in your browser at `http://localhost:8501`. Type any compliance question or click an example. The UI shows retrieval steps in real time — searching documents, reading relevant sections, then delivering a clean conversational answer.

**4b. Run the command-line demo**
```bash
python3 main.py
```

Runs 4 preset queries automatically, then enters an interactive prompt. Results are saved to `results.csv` for import into Airtable or Google Sheets.

---

## Tech Stack

| Component | Library | Why |
|---|---|---|
| Web UI | `streamlit` | Python-native, deploys to Streamlit Cloud for free |
| Generation | `anthropic` / `claude-haiku-4-5` | Fast, cost-efficient, strong instruction-following |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` | Local inference, no API cost, strong semantic search |
| Vector search | `numpy` (cosine similarity) | Zero overhead for demo-scale corpus (~100 chunks) |
| Document storage | Plain `.txt` files | Simple, auditable, easy to update |

---

## Where This Would Be Used in Practice

This demo shows the core pattern that would underpin several real workflows at a financial institution:

**1. Compliance help desk automation**
Today, junior compliance analysts field repetitive policy questions from the front office — "can we onboard this type of entity?", "what's the wire reporting threshold?". A RAG system answers these instantly, with citations, freeing analysts for higher-judgment work. Estimated time savings: 30–50% of tier-1 compliance queries.

**2. Regulatory change management**
When a regulator issues updated guidance (new FinCEN thresholds, revised FATF country lists), compliance teams must identify which internal policies are affected. RAG over both old and new regulatory text surfaces the delta automatically, reducing manual gap analysis from days to minutes.

**3. Audit and examination support**
During a regulatory examination, examiners ask banks to produce evidence that specific policies exist and are followed. A RAG system over internal policy documents can instantly surface the relevant clause and its source, supporting faster examination response.

**4. Onboarding workflow guidance**
Relationship managers onboarding a new client type they haven't seen before (a foreign PEP, a crypto MSB) can query the KYC system in plain English and get a step-by-step requirements list with document citations — instead of escalating to Compliance or reading 40-page manuals.

**5. Employee self-service compliance portal**
A bank-wide internal chatbot where any employee can ask "am I allowed to trade this security?" or "what's the personal account dealing holding period?" — with answers grounded in current policy, not general LLM knowledge.

**Why this matters for strategy roles:** The business case is straightforward — compliance headcount is expensive, regulatory risk is existential, and these queries are high-volume and highly repetitive. RAG is one of the few AI patterns where the ROI is immediate and the auditability requirement is already met by design.

---

## Project Structure

```
compliance-rag-demo/
├── README.md                    # This file
├── requirements.txt             # anthropic, sentence-transformers, numpy, streamlit
├── app.py                       # Streamlit chatbot UI — step-by-step retrieval + chat history
├── main.py                      # CLI demo — runs 4 preset queries, exports results.csv
├── rag_pipeline.py              # RAGPipeline class: embed, retrieve, generate
├── utils.py                     # Document loading and chunking helpers
└── documents/
    ├── aml_policy.txt           # Synthetic AML policy (SAMPLE/FICTIONAL)
    ├── kyc_guidelines.txt       # Synthetic KYC guidelines (SAMPLE/FICTIONAL)
    ├── trade_compliance.txt     # Synthetic trade surveillance policy (SAMPLE/FICTIONAL)
    └── data_governance.txt      # Synthetic data governance framework (SAMPLE/FICTIONAL)
```

---

*All compliance documents in this repository are synthetic and fictional, created for demonstration purposes only. They do not represent the policies of any real institution.*
