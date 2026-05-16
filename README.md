# Compliance RAG Demo — Retrieval-Augmented Generation for Financial Services

AI chatbot that answers financial services compliance questions using internal policy documents — built with Claude.

**[Live Demo](https://compliance-rag-demo-mrwtbs4k7gvdvmiuck8mdn.streamlit.app)**

---

## What This Is

Retrieval-Augmented Generation (RAG) is a technique for grounding large language model responses in a specific corpus of documents. Rather than relying on a model's parametric memory — which can be stale, hallucinated, or unverifiable — RAG retrieves the most relevant document excerpts at query time and passes them as context to the model. The model then generates an answer that is directly traceable to source material. This is fundamentally different from fine-tuning: fine-tuning bakes knowledge into model weights (expensive, slow, opaque), while RAG leaves the knowledge in documents (cheap, instant, auditable). For compliance use cases where accuracy and traceability are non-negotiable, RAG is the right architecture.

---

## Why Compliance Is the Highest-Value RAG Use Case in Financial Services

Compliance analysts at financial institutions spend significant time manually searching policy documents, regulatory circulars, and internal procedure manuals to answer operational questions — what's the SAR filing deadline, what documents are required for enhanced due diligence, what's the escalation path for a surveillance alert. These questions are high-stakes (wrong answers create regulatory exposure), highly repetitive, and perfectly structured for RAG: the answer exists verbatim somewhere in internal documentation, it just needs to be found and surfaced. RAG automates this with grounded, auditable answers — creating a paper trail that satisfies internal controls and regulatory examination requirements. This demo shows the core pattern.

---

## Features

- **Streaming responses** — answers stream token by token via Claude's streaming API, not delivered all at once after a pause
- **Conversation memory** — follow-up questions work correctly; the model sees prior turns and interprets context ("what about exceptions to that?")
- **Low-confidence fallback** — if no retrieved chunk scores above the similarity threshold, the system returns an honest "I couldn't find a clear answer" instead of a weak or hallucinated response
- **Cross-document synthesis callout** — when an answer draws from multiple policy areas (e.g. AML + KYC), a banner surfaces this explicitly to show cross-policy reasoning
- **Suggested follow-ups** — 3 clickable follow-up questions generated after each answer, the way a real analyst would continue the conversation
- **Export session as CSV** — download the full Q&A session for audit records, Airtable import, or sharing

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
Confidence Check
    │  If best match < 0.45 similarity, return fallback — no weak answers
    ▼
Top-3 Relevant Chunks Retrieved
    │  Returns most semantically similar passages; flags cross-document synthesis
    ▼
Claude API (claude-haiku-4-5) — streaming generation
    │  Prompt instructs Claude to answer only from retrieved context
    │  Response streams token by token via st.write_stream()
    ▼
Answer + Confidence indicator + Source tags + Follow-up suggestions
```

---

## Why RAG Over Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|---|---|---|
| **Document updates** | Instant — re-index the new document | Requires full retraining run |
| **Auditability** | Every answer traceable to a source chunk | No visibility into what the model "learned" |
| **Cost** | Embedding + inference only | GPU compute for training + inference |
| **Data requirements** | Works with any document corpus | Needs thousands of labelled Q&A pairs |
| **Latency** | Fast — haiku streams in ~1s | Same inference latency, higher training overhead |
| **Failure mode** | Returns honest fallback if answer isn't in docs | May confidently hallucinate if topic wasn't in training data |

Compliance documents change frequently — regulators update thresholds, internal policies are revised, new product lines require new procedures. RAG reflects those changes as soon as the updated document is ingested. Fine-tuning would require a new training run every time a threshold changes, which is operationally impractical.

RAG answers are auditable by design: the retrieved chunks are part of the system output, so a compliance officer or internal auditor can verify exactly which document passage generated the answer. This satisfies the model explainability requirements regulators increasingly impose under SR 11-7 and equivalent guidance.

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

The sentence-transformers model (~90MB) downloads automatically on first run.

**3. Set your Anthropic API key**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

**4a. Run the web UI (recommended)**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Type any compliance question or click an example. Answers stream in real time. Follow-up questions are suggested after each answer. Export the session from the sidebar.

**4b. Run the command-line demo**
```bash
python3 main.py
```

Runs 4 preset queries automatically, then enters an interactive prompt. Results are saved to `results.csv`.

---

## Tech Stack

| Component | Library | Why |
|---|---|---|
| Web UI | `streamlit` | Python-native, deploys to Streamlit Cloud for free |
| Generation | `anthropic` / `claude-haiku-4-5` | Streaming API, fast, cost-efficient |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` | Local inference, no API cost, 384-dim semantic vectors |
| Vector search | `numpy` cosine similarity | Zero overhead for demo-scale corpus |
| Document storage | Plain `.txt` files | Simple, auditable, easy to update |

---

## Where This Would Be Used in Practice

**1. Compliance help desk automation**
Front office asks repetitive policy questions — "can we onboard this entity type?", "what's the wire reporting threshold?" — today answered by a junior analyst manually searching PDFs. RAG answers in seconds. Estimated to cover 30–50% of tier-1 compliance queries.

**2. Regulatory change management**
When FinCEN updates a threshold or FATF revises a country list, compliance teams manually identify which internal policies are affected. RAG over old and new regulatory text surfaces the gaps automatically — days of work reduced to minutes.

**3. Audit and examination support**
During a regulatory exam, examiners ask banks to produce evidence that specific policies exist and are followed. RAG instantly surfaces the relevant clause and its source — faster examination response, less scrambling.

**4. Onboarding workflow guidance**
A relationship manager onboarding an unfamiliar client type (foreign PEP, crypto MSB) queries the system in plain English and gets a step-by-step requirements checklist — instead of escalating to Compliance or reading a 40-page manual.

**5. Employee self-service compliance portal**
Any employee can ask "am I allowed to trade this security?" or "what's the personal account dealing holding period?" — with answers grounded in current internal policy, not general LLM knowledge.

**Why this matters for strategy roles:** Compliance headcount is expensive, regulatory risk is existential, and these queries are high-volume and highly repetitive. RAG is one of the few AI patterns where the ROI is immediate and the auditability requirement is already met by design.

---

## Project Structure

```
compliance-rag-demo/
├── README.md                    # This file
├── requirements.txt             # anthropic, sentence-transformers, numpy, streamlit
├── app.py                       # Streamlit chatbot — streaming, memory, export, follow-ups
├── main.py                      # CLI demo — 4 preset queries, exports results.csv
├── rag_pipeline.py              # RAGPipeline: embed, retrieve, stream, confidence check
├── utils.py                     # Document loading and chunking helpers
└── documents/
    ├── aml_policy.txt           # Synthetic AML policy (SAMPLE/FICTIONAL)
    ├── kyc_guidelines.txt       # Synthetic KYC guidelines (SAMPLE/FICTIONAL)
    ├── trade_compliance.txt     # Synthetic trade surveillance policy (SAMPLE/FICTIONAL)
    └── data_governance.txt      # Synthetic data governance framework (SAMPLE/FICTIONAL)
```

---

*All compliance documents in this repository are synthetic and fictional, created for demonstration purposes only. They do not represent the policies of any real institution.*
