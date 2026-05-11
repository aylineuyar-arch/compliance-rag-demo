# Compliance RAG Demo — Retrieval-Augmented Generation for Financial Services

A working end-to-end RAG pipeline that answers compliance questions using synthetic financial services policy documents, with cited, auditable answers powered by Claude.

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
Claude API (claude-haiku-4-5) — grounded generation with citations
    │  Prompt instructs Claude to answer only from retrieved context
    ▼
Answer + Source Citations
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

**4. Run the demo**
```bash
python main.py
```

The pipeline will load and chunk the four synthetic compliance documents, embed all chunks locally, then run four demo queries and print retrieved context plus Claude's cited answer for each.

---

## Sample Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUERY: What are the transaction monitoring thresholds for suspicious activity reporting?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RETRIEVED CONTEXT:
[1] aml_policy.txt (similarity: 0.87)
"Cash transactions of $10,000 or more in a single business day require Currency
Transaction Report (CTR) filing with FinCEN within 15 calendar days..."

[2] kyc_guidelines.txt (similarity: 0.71)
"..."

[3] data_governance.txt (similarity: 0.63)
"..."

ANSWER:
Based on the AML Policy [aml_policy.txt], transaction monitoring thresholds include:
cash transactions of $10,000 or more trigger a mandatory CTR filing within 15 days;
structured transactions where cumulative cash activity exceeds $9,000 across two or
more transactions are flagged; wire transfers of $3,000 or more require Travel Rule
recordkeeping; and unusual activity exceeding 300% of the customer's 90-day average
is escalated within 48 hours.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Tech Stack

| Component | Library | Why |
|---|---|---|
| Generation | `anthropic` / `claude-haiku-4-5` | Fast, cost-efficient, strong instruction-following |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` | Local inference, no API cost, strong semantic search |
| Vector search | `numpy` (cosine similarity) | Zero overhead for demo-scale corpus (~100 chunks) |
| Document storage | Plain `.txt` files | Simple, auditable, easy to update |

---

## Use Cases This Generalises To

- **Trade operations policy lookup** — "What's the settlement fail escalation procedure for equity trades?"
- **Onboarding workflow guidance** — "What documents are needed to onboard a PEP client in the UK?"
- **Regulatory change impact analysis** — Ingest updated regulatory text and query for differences vs. current internal policy
- **Internal audit support** — Map audit findings to specific policy clauses automatically
- **Employee compliance Q&A** — Self-service portal for staff to query procedures without emailing Compliance

---

## Project Structure

```
compliance-rag-demo/
├── README.md                    # This file
├── requirements.txt             # anthropic, sentence-transformers, numpy
├── main.py                      # Entry point — runs the 4 demo queries
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
