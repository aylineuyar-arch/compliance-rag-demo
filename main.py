"""
main.py — Demo entry point for the Compliance RAG pipeline.

Runs 4 representative compliance queries and prints retrieved context
plus Claude's cited answer for each one.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python main.py
"""

import os
from rag_pipeline import RAGPipeline

# Path to the synthetic compliance documents
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")

# Demo queries that span all four source documents
DEMO_QUERIES = [
    "What are the transaction monitoring thresholds for suspicious activity reporting?",
    "What identity documents are required for enhanced due diligence KYC?",
    "How should a compliance officer handle a trade surveillance alert?",
    "What is the approval process for deploying an AI model under the data governance framework?",
]

DIVIDER = "━" * 70


def print_result(result: dict) -> None:
    """Pretty-print a single RAG result: query, retrieved chunks, and answer."""
    print(f"\n{DIVIDER}")
    print(f"QUERY: {result['query']}")
    print(DIVIDER)

    print("\nRETRIEVED CONTEXT:")
    for i, chunk in enumerate(result["chunks"], 1):
        similarity = chunk["similarity"]
        source = chunk["source"]
        # Truncate long chunks for readability — show first 300 chars
        preview = chunk["text"][:300].strip()
        if len(chunk["text"]) > 300:
            preview += "..."
        print(f'\n[{i}] {source} (similarity: {similarity:.2f})')
        print(f'"{preview}"')

    print(f"\nANSWER:")
    print(result["answer"])
    print(DIVIDER)


def main():
    print("=" * 70)
    print("  Compliance RAG Demo — Retrieval-Augmented Generation")
    print("  Financial Services Policy Q&A with Claude + sentence-transformers")
    print("=" * 70)

    # Initialise the pipeline — loads docs, builds embeddings, checks API key
    pipeline = RAGPipeline(DOCUMENTS_DIR)

    print(f"\nRunning {len(DEMO_QUERIES)} demo queries...\n")

    for query in DEMO_QUERIES:
        result = pipeline.ask(query, top_k=3)
        print_result(result)

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
