"""
main.py — Interactive Compliance RAG demo.

Runs 4 demo queries automatically, then enters an interactive loop
where you can type your own questions. All results are saved to
results.csv for import into Airtable or Google Sheets.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python3 main.py
"""

import os
import csv
from datetime import datetime
from rag_pipeline import RAGPipeline

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
CSV_FILE = "results.csv"

DEMO_QUERIES = [
    "What are the transaction monitoring thresholds for suspicious activity reporting?",
    "What identity documents are required for enhanced due diligence KYC?",
    "How should a compliance officer handle a trade surveillance alert?",
    "What is the approval process for deploying an AI model under the data governance framework?",
]

DIVIDER = "━" * 70


def print_result(result: dict) -> None:
    print(f"\n{DIVIDER}")
    print(f"QUERY: {result['query']}")
    print(DIVIDER)
    print("\nRETRIEVED CONTEXT:")
    for i, chunk in enumerate(result["chunks"], 1):
        preview = chunk["text"][:300].strip()
        if len(chunk["text"]) > 300:
            preview += "..."
        print(f'\n[{i}] {chunk["source"]} (similarity: {chunk["similarity"]:.2f})')
        print(f'"{preview}"')
    print(f"\nANSWER:")
    print(result["answer"])
    print(DIVIDER)


def save_to_csv(result: dict, writer: csv.DictWriter) -> None:
    """Append one result row to the CSV."""
    top_chunk = result["chunks"][0] if result["chunks"] else {}
    writer.writerow({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": result["query"],
        "top_source": top_chunk.get("source", ""),
        "top_similarity": f"{top_chunk.get('similarity', 0):.2f}",
        "sources_retrieved": ", ".join(c["source"] for c in result["chunks"]),
        "answer": result["answer"],
    })


def main():
    print("=" * 70)
    print("  Compliance RAG Demo — Interactive Q&A")
    print("  Results saved to results.csv for Airtable / Google Sheets")
    print("=" * 70)

    pipeline = RAGPipeline(DOCUMENTS_DIR)

    # Open CSV and write header
    csv_file = open(CSV_FILE, "w", newline="", encoding="utf-8")
    fieldnames = ["timestamp", "query", "top_source", "top_similarity", "sources_retrieved", "answer"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Run demo queries first
    print(f"\nRunning {len(DEMO_QUERIES)} demo queries...\n")
    for query in DEMO_QUERIES:
        result = pipeline.ask(query, top_k=3)
        print_result(result)
        save_to_csv(result, writer)
        csv_file.flush()  # Write after each query so results aren't lost

    # Interactive loop
    print(f"\n{DIVIDER}")
    print("Demo queries complete. Now enter your own questions.")
    print("Type 'quit' or press Ctrl+C to exit.")
    print(f"{DIVIDER}\n")

    try:
        while True:
            query = input("Your question: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            result = pipeline.ask(query, top_k=3)
            print_result(result)
            save_to_csv(result, writer)
            csv_file.flush()
    except KeyboardInterrupt:
        pass

    csv_file.close()
    print(f"\nResults saved to {CSV_FILE} — import into Airtable or Google Sheets.")


if __name__ == "__main__":
    main()
