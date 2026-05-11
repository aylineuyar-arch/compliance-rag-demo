"""
utils.py — Document chunking and text preprocessing helpers.

Chunking strategy: fixed-size word windows with overlap.
- Overlap ensures that sentences spanning a chunk boundary are captured in at least one chunk.
- 200-word chunks are large enough to contain meaningful regulatory context,
  small enough that retrieved chunks stay focused and don't overwhelm the prompt.
"""

import os
import re


def load_documents(documents_dir: str) -> dict[str, str]:
    """
    Load all .txt files from a directory and return a dict of {filename: text}.
    Strips leading/trailing whitespace from each document.
    """
    documents = {}
    for filename in sorted(os.listdir(documents_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(documents_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents[filename] = f.read().strip()
    return documents


def clean_text(text: str) -> str:
    """
    Normalize whitespace in a text string.
    - Collapses multiple spaces and tabs into a single space.
    - Preserves paragraph breaks (double newlines) as single newlines.
    """
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-count windows.

    Args:
        text:       The input document text.
        chunk_size: Target number of words per chunk (default 200).
        overlap:    Number of words shared between consecutive chunks (default 50).
                    Overlap prevents context loss at chunk boundaries.

    Returns:
        List of chunk strings. Each chunk is a contiguous run of words
        joined back into a space-separated string.
    """
    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap  # How far to advance the window each iteration

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks


def chunk_documents(documents: dict[str, str],
                    chunk_size: int = 200,
                    overlap: int = 50) -> list[dict]:
    """
    Chunk all documents and return a flat list of chunk records.

    Each record is a dict with:
        - "source":  filename the chunk came from (used for citations)
        - "chunk_id": index of the chunk within its source document
        - "text":    the chunk text

    This flat list is what gets embedded and stored in memory for retrieval.
    """
    all_chunks = []
    for filename, text in documents.items():
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": filename,
                "chunk_id": i,
                "text": chunk,
            })
    return all_chunks
