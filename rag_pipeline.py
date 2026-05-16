"""
rag_pipeline.py — Core RAG logic: embed, index, retrieve, and generate.

Architecture:
  1. At init time, all document chunks are embedded using sentence-transformers
     and stored in a numpy matrix. This is the "index."
  2. At query time, the user's query is embedded with the same model,
     then cosine similarity is computed against every chunk vector.
  3. The top-K most similar chunks are retrieved and passed to Claude
     as grounding context. Claude generates a cited answer.

Why no vector database?
  For a demo with ~100 chunks, in-memory numpy is fast and dependency-free.
  In production you'd swap the numpy search for Pinecone, pgvector, or Chroma.
"""

import os
import sys
from typing import Optional
import numpy as np
import anthropic
from sentence_transformers import SentenceTransformer

from utils import load_documents, chunk_documents


# Embedding model — runs locally, no API key needed.
# all-MiniLM-L6-v2 is a strong lightweight model (22M params, 384-dim embeddings).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Claude model for generation — fast and cost-efficient.
GENERATION_MODEL = "claude-haiku-4-5-20251001"


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline for compliance documents.

    Usage:
        pipeline = RAGPipeline("documents/")
        answer = pipeline.ask("What are the SAR filing deadlines?")
        print(answer)
    """

    def __init__(self, documents_dir: str, chunk_size: int = 200, overlap: int = 50):
        """
        Load documents, chunk them, embed all chunks, and store the index in memory.
        Also initialises the Anthropic client (reads ANTHROPIC_API_KEY from env).
        """
        # --- Validate API key early so the error is clear ---
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it before running: export ANTHROPIC_API_KEY=your_key_here"
            )

        # --- Load and chunk documents ---
        print(f"\nLoading documents from '{documents_dir}'...")
        documents = load_documents(documents_dir)
        if not documents:
            raise ValueError(f"No .txt files found in '{documents_dir}'")

        for name in documents:
            print(f"  Loaded: {name}")

        self.chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        print(f"\nIndexed {len(self.chunks)} chunks from {len(documents)} documents.")

        # --- Embed all chunks ---
        print(f"\nLoading embedding model '{EMBEDDING_MODEL}'...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("Embedding document chunks (this runs locally, no API call)...")
        chunk_texts = [c["text"] for c in self.chunks]
        # Shape: (num_chunks, embedding_dim) — each row is one chunk's vector
        self.embeddings = self.embedder.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        # L2-normalise so cosine similarity reduces to a dot product (faster)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-10)

        # --- Anthropic client ---
        self.client = anthropic.Anthropic(api_key=api_key)

        # Expose metadata so the UI can display real stats
        self.num_documents = len(documents)
        self.num_chunks = len(self.chunks)
        self.embedding_dim = self.embeddings.shape[1]  # 384 for all-MiniLM-L6-v2

        print("\nRAG pipeline ready.\n")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find the top_k chunks most semantically similar to the query.

        Steps:
          1. Embed the query with the same model used to embed chunks.
          2. Compute cosine similarity (dot product after L2 normalisation).
          3. Return the top_k chunks sorted by descending similarity score.

        Returns:
            List of chunk dicts, each augmented with a "similarity" float.
        """
        # Embed and normalise query vector
        query_vec = self.embedder.encode([query], convert_to_numpy=True)[0]
        query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-10)

        # Dot product against all chunk vectors → similarity scores
        scores = self.embeddings @ query_norm  # shape: (num_chunks,)

        # Get indices of top_k scores, highest first
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])          # copy so we don't mutate the index
            chunk["similarity"] = float(scores[idx])
            results.append(chunk)

        return results

    def generate(self, query: str, retrieved_chunks: list,
                 chat_history: Optional[list] = None) -> str:
        """
        Send the query, retrieved context, and optional conversation history to Claude.

        chat_history is a list of {"role": "user"|"assistant", "content": str} dicts
        representing prior turns. This gives the model memory of the conversation so
        follow-up questions like "what about exceptions to that?" work correctly.
        """
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[{i}] Source: {chunk['source']}\n{chunk['text']}"
            )
        context_block = "\n\n".join(context_parts)

        system_prompt = f"""You are a senior compliance analyst assistant at a financial institution.
Your role is to answer operational compliance questions accurately and concisely, the way an experienced analyst would brief a colleague or relationship manager.

Rules:
- Answer using ONLY the provided policy document excerpts. Do not draw on outside knowledge.
- Write in clear, direct professional prose. No bullet points unless listing distinct items.
- Do not mention document names, file references, or citations in your answer.
- If the context is insufficient to answer fully, say so explicitly — do not speculate.
- If a question involves a threshold, procedure, or deadline, be precise. Numbers matter in compliance.
- Keep answers concise. A compliance officer reading this is busy.
- You have access to the conversation history. Use it to interpret follow-up questions correctly.

--- POLICY CONTEXT ---
{context_block}
--- END CONTEXT ---"""

        # Build messages: prior turns + current question
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": query})

        message = self.client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=600,
            system=system_prompt,
            messages=messages,
        )

        return message.content[0].text.strip()

    def suggest_followups(self, query: str, answer: str) -> list[str]:
        """
        Ask Claude to generate 3 short follow-up questions a compliance officer
        might naturally ask after receiving this answer. Makes the chat feel like
        a real analyst conversation rather than isolated Q&A.
        """
        prompt = f"""A compliance officer asked: "{query}"

The answer they received was:
{answer}

Generate exactly 3 short follow-up questions they might naturally ask next.
Each question should be on its own line. No numbering, no bullet points, no explanation.
Questions should be specific and operationally useful — not generic."""

        message = self.client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        lines = message.content[0].text.strip().split("\n")
        return [q.strip("- •123.").strip() for q in lines if q.strip()][:3]

    def ask(self, query: str, top_k: int = 3,
            chat_history: Optional[list] = None) -> dict:
        """
        Full RAG pipeline: retrieve → generate → suggest follow-ups.

        Returns:
            Dict with keys:
              - "query":      the original question
              - "chunks":     list of retrieved chunk dicts (with similarity scores)
              - "answer":     Claude's generated answer string
              - "followups":  list of 3 suggested follow-up questions
        """
        retrieved = self.retrieve(query, top_k=top_k)
        answer = self.generate(query, retrieved, chat_history=chat_history)
        followups = self.suggest_followups(query, answer)
        return {
            "query": query,
            "chunks": retrieved,
            "answer": answer,
            "followups": followups,
        }
