"""Vector-based retrieval: query -> embedding -> FAISS search -> Top-K.

Thin orchestration layer that ties the Embedder and VectorStore together.

Usage
-----
    from src.retriever import VectorRetriever
    from src.config import Config

    cfg = Config()
    vr = VectorRetriever(cfg)
    vr.build_index(chunks)            # from JSONL chunk list
    results = vr.search("some query", top_k=10)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import Config
from src.embeddings import Embedder
from src.vector_store import VectorStore


class VectorRetriever:
    """End-to-end vector retrieval: embed query -> search index."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.embedder = Embedder(self.config)
        self.store = VectorStore(self.config)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def load_chunks(self, chunks_path: Optional[str] = None) -> List[Dict]:
        """Load chunk JSONL into a list of dicts."""
        path = chunks_path or self.config.chunks_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunks file not found: {path}")
        chunks: List[Dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def build_index(self, chunks: Optional[List[Dict]] = None,
                    chunks_path: Optional[str] = None) -> int:
        """Build the FAISS index from chunk data.

        If ``chunks`` is not provided, loads from ``chunks_path`` (or the
        path implied by the current config).

        Returns the number of indexed vectors.
        """
        if chunks is None:
            chunks = self.load_chunks(chunks_path)

        if not chunks:
            raise ValueError("No chunks to index")

        texts = [c["chunk_text"] for c in chunks]
        vectors = self.embedder.encode_texts(texts, show_progress=True)

        metadata = []
        for c in chunks:
            metadata.append({
                "chunk_id": c["chunk_id"],
                "chunk_text": c["chunk_text"],
                "query_id": c.get("query_id"),
                "query": c.get("query"),
                "query_type": c.get("query_type"),
                "source_lang": c.get("source_lang"),
                "target_lang": c.get("target_lang"),
                "passage_index": c.get("passage_index"),
                "chunk_index": c.get("chunk_index"),
                "is_selected": c.get("is_selected", False),
                "num_words": c.get("num_words"),
            })

        self.store.build(vectors, metadata)
        self.store.save()
        return self.store.size

    def load_index(self) -> None:
        """Load a previously saved index from disk."""
        self.store.load()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search the index for a query string.

        Returns a list of result dicts sorted by descending score, each
        containing at least: ``rank``, ``score``, ``chunk_id``, ``chunk_text``.
        """
        k = top_k or self.config.vector_top_k
        q_vec = self.embedder.encode_query(query)
        return self.store.search(q_vec, top_k=k)
