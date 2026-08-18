"""BM25 keyword-based retrieval.

Builds a BM25 index from the chunk corpus and returns ranked results for a
query string.  BM25 excels at exact keyword matching (e.g. proper nouns,
technical terms, numbers) that dense embeddings sometimes miss.

Usage
-----
    from src.bm25 import BM25Retriever
    from src.config import Config

    cfg = Config()
    bm = BM25Retriever(cfg)
    bm.build_index(chunks)
    results = bm.search("potassium chart", top_k=10)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import Config

# Simple Unicode-aware tokenizer: split on whitespace and non-word characters
# while keeping multi-word tokens like "state-of-the-art" intact.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase + Unicode-aware whitespace/word split."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """BM25Okapi index backed by chunk metadata."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._bm25: Optional[BM25Okapi] = None
        self._metadata: List[Dict] = []
        self._chunk_texts: List[str] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None and len(self._metadata) > 0

    @property
    def size(self) -> int:
        return len(self._metadata)

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict]) -> None:
        """Build BM25 index from a list of chunk dicts.

        Parameters
        ----------
        chunks : list of dicts, each containing at least ``chunk_text``
                 and ``chunk_id``.
        """
        if not chunks:
            raise ValueError("No chunks provided for BM25 index")

        self._chunk_texts = [c["chunk_text"] for c in chunks]
        tokenized = [tokenize(t) for t in self._chunk_texts]
        self._bm25 = BM25Okapi(
            tokenized,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        self._metadata = []
        for c in chunks:
            self._metadata.append({
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

    def save(self, path: Optional[str] = None) -> None:
        """Persist the index metadata and chunk texts to disk.

        Note: rank_bm25 does not natively support serialization, so we
        save the raw data and rebuild on load.  For large corpora this
        is still fast (~1-2s for 5k chunks).
        """
        out = path or os.path.join(
            self.config.index_dir, "bm25_index.json"
        )
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        payload = {
            "chunks": self._metadata,
            "config_k1": self.config.bm25_k1,
            "config_b": self.config.bm25_b,
        }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    def load(self, path: Optional[str] = None) -> None:
        """Load and rebuild from a saved JSON index."""
        src = path or os.path.join(self.config.index_dir, "bm25_index.json")
        if not os.path.exists(src):
            raise FileNotFoundError(f"BM25 index not found: {src}")
        with open(src, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.build_index(payload["chunks"])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """BM25 search returning ranked results.

        Returns a list of dicts sorted by descending BM25 score, each
        containing at least: ``rank``, ``score``, ``chunk_id``, ``chunk_text``.
        """
        if self._bm25 is None:
            return []

        k = top_k or self.config.bm25_top_k
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            entry = {"rank": rank, "score": float(scores[idx])}
            entry.update(self._metadata[idx])
            results.append(entry)
        return results
