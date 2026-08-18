"""FAISS-based vector storage and retrieval.

Provides :class:`VectorStore`, a thin wrapper around ``faiss-cpu`` that
stores chunk vectors alongside their metadata and supports approximate
nearest-neighbor search with configurable Top-K.

Design goals
------------
- Simple local development solution (no external services).
- Metadata stored in a parallel JSON sidecar so the FAISS index stays
  compact and portable.
- Abstract enough that swapping the backend later (e.g. to Milvus, Chroma)
  requires changes only in this file.
- Supports adding vectors incrementally (for future streaming use).

Usage
-----
    from src.vector_store import VectorStore
    from src.config import Config

    cfg = Config()
    store = VectorStore(cfg)
    store.build(vectors, metadata_list)   # or store.add() in batches
    results = store.search(query_vec, top_k=10)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from src.config import Config


class VectorStore:
    """FAISS inner-product index backed by a JSON metadata sidecar.

    Parameters
    ----------
    config : Config
        Reads ``embedding_dim``, ``index_dir``, ``vector_index_name``,
        ``metadata_name``.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._dim = self.config.embedding_dim
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[Dict[str, Any]] = []
        os.makedirs(self.config.index_dir, exist_ok=True)

    @property
    def is_built(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    # ------------------------------------------------------------------
    # Build / load / save
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """Build the index from scratch.

        Parameters
        ----------
        vectors : (N, dim) float32, L2-normalized.
        metadata : list of N dicts, one per vector.
        """
        if vectors.shape[0] != len(metadata):
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) != metadata count ({len(metadata)})"
            )
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"Vector dim ({vectors.shape[1]}) != expected ({self._dim})"
            )

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._metadata = list(metadata)

    def add(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """Append vectors to an existing (or empty) index."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"Vector dim ({vectors.shape[1]}) != expected ({self._dim})"
            )
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._metadata.extend(metadata)

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self._index is None:
            raise RuntimeError("No index to save -- call build() first")
        idx_path = os.path.join(self.config.index_dir, self.config.vector_index_name)
        meta_path = os.path.join(self.config.index_dir, self.config.metadata_name)
        faiss.write_index(self._index, idx_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False)

    def load(self) -> None:
        """Load index and metadata from disk."""
        idx_path = os.path.join(self.config.index_dir, self.config.vector_index_name)
        meta_path = os.path.join(self.config.index_dir, self.config.metadata_name)
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Index file not found: {idx_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self._index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

        if self._index.ntotal != len(self._metadata):
            raise ValueError(
                f"Index vectors ({self._index.ntotal}) != "
                f"metadata entries ({len(self._metadata)})"
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Nearest-neighbor search.

        Parameters
        ----------
        query_vector : (dim,) or (1, dim) float32, L2-normalized.
        top_k : number of results to return (clipped to index size).

        Returns
        -------
        List of dicts with keys: ``rank``, ``score``, ``chunk_id``, and any
        other fields stored in the metadata.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            entry = {"rank": rank, "score": float(score)}
            entry.update(self._metadata[idx])
            results.append(entry)
        return results
