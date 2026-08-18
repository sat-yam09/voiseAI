"""Multilingual embedding module for the RAG retrieval pipeline.

Wraps a sentence-transformers model to produce normalized embeddings from
chunk texts and queries.  Supports batch processing, on-disk caching, and
configurable model selection via :class:`src.config.Config`.

Default model
-------------
``intfloat/multilingual-e5-base``

Why this model?
- 100+ languages including all major Indic scripts (Hindi, Bengali, Assamese,
  Tamil, Telugu, ...).
- 1024 dimensions -- strong multilingual retrieval representations.
- Trained on multilingual retrieval datasets (better IR quality than
  paraphrase-based models for Assamese and other Indic languages).
- E5-style: requires ``"query: "`` prefix for queries and
  ``"passage: "`` prefix for documents.
- HuggingFace Hub caching means the model weights are downloaded once and
  reused across runs.

Non-E5 models (e.g. paraphrase-multilingual-MiniLM-L12-v2) remain
supported by setting ``embedding_query_prefix`` and
``embedding_passage_prefix`` to ``""`` in the config.

Usage
-----
    from src.embeddings import Embedder
    from src.config import Config

    cfg = Config()
    embedder = Embedder(cfg)

    vectors = embedder.encode_chunks(chunks)          # list of dicts
    q_vec    = embedder.encode_query("some query")     # 1-D numpy array
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np

from src.config import Config

# Lazy imports -- only loaded when Embedder is first instantiated so the rest
# of the pipeline can be imported without downloading torch/transformers.
# Per-model singleton: maps model_name -> loaded SentenceTransformer instance.
_loaded_models: Dict[str, object] = {}


def _lazy_load_model(model_name: str):
    """Load a SentenceTransformer model once per model name (module-level cache)."""
    if model_name not in _loaded_models:
        from sentence_transformers import SentenceTransformer
        _loaded_models[model_name] = SentenceTransformer(model_name)
    return _loaded_models[model_name]


class Embedder:
    """Produce normalized embeddings for chunks and queries.

    Parameters come from :class:`Config`.  Vectors are L2-normalized so that
    cosine similarity equals dot product, which FAISS can exploit efficiently.

    E5-style prefixing
    ------------------
    When ``embedding_query_prefix`` / ``embedding_passage_prefix`` are set in
    the config, the corresponding prefix is prepended to every text before
    encoding.  This is required for E5-family models and ignored (empty
    strings) for non-E5 models.

    Caching
    -------
    ``embed_texts`` stores its result on disk (numpy) keyed by a SHA-256 hash
    of the prefixed texts.  Changing the prefix invalidates the cache
    automatically.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._model = None
        self._cache_dir = self.config.cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

    @property
    def model(self):
        if self._model is None:
            self._model = _lazy_load_model(self.config.embedding_model)
        return self._model

    @property
    def dimension(self) -> int:
        return self.config.embedding_dim

    # ------------------------------------------------------------------
    # Prefix helpers
    # ------------------------------------------------------------------

    def _apply_query_prefix(self, texts: List[str]) -> List[str]:
        """Prepend the query prefix to each text (E5-style)."""
        prefix = self.config.embedding_query_prefix
        if prefix:
            return [prefix + t for t in texts]
        return texts

    def _apply_passage_prefix(self, texts: List[str]) -> List[str]:
        """Prepend the passage/document prefix to each text (E5-style)."""
        prefix = self.config.embedding_passage_prefix
        if prefix:
            return [prefix + t for t in texts]
        return texts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into a (N, dim) float32 numpy array.

        If ``is_query`` is True, the query prefix is applied; otherwise the
        passage prefix is applied.  The prefix is included in the cache key
        so switching prefixes invalidates stale caches.

        If a cached result exists for the exact same (prefixed) texts it is
        loaded directly from disk, skipping model inference entirely.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # Apply the appropriate prefix
        if is_query:
            prefixed = self._apply_query_prefix(texts)
        else:
            prefixed = self._apply_passage_prefix(texts)

        # Cache key includes the prefixed texts (prefix change = cache miss)
        cache_key = self._text_list_hash(prefixed)
        cache_path = os.path.join(self._cache_dir, f"embed_{cache_key}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        bs = batch_size or self.config.embed_batch_size
        embeddings = self.model.encode(
            prefixed,
            batch_size=bs,
            show_progress_bar=show_progress and len(prefixed) > bs,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(cache_path, embeddings)
        return embeddings

    def encode_chunks(self, chunks: List[Dict], show_progress: bool = True) -> np.ndarray:
        """Encode chunk dicts (must contain ``chunk_text``).

        Returns (N, dim) float32 array aligned with the input list.
        Passages use the passage prefix.
        """
        texts = [c["chunk_text"] for c in chunks]
        return self.encode_texts(texts, is_query=False, show_progress=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a (dim,) float32 vector.

        The query prefix is applied automatically.
        """
        vec = self.model.encode(
            self._apply_query_prefix([query]),
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return np.asarray(vec[0], dtype=np.float32)

    def clear_cache(self) -> int:
        """Delete all cached embedding files. Returns number removed."""
        removed = 0
        for fname in os.listdir(self._cache_dir):
            if fname.startswith("embed_") and fname.endswith(".npy"):
                os.remove(os.path.join(self._cache_dir, fname))
                sidecar = fname.replace(".npy", ".json")
                sidecar_path = os.path.join(self._cache_dir, sidecar)
                if os.path.exists(sidecar_path):
                    os.remove(sidecar_path)
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _text_list_hash(texts: List[str]) -> str:
        """Deterministic SHA-256 of the text list (content-only)."""
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\x00")  # separator
        return h.hexdigest()[:16]
