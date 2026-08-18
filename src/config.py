"""Central configuration for the RAG retrieval pipeline (Member 2).

Every pipeline stage reads its parameters from a :class:`Config` object so the
system can be re-tuned without rewriting code. Configuration comes from three
sources, lowest to highest precedence:

1. Built-in defaults (below).
2. An optional JSON file passed via ``--config`` / ``Config.from_file``.
3. Explicit keyword overrides at construction time.

Nothing here is model-specific; the embedding and reranker models are named in
the config so they can be swapped (e.g. ``multilingual-e5-small`` or
``bge-m3``) without touching any pipeline code.

Usage
-----
    from src.config import Config

    cfg = Config()                       # defaults
    cfg = Config.from_file("config.json")
    cfg = Config(top_k=10, rrf_k=100)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

# Paths are relative to the repository root by default, so the scripts work
# whether they are invoked as ``python src/...`` or ``python -m src.retriever``.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _join(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


@dataclass
class Config:
    """All tunable parameters for the retrieval pipeline."""

    # --- data paths -----------------------------------------------------
    cleaned_jsonl: str = _join("data", "cleaned", "preprocessed.jsonl")
    chunks_dir: str = _join("data", "chunks")
    index_dir: str = _join("data", "index")
    cache_dir: str = _join("data", "cache")

    # --- chunking -------------------------------------------------------
    chunk_size_words: int = 256
    overlap_words: int = 32

    # --- embeddings -----------------------------------------------------
    # E5-style multilingual bi-encoder: 100+ languages incl. Indic scripts,
    # 1024 dims, trained on multilingual retrieval data. Swap freely.
    # For E5 models: set query_prefix="query: " and passage_prefix="passage: ".
    # For non-E5 models (e.g. paraphrase-*): set both prefixes to "".
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768           # kept in sync with the default model
    embed_batch_size: int = 32
    normalize_embeddings: bool = True  # cosine = dot product after L2 norm
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    # --- vector store ---------------------------------------------------
    vector_index_name: str = "vectors.npz"
    metadata_name: str = "metadata.json"
    vector_backend: str = "numpy"     # swappable: "faiss", "numpy", ...

    # --- retrieval ------------------------------------------------------
    vector_top_k: int = 20            # candidates from vector search
    bm25_top_k: int = 20              # candidates from BM25
    hybrid_top_k: int = 20            # candidates after RRF fusion
    rerank_top_n: int = 20            # candidates handed to the reranker
    top_k: int = 5                    # final context returned to the LLM

    # --- BM25 -----------------------------------------------------------
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # --- RRF (Reciprocal Rank Fusion) ------------------------------------
    rrf_k: int = 60                   # RRF score = 1 / (k + rank)

    # --- reranker -------------------------------------------------------
    # Multilingual mMARCO cross-encoder for cross-lingual reranking.
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_batch_size: int = 8

    # --- evaluation -----------------------------------------------------
    eval_metrics_k: tuple = (1, 3, 5, 10)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load config from a JSON file, falling back to defaults for keys
        that are absent. Unknown keys are ignored so config files stay
        forward-compatible."""
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path!r} must contain a JSON object")
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict (tuples become lists for JSON)."""
        return json.loads(json.dumps(asdict(self)))

    def save(self, path: str) -> None:
        """Persist this configuration to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    def chunks_path(self) -> str:
        """Path of the chunk file implied by the chunking config."""
        return os.path.join(
            self.chunks_dir,
            f"chunks_{self.chunk_size_words}_{self.overlap_words}.jsonl",
        )
