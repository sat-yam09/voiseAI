"""Clean retrieval interface for Member 1 (Voice/LLM Engineer).

Single public function that takes a query and returns Top-K relevant chunks
with all metadata needed for the LLM prompt.

Usage
-----
    from src.retrieval import retrieve
    from src.config import Config

    cfg = Config()
    context = retrieve("What is a corporation?", config=cfg)
    # context is a list of dicts, each with: rank, score, text, source metadata

    # Or use the class directly for more control:
    from src.retrieval import RetrievalPipeline
    pipe = RetrievalPipeline(cfg)
    context = pipe.retrieve("What is a corporation?", top_k=5)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.config import Config
from src.hybrid_search import HybridSearch
from src.reranker import Reranker


class RetrievalPipeline:
    """Full retrieval pipeline: hybrid search -> reranking -> Top-K context.

    This is the main class Member 1 should instantiate.  It lazily loads
    indices on first use to avoid expensive startup when not needed.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._hybrid: Optional[HybridSearch] = None
        self._reranker: Optional[Reranker] = None
        self._built = False

    def _ensure_loaded(self) -> None:
        """Lazy-load both the hybrid index and reranker model."""
        if self._built:
            return
        self._hybrid = HybridSearch(self.config)
        self._hybrid.load_index()
        self._reranker = Reranker(self.config)
        self._built = True

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """End-to-end retrieval: query -> hybrid search -> rerank -> Top-K.

        Parameters
        ----------
        query : natural-language query string.
        top_k : final number of results (default ``config.top_k``).

        Returns
        -------
        List of dicts, each containing:
        - ``rank``        : 1-based final rank
        - ``score``       : reranker relevance score
        - ``text``        : chunk text (ready for the LLM prompt)
        - ``chunk_id``    : stable identifier
        - ``query_id``    : original query ID from the dataset
        - ``source_lang`` / ``target_lang``
        - ``is_selected`` : whether this chunk was a gold passage
        - ``latency_ms``  : total retrieval + reranking latency in ms

        Raises
        ------
        FileNotFoundError  if the index files have not been built yet.
        """
        k = top_k or self.config.top_k
        t0 = time.perf_counter()

        self._ensure_loaded()

        # Step 1: hybrid retrieval (vector + BM25 + RRF).
        candidates = self._hybrid.search(
            query, top_k=self.config.rerank_top_n
        )

        # Step 2: rerank the candidates.
        reranked = self._reranker.rerank(
            query, candidates, top_n=self.config.rerank_top_n
        )

        # Step 3: trim to final Top-K and reshape for Member 1.
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results = []
        for item in reranked[:k]:
            results.append({
                "rank": item["rank"],
                "score": item.get("rerank_score", item.get("rrf_score", 0.0)),
                "text": item.get("chunk_text", ""),
                "chunk_id": item.get("chunk_id"),
                "query_id": item.get("query_id"),
                "query_type": item.get("query_type"),
                "source_lang": item.get("source_lang"),
                "target_lang": item.get("target_lang"),
                "is_selected": item.get("is_selected", False),
                "num_words": item.get("num_words"),
                "vector_rank": item.get("vector_rank"),
                "bm25_rank": item.get("bm25_rank"),
                "pre_rerank_rank": item.get("pre_rerank_rank"),
                "latency_ms": round(elapsed_ms, 1),
            })

        return results

    def retrieve_raw(
        self, query: str, top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Retrieve and return detailed timing + full candidate info.

        Useful for evaluation and benchmarking.  Returns a dict with:
        - ``query``
        - ``top_k``
        - ``latency_ms``
        - ``vector_latency_ms``
        - ``bm25_latency_ms``
        - ``rrf_latency_ms``
        - ``rerank_latency_ms``
        - ``candidates_count``
        - ``results``
        """
        k = top_k or self.config.top_k
        timings: Dict[str, float] = {}

        self._ensure_loaded()

        t0 = time.perf_counter()
        candidates = self._hybrid.search(
            query, top_k=self.config.rerank_top_n
        )
        timings["hybrid_latency_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        reranked = self._reranker.rerank(
            query, candidates, top_n=self.config.rerank_top_n
        )
        timings["rerank_latency_ms"] = (time.perf_counter() - t1) * 1000

        results = []
        for item in reranked[:k]:
            results.append({
                "rank": item["rank"],
                "score": item.get("rerank_score", item.get("rrf_score", 0.0)),
                "text": item.get("chunk_text", ""),
                "chunk_id": item.get("chunk_id"),
                "query_id": item.get("query_id"),
                "source_lang": item.get("source_lang"),
                "target_lang": item.get("target_lang"),
                "is_selected": item.get("is_selected", False),
            })

        return {
            "query": query,
            "top_k": k,
            "latency_ms": timings["hybrid_latency_ms"] + timings["rerank_latency_ms"],
            "hybrid_latency_ms": timings["hybrid_latency_ms"],
            "rerank_latency_ms": timings["rerank_latency_ms"],
            "candidates_count": len(candidates),
            "results": results,
        }


# ------------------------------------------------------------------
# Module-level convenience function (Member 1's main entry point)
# ------------------------------------------------------------------
_default_pipeline: Optional[RetrievalPipeline] = None


def retrieve(
    query: str,
    top_k: int = 5,
    config: Optional[Config] = None,
) -> List[Dict[str, Any]]:
    """High-level retrieval function for Member 1.

    Returns a list of top-K relevant chunk dicts ready to be inserted into
    the LLM prompt.  Each dict has at least: ``rank``, ``score``, ``text``,
    ``chunk_id``, and source metadata.

    The first call loads indices and the reranker model; subsequent calls
    reuse the cached pipeline.
    """
    global _default_pipeline
    if _default_pipeline is None or (config is not None and config is not _default_pipeline.config):
        _default_pipeline = RetrievalPipeline(config)
    return _default_pipeline.retrieve(query, top_k=top_k)
