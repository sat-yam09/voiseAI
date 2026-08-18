"""Hybrid search combining vector and BM25 retrieval via Reciprocal Rank Fusion.

Why hybrid?
-----------
- **BM25** excels at exact keyword matching: proper nouns, product codes,
  numbers, rare technical terms.  It scores documents based on term frequency
  and inverse document frequency.
- **Vector (dense) retrieval** excels at semantic matching: paraphrases,
  synonyms, cross-lingual queries, and conceptual similarity.

Neither approach alone covers all retrieval scenarios.  Hybrid search merges
both ranked lists, producing a candidate set that is more robust than either
individual method.

Reciprocal Rank Fusion (RRF)
-----------------------------
For each document ``d`` appearing in a ranked list:

    RRF_score(d) = sum_over_lists  1 / (k + rank_i(d))

where ``k`` (default 60) controls how much weight lower-ranked documents
receive.  A higher ``k`` makes the融合 more uniform; ``k = 0`` reduces to
pure rank-1 voting.

Usage
-----
    from src.hybrid_search import HybridSearch
    from src.config import Config

    cfg = Config()
    hs = HybridSearch(cfg)
    hs.build(vector_chunks, bm25_chunks)
    results = hs.search("query text", top_k=10)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.config import Config
from src.retriever import VectorRetriever
from src.bm25 import BM25Retriever


class HybridSearch:
    """Combine vector and BM25 retrieval using RRF."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.vector_retriever = VectorRetriever(self.config)
        self.bm25_retriever = BM25Retriever(self.config)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(self, chunks: Optional[List[Dict]] = None,
                    chunks_path: Optional[str] = None) -> int:
        """Build both vector and BM25 indices from the same chunk corpus.

        Returns the number of indexed chunks.
        """
        if chunks is None:
            chunks = self.vector_retriever.load_chunks(chunks_path)

        if not chunks:
            raise ValueError("No chunks provided for hybrid index")

        self.vector_retriever.build_index(chunks)
        self.bm25_retriever.build_index(chunks)
        return len(chunks)

    def load_index(self) -> None:
        """Load previously saved vector + BM25 indices."""
        self.vector_retriever.load_index()
        self.bm25_retriever.load()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Run vector + BM25 retrieval and fuse with RRF.

        Parameters
        ----------
        query : search query string.
        top_k : final number of results (default ``config.hybrid_top_k``).

        Returns
        -------
        List of dicts sorted by descending RRF score, each containing at
        least: ``rank``, ``rrf_score``, ``chunk_id``, ``chunk_text``,
        ``vector_rank``, ``bm25_rank``.
        """
        k = top_k or self.config.hybrid_top_k
        rrf_k = self.config.rrf_k

        # Fetch generous candidate pools from each retriever so RRF has
        # enough overlap to work with.
        vec_candidates = self.vector_retriever.search(
            query, top_k=max(k * 2, self.config.vector_top_k)
        )
        bm25_candidates = self.bm25_retriever.search(
            query, top_k=max(k * 2, self.config.bm25_top_k)
        )

        return self._rrf_fusion(
            vec_candidates, bm25_candidates, rrf_k=rrf_k, top_k=k
        )

    @staticmethod
    def _rrf_fusion(
        vec_results: List[Dict],
        bm25_results: List[Dict],
        rrf_k: int = 60,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Merge two ranked lists with Reciprocal Rank Fusion.

        Documents that appear in both lists get the highest RRF scores
        because their reciprocal ranks are summed.
        """
        # Accumulate RRF scores per chunk_id.
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_lookup: Dict[str, Dict] = {}
        vec_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        for item in vec_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            rrf_scores[cid] += 1.0 / (rrf_k + rank)
            chunk_lookup[cid] = item
            vec_ranks[cid] = rank

        for item in bm25_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            rrf_scores[cid] += 1.0 / (rrf_k + rank)
            if cid not in chunk_lookup:
                chunk_lookup[cid] = item
            bm25_ranks[cid] = rank

        # Sort by descending RRF score.
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for rank, cid in enumerate(sorted_ids[:top_k], start=1):
            entry = {
                "rank": rank,
                "rrf_score": rrf_scores[cid],
                "chunk_id": cid,
                "vector_rank": vec_ranks.get(cid),
                "bm25_rank": bm25_ranks.get(cid),
            }
            # Include chunk metadata from whichever retriever found it.
            base = chunk_lookup[cid]
            for key in ("chunk_text", "query_id", "query", "query_type",
                        "source_lang", "target_lang", "passage_index",
                        "chunk_index", "is_selected", "num_words"):
                if key in base:
                    entry[key] = base[key]
            results.append(entry)

        return results
