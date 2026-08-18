"""Cross-encoder reranking for improved precision.

After initial retrieval (vector or hybrid) returns a candidate pool, the
reranker scores each (query, passage) pair with a multilingual cross-encoder
and re-sorts by relevance.

Default model
-------------
``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``

Why this model?
- Trained on mMARCO (multilingual MS MARCO), matching our dataset domain.
- Supports 50+ languages including Indic scripts.
- 12-layer MiniLM architecture: fast enough for reranking Top-20 candidates
  on CPU, yet significantly more accurate than bi-encoder alone.

Usage
-----
    from src.reranker import Reranker
    from src.config import Config

    cfg = Config()
    rr = Reranker(cfg)
    reranked = rr.rerank(query, candidates, top_n=20)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.config import Config

# Lazy import to avoid loading torch at module-import time.
_cross_encoder = None


class Reranker:
    """Cross-encoder reranker for (query, passage) pairs."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            global _cross_encoder
            if _cross_encoder is None:
                from sentence_transformers import CrossEncoder
                _cross_encoder = CrossEncoder(self.config.reranker_model)
            self._model = _cross_encoder
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates by cross-encoder relevance score.

        Parameters
        ----------
        query : the search query.
        candidates : list of dicts, each with at least ``chunk_text``.
        top_n : how many candidates to score (default ``config.rerank_top_n``).

        Returns
        -------
        List of dicts re-sorted by descending ``rerank_score``, with original
        rank information preserved in ``pre_rerank_rank``.
        """
        if not candidates:
            return []

        n = min(top_n or self.config.rerank_top_n, len(candidates))
        subset = candidates[:n]

        pairs = [(query, c["chunk_text"]) for c in subset]
        scores = self.model.predict(
            pairs,
            batch_size=self.config.reranker_batch_size,
            show_progress_bar=False,
        )
        scores = np.asarray(scores, dtype=np.float32)

        # Sort by descending score.
        order = np.argsort(scores)[::-1]
        results = []
        for new_rank, idx in enumerate(order, start=1):
            entry = {
                "rank": new_rank,
                "rerank_score": float(scores[idx]),
                "pre_rerank_rank": subset[idx].get("rank"),
            }
            entry.update({
                k: v for k, v in subset[idx].items()
                if k not in ("rank", "score", "rrf_score")
            })
            results.append(entry)

        return results
