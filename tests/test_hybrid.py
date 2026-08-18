"""Tests for RRF hybrid fusion (unit-level, no index needed)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_search import HybridSearch


class TestRRFFusion:
    """Test the static _rrf_fusion method directly."""

    def test_both_lists_same_doc(self):
        vec = [{"chunk_id": "a", "rank": 1, "chunk_text": "text a"}]
        bm25 = [{"chunk_id": "a", "rank": 1, "chunk_text": "text a"}]
        results = HybridSearch._rrf_fusion(vec, bm25, rrf_k=60, top_k=5)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "a"
        assert results[0]["rrf_score"] > 0

    def test_no_overlap(self):
        vec = [
            {"chunk_id": "a", "rank": 1, "chunk_text": "text a"},
            {"chunk_id": "b", "rank": 2, "chunk_text": "text b"},
        ]
        bm25 = [
            {"chunk_id": "c", "rank": 1, "chunk_text": "text c"},
            {"chunk_id": "d", "rank": 2, "chunk_text": "text d"},
        ]
        results = HybridSearch._rrf_fusion(vec, bm25, rrf_k=60, top_k=4)
        assert len(results) == 4
        # Doc in both lists should rank highest; docs in one list tie
        chunk_ids = [r["chunk_id"] for r in results]
        assert set(chunk_ids) == {"a", "b", "c", "d"}

    def test_overlapping_docs_ranked_higher(self):
        vec = [
            {"chunk_id": "x", "rank": 1, "chunk_text": "text x"},
            {"chunk_id": "y", "rank": 2, "chunk_text": "text y"},
        ]
        bm25 = [
            {"chunk_id": "x", "rank": 1, "chunk_text": "text x"},
            {"chunk_id": "z", "rank": 2, "chunk_text": "text z"},
        ]
        results = HybridSearch._rrf_fusion(vec, bm25, rrf_k=60, top_k=3)
        # x appears in both -> highest RRF score
        assert results[0]["chunk_id"] == "x"
        assert results[0]["vector_rank"] == 1
        assert results[0]["bm25_rank"] == 1

    def test_top_k_limit(self):
        vec = [{"chunk_id": f"c{i}", "rank": i+1, "chunk_text": f"text {i}"} for i in range(10)]
        bm25 = [{"chunk_id": f"d{i}", "rank": i+1, "chunk_text": f"text {i}"} for i in range(10)]
        results = HybridSearch._rrf_fusion(vec, bm25, rrf_k=60, top_k=5)
        assert len(results) == 5

    def test_empty_lists(self):
        results = HybridSearch._rrf_fusion([], [], rrf_k=60, top_k=5)
        assert results == []

    def test_rrf_k_affects_ranking(self):
        vec = [
            {"chunk_id": "a", "rank": 1, "chunk_text": "text a"},
            {"chunk_id": "b", "rank": 2, "chunk_text": "text b"},
        ]
        bm25 = [
            {"chunk_id": "b", "rank": 1, "chunk_text": "text b"},
            {"chunk_id": "a", "rank": 2, "chunk_text": "text a"},
        ]
        # With very large k, both docs should have similar scores
        results_high_k = HybridSearch._rrf_fusion(vec, bm25, rrf_k=10000, top_k=2)
        # a and b each appear once in each list at different ranks;
        # total RRF should be very close
        scores = {r["chunk_id"]: r["rrf_score"] for r in results_high_k}
        assert abs(scores["a"] - scores["b"]) < 0.001
