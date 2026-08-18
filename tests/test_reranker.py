"""Tests for the reranker module (using mocked model for fast tests).

These tests verify the Reranker API, output structure, and sorting
without downloading the cross-encoder model.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.config import Config


class FakeCrossEncoder:
    """Mock cross-encoder that returns predictable scores."""

    def __init__(self, scores=None):
        self._scores = scores or [0.9, 0.1, 0.7, 0.3, 0.5]
        self._call_count = 0

    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        """Return scores based on chunk_text length (longer = higher)."""
        scores = []
        for query, text in pairs:
            # Simple heuristic: longer text gets higher score
            scores.append(float(len(text)))
        return np.array(scores, dtype=np.float32)


class TestRerankerOutput:
    """Test reranker output structure and sorting."""

    @pytest.fixture
    def reranker_with_mock(self, tmp_path):
        from src.reranker import Reranker
        cfg = Config()
        rr = Reranker(cfg)
        rr._model = FakeCrossEncoder()
        return rr

    def test_rerank_returns_list(self, reranker_with_mock):
        candidates = [
            {"chunk_id": "c1", "chunk_text": "short", "rank": 1},
            {"chunk_id": "c2", "chunk_text": "medium length text", "rank": 2},
            {"chunk_id": "c3", "chunk_text": "a very long chunk of text here", "rank": 3},
        ]
        results = reranker_with_mock.rerank("test query", candidates)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_rerank_sorts_by_score_descending(self, reranker_with_mock):
        candidates = [
            {"chunk_id": "c1", "chunk_text": "ab", "rank": 1},
            {"chunk_id": "c2", "chunk_text": "abcdefghij", "rank": 2},
            {"chunk_id": "c3", "chunk_text": "abc", "rank": 3},
        ]
        results = reranker_with_mock.rerank("q", candidates)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_preserves_metadata(self, reranker_with_mock):
        candidates = [
            {
                "chunk_id": "c1",
                "chunk_text": "some text",
                "query_id": 42,
                "source_lang": "eng_Latn",
                "is_selected": True,
                "num_words": 10,
                "rank": 1,
            },
        ]
        results = reranker_with_mock.rerank("q", candidates)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["query_id"] == 42
        assert results[0]["source_lang"] == "eng_Latn"
        assert results[0]["is_selected"] is True
        assert results[0]["num_words"] == 10

    def test_rerank_adds_rerank_score(self, reranker_with_mock):
        candidates = [
            {"chunk_id": "c1", "chunk_text": "text a", "rank": 1},
        ]
        results = reranker_with_mock.rerank("q", candidates)
        assert "rerank_score" in results[0]
        assert isinstance(results[0]["rerank_score"], float)

    def test_rerank_adds_pre_rerank_rank(self, reranker_with_mock):
        candidates = [
            {"chunk_id": "c1", "chunk_text": "text a", "rank": 3},
            {"chunk_id": "c2", "chunk_text": "text b", "rank": 1},
        ]
        results = reranker_with_mock.rerank("q", candidates)
        # pre_rerank_rank should reflect original rank from candidate list
        for r in results:
            assert "pre_rerank_rank" in r

    def test_rerank_resets_rank_to_new_order(self, reranker_with_mock):
        candidates = [
            {"chunk_id": "c1", "chunk_text": "ab", "rank": 1},
            {"chunk_id": "c2", "chunk_text": "abcdefghij", "rank": 2},
        ]
        results = reranker_with_mock.rerank("q", candidates)
        # New rank should be 1-based sequential
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_rerank_empty_candidates(self, reranker_with_mock):
        results = reranker_with_mock.rerank("query", [])
        assert results == []

    def test_rerank_respects_top_n(self, reranker_with_mock):
        candidates = [
            {"chunk_id": f"c{i}", "chunk_text": f"text {i}" * (i + 1), "rank": i + 1}
            for i in range(10)
        ]
        results = reranker_with_mock.rerank("q", candidates, top_n=3)
        # Only top_n candidates should be scored and returned
        assert len(results) == 3

    def test_rerank_top_n_larger_than_candidates(self, reranker_with_mock):
        candidates = [
            {"chunk_id": "c1", "chunk_text": "text", "rank": 1},
        ]
        results = reranker_with_mock.rerank("q", candidates, top_n=20)
        assert len(results) == 1  # can't rerank more than we have

    def test_rerank_does_not_mutate_input(self, reranker_with_mock):
        candidates = [
            {"chunk_id": "c1", "chunk_text": "text a", "rank": 1},
            {"chunk_id": "c2", "chunk_text": "text b", "rank": 2},
        ]
        original_ids = [c["chunk_id"] for c in candidates]
        reranker_with_mock.rerank("q", candidates)
        # Original list should be unchanged
        assert [c["chunk_id"] for c in candidates] == original_ids
