"""Tests for the retrieval pipeline (Top-K output, Member 1 interface).

Tests the RetrievalPipeline and the module-level retrieve() function
with mocked sub-components (no model downloads required).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.config import Config


class FakeHybridSearch:
    """Mock hybrid search that returns predictable candidates."""

    def __init__(self, candidates=None):
        self._candidates = candidates if candidates is not None else [
            {
                "chunk_id": "c1",
                "chunk_text": "A corporation is a legal entity.",
                "rrf_score": 0.05,
                "query_id": 1,
                "query": "what is a corporation",
                "query_type": "DESCRIPTION",
                "source_lang": "eng_Latn",
                "target_lang": "hin_Deva",
                "passage_index": 0,
                "chunk_index": 0,
                "is_selected": True,
                "num_words": 10,
                "vector_rank": 1,
                "bm25_rank": 2,
                "rank": 1,
            },
            {
                "chunk_id": "c2",
                "chunk_text": "B corps are certified companies.",
                "rrf_score": 0.03,
                "query_id": 1,
                "query": "what is a corporation",
                "query_type": "DESCRIPTION",
                "source_lang": "eng_Latn",
                "target_lang": "hin_Deva",
                "passage_index": 1,
                "chunk_index": 0,
                "is_selected": False,
                "num_words": 8,
                "vector_rank": 2,
                "bm25_rank": 1,
                "rank": 2,
            },
            {
                "chunk_id": "c3",
                "chunk_text": "Potassium is found in bananas.",
                "rrf_score": 0.02,
                "query_id": 2,
                "query": "foods high in potassium",
                "query_type": "ENTITY",
                "source_lang": "eng_Latn",
                "target_lang": "tam_Taml",
                "passage_index": 0,
                "chunk_index": 0,
                "is_selected": True,
                "num_words": 7,
                "vector_rank": 3,
                "bm25_rank": 3,
                "rank": 3,
            },
        ]

    def load_index(self):
        pass

    def search(self, query, top_k=20):
        return self._candidates[:top_k]


class FakeReranker:
    """Mock reranker that sorts by chunk_text length (longer = better)."""

    def __init__(self, candidates=None):
        self._candidates = candidates

    def rerank(self, query, candidates, top_n=20):
        subset = candidates[:top_n]
        # Sort by text length (simulating reranking)
        sorted_cands = sorted(subset, key=lambda c: len(c.get("chunk_text", "")), reverse=True)
        results = []
        for i, c in enumerate(sorted_cands, start=1):
            entry = {
                "rank": i,
                "rerank_score": float(len(c.get("chunk_text", ""))) / 100.0,
                "pre_rerank_rank": c.get("rank"),
            }
            entry.update({k: v for k, v in c.items() if k not in ("rank", "score", "rrf_score")})
            results.append(entry)
        return results


class TestRetrievalPipeline:
    """Test the RetrievalPipeline class with mocked sub-components."""

    @pytest.fixture
    def pipeline_with_mocks(self, tmp_path):
        from src.retrieval import RetrievalPipeline
        cfg = Config()
        cfg.index_dir = str(tmp_path / "index")
        os.makedirs(cfg.index_dir, exist_ok=True)

        pipe = RetrievalPipeline(cfg)
        pipe._hybrid = FakeHybridSearch()
        pipe._reranker = FakeReranker()
        pipe._built = True
        return pipe

    def test_retrieve_returns_list(self, pipeline_with_mocks):
        results = pipeline_with_mocks.retrieve("what is a corporation")
        assert isinstance(results, list)

    def test_retrieve_respects_top_k(self, pipeline_with_mocks):
        results = pipeline_with_mocks.retrieve("query", top_k=2)
        assert len(results) <= 2

    def test_retrieve_top_k_1(self, pipeline_with_mocks):
        results = pipeline_with_mocks.retrieve("query", top_k=1)
        assert len(results) == 1

    def test_result_has_required_keys(self, pipeline_with_mocks):
        results = pipeline_with_mocks.retrieve("query", top_k=3)
        required_keys = {"rank", "score", "text", "chunk_id", "query_id",
                         "source_lang", "target_lang", "is_selected", "latency_ms"}
        for r in results:
            assert required_keys.issubset(r.keys()), f"Missing keys: {required_keys - r.keys()}"

    def test_result_text_is_chunk_text(self, pipeline_with_mocks):
        results = pipeline_with_mocks.retrieve("query", top_k=1)
        # FakeReranker sorts by text length; longest text is "A corporation is a legal entity."
        assert results[0]["text"] == "A corporation is a legal entity."

    def test_result_rank_is_sequential(self, pipeline_with_mocks):
        results = pipeline_with_mocks.retrieve("query", top_k=3)
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_result_latency_is_positive(self, pipeline_with_mocks):
        results = pipeline_with_mocks.retrieve("query")
        assert all(r["latency_ms"] >= 0 for r in results)

    def test_retrieve_raw_returns_dict(self, pipeline_with_mocks):
        result = pipeline_with_mocks.retrieve_raw("query", top_k=2)
        assert isinstance(result, dict)
        assert "query" in result
        assert "results" in result
        assert "latency_ms" in result
        assert "hybrid_latency_ms" in result
        assert "rerank_latency_ms" in result

    def test_retrieve_raw_candidates_count(self, pipeline_with_mocks):
        result = pipeline_with_mocks.retrieve_raw("query")
        assert result["candidates_count"] == 3  # all 3 candidates from FakeHybridSearch

    def test_retrieve_raw_top_k(self, pipeline_with_mocks):
        result = pipeline_with_mocks.retrieve_raw("query", top_k=1)
        assert len(result["results"]) == 1


class TestModuleLevelRetrieve:
    """Test the module-level retrieve() convenience function."""

    def test_retrieve_function_exists(self):
        from src.retrieval import retrieve
        assert callable(retrieve)

    def test_retrieve_returns_list(self):
        from src.retrieval import retrieve, RetrievalPipeline
        import src.retrieval as ret_mod

        # Save original default pipeline
        original = ret_mod._default_pipeline

        try:
            # Mock the pipeline
            fake_pipe = RetrievalPipeline.__new__(RetrievalPipeline)
            fake_pipe.config = Config()
            fake_pipe._hybrid = FakeHybridSearch()
            fake_pipe._reranker = FakeReranker()
            fake_pipe._built = True
            ret_mod._default_pipeline = fake_pipe

            results = retrieve("what is a corporation", top_k=2)
            assert isinstance(results, list)
            assert len(results) <= 2
        finally:
            ret_mod._default_pipeline = original


class TestRetrievalErrorHandling:
    """Test error handling in the retrieval pipeline."""

    def test_empty_candidates_returns_empty(self, tmp_path):
        from src.retrieval import RetrievalPipeline
        cfg = Config()
        pipe = RetrievalPipeline(cfg)
        pipe._hybrid = FakeHybridSearch(candidates=[])
        pipe._reranker = FakeReranker()
        pipe._built = True

        results = pipe.retrieve("query")
        assert results == []

    def test_top_k_zero_uses_default(self, tmp_path):
        """top_k=0 is falsy so pipeline falls back to config.top_k."""
        from src.retrieval import RetrievalPipeline
        cfg = Config()
        pipe = RetrievalPipeline(cfg)
        pipe._hybrid = FakeHybridSearch()
        pipe._reranker = FakeReranker()
        pipe._built = True

        results = pipe.retrieve("query", top_k=0)
        # 0 is falsy, so `top_k or config.top_k` -> config.top_k (5)
        # but only 3 candidates exist
        assert len(results) == 3
