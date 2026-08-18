"""Tests for BM25 retrieval (no model downloads required)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.bm25 import BM25Retriever, tokenize
from src.config import Config


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_unicode(self):
        tokens = tokenize("নমস্কাৰ বিশ্ব hello")
        assert len(tokens) > 0
        # All tokens should be lowercase
        for t in tokens:
            assert t == t.lower()

    def test_punctuation(self):
        tokens = tokenize("state-of-the-art AI!")
        # The regex matches word chars; hyphens split differently
        assert "ai" in tokens

    def test_empty(self):
        assert tokenize("") == []


@pytest.fixture
def bm25():
    cfg = Config()
    return BM25Retriever(cfg)


class TestBM25Retriever:
    def test_build_and_search(self, bm25, sample_chunks):
        bm25.build_index(sample_chunks)
        assert bm25.size == 6

        results = bm25.search("corporation", top_k=3)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "1_p0_c0"
        assert "chunk_text" in results[0]

    def test_keyword_matching(self, bm25, sample_chunks):
        bm25.build_index(sample_chunks)
        results = bm25.search("potassium foods", top_k=2)
        assert len(results) == 2
        # Should find the potassium-related chunks
        chunk_ids = [r["chunk_id"] for r in results]
        assert any("2_" in cid for cid in chunk_ids)

    def test_save_and_load(self, bm25, sample_chunks, tmp_path):
        bm25.build_index(sample_chunks)
        path = str(tmp_path / "bm25.json")
        bm25.save(path)

        bm25_new = BM25Retriever(bm25.config)
        bm25_new.load(path)
        assert bm25_new.size == 6

        results = bm25_new.search("honesty", top_k=2)
        assert len(results) > 0

    def test_empty_query_returns_empty(self, bm25, sample_chunks):
        bm25.build_index(sample_chunks)
        results = bm25.search("", top_k=5)
        assert results == []

    def test_build_empty_raises(self, bm25):
        with pytest.raises(ValueError, match="No chunks"):
            bm25.build_index([])
