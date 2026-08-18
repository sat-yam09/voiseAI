"""Tests for vector store (using synthetic vectors, no model download)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.vector_store import VectorStore
from src.config import Config


@pytest.fixture
def dim():
    return 8  # small dimension for fast tests


@pytest.fixture
def store(tmp_path, dim):
    cfg = Config()
    cfg.embedding_dim = dim
    cfg.index_dir = str(tmp_path / "index")
    os.makedirs(cfg.index_dir, exist_ok=True)
    return VectorStore(cfg)


@pytest.fixture
def vectors_and_meta(dim):
    rng = np.random.RandomState(42)
    n = 10
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    meta = [{"chunk_id": f"c{i}", "chunk_text": f"text {i}"} for i in range(n)]
    return vecs, meta


class TestVectorStore:
    def test_build_and_search(self, store, vectors_and_meta, dim):
        vecs, meta = vectors_and_meta
        store.build(vecs, meta)
        assert store.size == 10

        q = vecs[0:1]  # query with first vector -> should find c0 as top-1
        results = store.search(q, top_k=3)
        assert len(results) == 3
        assert results[0]["chunk_id"] == "c0"
        assert results[0]["rank"] == 1
        assert results[0]["score"] > 0.99  # near-perfect match

    def test_add(self, store, vectors_and_meta, dim):
        vecs, meta = vectors_and_meta
        store.add(vecs[:5], meta[:5])
        store.add(vecs[5:], meta[5:])
        assert store.size == 10

    def test_save_and_load(self, store, vectors_and_meta):
        vecs, meta = vectors_and_meta
        store.build(vecs, meta)
        store.save()

        store2 = VectorStore(store.config)
        store2.load()
        assert store2.size == 10

        q = vecs[0:1]
        results = store2.search(q, top_k=3)
        assert results[0]["chunk_id"] == "c0"

    def test_empty_store_returns_empty(self, store):
        q = np.zeros((1, 8), dtype=np.float32)
        results = store.search(q, top_k=5)
        assert results == []

    def test_build_mismatch_raises(self, store, dim):
        vecs = np.random.randn(5, dim).astype(np.float32)
        meta = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        with pytest.raises(ValueError, match="Vector count"):
            store.build(vecs, meta)

    def test_dim_mismatch_raises(self, store):
        vecs = np.random.randn(5, 99).astype(np.float32)
        meta = [{"chunk_id": f"c{i}"} for i in range(5)]
        with pytest.raises(ValueError, match="Vector dim"):
            store.build(vecs, meta)
