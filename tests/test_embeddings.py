"""Tests for the embedding module (using mocked model for fast tests).

These tests verify the Embedder API, caching, and dimension correctness
without downloading any real models.
"""

import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.config import Config


class TestEmbedderAPI:
    """Test the Embedder class structure and error handling."""

    def test_config_defaults(self):
        cfg = Config()
        assert cfg.embedding_dim == 768
        assert cfg.normalize_embeddings is True
        assert cfg.embed_batch_size == 32
        assert cfg.embedding_query_prefix == "query: "
        assert cfg.embedding_passage_prefix == "passage: "

    def test_cache_dir_created(self, tmp_path):
        cfg = Config()
        cfg.cache_dir = str(tmp_path / "my_cache")
        os.makedirs(cfg.cache_dir, exist_ok=True)
        assert os.path.isdir(cfg.cache_dir)

    def test_text_list_hash_deterministic(self):
        from src.embeddings import Embedder
        texts = ["hello world", "foo bar"]
        h1 = Embedder._text_list_hash(texts)
        h2 = Embedder._text_list_hash(texts)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 16  # truncated SHA-256

    def test_text_list_hash_order_matters(self):
        from src.embeddings import Embedder
        h1 = Embedder._text_list_hash(["a", "b"])
        h2 = Embedder._text_list_hash(["b", "a"])
        assert h1 != h2

    def test_empty_texts_returns_zero_array(self, tmp_path):
        from src.embeddings import Embedder
        cfg = Config()
        cfg.cache_dir = str(tmp_path / "cache")
        os.makedirs(cfg.cache_dir, exist_ok=True)
        embedder = Embedder(cfg)
        result = embedder.encode_texts([])
        assert result.shape == (0, cfg.embedding_dim)
        assert result.dtype == np.float32

    def test_encode_query_returns_correct_shape(self, tmp_path):
        """Test with a mock model that returns fake embeddings."""
        from src.embeddings import Embedder
        import src.embeddings as emb_mod

        cfg = Config()
        cfg.cache_dir = str(tmp_path / "cache")
        os.makedirs(cfg.cache_dir, exist_ok=True)

        # Create a fake model that returns random vectors
        class FakeModel:
            def encode(self, texts, **kwargs):
                n = len(texts)
                dim = cfg.embedding_dim
                rng = np.random.RandomState(0)
                vecs = rng.randn(n, dim).astype(np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / norms

        # Patch the lazy load to return our fake model
        embedder = Embedder(cfg)
        embedder._model = FakeModel()

        # encode_query returns (dim,) array
        q = embedder.encode_query("test query")
        assert q.shape == (cfg.embedding_dim,)
        assert q.dtype == np.float32

    def test_encode_texts_returns_correct_shape(self, tmp_path):
        from src.embeddings import Embedder

        cfg = Config()
        cfg.cache_dir = str(tmp_path / "cache")
        os.makedirs(cfg.cache_dir, exist_ok=True)

        class FakeModel:
            def encode(self, texts, **kwargs):
                n = len(texts)
                dim = cfg.embedding_dim
                rng = np.random.RandomState(1)
                vecs = rng.randn(n, dim).astype(np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / norms

        embedder = Embedder(cfg)
        embedder._model = FakeModel()

        texts = ["hello", "world", "test"]
        result = embedder.encode_texts(texts, show_progress=False)
        assert result.shape == (3, cfg.embedding_dim)
        assert result.dtype == np.float32

    def test_encode_chunks_extracts_chunk_text(self, tmp_path):
        from src.embeddings import Embedder

        cfg = Config()
        cfg.cache_dir = str(tmp_path / "cache")
        os.makedirs(cfg.cache_dir, exist_ok=True)

        class FakeModel:
            def encode(self, texts, **kwargs):
                n = len(texts)
                dim = cfg.embedding_dim
                rng = np.random.RandomState(2)
                vecs = rng.randn(n, dim).astype(np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / norms

        embedder = Embedder(cfg)
        embedder._model = FakeModel()

        chunks = [
            {"chunk_text": "first chunk", "chunk_id": "c1"},
            {"chunk_text": "second chunk", "chunk_id": "c2"},
        ]
        result = embedder.encode_chunks(chunks)
        assert result.shape == (2, cfg.embedding_dim)

    def test_caching_works(self, tmp_path):
        from src.embeddings import Embedder

        cfg = Config()
        cfg.cache_dir = str(tmp_path / "cache")
        os.makedirs(cfg.cache_dir, exist_ok=True)

        call_count = 0

        class CountingModel:
            def encode(self, texts, **kwargs):
                nonlocal call_count
                call_count += 1
                n = len(texts)
                dim = cfg.embedding_dim
                return np.ones((n, dim), dtype=np.float32)

        embedder = Embedder(cfg)
        embedder._model = CountingModel()

        texts = ["cached text one", "cached text two"]

        # First call: hits the model
        result1 = embedder.encode_texts(texts, show_progress=False)
        assert call_count == 1

        # Second call: should load from cache, not hit the model
        result2 = embedder.encode_texts(texts, show_progress=False)
        assert call_count == 1  # still 1, not 2

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_clear_cache(self, tmp_path):
        from src.embeddings import Embedder

        cfg = Config()
        cfg.cache_dir = str(tmp_path / "cache")
        os.makedirs(cfg.cache_dir, exist_ok=True)

        class FakeModel:
            def encode(self, texts, **kwargs):
                n = len(texts)
                dim = cfg.embedding_dim
                return np.ones((n, dim), dtype=np.float32)

        embedder = Embedder(cfg)
        embedder._model = FakeModel()

        texts = ["to be cached"]
        embedder.encode_texts(texts, show_progress=False)

        # Cache file should exist
        cache_key = Embedder._text_list_hash(
    embedder._apply_passage_prefix(texts)
)
        cache_path = os.path.join(cfg.cache_dir, f"embed_{cache_key}.npy")
        assert os.path.exists(cache_path)

        removed = embedder.clear_cache()
        assert removed >= 1
        assert not os.path.exists(cache_path)


class TestEmbeddingNormalization:
    """Verify that normalized embeddings have unit length."""

    def test_normalized_embeddings_are_unit_length(self, tmp_path):
        from src.embeddings import Embedder

        cfg = Config()
        cfg.normalize_embeddings = True
        cfg.cache_dir = str(tmp_path / "cache")
        os.makedirs(cfg.cache_dir, exist_ok=True)

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True, **kwargs):
                n = len(texts)
                dim = cfg.embedding_dim
                rng = np.random.RandomState(42)
                vecs = rng.randn(n, dim).astype(np.float32)
                if normalize_embeddings:
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    vecs = vecs / norms
                return vecs

        embedder = Embedder(cfg)
        embedder._model = FakeModel()

        texts = ["hello", "world", "test query"]
        result = embedder.encode_texts(texts, show_progress=False)

        # Each vector should have unit norm
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
