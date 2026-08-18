"""Tests for the configuration module."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.chunk_size_words == 256
        assert cfg.overlap_words == 32
        assert cfg.embedding_dim == 768
        assert cfg.top_k == 5
        assert cfg.rrf_k == 60

    def test_from_file(self, tmp_path):
        path = tmp_path / "config.json"
        data = {"chunk_size_words": 512, "overlap_words": 64, "top_k": 10}
        path.write_text(json.dumps(data), encoding="utf-8")

        cfg = Config.from_file(str(path))
        assert cfg.chunk_size_words == 512
        assert cfg.overlap_words == 64
        assert cfg.top_k == 10
        # Unset values remain defaults
        assert cfg.rrf_k == 60

    def test_from_file_ignores_unknown_keys(self, tmp_path):
        path = tmp_path / "config.json"
        data = {"unknown_key": 42, "chunk_size_words": 1024}
        path.write_text(json.dumps(data), encoding="utf-8")

        cfg = Config.from_file(str(path))
        assert cfg.chunk_size_words == 1024

    def test_save_and_reload(self, tmp_path):
        cfg = Config(chunk_size_words=512, overlap_words=64)
        path = str(tmp_path / "saved.json")
        cfg.save(path)

        loaded = Config.from_file(path)
        assert loaded.chunk_size_words == 512
        assert loaded.overlap_words == 64

    def test_chunks_path(self):
        cfg = Config(chunk_size_words=512, overlap_words=64)
        path = cfg.chunks_path()
        assert "chunks_512_64.jsonl" in path

    def test_to_dict(self):
        cfg = Config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["embedding_model"] == cfg.embedding_model
