"""Pre-encode passages for all benchmark models.

Bypasses the Embedder's internal cache (which is model-agnostic and
would cause collisions when two models share the same prefix) by
calling model.encode() directly.  Saves to per-model cache dirs.

Usage:
    python experiments/preencode_passages.py
"""
import gc
import os
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
from src.config import Config, PROJECT_ROOT
from src.embeddings import Embedder, _loaded_models

MODELS = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "multilingual-e5-small",
        "model_id": "intfloat/multilingual-e5-small",
        "dim": 384,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    {
        "name": "multilingual-e5-base",
        "model_id": "intfloat/multilingual-e5-base",
        "dim": 1024,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
]


def main():
    base_cfg = Config()
    chunks_path = base_cfg.chunks_path()

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print("Loaded {} chunks".format(len(chunks)))
    print()

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        cache_dir = os.path.join(
            PROJECT_ROOT, "data", "cache",
            "bench_{}".format(model_name.replace("/", "_")))
        cache_path = os.path.join(cache_dir, "passages.npy")

        if os.path.exists(cache_path):
            vecs = np.load(cache_path)
            print("SKIP {} -- cached shape={}".format(model_name, vecs.shape))
            continue

        print("Encoding: {} (dim={})".format(model_name, model_cfg["dim"]))
        cfg = Config()
        cfg.embedding_model = model_cfg["model_id"]
        cfg.embedding_dim = model_cfg["dim"]
        cfg.embedding_query_prefix = model_cfg["query_prefix"]
        cfg.embedding_passage_prefix = model_cfg["passage_prefix"]
        cfg.embed_batch_size = 64

        _loaded_models.clear()
        gc.collect()

        embedder = Embedder(cfg)

        # Apply passage prefix manually, then encode directly via the
        # SentenceTransformer model to avoid the Embedder's internal
        # cache (which is keyed on text content, not model identity).
        prefix = model_cfg["passage_prefix"]
        texts = [c["chunk_text"] for c in chunks]
        if prefix:
            texts = [prefix + t for t in texts]

        t0 = time.perf_counter()
        vecs = embedder.model.encode(
            texts,
            batch_size=cfg.embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=cfg.normalize_embeddings,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        enc_s = time.perf_counter() - t0

        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, vecs)
        print("  Saved: shape={} time={:.1f}s".format(vecs.shape, enc_s))
        print()

        _loaded_models.clear()
        gc.collect()

    print("All passages pre-encoded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
