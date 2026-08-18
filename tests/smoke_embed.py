"""Quick test: embed just 10 chunks + search."""

import json
import sys
import os
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.embeddings import Embedder

cfg = Config()
embedder = Embedder(cfg)

# Load just 10 chunks
chunks = []
with open(cfg.chunks_path(), "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if line.strip() and i < 10:
            chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks")
print(f"Model: {cfg.embedding_model}")

t0 = time.perf_counter()
vectors = embedder.encode_chunks(chunks)
t1 = time.perf_counter()
print(f"Encoded {len(chunks)} chunks in {t1-t0:.2f}s")
print(f"Vector shape: {vectors.shape}")
print(f"Vector dtype: {vectors.dtype}")

# Search
q = embedder.encode_query(chunks[0]["query"])
print(f"\nQuery vector shape: {q.shape}")

import numpy as np
scores = vectors @ q
order = np.argsort(scores)[::-1]
for rank, idx in enumerate(order[:5], 1):
    print(f"  [{rank}] score={scores[idx]:.3f} | {chunks[idx]['chunk_text'][:60]}")

print("\nEmbedding test complete!")
