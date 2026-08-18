"""Build the full retrieval index (embeddings + vector store + BM25).

This script:
1. Loads chunks from the JSONL file.
2. Embeds all chunks using the multilingual embedding model.
3. Builds a FAISS vector index.
4. Builds a BM25 index.
5. Saves both to disk.
6. Runs a demo retrieval.

Usage:
    python tests/build_index.py
"""

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
from src.retriever import VectorRetriever
from src.bm25 import BM25Retriever

def main():
    cfg = Config()
    chunks_path = cfg.chunks_path()
    
    print(f"Loading chunks from: {chunks_path}")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks")

    # Build vector index
    print("\n--- Building Vector Index ---")
    vr = VectorRetriever(cfg)
    t0 = time.perf_counter()
    n = vr.build_index(chunks)
    vec_time = time.perf_counter() - t0
    print(f"Vector index built: {n} vectors in {vec_time:.1f}s")

    # Build BM25 index
    print("\n--- Building BM25 Index ---")
    bm = BM25Retriever(cfg)
    t1 = time.perf_counter()
    bm.build_index(chunks)
    bm.save()
    bm25_time = time.perf_counter() - t1
    print(f"BM25 index built: {bm.size} docs in {bm25_time:.1f}s")

    # Demo retrieval
    print("\n--- Demo Retrieval ---")
    sample_queries = [
        chunks[0]["query"],  # Assamese query
        chunks[100]["query"],  # Another query
    ]
    
    for query in sample_queries:
        print(f"\nQuery: {query}")
        results = vr.search(query, top_k=3)
        for r in results:
            print(f"  [{r['rank']}] score={r['score']:.3f} | {r['chunk_text'][:60]}")

    print("\nIndex build complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
