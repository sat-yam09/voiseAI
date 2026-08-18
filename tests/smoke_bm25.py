"""Extended smoke test for BM25 with matching-language queries."""

import json
import sys
import os

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.bm25 import BM25Retriever

chunks = []
with open("data/chunks/chunks_256_32.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

cfg = Config()
bm = BM25Retriever(cfg)
bm.build_index(chunks)

# Use the query from the chunk metadata (which is in the same language)
sample_query = chunks[0]["query"]
print(f"Sample query from data: {sample_query}")

results = bm.search(sample_query, top_k=5)
for r in results:
    print(f"  [{r['rank']}] score={r['score']:.2f} selected={r['is_selected']} | {r['chunk_text'][:60]}")

# Also test save/load
bm.save("data/index/bm25_test.json")
bm2 = BM25Retriever(cfg)
bm2.load("data/index/bm25_test.json")
print(f"\nSave/load OK. Index size: {bm2.size}")

# Test with chunk_ids from same query
same_q_chunks = [c for c in chunks if c["query_id"] == chunks[0]["query_id"]]
relevant_ids = [c["chunk_id"] for c in same_q_chunks if c.get("is_selected")]
retrieved_ids = [r["chunk_id"] for r in results]

from src.evaluate import evaluate_query
metrics = evaluate_query(relevant_ids, retrieved_ids, k=5)
print(f"\nEvaluation (same-query): {metrics}")
print("Smoke test complete.")
