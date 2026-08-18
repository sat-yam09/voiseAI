"""Incremental E5-base encoding: encode chunks in small batches, save progress.

Resumes from where it left off if interrupted. Each batch is saved to a
separate .npy file, and final assembly happens after all batches complete.

Usage:
    python experiments/build_e5_index.py
"""
import json, os, sys, time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.config import Config
from src.embeddings import Embedder
from src.vector_store import VectorStore

PROGRESS_DIR = "data/cache/e5_progress"
BATCH_SIZE = 32

def main():
    cfg = Config()
    chunks_path = cfg.chunks_path()
    e5_index_dir = "data/index/e5_base"

    # Load chunks
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    total = len(chunks)
    print("Total chunks: {}".format(total))

    # E5 config
    e5_cfg = Config()
    e5_cfg.embedding_model = "intfloat/multilingual-e5-base"
    e5_cfg.embedding_dim = 768
    e5_cfg.embedding_query_prefix = "query: "
    e5_cfg.embedding_passage_prefix = "passage: "

    embedder = Embedder(e5_cfg)

    # Apply passage prefix to all texts
    texts = [c["chunk_text"] for c in chunks]
    prefixed = embedder._apply_passage_prefix(texts)

    # Check progress
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    done = 0
    for i in range(n_batches):
        path = os.path.join(PROGRESS_DIR, "batch_{}.npy".format(i))
        if os.path.exists(path):
            done += 1

    print("Already encoded: {}/{} batches".format(done, n_batches))

    if done < n_batches:
        print("Encoding remaining batches ...")
        t_start = time.perf_counter()
        for i in range(n_batches):
            path = os.path.join(PROGRESS_DIR, "batch_{}.npy".format(i))
            if os.path.exists(path):
                continue
            start = i * BATCH_SIZE
            end = min(start + BATCH_SIZE, total)
            batch = prefixed[start:end]

            t0 = time.perf_counter()
            vecs = embedder.model.encode(
                batch,
                normalize_embeddings=e5_cfg.normalize_embeddings,
            )
            vecs = np.asarray(vecs, dtype=np.float32)
            np.save(path, vecs)
            batch_ms = (time.perf_counter() - t0) * 1000
            elapsed = time.perf_counter() - t_start
            remaining = (n_batches - i - 1) * (elapsed / (i - done + 1)) if i > done else 0
            print("  Batch {}/{} done ({:.0f}ms)  elapsed={:.0f}s  eta={:.0f}s".format(
                i + 1, n_batches, batch_ms, elapsed, remaining))
            sys.stdout.flush()

    # Assemble all batches
    print("Assembling {} batches ...".format(n_batches))
    all_vecs = []
    for i in range(n_batches):
        path = os.path.join(PROGRESS_DIR, "batch_{}.npy".format(i))
        all_vecs.append(np.load(path))
    vectors = np.vstack(all_vecs)
    print("Final shape: {}".format(vectors.shape))

    # Build and save index
    os.makedirs(e5_index_dir, exist_ok=True)
    e5_cfg.index_dir = e5_index_dir

    metadata = [{
        "chunk_id": c["chunk_id"],
        "chunk_text": c["chunk_text"],
        "query_id": c.get("query_id"),
        "query": c.get("query"),
        "is_selected": c.get("is_selected", False),
    } for c in chunks]

    store = VectorStore(e5_cfg)
    store.build(vectors, metadata)
    store.save()
    print("Index saved to {}/  ({} vectors)".format(e5_index_dir, store.size))

if __name__ == "__main__":
    main()
