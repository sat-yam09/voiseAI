<<<<<<< HEAD
# Voice RAG Project — Member 2: RAG/Retrieval Engineer

## Project Purpose

Build a robust multilingual RAG (Retrieval-Augmented Generation) retrieval pipeline using the **ai4bharat/MSMARCO-XI** dataset. The pipeline retrieves relevant passages for multilingual queries (including Assamese, Hindi, Tamil, Bengali, etc.) and prepares context for an LLM.

## Member 2 Responsibilities

- Dataset analysis and preprocessing
- Word-based chunking with configurable size/overlap
- Multilingual embedding generation
- Vector database indexing and retrieval
- BM25 keyword retrieval
- Hybrid search (vector + BM25) with Reciprocal Rank Fusion
- Cross-encoder reranking
- Top-K context generation
- Retrieval evaluation and benchmarking

## Architecture

```
Query
  |
  v
[Embedder] --> query vector
  |
  v
[Vector Store] --+--> Top-K vector candidates
[BM25 Index]  ---+--> Top-K BM25 candidates
  |
  v
[RRF Fusion] --> unified ranked candidates
  |
  v
[Reranker] --> Top-N re-scored candidates
  |
  v
[Top-K Selector] --> final context for LLM
```

## Installation

```bash
# Activate the virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Packages installed

| Package | Purpose |
|---|---|
| `sentence-transformers` | Multilingual embedding model (XLM-R based) |
| `faiss-cpu` | Fast local vector similarity search |
| `rank-bm25` | BM25 keyword retrieval |
| `datasets` | Loading MSMARCO-XI parquet shards |
| `numpy` | Numerical operations |
| `scikit-learn` | ML utilities |
| `pytest` | Unit testing |

## Dataset Preparation

```bash
# Analyze the dataset (one-time download, cached by HuggingFace)
python src/dataset_analysis.py --max-records 500

# Preprocess into clean JSONL
python src/preprocess.py --max-records 500
```

Output: `data/cleaned/preprocessed.jsonl` (500 records)

## Chunking

```bash
# Baseline: 256 words, 32 overlap
python src/chunking.py

# Alternative configurations
python src/chunking.py --chunk-size-words 512 --overlap-words 64
python src/chunking.py --chunk-size-words 1024 --overlap-words 128
```

Output: `data/chunks/chunks_{size}_{overlap}.jsonl`

## Embeddings

The default model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions). This is a multilingual bi-encoder covering 50+ languages including all major Indic scripts.

To change the model, modify `embedding_model` and `embedding_dim` in the `Config` dataclass or pass a config file.

Embeddings are cached to disk -- re-running the same corpus skips model inference.

## Vector Search

Built automatically when using the retrieval pipeline. FAISS IndexFlatIP (inner product) is used for exact nearest-neighbor search on L2-normalized vectors.

## BM25

BM25Okapi keyword retrieval with configurable `k1` and `b` parameters (default: 1.5, 0.75). Builds from the same chunk corpus as the vector index.

## Hybrid Search

Combines vector (semantic) and BM25 (keyword) retrieval using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = sum(1 / (k + rank_i(d)))
```

Default `rrf_k = 60`. Documents appearing in both lists get the highest scores.

**Why hybrid?** BM25 excels at exact keywords (proper nouns, codes, numbers). Vector retrieval excels at semantic matching (paraphrases, synonyms, cross-lingual queries). Neither alone covers all scenarios.

## Reranking

After initial retrieval, a cross-encoder reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) scores each (query, passage) pair and re-sorts by relevance. This significantly improves precision over bi-encoder-only retrieval.

## Evaluation

```python
from src.evaluate import evaluate_query, evaluate_dataset

# Single query evaluation
relevant_ids = ["chunk_1", "chunk_5"]
retrieved_ids = ["chunk_1", "chunk_3", "chunk_5"]
metrics = evaluate_query(relevant_ids, retrieved_ids, k=5)
# Returns: recall@5, precision@5, mrr, hit@5
```

Ground truth: `is_selected=True` from MSMARCO-XI annotations.

## Running Experiments

```bash
# Compare chunking configurations (256/32, 512/64, 1024/128)
python experiments/chunking_comparison.py
```

Results saved to `experiments/chunking_comparison.json`.

## How Member 1 Calls Retrieval

```python
from src.retrieval import retrieve
from src.config import Config

# Simple one-liner
context = retrieve("What is a corporation?", top_k=5)

# Each result contains:
# - rank: 1-based final rank
# - score: reranker relevance score
# - text: chunk text (ready for LLM prompt)
# - chunk_id: stable identifier
# - source_lang, target_lang
# - latency_ms: total pipeline latency

# For detailed timing and diagnostics:
from src.retrieval import RetrievalPipeline

pipeline = RetrievalPipeline(Config())
detailed = pipeline.retrieve_raw("query", top_k=5)
# Returns: query, latency_ms, hybrid_latency_ms, rerank_latency_ms, results
```

**Note:** First call loads indices and models (may take 30-60s). Subsequent calls are fast (~50-200ms per query).

## How to Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_chunking.py -v

# Run with coverage
pytest tests/ -v --tb=short
```

## How to Run the Full Pipeline

```bash
# 1. Preprocess (first time only)
python src/preprocess.py --max-records 500

# 2. Chunk (first time only)
python src/chunking.py

# 3. Build index and test retrieval
python -c "
from src.retrieval import retrieve
results = retrieve('What is a corporation?')
for r in results:
    print(f'[{r[\"rank\"]}] score={r[\"score\"]:.3f} | {r[\"text\"][:80]}...')
"
```

## Known Limitations

1. **Sample size**: Currently uses 500 records / 5013 chunks. Full dataset is ~147 GB.
2. **Vector index**: Uses FAISS IndexFlatIP (exact search). For large corpora, consider HNSW or IVF.
3. **Embedding model**: MiniLM-L12 is fast but not state-of-the-art. Consider `bge-m3` or `multilingual-e5-large` for production.
4. **Evaluation**: Uses binary relevance (is_selected). No graded relevance available.
5. **BM25 tokenization**: Simple regex-based tokenizer. Language-specific analyzers (e.g. Hindi morphological analysis) would improve BM25 for Indic scripts.

## Recommended Next Improvements

1. **Scale up**: Increase sample size to 5,000+ records and re-evaluate.
2. **Try larger embedding models**: `BAAI/bge-m3` or `intfloat/multilingual-e5-large`.
3. **Add language-specific BM25 tokenization** for Indic scripts.
4. **Implement HNSW index** for faster retrieval at scale.
5. **Add query expansion** or HyDE for improved recall.
6. **A/B test** different reranker models.
=======
# voiseAI
>>>>>>> 9d13a05839f18c163f0061f04ab005f127a13366
