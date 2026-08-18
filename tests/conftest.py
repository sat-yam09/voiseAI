"""Shared test fixtures for the retrieval pipeline tests.

All fixtures use tiny synthetic data -- no downloads from HuggingFace required.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Make sure the project root is on sys.path so ``from src.xxx`` works.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config


# ------------------------------------------------------------------
# Synthetic chunk data (tiny, multilingual, realistic structure)
# ------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "chunk_id": "1_p0_c0",
        "query_id": 1,
        "query": "what is a corporation",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "asm_Beng",
        "passage_index": 0,
        "chunk_index": 0,
        "is_selected": True,
        "num_words": 20,
        "chunk_text": "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.",
    },
    {
        "chunk_id": "1_p1_c0",
        "query_id": 1,
        "query": "what is a corporation",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "asm_Beng",
        "passage_index": 1,
        "chunk_index": 0,
        "is_selected": False,
        "num_words": 15,
        "chunk_text": "Today there is a growing community of certified B corps from 50 countries working together.",
    },
    {
        "chunk_id": "2_p0_c0",
        "query_id": 2,
        "query": "chart for foods low in potassium",
        "query_type": "ENTITY",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passage_index": 0,
        "chunk_index": 0,
        "is_selected": True,
        "num_words": 18,
        "chunk_text": "Low sodium low potassium foods list with nutritional data on thousands of foods for healthy diet.",
    },
    {
        "chunk_id": "2_p1_c0",
        "query_id": 2,
        "query": "chart for foods low in potassium",
        "query_type": "ENTITY",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passage_index": 1,
        "chunk_index": 0,
        "is_selected": False,
        "num_words": 22,
        "chunk_text": "High potassium foods include beans dark leafy greens potatoes squash yogurt fish avocados mushrooms and bananas.",
    },
    {
        "chunk_id": "3_p0_c0",
        "query_id": 3,
        "query": "honesty or integrity definition",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "tam_Taml",
        "passage_index": 0,
        "chunk_index": 0,
        "is_selected": True,
        "num_words": 16,
        "chunk_text": "Integrity is about conduct honesty is about adherence to the facts.",
    },
    {
        "chunk_id": "3_p1_c0",
        "query_id": 3,
        "query": "honesty or integrity definition",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "tam_Taml",
        "passage_index": 1,
        "chunk_index": 0,
        "is_selected": False,
        "num_words": 14,
        "chunk_text": "The Wikipedia page about integrity defines consistency of actions values methods.",
    },
]


@pytest.fixture
def sample_chunks():
    """Return the synthetic chunk list."""
    return [dict(c) for c in SAMPLE_CHUNKS]


@pytest.fixture
def chunks_jsonl_path(tmp_path):
    """Write sample chunks to a temp JSONL file and return the path."""
    path = tmp_path / "test_chunks.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for c in SAMPLE_CHUNKS:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture
def test_config(tmp_path):
    """Return a Config pointing to temporary directories."""
    cfg = Config()
    cfg.cache_dir = str(tmp_path / "cache")
    cfg.index_dir = str(tmp_path / "index")
    cfg.chunks_dir = str(tmp_path / "chunks")
    return cfg


@pytest.fixture
def synthetic_vectors():
    """Random but deterministic vectors matching the sample chunk count."""
    rng = np.random.RandomState(42)
    dim = 768  # matches default embedding_dim
    vecs = rng.randn(len(SAMPLE_CHUNKS), dim).astype(np.float32)
    # L2-normalize so dot product = cosine similarity.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    return vecs
