"""Tests for the chunking module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.chunking import validate_params, chunk_words, chunk_passage


class TestValidateParams:
    def test_valid_params(self):
        validate_params(256, 32)

    def test_zero_overlap(self):
        validate_params(256, 0)

    def test_chunk_size_too_small(self):
        with pytest.raises(ValueError, match=">= 1"):
            validate_params(0, 0)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match=">= 0"):
            validate_params(256, -1)

    def test_overlap_equals_chunk_size(self):
        with pytest.raises(ValueError, match="strictly smaller"):
            validate_params(256, 256)

    def test_overlap_greater_than_chunk_size(self):
        with pytest.raises(ValueError, match="strictly smaller"):
            validate_params(256, 300)


class TestChunkWords:
    def test_short_text(self):
        words = ["a", "b", "c"]
        slices = list(chunk_words(words, 5, 1))
        assert len(slices) == 1
        assert slices[0] == (0, 3)

    def test_exact_chunk_size(self):
        words = list(range(10))
        slices = list(chunk_words(words, 10, 0))
        assert len(slices) == 1
        assert slices[0] == (0, 10)

    def test_overlap_produces_overlapping_chunks(self):
        words = list(range(20))
        slices = list(chunk_words(words, 10, 3))
        # First chunk: 0..10, second chunk starts at 7
        assert slices[0] == (0, 10)
        assert slices[1] == (7, 17)
        assert slices[2] == (14, 20)

    def test_no_overlap(self):
        words = list(range(20))
        slices = list(chunk_words(words, 10, 0))
        assert slices[0] == (0, 10)
        assert slices[1] == (10, 20)
        assert len(slices) == 2

    def test_single_word_chunks(self):
        words = ["a", "b", "c"]
        slices = list(chunk_words(words, 1, 0))
        assert len(slices) == 3


class TestChunkPassage:
    def test_short_passage(self):
        text = "hello world test"
        chunks = chunk_passage(text, 256, 32)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_passage(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = chunk_passage(text, 100, 20)
        assert len(chunks) > 1
        # Verify overlap: last 20 words of chunk 0 == first 20 words of chunk 1
        w0 = chunks[0].split()
        w1 = chunks[1].split()
        assert w0[-20:] == w1[:20]

    def test_preserves_content(self):
        text = "the quick brown fox jumps over the lazy dog"
        chunks = chunk_passage(text, 5, 2)
        joined = " ".join(chunks)
        # All original words should appear in order
        for word in text.split():
            assert word in joined
