"""Tests for the evaluation module (unit-level, no model downloads)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.evaluate import evaluate_query, _mean, _median, _percentile


class TestEvaluateQuery:
    def test_perfect_retrieval(self):
        relevant = ["a", "b", "c"]
        retrieved = ["a", "b", "c", "d", "e"]
        result = evaluate_query(relevant, retrieved, k=5)
        assert result["recall_k"] == 1.0
        assert result["precision_k"] == 0.6  # 3/5
        assert result["mrr"] == 1.0
        assert result["hit_k"] == 1.0

    def test_no_relevant_in_top_k(self):
        relevant = ["x", "y"]
        retrieved = ["a", "b", "c"]
        result = evaluate_query(relevant, retrieved, k=3)
        assert result["recall_k"] == 0.0
        assert result["precision_k"] == 0.0
        assert result["mrr"] == 0.0
        assert result["hit_k"] == 0.0

    def test_first_relevant_at_rank_2(self):
        relevant = ["b"]
        retrieved = ["a", "b", "c"]
        result = evaluate_query(relevant, retrieved, k=5)
        assert result["recall_k"] == 1.0
        assert result["mrr"] == 0.5  # 1/2

    def test_empty_relevant(self):
        result = evaluate_query([], ["a", "b"], k=5)
        assert result["recall_k"] == 0.0

    def test_partial_recall(self):
        relevant = ["a", "b", "c"]
        retrieved = ["a", "d", "e"]
        result = evaluate_query(relevant, retrieved, k=3)
        assert result["recall_k"] == pytest.approx(1 / 3)


class TestHelpers:
    def test_mean(self):
        assert _mean([1, 2, 3]) == 2.0
        assert _mean([]) == 0.0

    def test_median(self):
        assert _median([1, 3, 2]) == 2.0
        assert _median([1, 2]) == 1.5

    def test_percentile(self):
        assert _percentile([10, 20, 30], 50) == 20.0
        assert _percentile([10, 20, 30], 0) == 10.0
        assert _percentile([10, 20, 30], 100) == 30.0
