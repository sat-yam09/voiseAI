"""Tests for the preprocessing module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import normalize_text, clean_language, clean_passages


class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert normalize_text("  hello  ") == "hello"

    def test_handles_nbsp(self):
        assert normalize_text("hello\u00a0world") == "hello world"

    def test_removes_zero_width(self):
        assert normalize_text("hel\u200blo") == "hello"

    def test_preserves_case(self):
        assert normalize_text("Hello WORLD") == "Hello WORLD"

    def test_preserves_unicode(self):
        assert normalize_text("নমস্কাৰ বিশ্ব") == "নমস্কাৰ বিশ্ব"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_tabs_and_newlines(self):
        assert normalize_text("hello\t\nworld") == "hello world"


class TestCleanLanguage:
    def test_removes_empty_passages(self):
        raw = ["hello", "", "world"]
        flags = [False, False, False]
        cleaned, dup, empty, empty_sel = clean_language(raw, flags)
        assert len(cleaned) == 2
        assert empty == 1

    def test_deduplicates(self):
        raw = ["hello", "hello", "world"]
        flags = [False, False, False]
        cleaned, dup, empty, empty_sel = clean_language(raw, flags)
        assert len(cleaned) == 2
        assert dup == 1

    def test_preserves_selected_flag(self):
        raw = ["hello", "world"]
        flags = [True, False]
        cleaned, _, _, _ = clean_language(raw, flags)
        assert cleaned[0]["is_selected"] is True
        assert cleaned[1]["is_selected"] is False

    def test_counts_empty_selected(self):
        raw = ["hello", ""]
        flags = [True, True]
        cleaned, _, _, empty_sel = clean_language(raw, flags)
        assert empty_sel == 1


class TestCleanPassages:
    def test_handles_missing_keys(self):
        result = clean_passages({})
        assert result["english"] == []
        assert result["translated"] == []

    def test_cleans_both_languages(self):
        passages = {
            "English_passages": ["hello world", "test passage"],
            "Translated_passages": ["নমস্কাৰ"],
            "is_selected": [True, False],
        }
        result = clean_passages(passages)
        assert len(result["english"]) == 2
        assert len(result["translated"]) == 1

    def test_handles_none_passages(self):
        result = clean_passages(None)
        assert result["english"] == []
        assert result["translated"] == []
