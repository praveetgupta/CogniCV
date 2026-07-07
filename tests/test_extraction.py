"""Tests for file text extraction."""

import pytest

from cognicv.extraction import ExtractionError, extract_text


class TestTxt:
    def test_utf8(self):
        assert extract_text("a.txt", "Python développeur".encode()) == "Python développeur"

    def test_latin1_fallback(self):
        text = extract_text("a.txt", "café résumé".encode("latin-1"))
        assert "caf" in text

    def test_whitespace_normalized(self):
        out = extract_text("a.txt", b"a\r\nb\n\n\n\nc\t\td")
        assert out == "a\nb\n\nc d"


class TestUnsupported:
    def test_unknown_extension_raises(self):
        with pytest.raises(ExtractionError, match="Unsupported file type"):
            extract_text("resume.xyz", b"data")

    def test_corrupt_pdf_raises(self):
        with pytest.raises(ExtractionError):
            extract_text("bad.pdf", b"this is not a pdf at all")


class TestSemanticBackend:
    def test_tfidf_scores_sane(self):
        from cognicv.semantic import _tfidf_scores

        jd = "Looking for a Python engineer with Django and PostgreSQL experience."
        good = "Python engineer, five years of Django and PostgreSQL development."
        bad = "Pastry chef specializing in sourdough bread and croissants."
        scores = _tfidf_scores(jd, [good, bad])
        assert scores[0] > scores[1]
        assert 0 <= scores[1] <= 100
        assert scores[0] > 10

    def test_empty_inputs(self):
        from cognicv.semantic import _tfidf_scores

        assert _tfidf_scores("", ["resume"]) == [0.0]
        assert _tfidf_scores("jd text here", [""]) == [0.0]
