"""Semantic similarity between resumes and a job description.

Two backends, chosen automatically:
  * "embeddings" — sentence-transformers (all-MiniLM-L6-v2) if installed.
    Understands meaning, not just shared words ("built predictive models"
    ≈ "machine learning experience").
  * "tfidf" — scikit-learn TF-IDF + cosine (always available).

Both return calibrated 0–100 scores. Raw cosines are not comparable across
backends or document lengths, so each backend rescales against an empirical
range for the resume-vs-JD task; the constants are documented inline.
"""

from __future__ import annotations

import re
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF cosine between a multi-page resume and a JD is structurally low
# (documents differ hugely in length/vocabulary). ~0.35 is an empirical
# ceiling for a strongly matching pair, so it maps to 100.
_TFIDF_CEILING = 0.35

# MiniLM cosine for unrelated resume/JD pairs sits around 0.2-0.3;
# strong matches reach ~0.7. Map that band onto 0-100.
_EMB_FLOOR, _EMB_CEILING = 0.25, 0.70

# Embedding models have token limits; the first ~2500 words carry the
# signal (skills, recent roles) anyway.
_EMB_MAX_WORDS = 2500


@lru_cache(maxsize=1)
def _load_embedding_model():
    """Return a SentenceTransformer or None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def backend_name() -> str:
    return "embeddings" if _load_embedding_model() is not None else "tfidf"


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^\w\s\-.+#/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)


def _tfidf_scores(jd_text: str, resume_texts: list[str]) -> list[float]:
    jd_clean = _clean(jd_text)
    resumes_clean = [_clean(t) for t in resume_texts]
    if not jd_clean or all(not r for r in resumes_clean):
        return [0.0] * len(resume_texts)

    corpus = [jd_clean] + [r or " " for r in resumes_clean]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
        matrix = vectorizer.fit_transform(corpus)
        cosines = cosine_similarity(matrix[0:1], matrix[1:])[0]
    except ValueError:
        return [0.0] * len(resume_texts)
    return [min(float(c) / _TFIDF_CEILING, 1.0) * 100 for c in cosines]


def _embedding_scores(jd_text: str, resume_texts: list[str]) -> list[float]:
    model = _load_embedding_model()
    truncate = lambda t: " ".join(t.split()[:_EMB_MAX_WORDS])  # noqa: E731
    texts = [truncate(jd_text)] + [truncate(t) for t in resume_texts]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    jd_vec, resume_vecs = embeddings[0], embeddings[1:]
    scores = []
    for vec in resume_vecs:
        cos = float(jd_vec @ vec)
        scaled = (cos - _EMB_FLOOR) / (_EMB_CEILING - _EMB_FLOOR)
        scores.append(max(0.0, min(scaled, 1.0)) * 100)
    return scores


def semantic_scores(jd_text: str, resume_texts: list[str]) -> list[float]:
    """Score each resume's full-text alignment with the JD (0-100 each)."""
    if not resume_texts:
        return []
    if backend_name() == "embeddings":
        try:
            return _embedding_scores(jd_text, resume_texts)
        except Exception:
            pass  # fall back to TF-IDF on any runtime failure
    return _tfidf_scores(jd_text, resume_texts)
