"""Skill extraction: match taxonomy skills in free text with precise regexes.

Two classes of patterns:
  * Generic — built from every canonical + alias with word-ish boundaries
    that tolerate `+`, `#`, `.` (so "c++", "c#", ".net" work) and treat
    spaces/hyphens interchangeably ("scikit-learn" == "scikit learn").
  * Overrides — hand-tuned regexes for ambiguous tokens that would
    otherwise produce false positives ("Go" the language vs "go to",
    "Spring" the framework vs "Spring 2024", "Excel" vs "excel in ...").
"""

from __future__ import annotations

import re

from .taxonomy import ALIASES, ALL_CANONICALS, CATEGORY_ORDER, category_of, display_name

# Characters that can be part of a skill token; a match must not be
# embedded inside a longer token made of these.
_B_L = r"(?<![\w+#])"   # left boundary
_B_R = r"(?![\w+#])"    # right boundary

# List context: the token sits in a comma/slash/bullet-separated list,
# e.g. "Python, Go, Rust" or "C/C++" or "• R". Used for one/two-letter
# language names that are common English words or letters.
_LIST_BEFORE = r"(?:(?<=[,;/(|•·–-]\s)|(?<=[,;/(|•·]))"
_LIST_AFTER = r"(?=\s*[,;/)|•·]|\s*$)"

# Hand-tuned patterns for ambiguous skills (canonical -> regex).
_OVERRIDES: dict[str, str] = {
    # "Go" only in list context or explicit "go programming/language/golang"
    "go": (
        rf"{_B_L}golang{_B_R}"
        rf"|{_B_L}go(?=\s*[,;/)|])"
        rf"|{_LIST_BEFORE}go{_B_R}"
        rf"|{_B_L}go\s+(?:programming|language|lang\b|developer)"
    ),
    # "R" only in list context or "R programming/language" or RStudio
    "r": (
        rf"{_B_L}r(?=\s*[,;/)|])"
        rf"|{_LIST_BEFORE}r{_B_R}(?![.+])"
        rf"|{_B_L}r\s+(?:programming|language)"
        rf"|{_B_L}rstudio{_B_R}"
    ),
    # "C" only in list context or "C programming"
    "c": (
        rf"(?<![\w+#.])c(?=\s*[,;/)|])(?!\s*[+#])"
        rf"|{_LIST_BEFORE}c(?![\w+#.])"
        rf"|{_B_L}c\s+programming"
    ),
    # "Spring" the framework, not the season/semester
    "spring": (
        rf"{_B_L}spring(?:\s?boot|\s+framework|\s+mvc|\s+cloud|\s+security|\s+data|\s+batch){_B_R}"
        rf"|{_B_L}spring{_B_R}(?!\s*(?:20\d\d|['’]\d\d|semester|quarter|term|break|of\s+\d{{4}}))"
    ),
    # "Excel" the tool, not the verb ("excel in/at ...")
    "excel": (
        rf"{_B_L}(?:microsoft|ms)\s+excel{_B_R}"
        rf"|{_B_L}excel{_B_R}(?!\s+(?:in|at)\b)"
    ),
    # "D3" needs the digit boundary relaxed ("d3.js" contains ".")
    "d3": rf"{_B_L}d3(?:\.?js)?{_B_R}",
    # "node" alone is risky ("node in a graph") — require .js-ish context
    "node.js": (
        rf"{_B_L}node[\s.]?js{_B_R}"
        rf"|{_B_L}node(?=\s*[,;/)|])"
        rf"|{_LIST_BEFORE}node{_B_R}"
    ),
    # "express" the framework, not "express interest"
    "express": (
        rf"{_B_L}express[\s.]?js{_B_R}"
        rf"|{_B_L}express(?=\s*[,;/)|])"
    ),
    # "sketch" the design tool: only in list context (too common a word)
    "sketch": rf"{_B_L}sketch(?=\s*[,;/)|])|{_LIST_BEFORE}sketch{_B_R}",
    # "oracle" the database, avoid "Oracle of ..." style prose is rare; keep simple
}


def _generic_pattern(term: str) -> str:
    """Build a boundary-safe pattern; spaces/hyphens are interchangeable."""
    esc = re.escape(term)
    # Single pass so the inserted class isn't rewritten by a second replace
    esc = re.sub(r"\\[ \-]", lambda _: r"[\s\-]+", esc)
    return rf"{_B_L}{esc}{_B_R}"


def _build_patterns() -> list[tuple[str, re.Pattern]]:
    compiled = []
    for canon in ALL_CANONICALS:
        if canon in _OVERRIDES:
            pattern = _OVERRIDES[canon]
        else:
            terms = (canon, *ALIASES[canon])
            pattern = "|".join(_generic_pattern(t) for t in terms)
        compiled.append((canon, re.compile(pattern, re.IGNORECASE)))
    return compiled


_PATTERNS: list[tuple[str, re.Pattern]] = _build_patterns()

# Cap the text scanned per document; resumes/JDs are far below this and it
# bounds worst-case latency on malformed extractions.
_MAX_SCAN_CHARS = 200_000


def extract_skills(text: str) -> set[str]:
    """Return the set of canonical skills found in `text`."""
    if not text:
        return set()
    scan = text[:_MAX_SCAN_CHARS]
    return {canon for canon, pat in _PATTERNS if pat.search(scan)}


def group_by_category(canons: set[str] | list[str]) -> dict[str, list[str]]:
    """Group canonical skills into {category: [display names]}, ordered."""
    buckets: dict[str, list[str]] = {}
    for canon in canons:
        buckets.setdefault(category_of(canon), []).append(display_name(canon))
    ordered = {}
    for cat in (*CATEGORY_ORDER, "Other"):
        if cat in buckets:
            ordered[cat] = sorted(buckets[cat], key=str.lower)
    return ordered


def displays(canons: set[str] | list[str]) -> list[str]:
    """Sorted display names for a set of canonical skills."""
    return sorted((display_name(c) for c in canons), key=str.lower)
