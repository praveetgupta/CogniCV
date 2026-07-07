"""Job-description parsing: skills split into must-have vs nice-to-have,
plus required experience and education.

Section-aware: skills under a "Requirements"-style heading are must-haves,
skills under a "Nice to have"-style heading are nice-to-haves. Skills in
untagged text default to must-have (JDs mention what they need). Inline
cues ("X is a plus") demote individual sentences to nice-to-have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .candidate import detect_education
from .skills import extract_skills
from .taxonomy import SOFT_GENERIC

_REQUIRED_HINTS = (
    "requirement", "required", "must have", "must-have", "qualifications",
    "what you'll need", "what you will need", "what we're looking for",
    "what we are looking for", "who you are", "your profile", "essential",
    "minimum", "basic qualifications", "you have", "you bring", "skills needed",
    "key skills", "technical skills", "responsibilities", "what you'll do",
    "what you will do", "about you", "core competencies",
)
_PREFERRED_HINTS = (
    "preferred", "nice to have", "nice-to-have", "nice to haves", "bonus",
    "plus", "good to have", "great to have", "desirable", "desired",
    "would be a plus", "optional", "additionally", "extra credit",
    "stand out", "even better",
)

# Demotion cues must be explicit phrases — a bare "plus"/"bonus" would
# wrongly demote lines like "5 plus years" or "annual bonus".
_INLINE_NICE = re.compile(
    r"\ba\s+(?:big\s+|huge\s+|strong\s+)?(?:plus|bonus|advantage)\b"
    r"|\bbonus points?\b"
    r"|\bnice[\s-]to[\s-]have\b"
    r"|\bpreferred but not required\b",
    re.I,
)

_YEARS_REQ = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:to|-|–|—)?\s*(?:\d{1,2})?\s*(?:years?|yrs?)",
    re.I,
)


def _is_heading(line: str) -> bool:
    stripped = line.strip().strip("#*_ ").rstrip(":")
    if not stripped or len(stripped) > 70:
        return False
    # Headings end with ':' or are very short; a longer sentence like
    # "Kubernetes experience is a plus" must NOT count as a heading
    # (it's handled by the inline-demotion rule instead).
    return line.strip().endswith(":") or len(stripped.split()) <= 5


def _heading_bucket(line: str) -> str | None:
    lowered = line.lower()
    if any(h in lowered for h in _PREFERRED_HINTS):
        return "nice"
    if any(h in lowered for h in _REQUIRED_HINTS):
        return "must"
    return None


@dataclass
class JobRequirements:
    text: str
    skills: set[str] = field(default_factory=set)        # all detected (canonical)
    must_have: set[str] = field(default_factory=set)
    nice_to_have: set[str] = field(default_factory=set)
    min_years: float | None = None
    education_level: int | None = None
    education_label: str | None = None


def _required_years(text: str) -> float | None:
    """Largest 'N years' figure that appears near the word 'experience'."""
    years: list[int] = []
    for m in _YEARS_REQ.finditer(text):
        window = text[max(0, m.start() - 60): m.end() + 60].lower()
        if "experience" in window or "exp." in window:
            value = int(m.group(1))
            if 0 < value <= 30:
                years.append(value)
    return float(max(years)) if years else None


def parse_jd(text: str) -> JobRequirements:
    """Parse a job description into structured requirements."""
    must_text_parts: list[str] = []
    nice_text_parts: list[str] = []

    bucket = "must"
    for line in text.split("\n"):
        if _is_heading(line):
            new_bucket = _heading_bucket(line)
            if new_bucket:
                bucket = new_bucket
                # scan the heading line itself in its own bucket, in case
                # it carries skills ("Bonus: Kubernetes, Terraform")
                (nice_text_parts if bucket == "nice" else must_text_parts).append(line)
                continue
        # Inline demotion: "Kubernetes experience is a plus"
        if bucket == "must" and _INLINE_NICE.search(line):
            nice_text_parts.append(line)
        elif bucket == "nice":
            nice_text_parts.append(line)
        else:
            must_text_parts.append(line)

    must_skills = extract_skills("\n".join(must_text_parts)) - SOFT_GENERIC
    nice_skills = extract_skills("\n".join(nice_text_parts)) - SOFT_GENERIC
    nice_skills -= must_skills  # a skill in both zones counts as required

    level, label = detect_education(text)
    return JobRequirements(
        text=text,
        skills=must_skills | nice_skills,
        must_have=must_skills,
        nice_to_have=nice_skills,
        min_years=_required_years(text),
        education_level=level,
        education_label=label,
    )
