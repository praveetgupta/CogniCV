"""Candidate profile parsing: name, contact, experience, education, skills.

Everything here is heuristic — resumes are unstructured. The rules favor
precision over recall (better to show "unknown" than a wrong number),
and every extracted figure is presented as an estimate in the UI.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field

from .skills import extract_skills

# ── Education ────────────────────────────────────────────────────────
# (level, label, pattern) — highest matching level wins.
EDUCATION_LEVELS: list[tuple[int, str, re.Pattern]] = [
    (4, "PhD / Doctorate", re.compile(r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctoral\b", re.I)),
    (3, "Master's", re.compile(
        r"\bmaster(?:'s|s)?\b"
        r"|\bm\.?\s?sc\.?\b|\bm\.?\s?tech\b|\bmba\b|\bm\.?eng\b|\bm\.?c\.?a\b"
        r"|\bm\.s\.(?!\w)|\bms\b\s+in\b",
        re.I)),
    (2, "Bachelor's", re.compile(
        r"\bbachelor(?:'s|s)?\b|\bundergraduate degree\b"
        r"|\bb\.?\s?sc\.?\b|\bb\.?\s?tech\b|\bb\.?\s?e\.?\b(?=\s+in\b|,|\s+\()"
        r"|\bb\.s\.(?!\w)|\bbs\b\s+in\b|\bba\b\s+in\b|\bb\.?c\.?a\b|\bbeng\b",
        re.I)),
    (1, "Associate / Diploma", re.compile(r"\bassociate(?:'s|s)?\s+degree\b|\bdiploma\b", re.I)),
]

EDUCATION_LABELS = {4: "PhD / Doctorate", 3: "Master's", 2: "Bachelor's", 1: "Associate / Diploma"}


def detect_education(text: str) -> tuple[int | None, str | None]:
    """Return (highest degree level 1-4, label) found in text, or (None, None)."""
    best = None
    for level, label, pattern in EDUCATION_LEVELS:
        if pattern.search(text) and (best is None or level > best[0]):
            best = (level, label)
    return best if best else (None, None)


# ── Experience estimation ────────────────────────────────────────────
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MON = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DASH = r"(?:–|—|−|-|to|through|until)"
_PRESENT = r"(?:present|current|now|today|ongoing|date)"

# "Jan 2020 – Mar 2023" / "January 2020 to Present"
_RANGE_MONTH = re.compile(
    rf"\b(?P<m1>{_MON})\.?,?\s*['’]?(?P<y1>(?:19|20)\d{{2}})\s*{_DASH}\s*"
    rf"(?:(?P<m2>{_MON})\.?,?\s*['’]?(?P<y2>(?:19|20)\d{{2}})|(?P<present>{_PRESENT}))\b",
    re.I,
)
# "03/2020 – 11/2023" / "3.2020 - present"
_RANGE_NUM = re.compile(
    rf"\b(?P<m1>0?[1-9]|1[0-2])[/.](?P<y1>(?:19|20)\d{{2}})\s*{_DASH}\s*"
    rf"(?:(?P<m2>0?[1-9]|1[0-2])[/.](?P<y2>(?:19|20)\d{{2}})|(?P<present>{_PRESENT}))\b",
    re.I,
)
# "2018 – 2022" / "2019 - Present"  (year-only; lowest confidence)
_RANGE_YEAR = re.compile(
    rf"\b(?P<y1>(?:19|20)\d{{2}})\s*{_DASH}\s*"
    rf"(?:(?P<y2>(?:19|20)\d{{2}})|(?P<present>{_PRESENT}))\b",
    re.I,
)
# "7+ years of experience", "over 5 years' relevant industry experience".
# Word-bounded repetition (max 5 words) keeps backtracking linear.
_STATED_YEARS = re.compile(
    r"(\d{1,2})(?:\.\d)?\s*\+?\s*(?:years?|yrs?)[’']?\s+(?:(?:of|in)\s+)?"
    r"(?:[\w,'-]+\s+){0,5}?(?:experience|exp\b)",
    re.I,
)


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _now_index() -> int:
    today = _dt.date.today()
    return _month_index(today.year, today.month)


def _parse_month_group(raw: str | None, default: int) -> int:
    if not raw:
        return default
    if raw.isdigit():
        return int(raw)
    return _MONTHS[raw[:3].lower()]


def _interval_from_match(
    g: dict, default_start_m: int, default_end_m: int, now: int,
) -> tuple[int, int] | None:
    """Build a sane (start, end) month-index interval from match groups."""
    start = _month_index(int(g["y1"]), _parse_month_group(g.get("m1"), default_start_m))
    if g.get("present"):
        end = now
    else:
        end = _month_index(int(g["y2"]), _parse_month_group(g.get("m2"), default_end_m))
    # Sanity: forward-in-time, not absurdly long, not in the future
    if end < start or end - start > 45 * 12 or start > now:
        return None
    return start, min(end, now)


def _collect_intervals(text: str) -> list[tuple[int, int]]:
    now = _now_index()
    intervals: list[tuple[int, int]] = []
    consumed: list[tuple[int, int]] = []  # char spans already matched

    def overlaps_consumed(span: tuple[int, int]) -> bool:
        return any(s < span[1] and span[0] < e for s, e in consumed)

    # Month-level patterns run first; the year-only pattern must not
    # re-match a span a more precise pattern already claimed.
    for regex, default_start_m, default_end_m in (
        (_RANGE_MONTH, 1, 12),
        (_RANGE_NUM, 1, 12),
        (_RANGE_YEAR, 1, 12),
    ):
        for m in regex.finditer(text):
            if overlaps_consumed(m.span()):
                continue
            interval = _interval_from_match(
                m.groupdict(), default_start_m, default_end_m, now,
            )
            if interval is not None:
                intervals.append(interval)
                consumed.append(m.span())
    return intervals


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def estimate_experience_years(text: str) -> float | None:
    """Estimate total professional experience in years.

    Combines two signals and returns the larger:
      1. Merged employment date ranges found in the text
         (overlapping jobs are not double-counted).
      2. Self-stated "N+ years of experience" claims.

    Returns None when neither signal is present. Note: date ranges in an
    education section count toward the total — treat as an estimate.
    """
    merged = _merge_intervals(_collect_intervals(text))
    from_ranges = sum(end - start + 1 for start, end in merged) / 12 if merged else None

    stated_matches = [int(m.group(1)) for m in _STATED_YEARS.finditer(text)]
    stated = float(max(stated_matches)) if stated_matches else None

    candidates = [v for v in (from_ranges, stated) if v is not None and v <= 45]
    if not candidates:
        return None
    return round(max(candidates), 1)


# ── Contact & identity ───────────────────────────────────────────────
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<![\d.])\+?\d[\d\s().\-]{7,16}\d(?![\d.])")
_LINKEDIN = re.compile(r"linkedin\.com/in/[\w\-%]+", re.I)

_NAME_STOPWORDS = {
    "resume", "curriculum", "vitae", "cv", "summary", "objective", "profile",
    "professional", "experience", "education", "skills", "contact", "about",
    "portfolio", "senior", "junior", "engineer", "developer", "manager",
    "analyst", "scientist", "designer", "consultant", "architect", "phone",
    "email", "address", "confidential", "page",
}
_NAME_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.'\-]*$")


def _line_as_name(line: str) -> str | None:
    """Return the line formatted as a name if it plausibly is one."""
    if "@" in line or "http" in line.lower() or any(ch.isdigit() for ch in line):
        return None
    if len(line) > 40:
        return None
    tokens = line.replace(",", " ").split()
    if not 2 <= len(tokens) <= 4:
        return None
    if not all(_NAME_TOKEN.match(t) and t[0].isupper() for t in tokens):
        return None
    if any(t.lower().strip(".") in _NAME_STOPWORDS for t in tokens):
        return None
    return " ".join(t if t.isupper() and len(t) <= 3 else t.title() for t in tokens)


def extract_name(text: str, filename: str = "") -> str:
    """Best-effort candidate name: scan the first lines, fall back to filename."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()][:8]
    for line in lines:
        name = _line_as_name(line)
        if name:
            return name

    # Fallback: clean up the filename ("john_doe_resume.pdf" -> "John Doe")
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
    stem = re.sub(r"[_\-.]+", " ", stem)  # split separators before word matching
    stem = re.sub(r"(?i)\b(resume|cv|curriculum|vitae|final|updated|new|copy|v?\d+)\b", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if stem:
        return " ".join(w.title() for w in stem.split()[:4])
    return "Unknown Candidate"


# ── Profile ──────────────────────────────────────────────────────────
@dataclass
class CandidateProfile:
    filename: str
    name: str
    email: str | None
    phone: str | None
    linkedin: str | None
    skills: set[str] = field(default_factory=set)  # canonical forms
    years_experience: float | None = None
    education_level: int | None = None
    education_label: str | None = None
    word_count: int = 0
    text: str = ""


def parse_candidate(filename: str, text: str) -> CandidateProfile:
    """Parse resume text into a structured CandidateProfile."""
    email_m = _EMAIL.search(text)
    phone_m = _PHONE.search(text)
    linkedin_m = _LINKEDIN.search(text)
    level, label = detect_education(text)
    return CandidateProfile(
        filename=filename,
        name=extract_name(text, filename),
        email=email_m.group(0) if email_m else None,
        phone=phone_m.group(0).strip() if phone_m else None,
        linkedin=linkedin_m.group(0) if linkedin_m else None,
        skills=extract_skills(text),
        years_experience=estimate_experience_years(text),
        education_level=level,
        education_label=label,
        word_count=len(text.split()),
        text=text,
    )
