"""Scoring engine: blend skill coverage, semantic fit, experience and
education into a ranked shortlist.

Design principles:
  * Must-have skills weigh 3x nice-to-haves in the coverage score.
  * Components the JD doesn't specify (e.g. no years requirement) are
    excluded and the remaining weights renormalized — candidates are never
    penalized for a requirement that doesn't exist.
  * Components the *candidate* is missing data for (e.g. no parseable
    dates) are also excluded, but surfaced as a screening flag so a human
    looks at them — unknown is not the same as unqualified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .candidate import CandidateProfile
from .jd import JobRequirements
from .semantic import semantic_scores
from .taxonomy import SOFT_GENERIC

MUST_HAVE_WEIGHT = 3.0  # relative to a nice-to-have's weight of 1.0


@dataclass
class ScoreWeights:
    """Relative importance of each component; normalized before use."""
    skills: float = 0.45
    semantic: float = 0.25
    experience: float = 0.20
    education: float = 0.10

    def normalized(self) -> "ScoreWeights":
        total = self.skills + self.semantic + self.experience + self.education
        if total <= 0:
            return ScoreWeights(1.0, 0.0, 0.0, 0.0)
        return ScoreWeights(
            self.skills / total, self.semantic / total,
            self.experience / total, self.education / total,
        )


@dataclass
class ScoredCandidate:
    profile: CandidateProfile
    total: float
    skill_score: float
    semantic_score: float
    experience_score: float | None   # None -> not assessed
    education_score: float | None    # None -> not assessed
    matched_must: list[str] = field(default_factory=list)    # canonical
    missing_must: list[str] = field(default_factory=list)
    matched_nice: list[str] = field(default_factory=list)
    missing_nice: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def _skill_score(profile: CandidateProfile, jd: JobRequirements) -> tuple[float, dict]:
    matched_must = profile.skills & jd.must_have
    missing_must = jd.must_have - profile.skills
    matched_nice = profile.skills & jd.nice_to_have
    missing_nice = jd.nice_to_have - profile.skills
    extra = profile.skills - jd.skills - SOFT_GENERIC

    denom = MUST_HAVE_WEIGHT * len(jd.must_have) + len(jd.nice_to_have)
    if denom == 0:
        score = 0.0
    else:
        score = 100 * (MUST_HAVE_WEIGHT * len(matched_must) + len(matched_nice)) / denom

    detail = {
        "matched_must": sorted(matched_must),
        "missing_must": sorted(missing_must),
        "matched_nice": sorted(matched_nice),
        "missing_nice": sorted(missing_nice),
        "extra": sorted(extra),
    }
    return score, detail


def _experience_score(profile: CandidateProfile, jd: JobRequirements) -> float | None:
    if jd.min_years is None:
        return None  # JD doesn't ask -> not assessed
    if profile.years_experience is None:
        return None  # unknown -> not assessed (flagged separately)
    ratio = profile.years_experience / jd.min_years
    return min(ratio, 1.0) * 100


def _education_score(profile: CandidateProfile, jd: JobRequirements) -> float | None:
    if jd.education_level is None:
        return None
    if profile.education_level is None:
        return None
    gap = profile.education_level - jd.education_level
    if gap >= 0:
        return 100.0
    if gap == -1:
        return 50.0
    return 25.0


def _blend(components: list[tuple[float | None, float]]) -> float:
    """Weighted average over available components, weights renormalized."""
    available = [(score, w) for score, w in components if score is not None and w > 0]
    if not available:
        return 0.0
    total_weight = sum(w for _, w in available)
    return sum(score * w for score, w in available) / total_weight


def _screening_flags(
    profile: CandidateProfile, jd: JobRequirements, detail: dict,
) -> list[str]:
    flags = []
    if detail["missing_must"]:
        n = len(detail["missing_must"])
        flags.append(f"Missing {n} must-have skill{'s' if n > 1 else ''}")
    if jd.min_years is not None and profile.years_experience is None:
        flags.append("Experience could not be estimated from resume dates")
    elif (
        jd.min_years is not None
        and profile.years_experience is not None
        and profile.years_experience < jd.min_years
    ):
        flags.append(
            f"~{profile.years_experience:g} yrs experience vs {jd.min_years:g} required"
        )
    if jd.education_level is not None and profile.education_level is None:
        flags.append("Education level not detected")
    if not profile.email and not profile.phone:
        flags.append("No contact details found")
    if profile.word_count < 120:
        flags.append("Very little text extracted — possibly a scanned/image PDF")
    return flags


def score_candidates(
    profiles: list[CandidateProfile],
    jd: JobRequirements,
    weights: ScoreWeights | None = None,
    semantic_pcts: list[float] | None = None,
) -> list[ScoredCandidate]:
    """Score and rank candidates against the JD (best first).

    `semantic_pcts` can be supplied to reuse precomputed semantic scores
    (e.g. cached across UI reruns); otherwise they are computed here.
    """
    if not profiles:
        return []
    w = (weights or ScoreWeights()).normalized()

    if semantic_pcts is None:
        semantic_pcts = semantic_scores(jd.text, [p.text for p in profiles])

    results = []
    for profile, sem_pct in zip(profiles, semantic_pcts):
        skill_pct, detail = _skill_score(profile, jd)
        exp_pct = _experience_score(profile, jd)
        edu_pct = _education_score(profile, jd)

        total = _blend([
            (skill_pct, w.skills),
            (sem_pct, w.semantic),
            (exp_pct, w.experience),
            (edu_pct, w.education),
        ])

        results.append(ScoredCandidate(
            profile=profile,
            total=round(total, 1),
            skill_score=round(skill_pct, 1),
            semantic_score=round(sem_pct, 1),
            experience_score=round(exp_pct, 1) if exp_pct is not None else None,
            education_score=round(edu_pct, 1) if edu_pct is not None else None,
            matched_must=detail["matched_must"],
            missing_must=detail["missing_must"],
            matched_nice=detail["matched_nice"],
            missing_nice=detail["missing_nice"],
            extra_skills=detail["extra"],
            flags=_screening_flags(profile, jd, detail),
        ))

    results.sort(key=lambda r: (-r.total, -r.skill_score, r.profile.name))
    return results


def score_label(score: float) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 55:
        return "Good match"
    if score >= 35:
        return "Partial match"
    return "Weak match"
