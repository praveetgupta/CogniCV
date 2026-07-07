"""CogniCV — resume screening and ranking engine.

Public API:
    extract_text        — file bytes -> plain text
    parse_candidate     — text -> CandidateProfile
    parse_jd            — JD text -> JobRequirements
    score_candidates    — profiles + JD -> ranked ScoredCandidate list
"""

__version__ = "2.0.0"

from .extraction import extract_text
from .candidate import CandidateProfile, parse_candidate
from .jd import JobRequirements, parse_jd
from .scoring import ScoreWeights, ScoredCandidate, score_candidates

__all__ = [
    "extract_text",
    "CandidateProfile",
    "parse_candidate",
    "JobRequirements",
    "parse_jd",
    "ScoreWeights",
    "ScoredCandidate",
    "score_candidates",
    "__version__",
]
