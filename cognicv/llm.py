"""Optional Claude-powered deep review of candidates.

Keyword matching cannot tell "8 years of PyTorch in production" apart from
"no PyTorch experience yet" — both contain "PyTorch". When enabled, this
module sends the JD and one resume to Claude, which judges the *evidence*
behind each claimed skill and returns a structured second opinion shown
alongside the deterministic score (it never replaces it).

Requires the optional `anthropic` package plus credentials — an
ANTHROPIC_API_KEY environment variable, an `ant auth login` profile, or a
key entered in the app sidebar. Everything degrades gracefully when absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

DEFAULT_MODEL = "claude-opus-4-8"

# Per-request text bounds. Generous — resumes are 1-3 pages, so these
# effectively never truncate; they only cap cost on malformed extractions.
_MAX_RESUME_CHARS = 60_000
_MAX_JD_CHARS = 20_000

_VERDICTS = ("strong", "good", "partial", "weak")

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "fit_score": {
            "type": "integer",
            "description": "Overall fit for the role, 0-100.",
        },
        "verdict": {"type": "string", "enum": list(_VERDICTS)},
        "evidence_backed_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "JD-relevant skills the resume demonstrates with real work.",
        },
        "unsupported_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "JD-relevant skills only name-dropped, aspirational, or mentioned negatively.",
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": [
        "fit_score", "verdict", "evidence_backed_skills", "unsupported_skills",
        "strengths", "concerns", "summary",
    ],
    "additionalProperties": False,
}

_SYSTEM_INSTRUCTIONS = """\
You are an experienced technical recruiter reviewing one candidate's resume \
against a job description. Judge EVIDENCE, not keywords:

- A skill is evidence-backed only if the resume shows it in use — a project, \
a role responsibility, a shipped deliverable, a metric. A bare mention in a \
skills list is weak evidence.
- Aspirational or negative mentions ("no ML experience yet", "eager to learn \
Kubernetes") are NOT evidence — list those skills as unsupported.
- Weigh recency and depth: daily use in the current role beats brief exposure \
years ago.
- fit_score is 0-100. verdict bands: strong >= 75, good >= 55, partial >= 35, \
otherwise weak. Keep fit_score and verdict consistent.
- strengths and concerns: at most 3 each, each concrete and grounded in the \
resume ("Led the LangChain RAG migration at StreamCart"), never generic.
- summary: 2-3 sentences a hiring manager can absorb in ten seconds.

Judge only job fit. Ignore name, gender, age, ethnicity, photos, and other \
protected characteristics entirely."""


@dataclass
class LLMReview:
    fit_score: int = 0
    verdict: str = "weak"
    evidence_backed_skills: list[str] = field(default_factory=list)
    unsupported_skills: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    summary: str = ""
    error: str | None = None  # set when the review failed; other fields empty


def llm_available() -> bool:
    """True when the optional `anthropic` SDK is installed."""
    try:
        import anthropic  # noqa: F401

        return True
    except ImportError:
        return False


def _failed(message: str) -> LLMReview:
    return LLMReview(error=message)


def _get_client(api_key: str | None):
    import anthropic

    # A bare Anthropic() resolves ANTHROPIC_API_KEY or an `ant auth login`
    # profile automatically — only inject a key when one was given.
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _build_request(jd_text: str, resume_text: str, model: str) -> dict:
    return {
        "model": model,
        # Adaptive thinking spends from max_tokens too, so leave headroom
        # even though the JSON answer itself is small.
        "max_tokens": 16000,
        "thinking": {"type": "adaptive"},
        "system": [
            {"type": "text", "text": _SYSTEM_INSTRUCTIONS},
            {
                "type": "text",
                "text": "JOB DESCRIPTION:\n\n" + jd_text[:_MAX_JD_CHARS],
                # Instructions + JD are identical for every candidate in a
                # batch — cache them so only the resume is re-processed.
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": "CANDIDATE RESUME:\n\n" + resume_text[:_MAX_RESUME_CHARS],
            }
        ],
        "output_config": {"format": {"type": "json_schema", "schema": _REVIEW_SCHEMA}},
    }


def _parse_response(response) -> LLMReview:
    if response.stop_reason == "refusal":
        return _failed("Claude declined to review this resume.")
    text = next((b.text for b in response.content if b.type == "text"), "")
    if response.stop_reason == "max_tokens" or not text:
        return _failed("The model's response was truncated or empty — try again.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _failed("Could not parse the model's response as JSON.")

    verdict = data.get("verdict", "")
    return LLMReview(
        fit_score=max(0, min(int(data.get("fit_score", 0)), 100)),
        verdict=verdict if verdict in _VERDICTS else "partial",
        evidence_backed_skills=[str(s) for s in data.get("evidence_backed_skills", [])],
        unsupported_skills=[str(s) for s in data.get("unsupported_skills", [])],
        strengths=[str(s) for s in data.get("strengths", [])][:3],
        concerns=[str(s) for s in data.get("concerns", [])][:3],
        summary=str(data.get("summary", "")),
    )


def review_candidate(
    jd_text: str,
    resume_text: str,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    _client=None,
) -> LLMReview:
    """Run one AI deep review. Never raises — failures come back as
    LLMReview(error=...) so one bad call can't break a batch."""
    if _client is None:
        if not llm_available():
            return _failed(
                "The `anthropic` package is not installed. "
                "Run `pip install anthropic` to enable AI review."
            )
        _client = _get_client(api_key)

    import anthropic

    try:
        response = _client.messages.create(**_build_request(jd_text, resume_text, model))
    except (anthropic.AuthenticationError, TypeError):
        # The SDK raises TypeError ("Could not resolve authentication method")
        # at request time when no key/profile is configured at all.
        return _failed(
            "No valid Anthropic credentials. Set ANTHROPIC_API_KEY, run "
            "`ant auth login`, or enter an API key in the sidebar."
        )
    except anthropic.RateLimitError:
        return _failed("Rate limited by the Claude API — wait a moment and retry.")
    except anthropic.APIStatusError as exc:
        return _failed(f"Claude API error ({exc.status_code}): {exc.message}")
    except anthropic.APIConnectionError:
        return _failed("Could not reach the Claude API — check your network connection.")

    return _parse_response(response)


def review_candidates(
    jd_text: str,
    resume_texts: list[str],
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    on_progress=None,
) -> list[LLMReview]:
    """Review several resumes against one JD, reusing a single client.

    `on_progress(done, total)` is called after each review completes.
    """
    if not llm_available():
        return [
            _failed("The `anthropic` package is not installed.")
            for _ in resume_texts
        ]
    client = _get_client(api_key)
    reviews = []
    for i, text in enumerate(resume_texts):
        reviews.append(
            review_candidate(jd_text, text, model=model, _client=client)
        )
        if on_progress:
            on_progress(i + 1, len(resume_texts))
    return reviews
