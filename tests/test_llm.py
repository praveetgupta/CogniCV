"""Tests for the optional Claude AI review module (mocked client — no network)."""

import json

import pytest

from cognicv.llm import (
    DEFAULT_MODEL,
    LLMReview,
    _build_request,
    llm_available,
    review_candidate,
)

GOOD_PAYLOAD = {
    "fit_score": 82,
    "verdict": "strong",
    "evidence_backed_skills": ["Python", "PyTorch"],
    "unsupported_skills": ["Kubernetes"],
    "strengths": ["Shipped RAG assistant to 25M users"],
    "concerns": ["No Terraform exposure"],
    "summary": "Strong ML engineer with production LLM experience.",
}


class FakeBlock:
    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text
        self.thinking = ""


class FakeResponse:
    def __init__(self, payload=None, stop_reason="end_turn", text=None):
        self.stop_reason = stop_reason
        body = text if text is not None else json.dumps(payload or {})
        self.content = [FakeBlock("thinking"), FakeBlock("text", body)]


class FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def review(response, jd="We need Python and PyTorch.", resume="resume text"):
    return review_candidate(jd, resume, _client=FakeClient(response))


class TestParsing:
    def test_good_response(self):
        r = review(FakeResponse(GOOD_PAYLOAD))
        assert r.error is None
        assert r.fit_score == 82
        assert r.verdict == "strong"
        assert r.evidence_backed_skills == ["Python", "PyTorch"]
        assert r.unsupported_skills == ["Kubernetes"]
        assert "RAG assistant" in r.strengths[0]

    def test_fit_score_clamped(self):
        r = review(FakeResponse({**GOOD_PAYLOAD, "fit_score": 150}))
        assert r.fit_score == 100
        r = review(FakeResponse({**GOOD_PAYLOAD, "fit_score": -5}))
        assert r.fit_score == 0

    def test_unknown_verdict_normalized(self):
        r = review(FakeResponse({**GOOD_PAYLOAD, "verdict": "amazing"}))
        assert r.verdict == "partial"

    def test_lists_capped_at_three(self):
        r = review(FakeResponse({**GOOD_PAYLOAD, "concerns": ["a", "b", "c", "d", "e"]}))
        assert len(r.concerns) == 3

    def test_refusal_returns_error(self):
        r = review(FakeResponse(GOOD_PAYLOAD, stop_reason="refusal"))
        assert r.error is not None
        assert "declined" in r.error

    def test_truncated_returns_error(self):
        r = review(FakeResponse(GOOD_PAYLOAD, stop_reason="max_tokens"))
        assert r.error is not None

    def test_invalid_json_returns_error(self):
        r = review(FakeResponse(text="not json at all"))
        assert r.error is not None
        assert "parse" in r.error.lower()


class TestRequestShape:
    def test_request_structure(self):
        client = FakeClient(FakeResponse(GOOD_PAYLOAD))
        review_candidate("JD TEXT HERE", "RESUME TEXT HERE", _client=client)
        kwargs = client.messages.last_kwargs

        assert kwargs["model"] == DEFAULT_MODEL == "claude-opus-4-8"
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert "temperature" not in kwargs  # rejected on Opus 4.8
        assert kwargs["output_config"]["format"]["type"] == "json_schema"

        # JD lives in the (cached) system blocks; resume in the user turn
        assert "JD TEXT HERE" in kwargs["system"][1]["text"]
        assert kwargs["system"][1]["cache_control"] == {"type": "ephemeral"}
        assert "RESUME TEXT HERE" in kwargs["messages"][0]["content"]

    def test_oversized_inputs_bounded(self):
        req = _build_request("j" * 100_000, "r" * 200_000, DEFAULT_MODEL)
        assert len(req["system"][1]["text"]) < 30_000
        assert len(req["messages"][0]["content"]) < 70_000

    def test_schema_is_strict(self):
        req = _build_request("jd", "resume", DEFAULT_MODEL)
        schema = req["output_config"]["format"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"].keys())


class TestAvailability:
    def test_llm_available_matches_import(self):
        try:
            import anthropic  # noqa: F401

            assert llm_available()
        except ImportError:
            assert not llm_available()

    def test_review_dataclass_defaults(self):
        r = LLMReview()
        assert r.error is None
        assert r.fit_score == 0


class TestErrorHandling:
    def test_api_errors_become_error_reviews(self):
        anthropic = pytest.importorskip("anthropic")
        import httpx

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

        class RaisingMessages:
            def create(self, **kwargs):
                raise anthropic.AuthenticationError(
                    "bad key",
                    response=httpx.Response(401, request=request),
                    body=None,
                )

        class RaisingClient:
            messages = RaisingMessages()

        r = review_candidate("jd", "resume", _client=RaisingClient())
        assert r.error is not None
        assert "credentials" in r.error.lower()

    def test_missing_credentials_entirely(self, monkeypatch):
        # With no key/profile the SDK raises TypeError at request time,
        # before any network I/O — must surface as a friendly error.
        anthropic = pytest.importorskip("anthropic")
        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
            monkeypatch.delenv(var, raising=False)
        client = anthropic.Anthropic()
        r = review_candidate("jd", "resume", _client=client)
        assert r.error is not None
        assert "credentials" in r.error.lower()
