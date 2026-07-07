"""CogniCV — AI-assisted resume screening for recruiters.

Upload a batch of resumes, paste a job description, and get a ranked
shortlist with skill-gap analysis, experience/education checks, and
CSV export. Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cognicv import __version__
from cognicv.candidate import CandidateProfile, parse_candidate
from cognicv.extraction import ExtractionError, extract_text
from cognicv.jd import JobRequirements, parse_jd
from cognicv.llm import DEFAULT_MODEL, LLMReview, llm_available, review_candidate
from cognicv.scoring import (
    ScoredCandidate,
    ScoreWeights,
    score_candidates,
    score_label,
)
from cognicv.semantic import backend_name, semantic_scores
from cognicv.skills import displays, group_by_category
from cognicv.taxonomy import ALL_CANONICALS, canonicalize, display_name

_NONE_HTML = "<i>None</i>"


# ─────────────────────────────────────────────────────────────────────
# Cached helpers (parsing and semantic scoring are the slow parts)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_parse(filename: str, data: bytes) -> CandidateProfile:
    return parse_candidate(filename, extract_text(filename, data))


@st.cache_data(show_spinner=False)
def cached_semantic(jd_text: str, resume_texts: tuple[str, ...]) -> list[float]:
    return semantic_scores(jd_text, list(resume_texts))


# ─────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────
def pills(items: list[str], kind: str, limit: int = 40) -> str:
    shown = items[:limit]
    html = "".join(f'<span class="skill-pill pill-{kind}">{s}</span>' for s in shown)
    if len(items) > limit:
        html += f'<span class="skill-pill pill-more">+{len(items) - limit} more</span>'
    return html


def categorized_pills(canons: list[str], kind: str) -> str:
    html = ""
    for cat, names in group_by_category(canons).items():
        html += f'<div class="category-label">{cat}</div>'
        html += pills(names, kind)
    return html


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="stApp"] { font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1200px; padding-top: 1.5rem; }

.cogni-header { text-align: center; padding: 1.2rem 0 .6rem; }
.cogni-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .2rem;
}
.cogni-header p { font-size: 1.02rem; opacity: .65; margin-top: 0; }

.skill-pill {
    display: inline-block; padding: 4px 13px; border-radius: 999px;
    font-size: .84rem; font-weight: 500; margin: 3px;
}
.pill-match   { background: rgba(34,197,94,.14);  color: #16a34a; border: 1px solid rgba(34,197,94,.3); }
.pill-missing { background: rgba(239,68,68,.12);  color: #dc2626; border: 1px solid rgba(239,68,68,.25); }
.pill-nice    { background: rgba(234,179,8,.13);  color: #a16207; border: 1px solid rgba(234,179,8,.3); }
.pill-extra   { background: rgba(99,102,241,.12); color: #6366f1; border: 1px solid rgba(99,102,241,.25); }
.pill-more    { background: rgba(128,128,128,.12); color: #6b7280; border: 1px solid rgba(128,128,128,.25); }

.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem; font-weight: 600;
    margin: 1.4rem 0 .5rem; padding-bottom: .35rem;
    border-bottom: 2px solid rgba(99,102,241,.25);
}
.category-label {
    font-size: .78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .05em; opacity: .55; margin: 8px 0 2px 4px;
}
.flag-box {
    background: rgba(234,179,8,.08); border-left: 4px solid #eab308;
    border-radius: 0 8px 8px 0; padding: 8px 14px; margin: 6px 0;
    font-size: .9rem;
}
.jd-summary {
    background: rgba(99,102,241,.05); border: 1px solid rgba(99,102,241,.15);
    border-radius: 10px; padding: 14px 18px; margin-top: .6rem;
}
.ai-box {
    background: rgba(139,92,246,.06); border: 1px solid rgba(139,92,246,.2);
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
    font-size: .92rem;
}
.ai-box .ai-title { font-weight: 600; color: #8b5cf6; margin-bottom: 4px; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────
# UI sections
# ─────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[ScoreWeights, bool, bool, int]:
    with st.sidebar:
        st.markdown("### ⚙️ Scoring weights")
        st.caption("Relative importance — normalized automatically.")
        weights = ScoreWeights(
            skills=st.slider("Skill coverage", 0, 100, 45),
            semantic=st.slider("Semantic fit (full text)", 0, 100, 25),
            experience=st.slider("Experience", 0, 100, 20),
            education=st.slider("Education", 0, 100, 10),
        )

        st.markdown("### 🛡️ Screening options")
        anonymize = st.toggle(
            "Anonymize candidates",
            help="Hide names and contact details to reduce unconscious bias "
                 "during the first screening pass.",
        )
        hard_filter = st.toggle(
            "Exclude candidates missing must-haves",
            help="Remove candidates missing more must-have skills than the "
                 "threshold below, instead of just ranking them lower.",
        )
        max_missing = 0
        if hard_filter:
            max_missing = st.number_input("Max missing must-haves allowed", 0, 20, 0)

        st.divider()
        if backend_name() == "embeddings":
            st.caption("🧬 Semantic backend: **sentence embeddings** (MiniLM)")
        else:
            st.caption(
                "📐 Semantic backend: **TF-IDF**. Install "
                "`sentence-transformers` for meaning-aware matching."
            )
        st.caption(
            f"CogniCV v{__version__} · Decision-support only — "
            "final hiring decisions require human review."
        )
    return weights, anonymize, hard_filter, max_missing


def render_inputs() -> tuple[str, list]:
    col_jd, col_files = st.columns(2, gap="large")
    with col_jd:
        st.markdown("##### 💼 Job description")
        jd_text = st.text_area(
            "Paste the job description",
            height=260,
            label_visibility="collapsed",
            placeholder="Paste the full job description here …",
        )
    with col_files:
        st.markdown("##### 📄 Candidate resumes")
        uploaded = st.file_uploader(
            "Upload resumes",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            st.caption(f"{len(uploaded)} file(s) ready.")
    return jd_text or "", uploaded or []


def render_requirements_panel(jd_text: str, expanded: bool) -> JobRequirements:
    """Show detected JD requirements and let the recruiter adjust must-haves."""
    detected = parse_jd(jd_text)

    with st.expander("🎯 Detected requirements — review & adjust", expanded=expanded):
        all_options = sorted(
            {display_name(c) for c in ALL_CANONICALS}
            | {display_name(c) for c in detected.skills},
            key=str.lower,
        )
        must_selection = st.multiselect(
            "Must-have skills (weighted 3× in scoring)",
            options=all_options,
            default=displays(detected.must_have),
            help="Auto-detected from the JD's requirements sections. "
                 "Add or remove skills to match what you actually need.",
        )
        must_canons = {c for s in must_selection if (c := canonicalize(s))}
        nice_canons = detected.skills - must_canons

        jd_req = JobRequirements(
            text=jd_text,
            skills=must_canons | nice_canons,
            must_have=must_canons,
            nice_to_have=nice_canons,
            min_years=detected.min_years,
            education_level=detected.education_level,
            education_label=detected.education_label,
        )

        summary_bits = []
        if jd_req.nice_to_have:
            summary_bits.append(
                "<b>Nice-to-have:</b> " + pills(displays(jd_req.nice_to_have), "nice")
            )
        if jd_req.min_years:
            summary_bits.append(f"<b>Experience:</b> {jd_req.min_years:g}+ years")
        if jd_req.education_label:
            summary_bits.append(f"<b>Education:</b> {jd_req.education_label}")
        if summary_bits:
            st.markdown(
                '<div class="jd-summary">' + "<br>".join(summary_bits) + "</div>",
                unsafe_allow_html=True,
            )
        if not jd_req.skills:
            st.warning(
                "No recognizable skills detected in this JD — scoring will rely "
                "on semantic fit only. Consider adding must-have skills above."
            )
    return jd_req


def parse_uploads(uploaded: list) -> list[CandidateProfile]:
    profiles: list[CandidateProfile] = []
    errors: list[str] = []
    with st.spinner(f"Parsing {len(uploaded)} resume(s) …"):
        for f in uploaded:
            try:
                profiles.append(cached_parse(f.name, f.getvalue()))
            except ExtractionError as exc:
                errors.append(f"**{f.name}** — {exc}")
    if errors:
        st.warning("Skipped unreadable files:\n\n" + "\n".join(f"- {e}" for e in errors))
    return profiles


def render_metrics(ranked: list[ScoredCandidate], n_excluded: int) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates", len(ranked) + n_excluded)
    m2.metric("Shortlist (≥55%)", sum(1 for r in ranked if r.total >= 55))
    m3.metric("Top score", f"{ranked[0].total:g}%" if ranked else "—")
    avg = sum(r.total for r in ranked) / len(ranked) if ranked else 0
    m4.metric("Average score", f"{avg:.1f}%")


def render_table(
    ranked: list[ScoredCandidate], names: dict[str, str], anonymize: bool,
    reviews: dict[str, LLMReview] | None = None,
) -> None:
    reviews = reviews or {}
    has_reviews = any(not r.error for r in reviews.values())
    rows = []
    for i, r in enumerate(ranked, 1):
        row = {
            "Rank": i,
            "Candidate": names[r.profile.filename],
            "Score": r.total,
            "Verdict": score_label(r.total),
            "Skills %": r.skill_score,
            "Semantic %": r.semantic_score,
            "Experience (yrs)": r.profile.years_experience,
            "Education": r.profile.education_label or "—",
            "Missing must-haves": len(r.missing_must),
        }
        if has_reviews:
            review = reviews.get(r.profile.filename)
            row["AI fit %"] = review.fit_score if review and not review.error else None
        if not anonymize:
            row["Email"] = r.profile.email or "—"
            row["File"] = r.profile.filename
        rows.append(row)

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", format="%.1f%%", min_value=0, max_value=100,
            ),
            "Skills %": st.column_config.NumberColumn(format="%.1f"),
            "Semantic %": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def _export_row(
    rank: int, r: ScoredCandidate, name: str, anonymize: bool,
    review: LLMReview | None, excluded_by_filter: bool,
) -> dict:
    return {
        "rank": rank,
        "candidate": name,
        "file": r.profile.filename if not anonymize else "",
        "email": (r.profile.email or "") if not anonymize else "",
        "phone": (r.profile.phone or "") if not anonymize else "",
        "total_score": r.total,
        "skill_score": r.skill_score,
        "semantic_score": r.semantic_score,
        "experience_years": r.profile.years_experience,
        "education": r.profile.education_label or "",
        "matched_must": "; ".join(displays(r.matched_must)),
        "missing_must": "; ".join(displays(r.missing_must)),
        "matched_nice": "; ".join(displays(r.matched_nice)),
        "flags": "; ".join(r.flags),
        "excluded_by_filter": excluded_by_filter,
        "ai_fit_score": review.fit_score if review else "",
        "ai_verdict": review.verdict if review else "",
        "ai_summary": review.summary if review else "",
        "ai_concerns": "; ".join(review.concerns) if review else "",
    }


def render_export(
    ranked: list[ScoredCandidate],
    excluded: list[ScoredCandidate],
    names: dict[str, str],
    anonymize: bool,
    reviews: dict[str, LLMReview] | None = None,
) -> None:
    reviews = reviews or {}
    rows = []
    for i, r in enumerate(ranked + excluded, 1):
        review = reviews.get(r.profile.filename)
        if review and review.error:
            review = None
        rows.append(_export_row(
            i, r, names[r.profile.filename], anonymize, review, r in excluded,
        ))
    st.download_button(
        "⬇️ Export results (CSV)",
        pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
        file_name="cognicv_results.csv",
        mime="text/csv",
    )


def _review_key(jd_text: str, profile: CandidateProfile) -> str:
    return f"{hash(jd_text)}:{profile.filename}:{hash(profile.text)}"


def current_reviews(
    ranked: list[ScoredCandidate], jd_text: str,
) -> dict[str, LLMReview]:
    """Reviews already run this session for the current JD + files, by filename."""
    stored = st.session_state.get("llm_reviews", {})
    return {
        r.profile.filename: stored[key]
        for r in ranked
        if (key := _review_key(jd_text, r.profile)) in stored
    }


def render_ai_review_section(
    ranked: list[ScoredCandidate], jd_text: str, names: dict[str, str],
) -> dict[str, LLMReview]:
    """Optional Claude-powered deep review of the top-ranked candidates."""
    with st.expander("🤖 AI deep review (optional) — Claude reads the resumes"):
        if not llm_available():
            st.info(
                "AI review needs the Claude SDK: `pip install anthropic`, then set "
                "`ANTHROPIC_API_KEY` (or run `ant auth login`) and restart the app. "
                "Claude judges the *evidence* behind each skill — catching things "
                "keyword matching can't, like “no ML experience yet”."
            )
            return current_reviews(ranked, jd_text)

        st.caption(
            "Claude reviews each resume against the JD and scores the evidence "
            "behind claimed skills. Resume text is sent to the Anthropic API for "
            "this feature only. Typical cost: a few cents per candidate."
        )
        c1, c2 = st.columns([1, 2])
        top_n = c1.number_input(
            "Review top N candidates", 1, min(20, len(ranked)), min(5, len(ranked)),
        )
        api_key = c2.text_input(
            "Anthropic API key (optional)",
            type="password",
            help="Leave blank to use ANTHROPIC_API_KEY or `ant auth login` credentials.",
        )

        if st.button("✨ Run AI review", type="primary"):
            stored = st.session_state.setdefault("llm_reviews", {})
            targets = ranked[: int(top_n)]
            progress = st.progress(0.0, text="Reviewing candidates …")
            for i, r in enumerate(targets):
                key = _review_key(jd_text, r.profile)
                if key not in stored or stored[key].error:
                    stored[key] = review_candidate(
                        jd_text, r.profile.text,
                        api_key=api_key or None, model=DEFAULT_MODEL,
                    )
                progress.progress(
                    (i + 1) / len(targets),
                    text=f"Reviewed {names[r.profile.filename]} ({i + 1}/{len(targets)})",
                )
            progress.empty()

        reviews = current_reviews(ranked, jd_text)
        errors = [r for r in reviews.values() if r.error]
        if errors:
            st.warning(errors[0].error)
        elif reviews:
            st.success(
                f"{len(reviews)} candidate(s) reviewed — see the AI fit column "
                "and each candidate's detail card below."
            )
    return reviews


def render_ai_review_panel(review: LLMReview) -> None:
    if review.error:
        st.markdown(
            f'<div class="ai-box">🤖 <i>AI review failed: {review.error}</i></div>',
            unsafe_allow_html=True,
        )
        return
    parts = [
        f'<div class="ai-title">🤖 AI review — fit {review.fit_score}% '
        f"({review.verdict} match)</div>"
    ]
    if review.summary:
        parts.append(f"<p>{review.summary}</p>")
    if review.evidence_backed_skills:
        parts.append(
            "<b>✔ Evidence-backed:</b> " + ", ".join(review.evidence_backed_skills)
        )
    if review.unsupported_skills:
        parts.append(
            "<br><b>✖ Claimed without evidence:</b> " + ", ".join(review.unsupported_skills)
        )
    for label, items in (("Strengths", review.strengths), ("Concerns", review.concerns)):
        if items:
            parts.append(f"<br><b>{label}:</b><ul>" + "".join(f"<li>{s}</li>" for s in items) + "</ul>")
    st.markdown(f'<div class="ai-box">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_candidate_detail(
    rank: int, r: ScoredCandidate, name: str, anonymize: bool, auto_expand: bool,
    review: LLMReview | None = None,
) -> None:
    header = f"#{rank} · {name} — {r.total:g}% · {score_label(r.total)}"
    with st.expander(header, expanded=auto_expand):
        if not anonymize:
            contact_bits = [
                b for b in (r.profile.email, r.profile.phone, r.profile.linkedin) if b
            ]
            meta = " · ".join(contact_bits) if contact_bits else "No contact details found"
            st.caption(f"{r.profile.filename} — {meta}")

        parts = [f"Skills **{r.skill_score:g}%**", f"Semantic **{r.semantic_score:g}%**"]
        if r.experience_score is not None:
            parts.append(f"Experience **{r.experience_score:g}%**")
        if r.education_score is not None:
            parts.append(f"Education **{r.education_score:g}%**")
        exp_txt = (
            f"~{r.profile.years_experience:g} yrs"
            if r.profile.years_experience is not None else "unknown"
        )
        parts.append(f"(est. experience: {exp_txt})")
        st.markdown(" · ".join(parts))

        for flag in r.flags:
            st.markdown(f'<div class="flag-box">⚠️ {flag}</div>', unsafe_allow_html=True)

        if review is not None:
            render_ai_review_panel(review)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Must-haves matched**")
            st.markdown(
                pills(displays(r.matched_must), "match") or _NONE_HTML,
                unsafe_allow_html=True,
            )
            st.markdown("**❌ Must-haves missing**")
            st.markdown(
                categorized_pills(r.missing_must, "missing") or "<i>None — full coverage</i>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**🟡 Nice-to-haves matched**")
            st.markdown(
                pills(displays(r.matched_nice), "nice") or _NONE_HTML,
                unsafe_allow_html=True,
            )
            st.markdown("**🔵 Additional skills**")
            st.markdown(
                pills(displays(r.extra_skills), "extra", limit=25) or _NONE_HTML,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="CogniCV — Resume Screening",
        page_icon="🧠",
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="cogni-header"><h1>🧠 CogniCV</h1>'
        "<p>Batch resume screening &amp; candidate ranking</p></div>",
        unsafe_allow_html=True,
    )

    weights, anonymize, hard_filter, max_missing = render_sidebar()
    jd_text, uploaded = render_inputs()

    jd_req: JobRequirements | None = None
    if len(jd_text.strip()) >= 30:
        jd_req = render_requirements_panel(jd_text, expanded=not uploaded)
    elif jd_text.strip():
        st.info("Job description looks too short — paste the full posting.")

    if not uploaded or jd_req is None:
        if not jd_text:
            st.info(
                "**Get started:** paste a job description and upload one or more "
                "resumes (PDF, DOCX, or TXT). CogniCV ranks candidates by skill "
                "coverage, semantic fit, experience, and education."
            )
        return

    profiles = parse_uploads(uploaded)
    if not profiles:
        st.error("No resumes could be parsed.")
        return

    with st.spinner("Scoring candidates …"):
        sem = cached_semantic(jd_req.text, tuple(p.text for p in profiles))
        ranked = score_candidates(profiles, jd_req, weights, semantic_pcts=sem)

    names = {
        p.filename: (f"Candidate {idx}" if anonymize else p.name)
        for idx, p in enumerate(profiles, 1)
    }

    excluded: list[ScoredCandidate] = []
    if hard_filter:
        excluded = [r for r in ranked if len(r.missing_must) > max_missing]
        ranked = [r for r in ranked if len(r.missing_must) <= max_missing]

    st.markdown("---")
    render_metrics(ranked, len(excluded))
    reviews = render_ai_review_section(ranked, jd_req.text, names)
    render_table(ranked, names, anonymize, reviews)
    render_export(ranked, excluded, names, anonymize, reviews)

    if excluded:
        with st.expander(f"🚫 {len(excluded)} candidate(s) excluded by must-have filter"):
            for r in excluded:
                st.markdown(
                    f"- **{names[r.profile.filename]}** ({r.total:g}%) — missing: "
                    + ", ".join(displays(r.missing_must))
                )

    st.markdown('<div class="section-title">📋 Candidate details</div>', unsafe_allow_html=True)
    only_one = len(ranked) == 1
    for i, r in enumerate(ranked, 1):
        render_candidate_detail(
            i, r, names[r.profile.filename], anonymize,
            auto_expand=(i == 1 and only_one),
            review=reviews.get(r.profile.filename),
        )


if __name__ == "__main__":
    main()
