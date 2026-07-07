"""Command-line batch screening: rank a folder of resumes against a JD.

Usage:
    python -m cognicv --jd jd.txt --resumes ./resumes/ --csv results.csv
    cognicv --jd jd.txt --resumes a.pdf b.pdf --top 10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .candidate import parse_candidate
from .extraction import SUPPORTED_EXTENSIONS, ExtractionError, extract_text
from .jd import parse_jd
from .scoring import ScoreWeights, score_candidates, score_label
from .semantic import backend_name
from .taxonomy import display_name


def _collect_resume_paths(inputs: list[str], jd_path: Path) -> list[Path]:
    jd_resolved = jd_path.resolve()
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(
                child for child in sorted(p.iterdir())
                if child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif p.is_file():
            paths.append(p)
        else:
            print(f"warning: {item!r} not found, skipping", file=sys.stderr)
    # The JD often lives in the same folder as the resumes — don't score it
    return [p for p in paths if p.resolve() != jd_resolved]


def _displays(canons: list[str]) -> str:
    return "; ".join(display_name(c) for c in canons)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cognicv",
        description="Rank resumes against a job description.",
    )
    parser.add_argument("--jd", required=True, help="Path to job description (.txt/.md/.pdf/.docx)")
    parser.add_argument("--resumes", required=True, nargs="+",
                        help="Resume files and/or directories containing them")
    parser.add_argument("--top", type=int, default=0, help="Show only the top N candidates")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Hide candidates scoring below this (0-100)")
    parser.add_argument("--csv", metavar="PATH", help="Write full results to a CSV file")
    parser.add_argument("--w-skills", type=float, default=0.45, help="Weight: skill coverage")
    parser.add_argument("--w-semantic", type=float, default=0.25, help="Weight: semantic fit")
    parser.add_argument("--w-experience", type=float, default=0.20, help="Weight: experience")
    parser.add_argument("--w-education", type=float, default=0.10, help="Weight: education")
    parser.add_argument("--llm-top", type=int, default=0, metavar="N",
                        help="Run a Claude AI deep review on the top N ranked candidates "
                             "(requires `pip install anthropic` and API credentials)")
    return parser


def _parse_profiles(resume_paths: list[Path]) -> tuple[list, list[str]]:
    profiles, errors = [], []
    for path in resume_paths:
        try:
            text = extract_text(path.name, path.read_bytes())
            profiles.append(parse_candidate(path.name, text))
        except ExtractionError as exc:
            errors.append(f"{path.name}: {exc}")
    return profiles, errors


def _print_jd_summary(jd, jd_name: str) -> None:
    print(f"\nJob requirements ({jd_name}):")
    print(f"  Must-have:    {_displays(sorted(jd.must_have)) or '—'}")
    print(f"  Nice-to-have: {_displays(sorted(jd.nice_to_have)) or '—'}")
    if jd.min_years:
        print(f"  Experience:   {jd.min_years:g}+ years")
    if jd.education_label:
        print(f"  Education:    {jd.education_label}")


def _print_table(ranked, top: int, min_score: float) -> None:
    shown = [r for r in ranked if r.total >= min_score]
    if top > 0:
        shown = shown[:top]

    print(f"\n{'#':>3}  {'Score':>6}  {'Skills':>6}  {'Sem.':>5}  {'Exp.':>5}  Candidate")
    print("-" * 78)
    for i, r in enumerate(shown, 1):
        exp = f"{r.profile.years_experience:g}y" if r.profile.years_experience else "—"
        print(f"{i:>3}  {r.total:>5.1f}%  {r.skill_score:>5.1f}%  {r.semantic_score:>4.0f}%  "
              f"{exp:>5}  {r.profile.name}  ({score_label(r.total)})")
        if r.missing_must:
            print(f"{'':>28}missing must-have: {_displays(r.missing_must)}")


def _write_csv(ranked, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank", "name", "file", "email", "phone", "total_score",
            "skill_score", "semantic_score", "experience_years",
            "education", "matched_must", "missing_must",
            "matched_nice", "missing_nice", "flags",
        ])
        for i, r in enumerate(ranked, 1):
            writer.writerow([
                i, r.profile.name, r.profile.filename,
                r.profile.email or "", r.profile.phone or "",
                r.total, r.skill_score, r.semantic_score,
                r.profile.years_experience if r.profile.years_experience is not None else "",
                r.profile.education_label or "",
                _displays(r.matched_must), _displays(r.missing_must),
                _displays(r.matched_nice), _displays(r.missing_nice),
                "; ".join(r.flags),
            ])
    print(f"\nFull results written to {path}")


def _run_llm_reviews(ranked, jd_text: str, top_n: int) -> None:
    from .llm import DEFAULT_MODEL, llm_available, review_candidates

    if not llm_available():
        print("\nAI review skipped: `pip install anthropic` to enable it.", file=sys.stderr)
        return
    targets = ranked[:top_n]
    print(f"\nRunning AI deep review on top {len(targets)} candidate(s) "
          f"({DEFAULT_MODEL}) …", file=sys.stderr)
    reviews = review_candidates(jd_text, [r.profile.text for r in targets])

    print("\nAI deep review:")
    for r, review in zip(targets, reviews):
        if review.error:
            print(f"  {r.profile.name}: review failed — {review.error}")
            continue
        print(f"  {r.profile.name}: AI fit {review.fit_score}% ({review.verdict} match)")
        if review.summary:
            print(f"    {review.summary}")
        if review.unsupported_skills:
            print(f"    Claimed without evidence: {', '.join(review.unsupported_skills)}")
        for concern in review.concerns:
            print(f"    ⚠ {concern}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    jd_path = Path(args.jd)
    if not jd_path.is_file():
        parser.error(f"JD file not found: {args.jd}")
    jd = parse_jd(extract_text(jd_path.name, jd_path.read_bytes()))

    resume_paths = _collect_resume_paths(args.resumes, jd_path)
    if not resume_paths:
        parser.error("no readable resume files found")

    profiles, errors = _parse_profiles(resume_paths)
    weights = ScoreWeights(args.w_skills, args.w_semantic, args.w_experience, args.w_education)
    print(f"Scoring {len(profiles)} resume(s) — semantic backend: {backend_name()}", file=sys.stderr)
    ranked = score_candidates(profiles, jd, weights)

    _print_jd_summary(jd, jd_path.name)
    _print_table(ranked, args.top, args.min_score)

    if args.llm_top > 0 and ranked:
        _run_llm_reviews(ranked, jd.text, args.llm_top)

    if errors:
        print(f"\nSkipped {len(errors)} unreadable file(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)

    if args.csv:
        _write_csv(ranked, args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
