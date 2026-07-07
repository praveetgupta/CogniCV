"""Tests for candidate profile parsing (name, contact, experience, education)."""

import datetime

from cognicv.candidate import (
    detect_education,
    estimate_experience_years,
    extract_name,
    parse_candidate,
)

RESUME = """John A. Smith
Senior Machine Learning Engineer
john.smith@example.com | +1 (415) 555-0134 | linkedin.com/in/johnsmith

SUMMARY
ML engineer with 7+ years of experience building production models.

EXPERIENCE
Senior ML Engineer — Acme Corp
Jan 2020 – Present
- Built RAG pipelines with LangChain and vector databases.

ML Engineer — DataWorks
Mar 2016 – Dec 2019
- Deployed PyTorch models to AWS with Docker and Kubernetes.

EDUCATION
M.S. in Computer Science, Stanford University, 2016
B.S. in Mathematics, UCLA, 2014
"""


class TestContactExtraction:
    def test_email(self):
        p = parse_candidate("resume.pdf", RESUME)
        assert p.email == "john.smith@example.com"

    def test_phone(self):
        p = parse_candidate("resume.pdf", RESUME)
        assert p.phone is not None
        assert "555" in p.phone

    def test_linkedin(self):
        p = parse_candidate("resume.pdf", RESUME)
        assert p.linkedin == "linkedin.com/in/johnsmith"


class TestNameExtraction:
    def test_name_from_first_line(self):
        assert extract_name(RESUME, "resume.pdf") == "John A. Smith"

    def test_skips_headings(self):
        text = "CURRICULUM VITAE\nJane Doe\njane@example.com"
        assert extract_name(text, "cv.pdf") == "Jane Doe"

    def test_fallback_to_filename(self):
        text = "Highly motivated professional seeking opportunities in engineering."
        assert extract_name(text, "priya_sharma_resume.pdf") == "Priya Sharma"

    def test_unknown_when_nothing_available(self):
        # "resume.pdf" cleans to nothing -> generic placeholder
        assert extract_name("....", "resume.pdf") == "Unknown Candidate"


class TestExperience:
    def test_month_ranges_merged(self):
        years = estimate_experience_years(RESUME)
        assert years is not None
        # Mar 2016 – Dec 2019 (~3.8y) + Jan 2020 – present (6.5y+ as of 2026)
        assert years >= 9

    def test_stated_years(self):
        years = estimate_experience_years(
            "Results-driven leader with 12+ years of experience in sales."
        )
        assert years == 12

    def test_overlapping_ranges_not_double_counted(self):
        text = (
            "Engineer, Jan 2020 - Dec 2021\n"
            "Consultant (concurrent), Jun 2020 - Jun 2021\n"
        )
        years = estimate_experience_years(text)
        assert years is not None
        assert years <= 2.5

    def test_year_only_range(self):
        this_year = datetime.date.today().year
        years = estimate_experience_years(f"Developer, {this_year - 3} - Present")
        assert years is not None
        assert 2.5 <= years <= 4.5

    def test_no_dates_returns_none(self):
        assert estimate_experience_years("Skills: Python, SQL") is None

    def test_future_or_invalid_ranges_ignored(self):
        assert estimate_experience_years("Planned 2098 - 2099 sabbatical") is None


class TestEducation:
    def test_phd(self):
        level, label = detect_education("Ph.D. in Physics, MIT")
        assert level == 4

    def test_masters(self):
        assert detect_education("M.S. in Computer Science")[0] == 3
        assert detect_education("MBA, Wharton")[0] == 3
        assert detect_education("Master of Science in Data Science")[0] == 3
        assert detect_education("M.Tech in AI")[0] == 3

    def test_bachelors(self):
        assert detect_education("Bachelor of Engineering")[0] == 2
        assert detect_education("B.Tech in Computer Science")[0] == 2
        assert detect_education("B.S. in Mathematics")[0] == 2

    def test_highest_degree_wins(self):
        level, label = detect_education(RESUME)
        assert level == 3
        assert label == "Master's"

    def test_none_detected(self):
        assert detect_education("Skills: Python") == (None, None)


class TestFullProfile:
    def test_skills_extracted(self):
        p = parse_candidate("resume.pdf", RESUME)
        assert {"langchain", "pytorch", "aws", "docker", "kubernetes", "rag"} <= p.skills

    def test_word_count(self):
        p = parse_candidate("resume.pdf", RESUME)
        assert p.word_count > 50
