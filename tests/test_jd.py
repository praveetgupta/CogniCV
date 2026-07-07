"""Tests for job-description parsing."""

from cognicv.jd import parse_jd

JD = """Senior Machine Learning Engineer

We are looking for a Senior ML Engineer to join our platform team.

Requirements:
- 5+ years of experience in machine learning
- Strong Python and SQL skills
- Production experience with PyTorch or TensorFlow
- Experience with AWS and Docker
- Bachelor's degree in Computer Science or related field

Nice to have:
- Kubernetes experience
- LangChain or RAG pipelines
- Terraform
"""


class TestSectionSplit:
    def test_must_haves_from_requirements(self):
        jd = parse_jd(JD)
        assert {"python", "sql", "pytorch", "tensorflow", "aws", "docker"} <= jd.must_have

    def test_nice_to_haves_from_preferred_section(self):
        jd = parse_jd(JD)
        assert {"kubernetes", "langchain", "terraform", "rag"} <= jd.nice_to_have

    def test_no_overlap_between_sets(self):
        jd = parse_jd(JD)
        assert not (jd.must_have & jd.nice_to_have)

    def test_untagged_text_defaults_to_must(self):
        jd = parse_jd("Looking for a Python developer with Django experience.")
        assert {"python", "django"} <= jd.must_have

    def test_inline_plus_demotes_to_nice(self):
        jd = parse_jd(
            "We need a strong Python engineer.\n"
            "Kubernetes experience is a plus.\n"
        )
        assert "python" in jd.must_have
        assert "kubernetes" in jd.nice_to_have

    def test_skill_in_both_zones_is_must(self):
        jd = parse_jd(
            "Requirements:\n- Python expertise\n\n"
            "Nice to have:\n- Advanced Python tricks\n"
        )
        assert "python" in jd.must_have
        assert "python" not in jd.nice_to_have


class TestRequirementFields:
    def test_min_years(self):
        assert parse_jd(JD).min_years == 5

    def test_min_years_takes_max_near_experience(self):
        jd = parse_jd(
            "7+ years of software experience required. "
            "2+ years of experience with Kubernetes."
        )
        assert jd.min_years == 7

    def test_years_without_experience_context_ignored(self):
        jd = parse_jd("Our company was founded 20 years ago. We build Python tools.")
        assert jd.min_years is None

    def test_education_requirement(self):
        jd = parse_jd(JD)
        assert jd.education_level == 2
        assert jd.education_label == "Bachelor's"

    def test_soft_skills_excluded(self):
        jd = parse_jd(
            "Requirements: excellent communication skills, teamwork, "
            "and strong Python knowledge."
        )
        assert "python" in jd.must_have
        assert "communication" not in jd.must_have
        assert "teamwork" not in jd.must_have
