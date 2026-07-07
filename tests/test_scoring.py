"""Tests for the scoring engine."""

from cognicv.candidate import CandidateProfile
from cognicv.jd import JobRequirements
from cognicv.scoring import ScoreWeights, score_candidates


def make_profile(name="Test", skills=None, years=None, edu=None, text="", **kw):
    return CandidateProfile(
        filename=f"{name}.pdf",
        name=name,
        email="t@example.com",
        phone="123-456-7890",
        linkedin=None,
        skills=set(skills or []),
        years_experience=years,
        education_level=edu,
        education_label=None,
        word_count=max(len(text.split()), 300),
        text=text or "generic resume text about software",
        **kw,
    )


def make_jd(must=None, nice=None, years=None, edu=None, text="job description text"):
    must, nice = set(must or []), set(nice or [])
    return JobRequirements(
        text=text,
        skills=must | nice,
        must_have=must,
        nice_to_have=nice,
        min_years=years,
        education_level=edu,
    )


# Fixed semantic scores let us test blending without a live backend.
def score(profiles, jd, weights=None, sem=None):
    sem = sem if sem is not None else [50.0] * len(profiles)
    return score_candidates(profiles, jd, weights, semantic_pcts=sem)


class TestSkillScore:
    def test_full_coverage_scores_100(self):
        jd = make_jd(must={"python", "sql"}, nice={"docker"})
        [r] = score([make_profile(skills={"python", "sql", "docker"})], jd,
                    weights=ScoreWeights(1, 0, 0, 0))
        assert r.total == 100.0
        assert r.skill_score == 100.0

    def test_no_coverage_scores_0(self):
        jd = make_jd(must={"python", "sql"})
        [r] = score([make_profile(skills={"photoshop"})], jd,
                    weights=ScoreWeights(1, 0, 0, 0))
        assert r.skill_score == 0.0

    def test_must_haves_weigh_more(self):
        jd = make_jd(must={"python"}, nice={"docker", "terraform", "helm"})
        [only_must] = score([make_profile(skills={"python"})], jd,
                            weights=ScoreWeights(1, 0, 0, 0))
        [only_nice] = score([make_profile(skills={"docker", "terraform", "helm"})], jd,
                            weights=ScoreWeights(1, 0, 0, 0))
        # 1 must-have (weight 3) beats all 3 nice-to-haves (weight 1 each)
        assert only_must.skill_score == 50.0
        assert only_nice.skill_score == 50.0
        jd2 = make_jd(must={"python"}, nice={"docker", "terraform"})
        [m] = score([make_profile(skills={"python"})], jd2, weights=ScoreWeights(1, 0, 0, 0))
        [n] = score([make_profile(skills={"docker", "terraform"})], jd2,
                    weights=ScoreWeights(1, 0, 0, 0))
        assert m.skill_score > n.skill_score

    def test_matched_missing_lists(self):
        jd = make_jd(must={"python", "sql"}, nice={"docker"})
        [r] = score([make_profile(skills={"python", "react"})], jd)
        assert r.matched_must == ["python"]
        assert r.missing_must == ["sql"]
        assert r.missing_nice == ["docker"]
        assert r.extra_skills == ["react"]


class TestExperienceAndEducation:
    def test_meets_experience(self):
        jd = make_jd(must={"python"}, years=5)
        [r] = score([make_profile(skills={"python"}, years=7)], jd,
                    weights=ScoreWeights(0, 0, 1, 0))
        assert r.total == 100.0

    def test_partial_experience_linear(self):
        jd = make_jd(must={"python"}, years=10)
        [r] = score([make_profile(skills={"python"}, years=5)], jd,
                    weights=ScoreWeights(0, 0, 1, 0))
        assert r.total == 50.0

    def test_unknown_experience_not_penalized_but_flagged(self):
        jd = make_jd(must={"python"}, years=5)
        [r] = score([make_profile(skills={"python"}, years=None)], jd,
                    weights=ScoreWeights(1, 0, 1, 0))
        # experience unknown -> blend falls back to skills only
        assert r.total == 100.0
        assert any("Experience could not be estimated" in f for f in r.flags)

    def test_education_meets(self):
        jd = make_jd(must={"python"}, edu=2)
        [r] = score([make_profile(skills={"python"}, edu=3)], jd,
                    weights=ScoreWeights(0, 0, 0, 1))
        assert r.total == 100.0

    def test_education_one_below(self):
        jd = make_jd(must={"python"}, edu=3)
        [r] = score([make_profile(skills={"python"}, edu=2)], jd,
                    weights=ScoreWeights(0, 0, 0, 1))
        assert r.total == 50.0

    def test_jd_without_requirements_excludes_components(self):
        jd = make_jd(must={"python"})  # no years, no education
        [r] = score([make_profile(skills={"python"}, years=1, edu=1)], jd,
                    weights=ScoreWeights(1, 0, 1, 1))
        assert r.experience_score is None
        assert r.education_score is None
        assert r.total == 100.0  # blend uses skills only


class TestRanking:
    def test_stronger_candidate_ranks_first(self):
        jd = make_jd(must={"python", "sql", "aws"}, years=5)
        strong = make_profile("Strong", skills={"python", "sql", "aws"}, years=8)
        weak = make_profile("Weak", skills={"python"}, years=1)
        ranked = score([weak, strong], jd)
        assert ranked[0].profile.name == "Strong"
        assert ranked[0].total > ranked[1].total

    def test_semantic_breaks_ties(self):
        jd = make_jd(must={"python"})
        a = make_profile("Alpha", skills={"python"})
        b = make_profile("Beta", skills={"python"})
        ranked = score([a, b], jd, sem=[20.0, 80.0],
                       weights=ScoreWeights(0.5, 0.5, 0, 0))
        assert ranked[0].profile.name == "Beta"

    def test_empty_input(self):
        assert score_candidates([], make_jd(must={"python"})) == []


class TestWeights:
    def test_normalization(self):
        w = ScoreWeights(2, 2, 0, 0).normalized()
        assert abs(w.skills - 0.5) < 1e-9
        assert abs(w.semantic - 0.5) < 1e-9

    def test_all_zero_falls_back_to_skills(self):
        w = ScoreWeights(0, 0, 0, 0).normalized()
        assert w.skills == 1.0


class TestFlags:
    def test_missing_must_flag(self):
        jd = make_jd(must={"python", "sql"})
        [r] = score([make_profile(skills={"python"})], jd)
        assert any("Missing 1 must-have" in f for f in r.flags)

    def test_short_resume_flag(self):
        jd = make_jd(must={"python"})
        profile = make_profile(skills={"python"})
        profile.word_count = 40
        [r] = score([profile], jd)
        assert any("scanned/image" in f for f in r.flags)
