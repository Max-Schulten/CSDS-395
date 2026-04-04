import pytest
import numpy as np
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4  # embedding dimensionality for all tests


def unit(v):
    """Return a unit-norm row vector."""
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def mock_embedder(*arrays):
    """Embedding model whose encode() returns arrays in round-robin order."""
    emb = MagicMock()
    emb.encode.side_effect = list(arrays)
    return emb


def make_resume(skills=None, education=None, raw_text="resume text"):
    r = MagicMock()
    r.skills = skills or {}
    r.education = education or {"degree": None, "majors": []}
    r.resume_raw = raw_text
    r.resume_text = raw_text
    r.to_dict.return_value = {"text": raw_text, "skills": r.skills, "education": r.education}
    return r


def make_job(skills=None, education=None, desc="job description"):
    j = MagicMock()
    j.skills = skills or {}
    j.education = education or {"degree": None, "majors": []}
    j.job_desc = desc
    j.to_dict.return_value = {"job_desc": desc, "skills": j.skills, "education": j.education}
    return j


# ---------------------------------------------------------------------------
# embed_doc
# ---------------------------------------------------------------------------

class TestEmbedDoc:
    from utils.scoring import embed_doc

    def test_returns_tuple_of_array_and_list(self):
        from utils.scoring import embed_doc
        emb = MagicMock()
        emb.encode.return_value = np.zeros((1, DIM))
        arr, snaps = embed_doc("short text", emb, window_size=30, stride=10)
        assert isinstance(arr, np.ndarray)
        assert isinstance(snaps, list)

    def test_short_text_falls_back_to_full_text(self):
        from utils.scoring import embed_doc
        emb = MagicMock()
        emb.encode.return_value = np.zeros((1, DIM))
        text = "only five words here"
        _, snaps = embed_doc(text, emb, window_size=30, stride=10)
        assert snaps == [text]

    def test_long_text_produces_multiple_snapshots(self):
        from utils.scoring import embed_doc
        emb = MagicMock()
        emb.encode.side_effect = lambda s, **kw: np.zeros((len(s), DIM))
        tokens = ["word"] * 50
        text = " ".join(tokens)
        _, snaps = embed_doc(text, emb, window_size=10, stride=5)
        assert len(snaps) > 1

    def test_exact_snapshot_count(self):
        from utils.scoring import embed_doc
        emb = MagicMock()
        emb.encode.side_effect = lambda s, **kw: np.zeros((len(s), DIM))
        n_tokens, window, stride = 50, 10, 5
        text = " ".join(["word"] * n_tokens)
        _, snaps = embed_doc(text, emb, window_size=window, stride=stride)
        expected = len(range(0, n_tokens - window + 1, stride))
        assert len(snaps) == expected

    def test_multi_whitespace_handled(self):
        from utils.scoring import embed_doc
        emb = MagicMock()
        emb.encode.return_value = np.zeros((1, DIM))
        # .split() (no arg) collapses all whitespace — should not crash
        _, snaps = embed_doc("python  \t  developer\n\nsql", emb, window_size=30)
        assert len(snaps) == 1

    def test_encode_called_with_list(self):
        from utils.scoring import embed_doc
        emb = MagicMock()
        emb.encode.return_value = np.zeros((1, DIM))
        embed_doc("some text", emb, window_size=30)
        args, kwargs = emb.encode.call_args
        assert isinstance(args[0], list)


# ---------------------------------------------------------------------------
# semantic_score
# ---------------------------------------------------------------------------

class TestSemanticScore:

    def test_returns_scalar(self):
        from utils.scoring import semantic_score
        res = np.array([unit([1, 0, 0, 0])])
        job = np.array([unit([1, 0, 0, 0])])
        result = semantic_score(res, job)
        assert np.isscalar(result) or result.ndim == 0

    def test_identical_embeddings_return_one(self):
        from utils.scoring import semantic_score
        v = unit([1, 2, 3, 4])
        res = np.array([v])
        job = np.array([v])
        assert pytest.approx(float(semantic_score(res, job)), abs=1e-6) == 1.0

    def test_orthogonal_embeddings_return_zero(self):
        from utils.scoring import semantic_score
        res = np.array([unit([1, 0, 0, 0])])
        job = np.array([unit([0, 1, 0, 0])])
        assert pytest.approx(float(semantic_score(res, job)), abs=1e-6) == 0.0

    def test_jd_coverage_only_is_asymmetric(self):
        """Swapping resume and job embs should give a different result
        when the distributions are unequal."""
        from utils.scoring import semantic_score
        res = np.array([unit([1, 0, 0, 0]), unit([0, 1, 0, 0])])
        job = np.array([unit([1, 0, 0, 0])])
        forward = float(semantic_score(res, job))   # 1 JD window vs 2 resume windows
        backward = float(semantic_score(job, res))  # 2 JD windows vs 1 resume window
        assert forward != backward

    def test_score_in_unit_interval(self):
        from utils.scoring import semantic_score
        rng = np.random.default_rng(42)
        res = rng.standard_normal((5, DIM))
        res /= np.linalg.norm(res, axis=1, keepdims=True)
        job = rng.standard_normal((3, DIM))
        job /= np.linalg.norm(job, axis=1, keepdims=True)
        s = float(semantic_score(res, job))
        assert 0.0 <= s <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# skill_coverage
# ---------------------------------------------------------------------------

class TestSkillCoverage:

    def test_empty_job_skills_returns_zero(self):
        from utils.scoring import skill_coverage
        res_emb = np.array([unit([1, 0, 0, 0])])
        job_emb = np.zeros((0, DIM))
        assert skill_coverage(["python"], [], res_emb, job_emb) == 0.0

    def test_no_match_below_tau_returns_zero(self):
        from utils.scoring import skill_coverage
        res_emb = np.array([unit([1, 0, 0, 0])])
        job_emb = np.array([unit([0, 1, 0, 0])])  # orthogonal → sim=0
        assert skill_coverage(["python"], ["java"], res_emb, job_emb, tau=0.8) == 0.0

    def test_perfect_match_returns_one(self):
        from utils.scoring import skill_coverage
        v = unit([1, 0, 0, 0])
        res_emb = np.array([v])
        job_emb = np.array([v])
        result = skill_coverage(["python"], ["python"], res_emb, job_emb, tau=0.0)
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_partial_coverage(self):
        from utils.scoring import skill_coverage
        # resume has python; job needs python + java (orthogonal → no match)
        v_py = unit([1, 0, 0, 0])
        v_java = unit([0, 1, 0, 0])
        res_emb = np.array([v_py])
        job_emb = np.array([v_py, v_java])
        result = skill_coverage(["python"], ["python", "java"], res_emb, job_emb, tau=0.0)
        # python matched perfectly (weight=1), java unmatched → coverage = 1/2
        assert pytest.approx(result, abs=1e-6) == 0.5

    def test_best_resume_skill_wins_per_job_skill(self):
        """Two resume skills match the same job skill — higher sim wins."""
        from utils.scoring import skill_coverage
        v_job = unit([1, 0, 0, 0])
        v_res_high = unit([1, 0, 0, 0])        # sim=1.0 with job
        v_res_low  = unit([0.9, 0.1, 0.1, 0])  # lower sim
        v_res_low /= np.linalg.norm(v_res_low)
        res_emb = np.array([v_res_high, v_res_low])
        job_emb = np.array([v_job])
        result = skill_coverage(["python", "py"], ["python"], res_emb, job_emb, tau=0.0)
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_weight_formula_at_midpoint(self):
        """At sim = (1 + tau) / 2, weight should be 0.5."""
        from utils.scoring import skill_coverage
        tau = 0.8
        sim = (1.0 + tau) / 2  # 0.9
        # Construct two unit vectors with dot product = sim
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        # v2 such that v1·v2 = sim: v2 = [sim, sqrt(1-sim^2), 0, 0]
        v2 = np.array([sim, np.sqrt(1 - sim**2), 0.0, 0.0])
        res_emb = v1.reshape(1, -1)
        job_emb = v2.reshape(1, -1)
        result = skill_coverage(["python"], ["python"], res_emb, job_emb, tau=tau)
        expected = (sim - tau) / (1.0 - tau)
        assert pytest.approx(result, abs=1e-4) == expected

    def test_returns_float(self):
        from utils.scoring import skill_coverage
        v = unit([1, 0, 0, 0])
        result = skill_coverage(["python"], ["python"], v.reshape(1,-1), v.reshape(1,-1))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# education_coverage
# ---------------------------------------------------------------------------

class TestEducationCoverage:

    def _edu(self, degree=None, majors=None):
        return {"degree": degree, "majors": majors or []}

    # ── Degree scoring ────────────────────────────────────────────────────────

    def test_no_job_degree_gives_full_degree_score(self):
        from utils.scoring import education_coverage
        emb = MagicMock()
        result = education_coverage(
            self._edu("bachelor's"), self._edu(None), emb
        )
        # degree_score=1.0, no majors → major_score=1.0 → 0.7+0.3=1.0
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_unknown_resume_degree_gives_partial(self):
        from utils.scoring import education_coverage
        emb = MagicMock()
        result = education_coverage(
            self._edu(None), self._edu("bachelor's"), emb
        )
        # degree_score=0.5, no majors → 0.7*0.5 + 0.3*1.0 = 0.65
        assert pytest.approx(result, abs=1e-6) == 0.65

    def test_exact_degree_match(self):
        from utils.scoring import education_coverage
        emb = MagicMock()
        result = education_coverage(
            self._edu("bachelor's"), self._edu("bachelor's"), emb
        )
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_resume_exceeds_job_degree(self):
        from utils.scoring import education_coverage
        emb = MagicMock()
        result = education_coverage(
            self._edu("master's"), self._edu("bachelor's"), emb
        )
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_one_level_under_gives_half_degree_score(self):
        from utils.scoring import education_coverage
        emb = MagicMock()
        result = education_coverage(
            self._edu("bachelor's"), self._edu("master's"), emb
        )
        # degree_score=0.5, no majors → 0.7*0.5 + 0.3*1.0 = 0.65
        assert pytest.approx(result, abs=1e-6) == 0.65

    def test_two_levels_under_gives_zero_degree_score(self):
        from utils.scoring import education_coverage
        emb = MagicMock()
        result = education_coverage(
            self._edu("bachelor's"), self._edu("phd"), emb
        )
        # degree_score=0.0, no majors → 0.7*0.0 + 0.3*1.0 = 0.3
        assert pytest.approx(result, abs=1e-6) == 0.3

    # ── Major scoring ─────────────────────────────────────────────────────────

    def test_no_job_majors_gives_full_major_score(self):
        from utils.scoring import education_coverage
        emb = MagicMock()
        result = education_coverage(
            self._edu("bachelor's", ["computer science"]),
            self._edu("bachelor's", []),
            emb,
            resume_major_embs=np.array([unit([1,0,0,0])])
        )
        # degree=1.0, major=1.0
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_unknown_resume_majors_gives_partial_major_score(self):
        from utils.scoring import education_coverage
        emb = MagicMock()
        result = education_coverage(
            self._edu("bachelor's", []),
            self._edu("bachelor's", ["computer science"]),
            emb,
            resume_major_embs=None   # no resume major embeddings
        )
        # degree=1.0, major=0.5 → 0.7 + 0.15 = 0.85
        assert pytest.approx(result, abs=1e-6) == 0.85

    def test_major_above_edu_tau_counted(self):
        from utils.scoring import education_coverage
        v = unit([1, 0, 0, 0])
        emb = MagicMock()
        emb.encode.return_value = np.array([v])  # job major emb
        result = education_coverage(
            self._edu("bachelor's", ["cs"]),
            self._edu("bachelor's", ["computer science"]),
            emb,
            resume_major_embs=np.array([v]),
            edu_tau=0.85
        )
        # sim=1.0 >= 0.85 → major matched → major_score=1.0 → total=1.0
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_major_below_edu_tau_not_counted(self):
        from utils.scoring import education_coverage
        v_res = unit([1, 0, 0, 0])
        v_job = unit([0, 1, 0, 0])  # orthogonal → sim=0
        emb = MagicMock()
        emb.encode.return_value = np.array([v_job])
        result = education_coverage(
            self._edu("bachelor's", ["cs"]),
            self._edu("bachelor's", ["unrelated"]),
            emb,
            resume_major_embs=np.array([v_res]),
            edu_tau=0.85
        )
        # sim=0 < 0.85 → major_score=0.0 → 0.7*1.0 + 0.3*0.0 = 0.7
        assert pytest.approx(result, abs=1e-6) == 0.7

    def test_final_score_is_weighted_combination(self):
        from utils.scoring import education_coverage
        # degree one level under (score=0.5), no majors (score=1.0)
        emb = MagicMock()
        result = education_coverage(
            self._edu("bachelor's"), self._edu("master's"), emb
        )
        assert pytest.approx(result, abs=1e-6) == 0.7 * 0.5 + 0.3 * 1.0

    def test_partial_major_coverage_two_of_two(self):
        from utils.scoring import education_coverage
        v = unit([1, 0, 0, 0])
        emb = MagicMock()
        # Both job majors embed to v; resume also embeds to v → both matched
        emb.encode.return_value = np.array([v, v])
        result = education_coverage(
            self._edu("bachelor's", ["cs", "math"]),
            self._edu("bachelor's", ["cs", "math"]),
            emb,
            resume_major_embs=np.array([v, v]),
            edu_tau=0.85,
        )
        # degree=1.0, major=1.0 (2/2 matched)
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_partial_major_coverage_one_of_two(self):
        from utils.scoring import education_coverage
        v_match = unit([1, 0, 0, 0])
        v_miss  = unit([0, 1, 0, 0])
        emb = MagicMock()
        # job encodes both majors; first matches, second is orthogonal
        emb.encode.return_value = np.array([v_match, v_miss])
        result = education_coverage(
            self._edu("bachelor's", ["cs"]),
            self._edu("bachelor's", ["cs", "unrelated"]),
            emb,
            resume_major_embs=np.array([v_match]),
            edu_tau=0.85,
        )
        # degree=1.0, major=0.5 (1/2 matched) → 0.7 + 0.15 = 0.85
        assert pytest.approx(result, abs=1e-6) == 0.85

    def test_partial_major_coverage_zero_of_two(self):
        from utils.scoring import education_coverage
        v_res = unit([1, 0, 0, 0])
        v_job = unit([0, 1, 0, 0])
        emb = MagicMock()
        emb.encode.return_value = np.array([v_job, v_job])
        result = education_coverage(
            self._edu("bachelor's", ["cs"]),
            self._edu("bachelor's", ["unrelated1", "unrelated2"]),
            emb,
            resume_major_embs=np.array([v_res]),
            edu_tau=0.85,
        )
        # degree=1.0, major=0.0 → 0.7 + 0.0 = 0.7
        assert pytest.approx(result, abs=1e-6) == 0.7


# ---------------------------------------------------------------------------
# score (integration)
# ---------------------------------------------------------------------------

class TestScore:

    def _make_inputs(self, n_jobs=1):
        """Minimal mocked Resume + Jobs with controlled embeddings."""
        resume = make_resume(
            skills={"python": [0, 6]},
            education={"degree": "bachelor's", "majors": ["computer science"]},
            raw_text="python developer with sql experience"
        )
        jobs = [
            make_job(
                skills={"python": [0, 6]},
                education={"degree": "bachelor's", "majors": ["computer science"]},
                desc="looking for a python developer"
            )
            for _ in range(n_jobs)
        ]

        v = unit([1, 0, 0, 0])
        arr = np.array([v])

        emb = MagicMock()
        emb.encode.return_value = arr
        return resume, jobs, emb

    def test_output_has_expected_keys(self):
        from utils.scoring import score
        resume, jobs, emb = self._make_inputs()
        result = score(resume, jobs, embedding_model=emb)
        for key in ("resume", "jobs", "skill_coverages", "semantic_scores",
                    "education_coverages", "scores"):
            assert key in result

    def test_single_job_accepted(self):
        from utils.scoring import score
        resume, jobs, emb = self._make_inputs()
        result = score(resume, jobs[0], embedding_model=emb)
        assert len(result["scores"]) == 1

    def test_multiple_jobs_produce_matching_length_results(self):
        from utils.scoring import score
        resume, jobs, emb = self._make_inputs(n_jobs=3)
        result = score(resume, jobs, embedding_model=emb)
        for key in ("jobs", "skill_coverages", "semantic_scores",
                    "education_coverages", "scores"):
            assert len(result[key]) == 3

    def test_invalid_weights_raise_assertion(self):
        from utils.scoring import score
        resume, jobs, emb = self._make_inputs()
        with pytest.raises(AssertionError):
            score(resume, jobs, embedding_model=emb, alpha=0.5, beta=0.5, gamma=0.5)

    def test_scores_are_non_negative(self):
        from utils.scoring import score
        resume, jobs, emb = self._make_inputs(n_jobs=2)
        result = score(resume, jobs, embedding_model=emb)
        assert all(s >= 0.0 for s in result["scores"])

    def test_resume_dict_included_in_output(self):
        from utils.scoring import score
        resume, jobs, emb = self._make_inputs()
        result = score(resume, jobs, embedding_model=emb)
        assert result["resume"] == resume.to_dict()


# ---------------------------------------------------------------------------
# score — semi-integration (controlled embeddings, real math)
# ---------------------------------------------------------------------------

class TestScoreSemiIntegration:
    """Uses a deterministic embedding model to verify the full scoring
    arithmetic end-to-end without loading any real ML models."""

    def _make_deterministic_embedder(self, mapping: dict):
        """Returns an embedder whose encode() maps known strings to fixed unit
        vectors. Any unknown input gets the zero vector."""
        emb = MagicMock()

        def encode(texts, **kwargs):
            out = []
            for t in texts:
                v = mapping.get(t.strip().lower(), np.zeros(DIM))
                if np.linalg.norm(v) > 0:
                    v = v / np.linalg.norm(v)
                out.append(v)
            return np.array(out)

        emb.encode.side_effect = encode
        return emb

    def test_perfect_match_scores_higher_than_mismatch(self):
        from utils.scoring import score

        v_py  = np.array([1.0, 0.0, 0.0, 0.0])
        v_java = np.array([0.0, 1.0, 0.0, 0.0])
        v_doc  = np.array([0.7, 0.7, 0.0, 0.0])  # semantically similar to python

        mapping = {
            "python": v_py, "java": v_java,
            "python developer": v_doc,
            "looking for a python developer": v_doc,
            "looking for a java developer": v_java,
        }
        emb = self._make_deterministic_embedder(mapping)

        resume = make_resume(
            skills={"python": [0, 6]},
            education={"degree": "bachelor's", "majors": []},
            raw_text="python developer"
        )
        job_match    = make_job(skills={"python": [0, 6]},
                                education={"degree": "bachelor's", "majors": []},
                                desc="looking for a python developer")
        job_mismatch = make_job(skills={"java": [0, 4]},
                                education={"degree": "bachelor's", "majors": []},
                                desc="looking for a java developer")

        result = score(resume, [job_match, job_mismatch], embedding_model=emb,
                       alpha=0.45, beta=0.45, gamma=0.1, tau=0.5)

        assert result["scores"][0] > result["scores"][1]

    def test_score_arithmetic_matches_manual_calculation(self):
        from utils.scoring import score, skill_coverage, semantic_score, education_coverage

        v = unit([1, 0, 0, 0])
        emb = MagicMock()
        emb.encode.return_value = np.array([v])

        resume = make_resume(
            skills={"python": [0, 6]},
            education={"degree": "bachelor's", "majors": []},
            raw_text="python"
        )
        job = make_job(
            skills={"python": [0, 6]},
            education={"degree": "bachelor's", "majors": []},
            desc="python"
        )

        alpha, beta, gamma, tau = 0.45, 0.45, 0.1, 0.0

        result = score(resume, [job], embedding_model=emb,
                       alpha=alpha, beta=beta, gamma=gamma, tau=tau)

        sk  = result["skill_coverages"][0]
        sem = result["semantic_scores"][0]
        edu = result["education_coverages"][0]
        expected = alpha * sk + beta * sem + gamma * edu

        assert pytest.approx(result["scores"][0], abs=1e-6) == expected

    def test_higher_skill_weight_amplifies_skill_signal(self):
        """Skill gap = 1.0, semantic gap = 0.5.
        skill-heavy weighting should produce a wider total gap than sem-heavy."""
        from utils.scoring import score

        v_py       = unit([1.0, 0.0,   0.0, 0.0])
        v_java     = unit([0.0, 1.0,   0.0, 0.0])   # orthogonal → skill_cov=0
        v_java_doc = unit([0.5, 0.866, 0.0, 0.0])   # 60° from v_py → sim≈0.5

        # encode() returns vectors based on whether "java" appears (and not "python")
        def controlled_encode(texts, **kwargs):
            out = []
            for t in texts:
                tl = t.strip().lower()
                if "java" in tl and "python" not in tl:
                    out.append(v_java if tl == "java" else v_java_doc)
                else:
                    out.append(v_py)
            return np.array(out)

        emb = MagicMock()
        emb.encode.side_effect = controlled_encode

        resume      = make_resume(skills={"python": [0,6]},
                                  education={"degree": "bachelor's", "majors": []},
                                  raw_text="python developer")
        job_match   = make_job(skills={"python": [0,6]},
                               education={"degree": "bachelor's", "majors": []},
                               desc="python role")
        job_nomatch = make_job(skills={"java": [0,4]},
                               education={"degree": "bachelor's", "majors": []},
                               desc="java role")

        res_skill_heavy = score(resume, [job_match, job_nomatch], embedding_model=emb,
                                alpha=0.8, beta=0.1, gamma=0.1, tau=0.5)
        emb.encode.side_effect = controlled_encode  # reset after first score() call
        res_sem_heavy   = score(resume, [job_match, job_nomatch], embedding_model=emb,
                                alpha=0.1, beta=0.8, gamma=0.1, tau=0.5)

        gap_skill_heavy = res_skill_heavy["scores"][0] - res_skill_heavy["scores"][1]
        gap_sem_heavy   = res_sem_heavy["scores"][0]   - res_sem_heavy["scores"][1]

        assert gap_skill_heavy > gap_sem_heavy
