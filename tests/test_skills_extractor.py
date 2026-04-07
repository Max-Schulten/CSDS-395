import pytest
import json
from unittest.mock import MagicMock, patch
from spacy.matcher import PhraseMatcher
import spacy


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nlp():
    """Lightweight real spaCy model for tests that need genuine NLP."""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def simple_skills_map():
    return {
        "python": "python",
        "machine learning": "machine learning",
        "docker": "docker",
        "postgresql": "postgresql",
        "flask": "flask",
    }


@pytest.fixture
def matcher(nlp, simple_skills_map):
    """A real PhraseMatcher built from simple_skills_map."""
    m = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(k) for k in simple_skills_map.keys()]
    m.add("SKILL", patterns)
    return m


@pytest.fixture
def extractor(nlp, matcher):
    """SkillsExtractor with injected nlp and matcher — no I/O."""
    from lib.skills_utils import SkillsExtractor
    return SkillsExtractor(nlp=nlp, matcher=matcher)


# ---------------------------------------------------------------------------
# load_skills_map
# ---------------------------------------------------------------------------

class TestLoadSkillsMap:
    def test_returns_dict_on_valid_file(self, tmp_path, simple_skills_map):
        from lib.skills_utils import load_skills_map
        p = tmp_path / "skill_map.json"
        p.write_text(json.dumps(simple_skills_map))
        result = load_skills_map(str(p))
        assert result == simple_skills_map

    def test_raises_on_missing_file(self, tmp_path):
        from lib.skills_utils import load_skills_map
        with pytest.raises(FileNotFoundError, match="Skills map json was not found"):
            load_skills_map(str(tmp_path / "nonexistent.json"))

    def test_raises_on_invalid_json(self, tmp_path):
        from lib.skills_utils import load_skills_map
        p = tmp_path / "bad.json"
        p.write_text("not valid json {{{")
        with pytest.raises(Exception):
            load_skills_map(str(p))


# ---------------------------------------------------------------------------
# load_skills_matcher
# ---------------------------------------------------------------------------

class TestLoadSkillsMatcher:
    
    def test_returns_phrase_matcher(self, nlp, tmp_path, simple_skills_map):
        from lib.skills_utils import load_skills_matcher
        p = tmp_path / "skill_map.json"
        p.write_text(json.dumps(simple_skills_map))
        result = load_skills_matcher(nlp_model=nlp, skills_map=simple_skills_map)
        assert isinstance(result, PhraseMatcher)

    def test_matcher_finds_known_skill(self, nlp, tmp_path, simple_skills_map):
        from lib.skills_utils import load_skills_matcher
        p = tmp_path / "skill_map.json"
        p.write_text(json.dumps(simple_skills_map))
        m = load_skills_matcher(nlp_model=nlp, skills_map=simple_skills_map)
        doc = nlp("I have experience with python and docker")
        matches = m(doc)
        matched_texts = {doc[s:e].text.lower() for _, s, e in matches}
        assert "python" in matched_texts
        assert "docker" in matched_texts

    def test_loads_nlp_if_not_provided(self, tmp_path, simple_skills_map):
        from lib.skills_utils import load_skills_matcher
        p = tmp_path / "skill_map.json"
        p.write_text(json.dumps(simple_skills_map))
        mock_nlp = MagicMock()
        mock_nlp.vocab = spacy.load("en_core_web_sm").vocab
        mock_nlp.make_doc = spacy.load("en_core_web_sm").make_doc
        with patch("utils.skills_utils.load_nlp", return_value=mock_nlp):
            result = load_skills_matcher(nlp_model=None, skills_map=simple_skills_map)
        assert isinstance(result, PhraseMatcher)


# ---------------------------------------------------------------------------
# SkillsExtractor.__init__
# ---------------------------------------------------------------------------

class TestSkillsExtractorInit:
    def test_adds_sentencizer_if_missing(self, nlp, matcher):
        from lib.skills_utils import SkillsExtractor
        # ensure sentencizer not present
        if nlp.has_pipe("sentencizer"):
            nlp.remove_pipe("sentencizer")
        ex = SkillsExtractor(nlp=nlp, matcher=matcher)
        assert ex.nlp.has_pipe("sentencizer")

    def test_does_not_add_duplicate_sentencizer(self, nlp, matcher):
        from lib.skills_utils import SkillsExtractor
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer", first=True)
        ex = SkillsExtractor(nlp=nlp, matcher=matcher)
        assert ex.nlp.pipe_names.count("sentencizer") == 1

    def test_uses_injected_matcher(self, nlp, matcher):
        from lib.skills_utils import SkillsExtractor
        ex = SkillsExtractor(nlp=nlp, matcher=matcher)
        assert ex.matcher is matcher

    def test_uses_injected_nlp(self, nlp, matcher):
        from lib.skills_utils import SkillsExtractor
        ex = SkillsExtractor(nlp=nlp, matcher=matcher)
        assert ex.nlp is nlp


# ---------------------------------------------------------------------------
# SkillsExtractor.extract_skills
# ---------------------------------------------------------------------------

class TestExtractSkills:
    def test_finds_single_skill(self, extractor):
        results = extractor.extract_skills("I have experience with Python.")
        assert "python (computer programming)" in results

    def test_finds_multiple_skills(self, extractor):
        results = extractor.extract_skills(
            "I am proficient in Python and Docker, and I have used Flask."
        )
        assert "python (computer programming)" in results
        assert "docker" in results
        assert "flask" in results

    def test_finds_multiword_skill(self, extractor):
        results = extractor.extract_skills(
            "I have worked on machine learning projects."
        )
        assert "machine learning" in results

    def test_is_case_insensitive(self, extractor):
        results = extractor.extract_skills("Expert in PYTHON and DOCKER.")
        assert "python (computer programming)" in results
        assert "docker" in results

    def test_returns_empty_dict_for_no_skills(self, extractor):
        results = extractor.extract_skills("I enjoy hiking and running")
        assert results == {}

    def test_deduplicates_repeated_skills(self, extractor):
        results = extractor.extract_skills(
            "Python is great. I love Python. Python forever."
        )
        assert list(results.keys()).count("python (computer programming)") == 1

    def test_empty_string_input(self, extractor):
        results = extractor.extract_skills("")
        assert results == {}

    def test_returns_dict(self, extractor):
        results = extractor.extract_skills("Skilled in Python.")
        assert isinstance(results, dict)

    # ── GLiNER path ───────────────────────────────────────────────────────────

    def test_gliner_skill_added_when_not_in_matcher(self, extractor):
        """A skill GLiNER finds that isn't in the skills_map is kept as-is."""
        extractor.gliner = MagicMock()
        extractor.gliner.predict_entities.return_value = [
            {"text": "kubernetes", "label": "technical software", "start": 0, "end": 10}
        ]
        results = extractor.extract_skills("kubernetes orchestration experience")
        assert "kubernetes" in results

    def test_gliner_skill_mapped_through_skills_map(self, extractor, simple_skills_map):
        """A skill GLiNER finds that IS in the skills_map is normalised."""
        extractor.gliner = MagicMock()
        extractor.gliner.predict_entities.return_value = [
            {"text": "python", "label": "job skill", "start": 0, "end": 6}
        ]
        extractor.skills_map = simple_skills_map
        results = extractor.extract_skills("python experience")
        # skills_map maps "python" → "python"; key present under that canonical
        assert "python" in results

    def test_gliner_deduplicates_with_phrase_matcher(self, extractor):
        """GLiNER should not add a skill already found by PhraseMatcher."""
        extractor.gliner = MagicMock()
        extractor.gliner.predict_entities.return_value = [
            {"text": "python", "label": "job skill", "start": 0, "end": 6}
        ]
        results = extractor.extract_skills("Python developer")
        # PhraseMatcher already adds python → GLiNER path should not duplicate
        assert list(results.keys()).count("python (computer programming)") == 1

    def test_gliner_education_label_excluded(self, extractor):
        """Matches labelled 'education' should not appear as skills."""
        extractor.gliner = MagicMock()
        extractor.gliner.predict_entities.return_value = [
            {"text": "B.S. Computer Science", "label": "education", "start": 0, "end": 21}
        ]
        results = extractor.extract_skills("B.S. Computer Science")
        assert "b.s. computer science" not in results

    def test_gliner_stores_char_positions(self, extractor):
        """GLiNER-sourced skills store the character offsets from the match."""
        extractor.gliner = MagicMock()
        extractor.gliner.predict_entities.return_value = [
            {"text": "terraform", "label": "technical software", "start": 5, "end": 14}
        ]
        results = extractor.extract_skills("use terraform daily")
        assert results["terraform"] == [5, 14]


# ---------------------------------------------------------------------------
# SkillsExtractor.extract_education
# ---------------------------------------------------------------------------

class TestExtractEducation:
    """GLiNER is mocked throughout so tests are fast and deterministic.
    We test the extraction and filtering logic, not GLiNER's model weights."""

    def _match(self, text, label="education degree", score=0.95):
        return {"text": text, "label": label, "score": score, "start": 0, "end": len(text)}

    def _mock_gliner(self, extractor, matches):
        extractor.gliner = MagicMock()
        extractor.gliner.predict_entities.return_value = matches

    # ── Return structure ───────────────────────────────────────────────────────

    def test_returns_dict_with_expected_keys(self, extractor):
        self._mock_gliner(extractor, [])
        result = extractor.extract_education("No education here.")
        assert isinstance(result, dict)
        assert "degree" in result and "majors" in result

    def test_empty_text_returns_none_degree_and_empty_majors(self, extractor):
        self._mock_gliner(extractor, [])
        result = extractor.extract_education("")
        assert result == {"degree": None, "majors": []}

    # ── Degree normalization ───────────────────────────────────────────────────

    def test_extracts_bachelors_from_bs_span(self, extractor):
        self._mock_gliner(extractor, [self._match("B.S. Computer Science")])
        assert extractor.extract_education("...")["degree"] == "bachelor's"

    def test_extracts_masters_from_ms_span(self, extractor):
        self._mock_gliner(extractor, [self._match("M.S. Machine Learning")])
        assert extractor.extract_education("...")["degree"] == "master's"

    def test_extracts_phd(self, extractor):
        self._mock_gliner(extractor, [self._match("Ph.D. Computer Science")])
        assert extractor.extract_education("...")["degree"] == "phd"

    def test_extracts_bachelors_full_word(self, extractor):
        self._mock_gliner(extractor, [self._match("Bachelor of Science in Engineering")])
        assert extractor.extract_education("...")["degree"] == "bachelor's"

    # ── Major extraction from bundled spans ───────────────────────────────────

    def test_extracts_major_from_bs_span(self, extractor):
        self._mock_gliner(extractor, [self._match("B.S. Computer Science")])
        assert "computer science" in extractor.extract_education("...")["majors"]

    def test_extracts_major_from_ba_span(self, extractor):
        self._mock_gliner(extractor, [self._match("B.A. Communications")])
        assert "communications" in extractor.extract_education("...")["majors"]

    def test_no_major_when_span_is_abbreviation_only(self, extractor):
        self._mock_gliner(extractor, [self._match("B.S.")])
        assert extractor.extract_education("...")["majors"] == []

    # ── Multiple degrees — take highest rank ──────────────────────────────────

    def test_takes_masters_over_bachelors(self, extractor):
        self._mock_gliner(extractor, [
            self._match("B.S. Computer Science", score=0.98),
            self._match("M.S. Machine Learning", score=0.95),
        ])
        assert extractor.extract_education("...")["degree"] == "master's"

    def test_takes_phd_over_masters(self, extractor):
        self._mock_gliner(extractor, [
            self._match("M.S. Statistics", score=0.97),
            self._match("Ph.D. Computer Science", score=0.92),
        ])
        assert extractor.extract_education("...")["degree"] == "phd"

    def test_collects_majors_from_all_degree_spans(self, extractor):
        self._mock_gliner(extractor, [
            self._match("B.S. Computer Science", score=0.98),
            self._match("M.S. Machine Learning", score=0.95),
        ])
        majors = extractor.extract_education("...")["majors"]
        assert "computer science" in majors
        assert "machine learning" in majors

    def test_breaks_rank_tie_by_confidence(self, extractor):
        """Two bachelor's spans — higher confidence one wins for degree;
        both contribute majors."""
        self._mock_gliner(extractor, [
            self._match("B.S. Computer Science", score=0.95),
            self._match("B.A. Mathematics", score=0.80),
        ])
        result = extractor.extract_education("...")
        assert result["degree"] == "bachelor's"
        assert "computer science" in result["majors"]
        assert "mathematics" in result["majors"]

    # ── Confidence floor for high-rank degrees ────────────────────────────────

    def test_master_below_floor_does_not_set_degree(self, extractor):
        """M.S. at 0.6 is below the 0.75 floor — should not beat a confident B.S."""
        self._mock_gliner(extractor, [
            self._match("B.S. Computer Science", score=0.95),
            self._match("M.S. Machine Learning", score=0.60),
        ])
        assert extractor.extract_education("...")["degree"] == "bachelor's"

    def test_phd_below_floor_does_not_set_degree(self, extractor):
        self._mock_gliner(extractor, [
            self._match("B.A. Economics", score=0.92),
            self._match("Ph.D.", score=0.55),
        ])
        assert extractor.extract_education("...")["degree"] == "bachelor's"

    def test_master_below_floor_still_contributes_major(self, extractor):
        """Low-confidence M.S. span should still yield its major."""
        self._mock_gliner(extractor, [
            self._match("M.S. Machine Learning", score=0.60),
        ])
        result = extractor.extract_education("...")
        assert result["degree"] is None          # didn't clear floor
        assert "machine learning" in result["majors"]

    def test_master_at_floor_sets_degree(self, extractor):
        self._mock_gliner(extractor, [self._match("M.S. Statistics", score=0.75)])
        assert extractor.extract_education("...")["degree"] == "master's"

    # ── Noise filtering ───────────────────────────────────────────────────────

    def test_school_name_does_not_set_degree(self, extractor):
        self._mock_gliner(extractor, [self._match("Harvard University", score=0.90)])
        assert extractor.extract_education("...")["degree"] is None

    def test_year_does_not_set_degree(self, extractor):
        self._mock_gliner(extractor, [self._match("2022", score=0.85)])
        assert extractor.extract_education("...")["degree"] is None

    def test_all_noise_returns_none_degree(self, extractor):
        self._mock_gliner(extractor, [
            self._match("State University", score=0.88),
            self._match("Riverside University", score=0.72),
        ])
        result = extractor.extract_education("...")
        assert result["degree"] is None
        assert result["majors"] == []

    # ── Major deduplication ───────────────────────────────────────────────────

    def test_deduplicates_identical_majors(self, extractor):
        self._mock_gliner(extractor, [
            self._match("B.S. Computer Science", score=0.97),
            self._match("B.S. Computer Science", score=0.85),
        ])
        majors = extractor.extract_education("...")["majors"]
        assert majors.count("computer science") == 1