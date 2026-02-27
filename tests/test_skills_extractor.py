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
    from utils.skills_utils import SkillsExtractor
    return SkillsExtractor(nlp=nlp, matcher=matcher)


# ---------------------------------------------------------------------------
# load_skills_map
# ---------------------------------------------------------------------------

class TestLoadSkillsMap:
    def test_returns_dict_on_valid_file(self, tmp_path, simple_skills_map):
        from utils.skills_utils import load_skills_map
        p = tmp_path / "skill_map.json"
        p.write_text(json.dumps(simple_skills_map))
        result = load_skills_map(str(p))
        assert result == simple_skills_map

    def test_raises_on_missing_file(self, tmp_path):
        from utils.skills_utils import load_skills_map
        with pytest.raises(FileNotFoundError, match="Skills map json was not found"):
            load_skills_map(str(tmp_path / "nonexistent.json"))

    def test_raises_on_invalid_json(self, tmp_path):
        from utils.skills_utils import load_skills_map
        p = tmp_path / "bad.json"
        p.write_text("not valid json {{{")
        with pytest.raises(Exception):
            load_skills_map(str(p))


# ---------------------------------------------------------------------------
# load_skills_matcher
# ---------------------------------------------------------------------------

class TestLoadSkillsMatcher:
    def test_returns_phrase_matcher(self, nlp, tmp_path, simple_skills_map):
        from utils.skills_utils import load_skills_matcher
        p = tmp_path / "skill_map.json"
        p.write_text(json.dumps(simple_skills_map))
        result = load_skills_matcher(nlp_model=nlp, skill_map_path=str(p))
        assert isinstance(result, PhraseMatcher)

    def test_matcher_finds_known_skill(self, nlp, tmp_path, simple_skills_map):
        from utils.skills_utils import load_skills_matcher
        p = tmp_path / "skill_map.json"
        p.write_text(json.dumps(simple_skills_map))
        m = load_skills_matcher(nlp_model=nlp, skill_map_path=str(p))
        doc = nlp("I have experience with python and docker")
        matches = m(doc)
        matched_texts = {doc[s:e].text.lower() for _, s, e in matches}
        assert "python" in matched_texts
        assert "docker" in matched_texts

    def test_loads_nlp_if_not_provided(self, tmp_path, simple_skills_map):
        from utils.skills_utils import load_skills_matcher
        p = tmp_path / "skill_map.json"
        p.write_text(json.dumps(simple_skills_map))
        mock_nlp = MagicMock()
        mock_nlp.vocab = spacy.load("en_core_web_sm").vocab
        mock_nlp.make_doc = spacy.load("en_core_web_sm").make_doc
        with patch("utils.skills_utils.load_nlp", return_value=mock_nlp):
            result = load_skills_matcher(nlp_model=None, skill_map_path=str(p))
        assert isinstance(result, PhraseMatcher)


# ---------------------------------------------------------------------------
# SkillsExtractor.__init__
# ---------------------------------------------------------------------------

class TestSkillsExtractorInit:
    def test_adds_sentencizer_if_missing(self, nlp, matcher):
        from utils.skills_utils import SkillsExtractor
        # ensure sentencizer not present
        if nlp.has_pipe("sentencizer"):
            nlp.remove_pipe("sentencizer")
        ex = SkillsExtractor(nlp=nlp, matcher=matcher)
        assert ex.nlp.has_pipe("sentencizer")

    def test_does_not_add_duplicate_sentencizer(self, nlp, matcher):
        from utils.skills_utils import SkillsExtractor
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer", first=True)
        ex = SkillsExtractor(nlp=nlp, matcher=matcher)
        assert ex.nlp.pipe_names.count("sentencizer") == 1

    def test_uses_injected_matcher(self, nlp, matcher):
        from utils.skills_utils import SkillsExtractor
        ex = SkillsExtractor(nlp=nlp, matcher=matcher)
        assert ex.matcher is matcher

    def test_uses_injected_nlp(self, nlp, matcher):
        from utils.skills_utils import SkillsExtractor
        ex = SkillsExtractor(nlp=nlp, matcher=matcher)
        assert ex.nlp is nlp


# ---------------------------------------------------------------------------
# SkillsExtractor.extract_skills
# ---------------------------------------------------------------------------

class TestExtractSkills:
    def test_finds_single_skill(self, extractor):
        results = extractor.extract_skills("I have experience with Python.")
        assert "python" in results

    def test_finds_multiple_skills(self, extractor):
        results = extractor.extract_skills(
            "I am proficient in Python and Docker, and I have used Flask."
        )
        assert "python" in results
        assert "docker" in results
        assert "flask" in results

    def test_finds_multiword_skill(self, extractor):
        results = extractor.extract_skills(
            "I have worked on machine learning projects."
        )
        assert "machine learning" in results

    def test_is_case_insensitive(self, extractor):
        results = extractor.extract_skills("Expert in PYTHON and DOCKER.")
        assert "python" in results
        assert "docker" in results

    def test_returns_empty_dict_for_no_skills(self, extractor):
        results = extractor.extract_skills("I enjoy hiking and cooking.")
        assert results == {}

    def test_deduplicates_repeated_skills(self, extractor):
        results = extractor.extract_skills(
            "Python is great. I love Python. Python forever."
        )
        assert list(results.keys()).count("python") == 1

    def test_result_values_are_source_sentences(self, extractor):
        text = "I have used Flask for web development."
        results = extractor.extract_skills(text)
        assert "flask" in results
        assert results["flask"] == text.strip()

    def test_empty_string_input(self, extractor):
        results = extractor.extract_skills("")
        assert results == {}

    def test_returns_dict(self, extractor):
        results = extractor.extract_skills("Skilled in Python.")
        assert isinstance(results, dict)