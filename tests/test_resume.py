# tests/test_resume.py
import pytest
from unittest.mock import MagicMock, patch
from lib.resume import Resume


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_classifier():
    c = MagicMock()
    c.clean_resume.return_value = "cleaned resume text"
    c.classify_resume.return_value = ["Technology", "Engineering"]
    return c


@pytest.fixture
def mock_extractor():
    e = MagicMock()
    e.extract_all.return_value = (
        {"python": 0.95, "sql": 0.80},
        [{"degree": "BS", "major": "Computer Science", "confidence": 1.0}],
    )
    return e


RAW_TEXT = "John Doe — Software Engineer with 5 years Python experience. BS Computer Science."


# ── __init__ ──────────────────────────────────────────────────────────────────

def test_resume_stores_raw_text(mock_classifier, mock_extractor):
    r = Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    assert r.resume_raw == RAW_TEXT


def test_resume_stores_cleaned_text(mock_classifier, mock_extractor):
    r = Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    assert r.resume_text == "cleaned resume text"


def test_resume_stores_categories(mock_classifier, mock_extractor):
    r = Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    assert r.categories == ["Technology", "Engineering"]


def test_resume_stores_skills(mock_classifier, mock_extractor):
    r = Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    assert "python" in r.skills


def test_resume_stores_education(mock_classifier, mock_extractor):
    r = Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    assert len(r.education) == 1
    assert r.education[0]["degree"] == "BS"


def test_resume_calls_extract_all_with_raw_text(mock_classifier, mock_extractor):
    Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    mock_extractor.extract_all.assert_called_once_with(RAW_TEXT, use_gliner=True, use_spacy=True)


def test_resume_calls_clean_resume(mock_classifier, mock_extractor):
    Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    mock_classifier.clean_resume.assert_called_once()


def test_resume_calls_classify_resume_with_cleaned_text(mock_classifier, mock_extractor):
    Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    mock_classifier.classify_resume.assert_called_once_with("cleaned resume text")


# ── to_dict ───────────────────────────────────────────────────────────────────

def test_to_dict_has_all_keys(mock_classifier, mock_extractor):
    r = Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    d = r.to_dict()
    assert set(d.keys()) == {"text", "skills", "education", "categories"}


def test_to_dict_values_match_attributes(mock_classifier, mock_extractor):
    r = Resume(RAW_TEXT, classifier=mock_classifier, skill_extractor=mock_extractor)
    d = r.to_dict()
    assert d["text"] == r.resume_text
    assert d["skills"] == r.skills
    assert d["education"] == r.education
    assert d["categories"] == r.categories


# ── default dependency loading ────────────────────────────────────────────────

def test_resume_calls_load_classifier_when_none_provided(mock_extractor):
    with patch("lib.resume.load_classifier") as mock_load, \
         patch("lib.resume.ResumeClassifier") as _:
        mock_load.return_value = MagicMock()
        mock_clf = MagicMock()
        mock_clf.clean_resume.return_value = "x"
        mock_clf.classify_resume.return_value = []
        mock_load.return_value = mock_clf
        Resume(RAW_TEXT, classifier=None, skill_extractor=mock_extractor)
        mock_load.assert_called_once()
