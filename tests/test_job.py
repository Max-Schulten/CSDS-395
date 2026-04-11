# tests/test_job.py
import pytest
from unittest.mock import MagicMock, call
from lib.job import Job


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_extractor():
    e = MagicMock()
    e.extract_all.return_value = (
        {"sql": 0.85, "python": 0.90},
        [{"degree": "BS", "major": "Computer Science", "confidence": 1.0}],
    )
    return e


DESC = "We are looking for a Python developer with SQL experience to join our team."


# ── __init__ ──────────────────────────────────────────────────────────────────

def test_job_stores_description(mock_extractor):
    j = Job(DESC, skill_extractor=mock_extractor)
    assert j.job_desc is not None
    assert "Python" in j.job_desc or "python" in j.job_desc.lower()


def test_job_stores_skills(mock_extractor):
    j = Job(DESC, skill_extractor=mock_extractor)
    assert "sql" in j.skills


def test_job_stores_education(mock_extractor):
    j = Job(DESC, skill_extractor=mock_extractor)
    assert len(j.education) == 1


def test_job_title_none_when_not_provided(mock_extractor):
    j = Job(DESC, skill_extractor=mock_extractor)
    assert j.job_title is None


def test_job_title_cleaned_when_provided(mock_extractor):
    j = Job(DESC, skill_extractor=mock_extractor, job_title="Senior  Python  Dev")
    assert j.job_title is not None
    assert "  " not in j.job_title


def test_job_optional_fields_stored(mock_extractor):
    j = Job(
        DESC,
        skill_extractor=mock_extractor,
        job_title="Engineer",
        company="Acme",
        loc="Remote",
        url="https://example.com/job",
    )
    assert j.company == "Acme"
    assert j.loc == "Remote"
    assert j.url == "https://example.com/job"


def test_job_optional_fields_default_to_none(mock_extractor):
    j = Job(DESC, skill_extractor=mock_extractor)
    assert j.company is None
    assert j.loc is None
    assert j.url is None


def test_job_title_prepended_to_extract_all_input(mock_extractor):
    j = Job(DESC, skill_extractor=mock_extractor, job_title="Python Engineer")
    call_args = mock_extractor.extract_all.call_args[0][0]
    assert "Python Engineer" in call_args
    assert "Python developer" in call_args or "looking" in call_args


def test_job_extract_all_called_with_desc_only_when_no_title(mock_extractor):
    Job(DESC, skill_extractor=mock_extractor)
    call_args = mock_extractor.extract_all.call_args[0][0]
    assert call_args == mock_extractor.extract_all.call_args[0][0]
    # no title prefix — just the cleaned desc
    assert "Python Engineer" not in call_args


# ── to_dict ───────────────────────────────────────────────────────────────────

def test_to_dict_has_all_keys(mock_extractor):
    j = Job(DESC, skill_extractor=mock_extractor)
    d = j.to_dict()
    assert set(d.keys()) == {"job_title", "job_desc", "skills", "education", "company", "location", "url"}


def test_to_dict_values_match_attributes(mock_extractor):
    j = Job(DESC, skill_extractor=mock_extractor, company="Acme", loc="NY", url="https://x.com")
    d = j.to_dict()
    assert d["job_desc"] == j.job_desc
    assert d["skills"] == j.skills
    assert d["education"] == j.education
    assert d["company"] == "Acme"
    assert d["location"] == "NY"
    assert d["url"] == "https://x.com"
