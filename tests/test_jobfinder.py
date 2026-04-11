# tests/test_jobfinder.py
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from lib.jobfinder import JobFinder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_resume():
    r = MagicMock()
    r.categories = ["Technology"]
    r.skills = {"python": 0.9, "sql": 0.8, "docker": 0.7}
    return r


@pytest.fixture
def mock_extractor():
    e = MagicMock()
    e.extract_all.return_value = ({"python": 0.9}, [])
    return e


@pytest.fixture
def finder(mock_resume, mock_extractor):
    return JobFinder(resume=mock_resume, skill_extractor=mock_extractor, n_jobs=5)


def _mock_response(json_data=None, text=None, content=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    if text is not None:
        resp.text = text
    if content is not None:
        resp.content = content
    return resp


# ── _skill_hit_count ──────────────────────────────────────────────────────────

def test_skill_hit_count_exact_match(finder):
    assert finder._skill_hit_count("we need python and sql experience") == 2


def test_skill_hit_count_case_insensitive(finder):
    assert finder._skill_hit_count("We need Python and SQL experience") == 2


def test_skill_hit_count_no_partial_word_match(finder):
    # "pythonic" should not count as "python"
    assert finder._skill_hit_count("pythonic code is preferred") == 0


def test_skill_hit_count_zero_when_no_match(finder):
    assert finder._skill_hit_count("no relevant skills here") == 0


def test_skill_hit_count_all_skills(finder):
    assert finder._skill_hit_count("python sql docker pipeline") == 3


# ── _build_job ────────────────────────────────────────────────────────────────

def test_build_job_returns_job_on_valid_input(finder):
    raw = {
        "desc": "Looking for a Python developer with SQL experience.",
        "job_title": "Python Dev",
        "company": "Acme",
        "loc": "Remote",
        "url": "https://example.com",
    }
    job = finder._build_job(raw)
    assert job is not None


def test_build_job_returns_none_on_empty_desc(finder):
    raw = {"desc": "", "job_title": "Dev", "company": "X", "loc": None, "url": None}
    result = finder._build_job(raw)
    assert result is None


def test_build_job_returns_none_on_exception(finder):
    finder.skill_extractor.extract_all.side_effect = RuntimeError("model failed")
    raw = {"desc": "Python dev role at Acme Corp.", "job_title": "Dev", "company": "Acme", "loc": None, "url": None}
    result = finder._build_job(raw)
    assert result is None
    finder.skill_extractor.extract_all.side_effect = None  # reset


# ── _fetch_and_rank ───────────────────────────────────────────────────────────

BASE_RAW = [
    {"desc": "python sql docker job", "job_title": "Dev A", "company": "Acme", "source": "X"},
    {"desc": "python sql docker job", "job_title": "Dev A", "company": "Acme", "source": "X"},  # duplicate
    {"desc": "marketing coordinator role", "job_title": "Mkt B", "company": "Corp", "source": "Y"},
    {"desc": "python developer role", "job_title": "Dev C", "company": "Biz", "source": "Z"},
]


@patch("lib.jobfinder.asyncio.run")
def test_fetch_and_rank_deduplicates(mock_run, finder):
    mock_run.return_value = BASE_RAW
    result = finder._fetch_and_rank()
    titles = [r["job_title"] for r in result]
    assert titles.count("Dev A") == 1


@patch("lib.jobfinder.asyncio.run")
def test_fetch_and_rank_sorts_by_skill_hits(mock_run, finder):
    mock_run.return_value = BASE_RAW
    result = finder._fetch_and_rank()
    # "python sql docker" has 3 hits, "python developer" has 1, "marketing" has 0
    assert result[0]["job_title"] == "Dev A"


@patch("lib.jobfinder.asyncio.run")
def test_fetch_and_rank_respects_n_jobs(mock_run, finder):
    many = [
        {"desc": f"job {i}", "job_title": f"T{i}", "company": f"C{i}", "source": "S"}
        for i in range(20)
    ]
    mock_run.return_value = many
    result = finder._fetch_and_rank()
    assert len(result) <= finder.n_jobs


@patch("lib.jobfinder.asyncio.run")
def test_fetch_and_rank_returns_all_when_n_jobs_none(mock_run, mock_resume, mock_extractor):
    finder_unlimited = JobFinder(resume=mock_resume, skill_extractor=mock_extractor, n_jobs=0)
    many = [
        {"desc": f"job desc {i}", "job_title": f"T{i}", "company": f"C{i}", "source": "S"}
        for i in range(10)
    ]
    mock_run.return_value = many
    result = finder_unlimited._fetch_and_rank()
    assert len(result) == 10


# ── fetch_jobicy ──────────────────────────────────────────────────────────────

@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_jobicy_returns_jobs(mock_get, mock_sleep, finder):
    mock_get.return_value = _mock_response(json_data={
        "jobs": [{
            "id": "1",
            "jobTitle": "Python Dev",
            "companyName": "Acme",
            "jobGeo": "Remote",
            "jobDescription": "<p>Python and SQL required for this remote role.</p>",
            "url": "https://jobicy.com/job/1",
        }]
    })
    jobs = finder.fetch_jobicy()
    assert len(jobs) == 1
    assert jobs[0]["source"] == "Jobicy"
    assert jobs[0]["job_title"] == "Python Dev"
    assert "<" not in jobs[0]["desc"]


@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_jobicy_skips_short_desc(mock_get, mock_sleep, finder):
    mock_get.return_value = _mock_response(json_data={
        "jobs": [{"id": "1", "jobTitle": "Dev", "companyName": "X", "jobGeo": "Remote",
                  "jobDescription": "Short", "url": "https://x.com"}]
    })
    jobs = finder.fetch_jobicy()
    assert jobs == []


@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_jobicy_handles_request_error(mock_get, mock_sleep, finder):
    mock_get.side_effect = Exception("network error")
    jobs = finder.fetch_jobicy()
    assert jobs == []


# ── fetch_remoteok ────────────────────────────────────────────────────────────

@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_remoteok_returns_jobs(mock_get, mock_sleep, finder):
    mock_get.return_value = _mock_response(json_data=[
        {"legal": True},  # header entry without "id" — should be skipped
        {
            "id": "42",
            "position": "Backend Engineer",
            "company": "Corp",
            "location": "Remote",
            "description": "<p>Python and Docker and SQL expertise needed for this remote role.</p>",
            "url": "https://remoteok.com/42",
        }
    ])
    jobs = finder.fetch_remoteok()
    assert len(jobs) == 1
    assert jobs[0]["source"] == "RemoteOK"
    assert jobs[0]["job_title"] == "Backend Engineer"


@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_remoteok_handles_error(mock_get, mock_sleep, finder):
    mock_get.side_effect = Exception("timeout")
    jobs = finder.fetch_remoteok()
    assert jobs == []


# ── fetch_himalayas ───────────────────────────────────────────────────────────

@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_himalayas_returns_jobs(mock_get, mock_sleep, finder):
    mock_get.return_value = _mock_response(json_data={
        "jobs": [{
            "slug": "abc",
            "title": "ML Engineer",
            "companyName": "AI Corp",
            "locationRestrictions": ["USA", "Canada"],
            "description": "<p>Machine learning and Python and SQL experience required for this position.</p>",
            "applicationLink": "https://himalayas.app/job/abc",
        }]
    })
    jobs = finder.fetch_himalayas()
    assert len(jobs) == 1
    assert jobs[0]["source"] == "Himalayas"
    assert jobs[0]["loc"] == "USA, Canada"


@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_himalayas_empty_data_returns_empty(mock_get, mock_sleep, finder):
    mock_get.return_value = _mock_response(json_data={"jobs": []})
    assert finder.fetch_himalayas() == []


# ── fetch_arbeitnow ───────────────────────────────────────────────────────────

@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_arbeitnow_returns_jobs(mock_get, mock_sleep, finder):
    page1 = _mock_response(json_data={"data": [{
        "slug": "job-1",
        "title": "Data Engineer",
        "company_name": "DataCo",
        "remote": True,
        "location": "Berlin",
        "description": "<p>SQL and Python and data pipeline experience required for this role.</p>",
        "url": "https://arbeitnow.com/job-1",
    }]})
    page2 = _mock_response(json_data={"data": []})
    mock_get.side_effect = [page1, page2]
    jobs = finder.fetch_arbeitnow()
    assert len(jobs) == 1
    assert jobs[0]["source"] == "Arbeitnow"
    assert jobs[0]["loc"] == "Remote"


# ── fetch_weworkremotely ──────────────────────────────────────────────────────

RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <guid>https://weworkremotely.com/job/1</guid>
      <title>Acme Corp: Senior Python Developer</title>
      <description>We need a Python and SQL expert to join our team remotely.</description>
      <link>https://weworkremotely.com/job/1</link>
    </item>
  </channel>
</rss>"""


@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_weworkremotely_returns_jobs(mock_get, mock_sleep, finder):
    mock_get.return_value = _mock_response(content=RSS_XML)
    jobs = finder.fetch_weworkremotely()
    assert len(jobs) >= 1
    assert jobs[0]["source"] == "We Work Remotely"
    assert jobs[0]["company"] == "Acme Corp"
    assert jobs[0]["job_title"] == "Senior Python Developer"


@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_weworkremotely_handles_error(mock_get, mock_sleep, finder):
    mock_get.side_effect = Exception("connection refused")
    jobs = finder.fetch_weworkremotely()
    assert jobs == []


# ── fetch_themuse ─────────────────────────────────────────────────────────────

@patch("lib.jobfinder.time.sleep")
@patch("lib.jobfinder.requests.get")
def test_fetch_themuse_returns_jobs(mock_get, mock_sleep, finder):
    page1 = _mock_response(json_data={"results": [{
        "id": "101",
        "name": "Frontend Engineer",
        "company": {"name": "Muse Co"},
        "locations": [{"name": "New York, NY"}],
        "contents": "<p>JavaScript and React and CSS skills required for this position.</p>",
        "refs": {"landing_page": "https://themuse.com/jobs/101"},
    }]})
    page2 = _mock_response(json_data={"results": []})
    mock_get.side_effect = [page1, page2]
    jobs = finder.fetch_themuse()
    assert len(jobs) == 1
    assert jobs[0]["source"] == "The Muse"
    assert jobs[0]["company"] == "Muse Co"


# ── key-gated fetchers skip when no key set ───────────────────────────────────

@patch("lib.jobfinder.FINDWORK_API_KEY", "")
def test_fetch_findwork_skips_without_key(finder):
    with patch("lib.jobfinder.requests.get") as mock_get:
        jobs = finder.fetch_findwork()
        assert jobs == []
        mock_get.assert_not_called()


@patch("lib.jobfinder.USAJOBS_API_KEY", "")
@patch("lib.jobfinder.USAJOBS_USER_AGENT", "")
def test_fetch_usajobs_skips_without_key(finder):
    with patch("lib.jobfinder.requests.get") as mock_get:
        jobs = finder.fetch_usajobs()
        assert jobs == []
        mock_get.assert_not_called()


@patch("lib.jobfinder.ADZUNA_APP_ID", "")
@patch("lib.jobfinder.ADZUNA_APP_KEY", "")
def test_fetch_adzuna_skips_without_key(finder):
    with patch("lib.jobfinder.requests.get") as mock_get:
        jobs = finder.fetch_adzuna()
        assert jobs == []
        mock_get.assert_not_called()
