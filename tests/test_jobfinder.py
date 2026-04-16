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
    # Plain text with no seniority keywords so _seniority_retrieval returns None
    # for the resume side, keeping coeff=1 and leaving skill-hit tests unaffected.
    r.resume_text = "software engineer with python and sql experience"
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


# ── _seniority_retrieval ──────────────────────────────────────────────────────

class TestSeniorityRetrieval:
    """Tests for JobFinder._seniority_retrieval static method."""

    def test_returns_none_for_no_seniority_signal(self):
        assert JobFinder._seniority_retrieval("software engineer position available") is None

    def test_ordinals_strictly_descend_from_executive_to_intern(self):
        texts = [
            "Chief Executive Officer",
            "Director of Engineering",
            "Senior Software Engineer",
            "Associate Engineer",
            "Junior Developer",
            "Software Engineering Internship",
        ]
        ordinals = [JobFinder._seniority_retrieval(t)[1] for t in texts]
        assert ordinals == sorted(ordinals, reverse=True)

    def test_returns_highest_level_when_multiple_match(self):
        # resume text that mentions "senior" but also contains "intern" from past experience
        result = JobFinder._seniority_retrieval("Senior Developer with past intern experience")
        assert result is not None and result[0] == "senior"

    # --- executive ---

    @pytest.mark.parametrize("text", [
        "Chief Executive Officer",
        "Chief Technology Officer",
        "Chief Financial Officer",
        "CEO", "CTO", "CFO", "COO", "CISO", "CHRO", "CDO",
        "VP of Engineering",
        "Vice President of Product",
        "Managing Director",
        "Executive Director",
        "Managing Partner",
        "Senior Partner",
        "Equity Partner",
        "Partner, Smith & Jones LLP",     # standalone partner at consulting/law firm
    ])
    def test_executive_matches(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is not None and result[0] == "executive"

    @pytest.mark.parametrize("text", [
        "Partner Engineering Manager",    # IC role at tech company
        "Partner Success Manager",
        "Partner Program Manager",
        "Partner Operations Lead",
        "Partner Development Representative",
        "Partner Marketing Manager",
        "Partner Relations Specialist",
    ])
    def test_executive_partner_ic_roles_do_not_match(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is None or result[0] != "executive"

    # --- director ---

    @pytest.mark.parametrize("text", [
        "Director of Engineering",
        "Director of Product",
        "Senior Director of Operations",
        "Associate Director of Product",  # should be director, not mid
        "Head of Data Science",
        "Head of Engineering",
        "Group Product Manager",
        "Group Manager",
    ])
    def test_director_matches(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is not None and result[0] == "director"

    # --- senior ---

    @pytest.mark.parametrize("text", [
        "Sr. Software Engineer",          # abbreviation with period
        "Sr Software Engineer",           # abbreviation without period
        "Senior Software Engineer",
        "Senior Product Manager",
        "Senior Data Scientist",
        "Staff Engineer",
        "Staff Software Engineer",
        "Staff Data Scientist",
        "Principal Engineer",
        "Principal Product Manager",
        "Principal Scientist",
        "Lead Developer",
        "Lead Engineer",
        "Cloud Architect",
        "Solutions Architect",
        "Enterprise Architect",
        "Subject Matter Expert",          # expert at end of title
        "Expert Systems Engineer",        # expert at start of title
        "AWS Expert",                     # expert at end of compound title
    ])
    def test_senior_matches(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is not None and result[0] == "senior"

    @pytest.mark.parametrize("text", [
        "senior year student",
        "senior class project",
        "senior thesis",
        "senior standing",
        "senior student researcher",
        "senior capstone project",
        "senior honors thesis",
        "senior dissertation",
        "senior seminar course",
        "senior research fellow",
        "senior research student",
        "senior faculty position",
        "senior fellow at the institute",
        "senior lecturer position",
        "senior prom committee",
        "senior week activities",
        "looking for a senior.",          # bare "senior" — no following role word
        "senior project",                 # academic capstone, bare
        "senior capstone",                # academic capstone, bare
        "my senior project: ML app",      # resume project section header
        "lead instructor position",
        "lead student researcher",
        "lead tutor",
        "lead advisor for the program",
        "lead counselor",
    ])
    def test_senior_student_contexts_do_not_match(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is None or result[0] != "senior"

    @pytest.mark.parametrize("text", [
        "Senior Project Manager",
        "Senior Project Engineer",
        "Senior Project Coordinator",
        "Senior Project Lead",
        "Senior Project Analyst",
        "Senior Project Architect",
        "Senior Capstone Engineer",
    ])
    def test_senior_project_compound_titles_match(self, text):
        """project/capstone in a compound job title must not suppress the senior match."""
        result = JobFinder._seniority_retrieval(text)
        assert result is not None and result[0] == "senior"

    # --- mid ---

    @pytest.mark.parametrize("text", [
        "Mid-Level Software Engineer",
        "Mid Level Developer",
        "Intermediate Developer",
        "Intermediate Data Analyst",
        "Software Engineer II",
        "Software Engineer 2",
        "Engineer III",
        "Associate Software Engineer",
        "Associate Product Manager",
        "Associate Data Analyst",
    ])
    def test_mid_matches(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is not None and result[0] == "mid"

    @pytest.mark.parametrize("text", [
        "Associate Professor of Computer Science",
        "Associate Degree in Information Technology",
        "Associate's Degree Program",           # possessive form
        "Associate's of Science",
        "Associate Dean of Students",
        "Associate Provost for Research",
        "Associate Vice President of Operations",
    ])
    def test_mid_academic_executive_do_not_match(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is None or result[0] not in ("mid",)

    def test_associate_director_caught_as_director_not_mid(self):
        result = JobFinder._seniority_retrieval("Associate Director of Product")
        assert result is not None and result[0] == "director"

    # --- junior ---

    @pytest.mark.parametrize("text", [
        "Jr. Developer",                  # abbreviation with period
        "Jr Developer",                   # abbreviation without period
        "Junior Software Engineer",
        "Junior Data Analyst",
        "Junior Product Designer",
        "Entry-Level Developer",
        "Entry Level Data Analyst",
        "Software Engineer I",
        "New Grad Software Engineer",
        "New Graduate Developer",
        "Early-Career Software Engineer",
    ])
    def test_junior_matches(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is not None and result[0] == "junior"

    @pytest.mark.parametrize("text", [
        "junior year of college",
        "junior class standing",
        "junior varsity coach",
        "junior varsity team",
        "junior standing at university",
        "junior student in CS",
        "junior students enrolled",
        "junior thesis project",
        "junior research fellow",
        "junior research assistant",
        "junior achievement award",
        "junior prom planning",
        "looking for a junior.",          # bare "junior" — no following role word
    ])
    def test_junior_student_contexts_do_not_match(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is None or result[0] != "junior"

    # --- intern ---

    @pytest.mark.parametrize("text", [
        "Software Engineering Internship",
        "Software Intern",
        "Summer Intern",
        "Software Co-op Position",
        "Software Coop",
        "Apprenticeship Program",
        "Developer Apprentice",
        "Trainee Developer",
        "Practicum Student",
    ])
    def test_intern_matches(self, text):
        result = JobFinder._seniority_retrieval(text)
        assert result is not None and result[0] == "intern"


# ── _heuristic_score ──────────────────────────────────────────────────────────
# _heuristic_score(desc, res_seniority) takes the resume's seniority ordinal
# pre-computed by _fetch_and_rank so it isn't re-derived on every call.
# Use _res_sen() in tests to derive it from a plain text string.

def _res_sen(text: str) -> "int | None":
    t = JobFinder._seniority_retrieval(text)
    return t[1] if t is not None else None


def test_heuristic_score_exact_match(finder):
    assert finder._heuristic_score("we need python and sql experience", None) == 2


def test_heuristic_score_case_insensitive(finder):
    assert finder._heuristic_score("We need Python and SQL experience", None) == 2


def test_heuristic_score_no_partial_word_match(finder):
    # "pythonic" should not count as "python"
    assert finder._heuristic_score("pythonic code is preferred", None) == 0


def test_heuristic_score_zero_when_no_match(finder):
    assert finder._heuristic_score("no relevant skills here", None) == 0


def test_heuristic_score_all_skills(finder):
    assert finder._heuristic_score("python sql docker pipeline", None) == 3


def test_heuristic_score_no_seniority_in_either_defaults_to_skill_hits(finder):
    # res_seniority=None → coeff stays 1
    score = finder._heuristic_score("software engineer needs python and sql", None)
    assert score == 2


def test_heuristic_score_no_seniority_in_job_defaults_to_skill_hits(finder):
    # job desc has no seniority signal → job_seniority=None → coeff stays 1
    score = finder._heuristic_score("python and sql required", _res_sen("Senior Software Engineer"))
    assert score == 2


def test_heuristic_score_full_when_seniority_matches(finder):
    # Same level — all 3 skills present
    score = finder._heuristic_score(
        "Senior Developer needs python sql docker",
        _res_sen("Senior Software Engineer"),
    )
    assert score == 3


def test_heuristic_score_allows_one_level_above(finder):
    # Mid is one level above junior — coeff should be 1
    score = finder._heuristic_score(
        "Associate Software Engineer needs python sql docker",
        _res_sen("Junior Software Engineer"),
    )
    assert score == 3


def test_heuristic_score_zeroes_when_job_too_senior(finder):
    # Senior is two levels above junior — coeff should be 0
    score = finder._heuristic_score(
        "Senior Developer needs python sql docker",
        _res_sen("Junior Software Engineer"),
    )
    assert score == 0


def test_heuristic_score_zeroes_when_job_is_executive_and_resume_is_junior(finder):
    score = finder._heuristic_score(
        "Chief Technology Officer needs python sql docker",
        _res_sen("Junior Software Engineer"),
    )
    assert score == 0


def test_heuristic_score_allows_same_level(finder):
    # coeff=1, only python matches among {python, sql, docker}
    score = finder._heuristic_score(
        "Senior Developer needs python sql docker",
        _res_sen("Senior Software Engineer"),
    )
    assert score == 3


def test_heuristic_score_does_not_penalize_applying_below_level(finder):
    # Junior role — lower than resume level, coeff should still be 1
    score = finder._heuristic_score(
        "Junior Developer needs python and sql",
        _res_sen("Senior Software Engineer"),
    )
    assert score == 2


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
def test_fetch_and_rank_sorts_by_heuristic_score(mock_run, finder):
    mock_run.return_value = BASE_RAW
    result = finder._fetch_and_rank()
    # "python sql docker" has 3 skill hits, "python developer" has 1, "marketing" has 0
    # resume_text has no seniority → coeff=1 for all, so ordering is purely by skill hits
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
