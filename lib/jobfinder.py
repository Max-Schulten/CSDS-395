from lib.job import Job
from lib.resume import Resume
from lib.skills_utils import SkillsExtractor
from lib.gen_utils import clean_html
from config import *
import xml.etree.ElementTree as ET
import re
import requests
import random
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class JobFinder:
    SENIORITY_PATTERNS = {
        "executive": re.compile(
            r"\b(?:"
            r"c[- ]?(?:suite|level|exec(?:utive)?)"
            r"|chief\s+\w+\s+officer"
            r"|(?:ceo|cto|cfo|coo|ciso|cpo|cmo|cro|chro|cdo)"         # expanded C-suite acronyms
            r"|(?:managing|executive)\s+director"
            r"|(?:vice\s+president|vp)\s+of"
            # partner: only qualified forms or standalone when NOT an IC/program role
            r"|(?:senior|managing|equity|founding|general)\s+partner"
            r"|partner(?!\s+(?:engineer(?:ing)?|manager|success|program|operations|development|marketing|relations))"
            r")\b",
            re.IGNORECASE,
        ),

        "director": re.compile(
            r"\b(?:"
            r"(?:associate\s+|senior\s+)?director(?:\s+of)?"           # includes Associate Director
            r"|head\s+of\s+\w+"
            r"|group\s+(?:product\s+)?manager"
            r")\b",
            re.IGNORECASE,
        ),

        # Split sr./jr. (abbreviations with period) out of the \b...\b group because a trailing
        # \b fails after a period — \b requires a word/non-word boundary, but "." and " " are both
        # non-word so no boundary exists. Abbreviations are always professional context.
        "senior": re.compile(
            r"(?:"
            r"\bsr\.(?=\s)"                                             # "Sr." before whitespace
            r"|\b(?:"
            r"sr\b"                                                     # "Sr" without period
            r"|senior(?=\s+\w)"                                        # requires a role word after
            r"|staff(?=\s+\w)"                                         # staff engineer / scientist
            r"|principal(?=\s+\w)"                                     # principal engineer / pm
            r"|lead(?!\s+(?:student|ta|teaching|instructor|tutor|advisor|counselor))"
            r"|expert"                                                  # subject matter expert, aws expert, etc.
            r"|architect"
            r")\b"
            r"(?!\s+(?:"                                               # NOT student/academic contexts
            r"year|class|standing|student|students"
            r"|thesis|dissertation|seminar|honors"
            # capstone/project are academic when bare, but valid in professional compound titles
            # (e.g. "Senior Project Manager", "Senior Capstone Engineer")
            r"|capstone(?!\s+(?:manager|engineer|architect|lead|coordinator|analyst|consultant))"
            r"|project(?!\s+(?:manager|management|engineer|lead|coordinator|analyst|architect|officer|specialist|consultant))"
            r"|research\s+(?:fellow|assistant|project|student)"
            r"|faculty|fellow|lecturer"
            r"|prom|week|activities|activity"
            r"))"
            r")",
            re.IGNORECASE,
        ),

        "mid": re.compile(
            r"\b(?:"
            r"mid(?:[- ]level)?"
            r"|intermediate"
            r"|(?:software\s+)?engineer\s+(?:ii|2|iii|3)"             # Engineer II / III
            r"|associate(?!"                                            # not academic or executive
            r"'s"                                                      # associate's degree (possessive, no space)
            r"|\s+(?:degree|professor"
            r"|of\s+(?:arts|science|applied\s+science)"
            r"|dean|provost|director|vice\s+\w+"
            r"))"
            r")\b",
            re.IGNORECASE,
        ),

        "junior": re.compile(
            r"(?:"
            r"\bjr\.(?=\s)"                                            # "Jr." before whitespace
            r"|\b(?:"
            r"jr\b"                                                    # "Jr" without period
            r"|junior(?=\s+\w)"                                       # requires a role word after
            r"|entry[- ]level"
            r"|(?:software\s+)?engineer\s+i(?!\w)"                    # Engineer I (not II/III)
            r"|new\s+grad(?:uate)?"
            r"|early[- ]career"
            r")\b"
            r"(?!\s+(?:"                                               # NOT student/academic contexts
            r"year|class|standing|student|students"
            r"|thesis|faculty|fellow|honors|capstone|project"
            r"|research\s+(?:fellow|assistant|student)"
            r"|varsity|league|olympics|achievement"
            r"|prom|activities|activity"
            r"))"
            r")",
            re.IGNORECASE,
        ),

        "intern": re.compile(
            r"\b(?:"
            r"intern(?:ship)?"
            r"|co[- ]?op"
            r"|apprentice(?:ship)?"
            r"|trainee"
            r"|practicum"
            r")\b",
            re.IGNORECASE,
        ),
    }
    SKILL_SAMPLE_SIZE = 3
    RESULTS_PER_QUERY = 10

    JOBICY_MAP = {
        "Business & Management": [
            "admin-support",       # Admin & Virtual Assistant
            "business",            # Business Development
            "management",          # Product & Operations
        ],
        "Sales & Marketing": [
            "marketing",           # Marketing & Sales
            "seller",              # Sales
            "seo",                 # SEO
            "smm",                 # Social Media Marketing
        ],
        "Technology": [
            "data-science",        # Data Science & Analytics
            "admin",               # DevOps & Infrastructure
            "dev",                 # Programming
            "engineering",         # Software Engineering
            "web-app-design",      # Web & App Design
        ],
        "Human & Social Services": [
            "supporting",          # Customer Success
            "education",           # Education & E-learning
            "hr",                  # HR & Recruiting
            "technical-support",   # Technical Support
            "legal"
        ],
        "Finance & Accounting": [
            "accounting-finance",  # Finance & Accounting
        ],
        "Health & Lifestyle": [
            "healthcare",          # Healthcare & Medical
        ],
        "Creative & Design": [
            "copywriting",         # Content & Editorial
            "design-multimedia",   # Creative & Design
        ],
    }
    # Remote OK is technology only
    THEMUSE_MAP = {
        "Business & Management": [
            "Account Management",
            "Account Management/Customer Success",
            "Administration and Office",
            "Business Operations",
            "Corporate",
            "Management",
            "Office Administration",
            "Product",
            "Product Management",
            "Project Management",
            "Real Estate",
        ],
        "Sales & Marketing": [
            "Advertising and Marketing",
            "Marketing",
            "Media, PR, and Communications",
            "Public Relations",
            "Retail",
            "Sales",
            "Videography",
        ],
        "Technology": [
            "Computer and IT",
            "Data and Analytics",
            "Data Science",
            "IT",
            "Science and Engineering",
            "Software Engineer",
            "Software Engineering",
            "UX",
        ],
        "Human & Social Services": [
            "Customer Service",
            "Education",
            "HR",
            "Human Resources and Recruitment",
            "Mental Health",
            "Personal Care and Services",
            "Recruiting",
            "Social Services",
            "Law",
            "Legal Services",
            "Food and Hospitality Services",
            "Entertainment and Travel Services",
        ],
        "Finance & Accounting": [
            "Accounting",
            "Accounting and Finance",
        ],
        "Health & Lifestyle": [
            "Healthcare",
            "Nurses",
            "Physical Assistant",
            "Sports, Fitness, and Recreation",
        ],
        "Creative & Design": [
            "Arts",
            "Design",
            "Design and UX",
            "Editor",
            "Writer",
            "Writing and Editing",
        ],
        "Automobile": ["Mechanic"],
        "Other": [
            "Animal Care",
            "Cleaning and Facilities",
            "Construction",
            "Energy Generation and Mining",
            "Farming and Outdoors",
            "Installation, Maintenance, and Repairs",
            "Manufacturing and Warehouse",
            "Protective Services",
            "Transportation and Logistics",
            "Unknown",
        ],
    }

    WWR_MAP = {
        "Business & Management": [
            "https://weworkremotely.com/categories/remote-management-and-finance-jobs.rss",
            "https://weworkremotely.com/categories/remote-product-jobs.rss",
        ],
        "Sales & Marketing": [
            "https://weworkremotely.com/categories/remote-sales-and-marketing-jobs.rss",
        ],
        "Technology": [
            "https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss",
            "https://weworkremotely.com/categories/remote-back-end-programming-jobs.rss",
            "https://weworkremotely.com/categories/remote-front-end-programming-jobs.rss",
            "https://weworkremotely.com/categories/remote-programming-jobs.rss",
            "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
        ],
        "Human & Social Services": [
            "https://weworkremotely.com/categories/remote-customer-support-jobs.rss",
        ],
        "Finance & Accounting": [
            "https://weworkremotely.com/categories/remote-management-and-finance-jobs.rss",
        ],
        "Creative & Design": [
            "https://weworkremotely.com/categories/remote-design-jobs.rss",
        ],
        "Other": [
            "https://weworkremotely.com/categories/all-other-remote-jobs.rss",
        ],
    }
    
    @staticmethod
    def _seniority_retrieval(text: str) -> tuple[str, int] | None:
        ord_level = len(JobFinder.SENIORITY_PATTERNS.keys()) # Highest possible ordinal level
        for level, pattern in JobFinder.SENIORITY_PATTERNS.items():
            m = pattern.search(text)
            if m:
                return level, ord_level
            ord_level -= 1 # Decrease ordinal level
        return None

    
    def __init__(self, resume: Resume, skill_extractor: SkillsExtractor | None = None, n_jobs=10) -> None:
        self.skill_extractor = skill_extractor if skill_extractor is not None else SkillsExtractor()
        self.n_jobs = n_jobs
        self.resume = resume
        
    def _build_job(self, raw: dict) -> Job | None:
        try:
            return Job(
                raw["desc"],
                skill_extractor=self.skill_extractor,
                job_title=raw.get("job_title"),
                company=raw.get("company"),
                loc=raw.get("loc"),
                url=raw.get("url"),
            )
        except Exception as e:
            print(f"  Failed to build Job: {e}")
            return None

    async def _fetch_all_async(self) -> list[dict]:
        loop = asyncio.get_running_loop()
        fetchers = [
            self.fetch_jobicy, self.fetch_remoteok, self.fetch_themuse,
            self.fetch_himalayas, self.fetch_weworkremotely,
            self.fetch_findwork, self.fetch_usajobs, self.fetch_adzuna,
        ]
        with ThreadPoolExecutor() as pool:
            tasks = [loop.run_in_executor(pool, f) for f in fetchers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        raw = []
        for r in results:
            if isinstance(r, Exception):
                print(f"  Fetcher failed: {r}")
            else:
                raw += r # type: ignore
        return raw

    def _heuristic_score(self, desc: str, res_seniority: int | None) -> int:
        """
        Count how many resume skills appear as whole-word matches in desc.
        Acts as a heuristic for score -- can't realistically score thousands of jobs retrieved
        """
        desc_lower = desc.lower()
        skill_hits = sum(
            1 for skill in self.resume.skills
            if re.search(r"\b" + re.escape(skill.lower()) + r"\b", desc_lower)
        )
        seniority_tuple = self._seniority_retrieval(desc_lower)
        coeff = 1
        if seniority_tuple is not None and res_seniority is not None:
            coeff = int(seniority_tuple[1]-1 <= res_seniority) # Need candidate to be within one seniority level
        print("Job:", seniority_tuple)
        return coeff * skill_hits

    def _fetch_and_rank(self) -> list[dict]:
        """Fetch from all sources, deduplicate, rank by skill hits, and return top n_jobs raw dicts."""
        raw: list[dict] = asyncio.run(self._fetch_all_async())
        print(f"jobs fetched, found {len(raw)} jobs")

        seen: set[tuple] = set()
        deduped: list[dict] = []
        for r in raw:
            key = (r.get("job_title"), r.get("company"), r.get("source"))
            if key not in seen:
                seen.add(key)
                deduped.append(r)
            
        resume_seniority_tuple = self._seniority_retrieval(self.resume.resume_text)
        resume_sen_int = resume_seniority_tuple[1] if resume_seniority_tuple is not None else None
        
        print("Res:", resume_seniority_tuple)

        deduped.sort(key=lambda r: self._heuristic_score(r["desc"], resume_sen_int), reverse=True)
        return deduped[:self.n_jobs] if self.n_jobs else deduped

    def find_jobs(self) -> list[Job]:
        sample = self._fetch_and_rank()
        print(f"constructing {len(sample)} jobs")
        return [job for r in sample if (job := self._build_job(r))]
    
    def fetch_arbeitnow(self) -> list[dict]:
        jobs, seen = [], set()
        for page in range(1, 4):
            try:
                r = requests.get(
                    f"https://arbeitnow.com/api/job-board-api?page={page}",
                    headers=HEADERS, timeout=15
                )
                r.raise_for_status()
                data = r.json().get("data", [])
            except Exception as e:
                print(f"  Error page {page}: {e}"); break
            if not data:
                break
            for j in data:
                uid      = str(j.get("slug", ""))
                if uid in seen:
                    continue
                title    = j.get("title", None)
                company  = j.get("company_name", None)
                location = "Remote" if j.get("remote") else j.get("location", None)
                desc     = clean_html(j.get("description", ""))
                url      = j.get("url", None)
                if len(desc.split()) >= 5:
                    seen.add(uid); jobs.append({"desc": desc, "job_title": title, "company": company, "loc": location, "url": url, "source": "Arbeitnow"})
            time.sleep(REQUEST_DELAY)
        return jobs

    # 2. Jobicy
    def fetch_jobicy(self) -> list[dict]:
        jobs, seen = [], set()
        industry_tags = []
        for cat in self.resume.categories:
            industry_tags += self.JOBICY_MAP.get(cat, [])
        for industry in industry_tags:
            api_url = f"https://jobicy.com/api/v2/remote-jobs?industry={requests.utils.quote(industry)}&count=50" # type: ignore
            try:
                r = requests.get(api_url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                data = r.json().get("jobs", [])
            except Exception as e:
                print(f"  Error '{industry}': {e}"); continue
            for j in data:
                uid      = str(j.get("id", ""))
                if uid in seen:
                    continue
                title    = j.get("jobTitle", None)
                company  = j.get("companyName", None)
                location = j.get("jobGeo", None)
                desc     = clean_html(j.get("jobDescription", ""))
                url      = j.get("url", None)
                if len(desc.split()) >= 5:
                    seen.add(uid); jobs.append({"desc": desc, "job_title": title, "company": company, "loc": location, "url": url, "source": "Jobicy"})
            time.sleep(REQUEST_DELAY)
        return jobs


    # 3. RemoteOK
    def fetch_remoteok(self) -> list[dict]:
        jobs, seen = [], set()
        try:
            r = requests.get(
                "https://remoteok.com/api",
                headers={**HEADERS, "Accept": "application/json"},
                timeout=20
            )
            r.raise_for_status()
            listings = [l for l in r.json() if isinstance(l, dict) and "id" in l]
        except Exception as e:
            print(f"  Error: {e}"); return jobs
        for j in listings:
            uid     = str(j.get("id", ""))
            if uid in seen:
                continue
            title   = j.get("position", None)
            company = j.get("company", None)
            location = j.get("location") or "Remote"
            desc    = clean_html(j.get("description", ""))
            url = j.get("url", None)
            if len(desc.split()) >= 5:
                seen.add(uid); jobs.append({"desc": desc, "job_title": title, "company": company, "loc": location, "url": url, "source": "RemoteOK"})
        time.sleep(REQUEST_DELAY)
        return jobs


    # 4. The Muse
    def fetch_themuse(self) -> list[dict]:
        jobs, seen = [], set()
        cats = []
        for cat in self.resume.categories:
            cats += self.THEMUSE_MAP.get(cat, [])
        for page in range(1, 4):
            params: dict = {"category": cats, "page": page, "descending": "true"}
            if THEMUSE_KEY:
                params["api_key"] = THEMUSE_KEY
            try:
                r = requests.get(
                    "https://www.themuse.com/api/public/jobs",
                    params=params, headers=HEADERS, timeout=15
                )
                r.raise_for_status()
                results = r.json().get("results", [])
            except Exception as e:
                print(f"  Error page {page}: {e}"); break
            if not results:
                break
            for j in results:
                uid      = str(j.get("id", ""))
                if uid in seen:
                    continue
                title    = j.get("name", None)
                company  = j.get("company", {}).get("name", None)
                locs     = j.get("locations", None)
                location = locs[0].get("name", None) if locs else None
                desc     = clean_html(j.get("contents", ""))
                url      = j.get("refs", {}).get("landing_page", None)
                if len(desc.split()) >= 5:
                    seen.add(uid); jobs.append({"desc": desc, "job_title": title, "company": company, "loc": location, "url": url, "source": "The Muse"})
            time.sleep(REQUEST_DELAY)
        return jobs


    # 5. Himalayas
    def fetch_himalayas(self) -> list[dict]:
        jobs, seen = [], set()
        page_size = 20
        try:
            r = requests.get(
                f"https://himalayas.app/jobs/api?limit={page_size}",
                headers=HEADERS, timeout=15
            )
            r.raise_for_status()
            data = r.json().get("jobs", [])
        except Exception as e:
            print(f"  Error: {e}")
            return []
        if not data:
            return []
        for j in data:
            uid     = str(j.get("slug", j.get("id", "")))
            if uid in seen:
                continue
            title    = j.get("title", None)
            company  = j.get("companyName", None)
            loc_restrictions = j.get("locationRestrictions", [])
            location = ", ".join(loc_restrictions) if loc_restrictions else "Remote (Worldwide)"
            desc     = clean_html(j.get("description", ""))
            url      = j.get("applicationLink", j.get("url", None))
            if len(desc.split()) >= 5:
                seen.add(uid); jobs.append({"desc": desc, "job_title": title, "company": company, "loc": location, "url": url, "source": "Himalayas"})
        time.sleep(REQUEST_DELAY)
        return jobs


    def fetch_weworkremotely(self) -> list[dict]:
        jobs, seen = [], set()
        
        feeds = set()
        for cat in self.resume.categories:
            feeds.update(self.WWR_MAP.get(cat, self.WWR_MAP["Other"]))
        
        for feed_url in list(feeds):
            try:
                r = requests.get(feed_url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                root = ET.fromstring(r.content)
            except Exception as e:
                continue

            for item in root.iter("item"):
                def tag(name: str) -> str | None:
                    el = item.find(name)
                    return (el.text or "").strip() if el is not None else None

                uid     = tag("guid") or tag("link")
                if uid in seen:
                    continue
                title_raw = tag("title")
                if title_raw is not None and ":" in title_raw:
                    company, title = [p.strip() for p in title_raw.split(":", 1)]
                else:
                    title, company = title_raw, ""
                location = tag("region") or "Remote (US-friendly)"
                url      = tag("link")
                desc     = clean_html(tag("description") or "")
                if len(desc.split()) >= 5:
                    seen.add(uid); jobs.append({"desc": desc, "job_title": title, "company": company, "loc": location, "url": url, "source": "We Work Remotely"})
            time.sleep(REQUEST_DELAY)
        return jobs


    # 7. Findwork.dev
    def fetch_findwork(self) -> list[dict]:
        if not FINDWORK_API_KEY:
            print("[Findwork.dev] Skipped – set FINDWORK_API_KEY in .env to enable.")
            return []
        jobs, seen = [], set()
        auth_headers = {**HEADERS, "Authorization": f"Token {FINDWORK_API_KEY}"}

        skill_sample = random.sample(list(self.resume.skills.keys()),
                                     min(self.SKILL_SAMPLE_SIZE, len(self.resume.skills)))
        queries = [""] + skill_sample

        for query in queries:
            base = "https://findwork.dev/api/jobs/"
            url = f"{base}?search={requests.utils.quote(query)}" if query else base  # type: ignore
            try:
                r = requests.get(url, headers=auth_headers, timeout=15)
                r.raise_for_status()
                body = r.json()
            except Exception as e:
                print(f"  Error '{query}': {e}"); continue

            for j in body.get("results", [])[:self.RESULTS_PER_QUERY]:
                uid      = str(j.get("id", ""))
                if uid in seen:
                    continue
                title    = j.get("role", None)
                company  = j.get("company_name", None)
                location = j.get("location", None)
                keywords = ", ".join(j.get("keywords", []))
                desc     = f"{j.get('text', '')} Skills: {keywords}".strip()
                url      = j.get("url", None)
                if len(desc.split()) >= 5:
                    seen.add(uid); jobs.append({"desc": desc, "job_title": title, "company": company, "loc": location, "url": url, "source": "Findwork"})
            time.sleep(REQUEST_DELAY)

        return jobs


    # 8. USAJobs
    def fetch_usajobs(self) -> list[dict]:
        if not USAJOBS_API_KEY or not USAJOBS_USER_AGENT:
            print("[USAJobs] Skipped – set USAJOBS_API_KEY and USAJOBS_USER_AGENT in .env to enable.")
            return []
        jobs, seen = [], set()
        usajobs_headers = {
            **HEADERS,
            "Authorization-Key" : USAJOBS_API_KEY,
            "User-Agent"        : USAJOBS_USER_AGENT,
            "Host"              : "data.usajobs.gov",
        }

        skill_sample = random.sample(list(self.resume.skills.keys()),
                                     min(self.SKILL_SAMPLE_SIZE, len(self.resume.skills)))
        queries = [""] + skill_sample

        for query in queries:
            params = {"ResultsPerPage": self.RESULTS_PER_QUERY}
            if query:
                params["Keyword"] = query # type: ignore
            try:
                r = requests.get(
                    "https://data.usajobs.gov/api/Search",
                    headers=usajobs_headers, params=params, timeout=20
                )
                r.raise_for_status()
                items = (
                    r.json()
                    .get("SearchResult", {})
                    .get("SearchResultItems", [])
                )
            except Exception as e:
                print(f"  Error '{query}': {e}"); continue
            for item in items:
                pos      = item.get("MatchedObjectDescriptor", {})
                uid      = str(pos.get("PositionID", ""))
                if uid in seen:
                    continue
                title    = pos.get("PositionTitle", None)
                company  = pos.get("OrganizationName", None)
                locs     = pos.get("PositionLocation", [{}])
                location = locs[0].get("LocationName", "USA") if locs else "USA"
                desc     = clean_html(
                    pos.get("QualificationSummary", "")
                    + " "
                    + pos.get("UserArea", {})
                    .get("Details", {})
                    .get("JobSummary", "")
                )
                url      = pos.get("PositionURI", None)
                if len(desc.split()) >= 5:
                    seen.add(uid); jobs.append({"desc": desc, "job_title": title, "company": company, "loc": location, "url": url, "source": "USAJobs"})
            time.sleep(REQUEST_DELAY)
        return jobs


    # 9. Adzuna
    def fetch_adzuna(self) -> list[dict]:
        if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
            print("[Adzuna] Skipped – set ADZUNA_APP_ID and ADZUNA_APP_KEY in .env to enable.")
            return []
        jobs, seen = [], set()

        skill_sample = random.sample(list(self.resume.skills.keys()),
                                     min(self.SKILL_SAMPLE_SIZE, len(self.resume.skills)))
        queries = [""] + skill_sample

        for query in queries:
            base = (
                f"https://api.adzuna.com/v1/api/jobs/us/search/1"
                f"?app_id={ADZUNA_APP_ID}&app_key={ADZUNA_APP_KEY}"
                f"&results_per_page={self.RESULTS_PER_QUERY}&content-type=application/json"
            )
            url = base + f"&what={requests.utils.quote(query)}" if query else base  # type: ignore
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                results = r.json().get("results", [])
            except Exception as e:
                print(f"  Error '{query}': {e}"); continue
            for j in results:
                uid      = str(j.get("id", ""))
                if uid in seen:
                    continue
                title    = j.get("title", None)
                company  = j.get("company", {}).get("display_name", None)
                location = j.get("location", {}).get("display_name", None)
                desc     = clean_html(j.get("description", ""))
                url      = j.get("redirect_url", None)
                if len(desc.split()) >= 5:
                    seen.add(uid); jobs.append({"desc": desc, "job_title": title, "company": company, "loc": location, "url": url, "source": "Adzuna"})
            time.sleep(REQUEST_DELAY)
        return jobs