"""
Job Aggregator:
Pulls from 9 job sources and scores every listing against listed skills.

Usage:
  python job_aggregator.py                              # interactive prompt
  python job_aggregator.py --skills "python, sql, dbt"
  python job_aggregator.py --skills "react, typescript" --max-pages 5

    Arbeitnow          remote + EU jobs
    Jobicy             remote jobs worldwide
    RemoteOK           remote tech jobs
    The Muse           global jobs
    Himalayas          remote jobs (new)
    We Work Remotely   remote jobs via RSS (new)
    Findwork.dev       dev / tech jobs (new)  → FINDWORK_API_KEY
    USAJobs            US federal + government jobs (new)
                          → USAJOBS_API_KEY  + USAJOBS_USER_AGENT
    Adzuna             US private-sector jobs
                          → ADZUNA_APP_ID + ADZUNA_APP_KEY

Output:
  jobs_dataframe.csv    sorted by skill match score (highest first)
  job_descriptions/     one .txt per job with the full description
"""

import os
import re
import sys
import time
import argparse
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# config

DESC_DIR      = Path("job_descriptions")
OUTPUT_CSV    = "jobs_dataframe.csv"
REQUEST_DELAY = 0.5

HEADERS = {"User-Agent": "JobAggregator/4.0 (personal research script)"}

# API keys
FINDWORK_API_KEY  = "key"
USAJOBS_API_KEY   = "key"
USAJOBS_USER_AGENT = "email"
ADZUNA_APP_ID  = "id"
ADZUNA_APP_KEY = "key"
THEMUSE_KEY    = "key"


# input parsing
def parse_args() -> tuple[list[str], int]:
    parser = argparse.ArgumentParser(
        description="Aggregate jobs worldwide and score them against your skills."
    )
    parser.add_argument(
        "--skills",
        type=str,
        default="",
        help='Comma-separated skills, e.g. "python, sql, machine learning"',
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Pages to fetch per source per skill (default: 3)",
    )
    args = parser.parse_args()

    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    if not skills:
        raw = input("\nEnter your skills (comma-separated):\n> ").strip()
        skills = [s.strip() for s in raw.split(",") if s.strip()]

    if not skills:
        print("No skills provided. Exiting.")
        sys.exit(1)

    return skills, args.max_pages


# helper functions

def sanitize_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", (text or "unknown")).strip()
    text = re.sub(r"\s+", "_", text)
    return text[:max_len] or "unnamed"


def clean_html(raw: str) -> str:
    return re.sub(r"<[^>]+>", " ", raw or "").strip()


def score_skills(skills: list[str], title: str, description: str) -> tuple[int, list[str]]:
    """Whole-word regex match each skill in title + description."""
    combined = ((title or "") + " " + (description or "")).lower()
    matched  = [
        s for s in skills
        if re.search(r"\b" + re.escape(s.lower()) + r"\b", combined)
    ]
    return len(matched), matched


def save_description(job_id: str, title: str, company: str, location: str,
                     description: str, matched_skills: list[str]) -> str:
    DESC_DIR.mkdir(exist_ok=True)
    fname = f"{sanitize_filename(job_id)}_{sanitize_filename(title)}.txt"
    fpath = DESC_DIR / fname
    content = (
        f"Title          : {title}\n"
        f"Company        : {company}\n"
        f"Location       : {location}\n"
        f"Matched Skills : {', '.join(matched_skills) if matched_skills else 'None'}\n"
        f"{'─' * 60}\n\n"
        f"{description}\n"
    )
    fpath.write_text(content, encoding="utf-8")
    return str(fpath)


def build_row(source: str, uid: str, title: str, company: str, location: str,
              url: str, desc: str, skills: list[str]) -> dict | None:
    """Score skills, return a row dict or None if no skills matched."""
    score, matched = score_skills(skills, title, desc)
    if score == 0:
        return None
    fp = save_description(
        f"{source.lower().replace(' ', '_')}_{uid}",
        title, company, location, desc, matched
    )
    return {
        "source"          : source,
        "job_id"          : uid,
        "title"           : title,
        "company"         : company,
        "location"        : location,
        "url"             : url,
        "skills_matched"  : score,
        "matched_skills"  : ", ".join(matched),
        "description_file": fp,
    }


#  SOURCES
# 1. Arbeitnow
def fetch_arbeitnow(skills: list[str], max_pages: int) -> list[dict]:
    print("[Arbeitnow] Fetching …")
    jobs, seen = [], set()
    for page in range(1, max_pages + 1):
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
            uid     = str(j.get("slug", ""))
            if uid in seen:
                continue
            title   = j.get("title", "")
            company = j.get("company_name", "")
            location = "Remote" if j.get("remote") else j.get("location", "")
            desc    = clean_html(j.get("description", ""))
            row     = build_row("Arbeitnow", uid, title, company,
                                location, j.get("url", ""), desc, skills)
            if row:
                seen.add(uid); jobs.append(row)
        time.sleep(REQUEST_DELAY)
    print(f"  → {len(jobs)} matching jobs")
    return jobs


# 2. Jobicy

def fetch_jobicy(skills: list[str], max_pages: int) -> list[dict]:
    print("[Jobicy] Fetching …")
    jobs, seen = [], set()
    for skill in skills:
        url = f"https://jobicy.com/api/v2/remote-jobs?tag={requests.utils.quote(skill)}&count=50"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json().get("jobs", [])
        except Exception as e:
            print(f"  Error '{skill}': {e}"); continue
        for j in data:
            uid     = str(j.get("id", ""))
            if uid in seen:
                continue
            title   = j.get("jobTitle", "")
            company = j.get("companyName", "")
            location = j.get("jobGeo", "Remote")
            desc    = clean_html(j.get("jobDescription", ""))
            row     = build_row("Jobicy", uid, title, company,
                                location, j.get("url", ""), desc, skills)
            if row:
                seen.add(uid); jobs.append(row)
        time.sleep(REQUEST_DELAY)
    print(f"  → {len(jobs)} matching jobs")
    return jobs


# 3. RemoteOK
def fetch_remoteok(skills: list[str], max_pages: int) -> list[dict]:
    print("[RemoteOK] Fetching …")
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
        title   = j.get("position", "")
        company = j.get("company", "")
        location = j.get("location") or "Remote"
        desc    = clean_html(j.get("description", ""))
        row     = build_row("RemoteOK", uid, title, company,
                            location, f"https://remoteok.com/remote-jobs/{uid}",
                            desc, skills)
        if row:
            seen.add(uid); jobs.append(row)
    time.sleep(REQUEST_DELAY)
    print(f"  → {len(jobs)} matching jobs")
    return jobs


# 4. The Muse
def fetch_themuse(skills: list[str], max_pages: int) -> list[dict]:
    print("[The Muse] Fetching …")
    jobs, seen = [], set()
    for skill in skills:
        for page in range(1, max_pages + 1):
            params: dict = {"category": skill, "page": page, "descending": "true"}
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
                print(f"  Error page {page} '{skill}': {e}"); break
            if not results:
                break
            for j in results:
                uid     = str(j.get("id", ""))
                if uid in seen:
                    continue
                title   = j.get("name", "")
                company = j.get("company", {}).get("name", "")
                locs    = j.get("locations", [])
                location = locs[0].get("name", "Remote") if locs else "Remote"
                desc    = clean_html(j.get("contents", ""))
                row     = build_row("The Muse", uid, title, company, location,
                                    j.get("refs", {}).get("landing_page", ""),
                                    desc, skills)
                if row:
                    seen.add(uid); jobs.append(row)
            time.sleep(REQUEST_DELAY)
    print(f"  → {len(jobs)} matching jobs")
    return jobs


# 5. Himalayas
def fetch_himalayas(skills: list[str], max_pages: int) -> list[dict]:
    """
    https://himalayas.app/api
    Free, no key. Max 20 results per request; use offset to paginate.
    """
    print("[Himalayas] Fetching …")
    jobs, seen = [], set()
    page_size = 20
    total_pages = max_pages * 2   # each page is only 20 jobs

    for offset in range(0, total_pages * page_size, page_size):
        try:
            r = requests.get(
                f"https://himalayas.app/jobs/api?limit={page_size}&offset={offset}",
                headers=HEADERS, timeout=15
            )
            r.raise_for_status()
            data = r.json().get("jobs", [])
        except Exception as e:
            print(f"  Error offset {offset}: {e}"); break
        if not data:
            break
        for j in data:
            uid     = str(j.get("slug", j.get("id", "")))
            if uid in seen:
                continue
            title   = j.get("title", "")
            company = j.get("companyName", "")
            # locationRestrictions is a list of country names
            loc_restrictions = j.get("locationRestrictions", [])
            location = ", ".join(loc_restrictions) if loc_restrictions else "Remote (Worldwide)"
            desc    = clean_html(j.get("description", ""))
            url     = j.get("applicationLink", j.get("url", ""))
            row     = build_row("Himalayas", uid, title, company, location, url, desc, skills)
            if row:
                seen.add(uid); jobs.append(row)
        time.sleep(REQUEST_DELAY)
    print(f"  → {len(jobs)} matching jobs")
    return jobs


# 6. We Work Remotely — RSS
WWR_FEEDS = {
    "Programming"          : "https://weworkremotely.com/categories/remote-programming-jobs.rss",
    "DevOps / Sysadmin"    : "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
    "Design"               : "https://weworkremotely.com/categories/remote-design-jobs.rss",
    "Data Science"         : "https://weworkremotely.com/categories/remote-data-science-jobs.rss",
    "Product"              : "https://weworkremotely.com/categories/remote-product-jobs.rss",
    "Management & Finance" : "https://weworkremotely.com/categories/remote-management-finance-legal-jobs.rss",
    "All Other Remote"     : "https://weworkremotely.com/categories/remote-jobs.rss",
}
def fetch_weworkremotely(skills: list[str], max_pages: int) -> list[dict]:
    """
    We Work Remotely public RSS feeds — no key, ~80% US companies.
    Each feed returns the 100 most recent listings in that category.
    """
    print("[We Work Remotely] Fetching …")
    jobs, seen = [], set()
    for category, feed_url in WWR_FEEDS.items():
        try:
            r = requests.get(feed_url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as e:
            print(f"  Error fetching '{category}' feed: {e}"); continue

        for item in root.iter("item"):
            def tag(name: str) -> str:
                el = item.find(name)
                return (el.text or "").strip() if el is not None else ""

            uid     = tag("guid") or tag("link")
            if uid in seen:
                continue
            title_raw = tag("title")
            # WWR title format: "Company: Job Title"
            if ":" in title_raw:
                company, title = [p.strip() for p in title_raw.split(":", 1)]
            else:
                title, company = title_raw, ""
            location = tag("region") or "Remote (US-friendly)"
            desc     = clean_html(tag("description"))
            url      = tag("link")
            row      = build_row("We Work Remotely", uid, title, company,
                                 location, url, desc, skills)
            if row:
                seen.add(uid); jobs.append(row)
        time.sleep(REQUEST_DELAY)
    print(f"  → {len(jobs)} matching jobs")
    return jobs


# 7. Findwork.dev

def fetch_findwork(skills: list[str], max_pages: int) -> list[dict]:
    """
    https://findwork.dev/developers/
    Free API key — register at https://findwork.dev/login/ then copy token from profile.
    Set env var: FINDWORK_API_KEY=your_token
    """
    if not FINDWORK_API_KEY:
        print("[Findwork.dev] Skipped – set FINDWORK_API_KEY env var to enable.")
        return []
    print("[Findwork.dev] Fetching …")
    jobs, seen = [], set()
    auth_headers = {**HEADERS, "Authorization": f"Token {FINDWORK_API_KEY}"}

    for skill in skills:
        url = f"https://findwork.dev/api/jobs/?search={requests.utils.quote(skill)}"
        for _ in range(max_pages):
            if not url:
                break
            try:
                r = requests.get(url, headers=auth_headers, timeout=15)
                r.raise_for_status()
                body = r.json()
            except Exception as e:
                print(f"  Error '{skill}': {e}"); break

            for j in body.get("results", []):
                uid     = str(j.get("id", ""))
                if uid in seen:
                    continue
                title   = j.get("role", "")
                company = j.get("company_name", "")
                location = j.get("location", "Remote")
                keywords = ", ".join(j.get("keywords", []))
                desc    = f"{j.get('text', '')} Skills: {keywords}".strip()
                row     = build_row("Findwork.dev", uid, title, company,
                                    location, j.get("url", ""), desc, skills)
                if row:
                    seen.add(uid); jobs.append(row)

            url = body.get("next", None)   # paginate via 'next' URL
            time.sleep(REQUEST_DELAY)

    print(f"  → {len(jobs)} matching jobs")
    return jobs


# 8. USAJobs

def fetch_usajobs(skills: list[str], max_pages: int) -> list[dict]:
    """
    https://developer.usajobs.gov/
    Free API key — apply at https://developer.usajobs.gov/API-Key/Apply-for-a-Developer-Account
    Set env vars:
      USAJOBS_API_KEY      your API key
      USAJOBS_USER_AGENT   the email address you registered with
    """
    if not USAJOBS_API_KEY or not USAJOBS_USER_AGENT:
        print("[USAJobs] Skipped – set USAJOBS_API_KEY and USAJOBS_USER_AGENT env vars to enable.")
        return []
    print("[USAJobs] Fetching US federal jobs …")
    jobs, seen = [], set()
    usajobs_headers = {
        **HEADERS,
        "Authorization-Key" : USAJOBS_API_KEY,
        "User-Agent"        : USAJOBS_USER_AGENT,
        "Host"              : "data.usajobs.gov",
    }
    for skill in skills:
        for page in range(1, max_pages + 1):
            params = {
                "Keyword"      : skill,
                "ResultsPerPage": 50,
                "Page"         : page,
            }
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
                print(f"  Error page {page} '{skill}': {e}"); break
            if not items:
                break
            for item in items:
                pos  = item.get("MatchedObjectDescriptor", {})
                uid  = str(pos.get("PositionID", ""))
                if uid in seen:
                    continue
                title   = pos.get("PositionTitle", "")
                company = pos.get("OrganizationName", "")
                locs    = pos.get("PositionLocation", [{}])
                location = locs[0].get("LocationName", "USA") if locs else "USA"
                desc    = clean_html(
                    pos.get("QualificationSummary", "")
                    + " "
                    + pos.get("UserArea", {})
                       .get("Details", {})
                       .get("JobSummary", "")
                )
                url     = pos.get("PositionURI", "")
                row     = build_row("USAJobs", uid, title, company,
                                    location, url, desc, skills)
                if row:
                    seen.add(uid); jobs.append(row)
            time.sleep(REQUEST_DELAY)
    print(f"  → {len(jobs)} matching jobs")
    return jobs


# 9. Adzuna
def fetch_adzuna(skills: list[str], max_pages: int) -> list[dict]:
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        print("[Adzuna] Skipped – set ADZUNA_APP_ID + ADZUNA_APP_KEY env vars to enable.")
        return []
    print("[Adzuna] Fetching …")
    jobs, seen = [], set()
    for skill in skills:
        for page in range(1, max_pages + 1):
            url = (
                f"https://api.adzuna.com/v1/api/jobs/us/search/{page}"
                f"?app_id={ADZUNA_APP_ID}&app_key={ADZUNA_APP_KEY}"
                f"&results_per_page=50&what={requests.utils.quote(skill)}"
                f"&content-type=application/json"
            )
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                results = r.json().get("results", [])
            except Exception as e:
                print(f"  Error page {page} '{skill}': {e}"); break
            if not results:
                break
            for j in results:
                uid     = str(j.get("id", ""))
                if uid in seen:
                    continue
                title   = j.get("title", "")
                company = j.get("company", {}).get("display_name", "")
                location = j.get("location", {}).get("display_name", "")
                desc    = clean_html(j.get("description", ""))
                row     = build_row("Adzuna", uid, title, company, location,
                                    j.get("redirect_url", ""), desc, skills)
                if row:
                    seen.add(uid); jobs.append(row)
            time.sleep(REQUEST_DELAY)
    print(f"  → {len(jobs)} matching jobs")
    return jobs


#  MAIN
def main():
    skills, max_pages = parse_args()

    print("\n" + "=" * 60)
    print("  Job Aggregator v4 — 9 Sources")
    print(f"  Skills    : {', '.join(skills)}")
    print(f"  Locations : All (worldwide)")
    print(f"  Started   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60 + "\n")

    all_jobs: list[dict] = []
    all_jobs += fetch_arbeitnow(skills, max_pages)
    all_jobs += fetch_jobicy(skills, max_pages)
    all_jobs += fetch_remoteok(skills, max_pages)
    all_jobs += fetch_themuse(skills, max_pages)
    all_jobs += fetch_himalayas(skills, max_pages)
    all_jobs += fetch_weworkremotely(skills, max_pages)
    all_jobs += fetch_findwork(skills, max_pages)        # optional key
    all_jobs += fetch_usajobs(skills, max_pages)         # optional key (US)
    all_jobs += fetch_adzuna(skills, max_pages)          # optional key (US)

    if not all_jobs:
        print("\nNo matching jobs found. Try different or broader skills.")
        return

    df = pd.DataFrame(all_jobs, columns=[
        "source", "job_id", "title", "company", "location",
        "url", "skills_matched", "matched_skills", "description_file"
    ])

    # Deduplicate by (title + company); keep highest skill score
    df.sort_values("skills_matched", ascending=False, inplace=True)
    df.drop_duplicates(subset=["title", "company"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print("\n" + "=" * 60)
    print(f"  Total unique jobs : {len(df)}")
    print(f"  CSV saved to      : {OUTPUT_CSV}")
    print(f"  Descriptions in   : {DESC_DIR}/")
    print("=" * 60)

    print("\nJobs by source:")
    for src, count in df["source"].value_counts().items():
        bar = "█" * min(count, 40)
        print(f"  {src:<22} {bar} ({count})")

    print("\nSkill match distribution:")
    dist = df["skills_matched"].value_counts().sort_index(ascending=False)
    for score, count in dist.items():
        bar = "█" * min(count, 40)
        print(f"  {int(score):2d} skill(s) : {bar} ({count})")

    print("\nTop 10 jobs by skill match:")
    print(
        df[["title", "company", "location", "skills_matched", "matched_skills"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()