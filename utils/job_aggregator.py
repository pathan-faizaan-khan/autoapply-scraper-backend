"""
Job Aggregator — Production-grade multi-source job scraper.

Strategy (ordered by reliability):
  1. Free public JSON APIs  — Remotive, Jobicy, RemoteOK, Arbeitnow, Jooble
  2. RSS feeds              — We Work Remotely, StackOverflow Jobs
  3. Stealth Playwright     — DuckDuckGo job results (much easier to bypass than Google)

All sources are normalised into the same job dict schema:
  { title, company_name, job_url, description, location, source, domain }

Deduplication is done by job_url before returning.
"""

import re
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import random
import urllib.parse
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── User-Agent Pool ──────────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

def _random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/html, */*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
    }

def _extract_domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else ""

def _normalize(job: dict) -> dict:
    """Ensure all jobs have a consistent schema."""
    return {
        "title":        job.get("title", "").strip(),
        "company_name": job.get("company_name", "").strip(),
        "job_url":      job.get("job_url", "").strip(),
        "description":  job.get("description", "")[:4000].strip(),
        "location":     job.get("location", "Remote").strip(),
        "source":       job.get("source", "unknown"),
        "domain":       _extract_domain(job.get("job_url", "")),
    }

def _keyword_match(text: str, query: str) -> bool:
    """Check if query keywords appear in text (case-insensitive)."""
    words = query.lower().split()
    text_lower = text.lower()
    return any(w in text_lower for w in words if len(w) > 2)


# Maximum number of jobs returned by a single aggregation call
MAX_RESULTS_CAP = 50

# ─── SOURCE 1: Remotive API ───────────────────────────────────────────────────

async def _fetch_remotive(query: str, location: str, session: aiohttp.ClientSession) -> list[dict]:
    """Remotive.com public API — free, no key, remote jobs."""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://remotive.com/api/remote-jobs?search={encoded}&limit=10"
        async with session.get(url, headers=_random_headers(), timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
        
        jobs = []
        for j in data.get("jobs", []):
            if not _keyword_match(j.get("title", "") + " " + j.get("category", ""), query):
                continue
            jobs.append(_normalize({
                "title":        j.get("title", ""),
                "company_name": j.get("company_name", ""),
                "job_url":      j.get("url", ""),
                "description":  re.sub(r"<[^>]+>", " ", j.get("description", "")),
                "location":     j.get("candidate_required_location", "Remote"),
                "source":       "remotive",
            }))
        logger.info(f"[Remotive] Found {len(jobs)} jobs for '{query}'")
        return jobs
    except Exception as e:
        logger.warning(f"[Remotive] Error: {e}")
        return []


# ─── SOURCE 2: Jobicy API ─────────────────────────────────────────────────────

async def _fetch_jobicy(query: str, location: str, session: aiohttp.ClientSession) -> list[dict]:
    """Jobicy.com public API — free, no key, supports geo."""
    try:
        params = {"count": 10, "tag": query}
        if location and location.lower() not in ("remote", ""):
            params["geo"] = location
        url = "https://jobicy.com/api/v2/remote-jobs?" + urllib.parse.urlencode(params)
        async with session.get(url, headers=_random_headers(), timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
        
        jobs = []
        for j in data.get("jobs", []):
            jobs.append(_normalize({
                "title":        j.get("jobTitle", ""),
                "company_name": j.get("companyName", ""),
                "job_url":      j.get("url", ""),
                "description":  re.sub(r"<[^>]+>", " ", j.get("jobDescription", "")),
                "location":     j.get("jobGeo", "Remote"),
                "source":       "jobicy",
            }))
        logger.info(f"[Jobicy] Found {len(jobs)} jobs for '{query}'")
        return jobs
    except Exception as e:
        logger.warning(f"[Jobicy] Error: {e}")
        return []


# ─── SOURCE 3: RemoteOK API ───────────────────────────────────────────────────

async def _fetch_remoteok(query: str, session: aiohttp.ClientSession) -> list[dict]:
    """RemoteOK public JSON API — free, no key."""
    try:
        headers = {**_random_headers(), "Accept": "application/json"}
        async with session.get("https://remoteok.com/api", headers=headers, timeout=aiohttp.ClientTimeout(total=12)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
        
        if not isinstance(data, list):
            return []
        
        jobs = []
        for j in data[1:]:   # first element is meta
            if not isinstance(j, dict):
                continue
            title = j.get("position", "")
            company = j.get("company", "")
            if not _keyword_match(title + " " + " ".join(j.get("tags", [])), query):
                continue
            jobs.append(_normalize({
                "title":        title,
                "company_name": company,
                "job_url":      j.get("url", f"https://remoteok.com/remote-jobs/{j.get('id', '')}"),
                "description":  re.sub(r"<[^>]+>", " ", j.get("description", "")),
                "location":     "Remote",
                "source":       "remoteok",
            }))
        logger.info(f"[RemoteOK] Found {len(jobs)} jobs for '{query}'")
        return jobs[:10]
    except Exception as e:
        logger.warning(f"[RemoteOK] Error: {e}")
        return []


# ─── SOURCE 4: Arbeitnow API ──────────────────────────────────────────────────

async def _fetch_arbeitnow(query: str, location: str, session: aiohttp.ClientSession) -> list[dict]:
    """Arbeitnow free API — supports EU + remote jobs."""
    try:
        params = {"q": query, "location": location or "remote"}
        url = "https://www.arbeitnow.com/api/job-board-api?" + urllib.parse.urlencode(params)
        async with session.get(url, headers=_random_headers(), timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
        
        jobs = []
        for j in data.get("data", []):
            jobs.append(_normalize({
                "title":        j.get("title", ""),
                "company_name": j.get("company_name", ""),
                "job_url":      j.get("url", ""),
                "description":  re.sub(r"<[^>]+>", " ", j.get("description", "")),
                "location":     j.get("location", "Remote"),
                "source":       "arbeitnow",
            }))
        logger.info(f"[Arbeitnow] Found {len(jobs)} jobs for '{query}'")
        return jobs
    except Exception as e:
        logger.warning(f"[Arbeitnow] Error: {e}")
        return []


# ─── SOURCE 5: We Work Remotely RSS ──────────────────────────────────────────

async def _fetch_wwr_rss(query: str, session: aiohttp.ClientSession) -> list[dict]:
    """We Work Remotely RSS feed — zero bot protection, always works."""
    FEEDS = [
        "https://weworkremotely.com/categories/remote-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
        "https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-back-end-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-front-end-programming-jobs.rss",
    ]
    
    jobs = []
    for feed_url in FEEDS:
        try:
            async with session.get(feed_url, headers=_random_headers(), timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    continue
                xml_text = await resp.text()
            
            root = ET.fromstring(xml_text)
            channel = root.find("channel")
            if channel is None:
                continue
            
            for item in channel.findall("item"):
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                desc = item.findtext("description", "")
                company_region = item.findtext("{https://weworkremotely.com}}company", "")
                
                # Parse "Company: Title" format from WWR titles
                company = ""
                job_title = title
                if ": " in title:
                    parts = title.split(": ", 1)
                    company = parts[0].strip()
                    job_title = parts[1].strip()
                
                if not _keyword_match(title + " " + re.sub(r"<[^>]+>", " ", desc), query):
                    continue
                
                jobs.append(_normalize({
                    "title":        job_title,
                    "company_name": company,
                    "job_url":      link,
                    "description":  re.sub(r"<[^>]+>", " ", desc)[:2000],
                    "location":     "Remote",
                    "source":       "weworkremotely",
                }))
        except Exception as e:
            logger.warning(f"[WWR] Feed {feed_url} error: {e}")
    
    logger.info(f"[WeWorkRemotely] Found {len(jobs)} matching jobs for '{query}'")
    return jobs


# ─── SOURCE 6: Stealth DuckDuckGo Job Scraper ────────────────────────────────

async def _fetch_duckduckgo_jobs(query: str, location: str) -> list[dict]:
    """
    Stealth Playwright scraper hitting DuckDuckGo HTML search.
    DuckDuckGo has far less bot protection than Google and serves job results.
    """
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth

    search_terms = f"{query} jobs"
    if location and location.lower() not in ("remote", ""):
        search_terms += f" {location}"
    else:
        search_terms += " remote"
    
    encoded = urllib.parse.quote(search_terms)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    jobs = []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ],
            )
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1366, "height": 768},
                locale="en-US",
                timezone_id="America/New_York",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            page = await context.new_page()
            await Stealth().apply_stealth_async(page)

            # Block heavy resources to speed up
            await page.route("**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,mp4}", lambda r: r.abort())

            await page.goto(url, timeout=20000, wait_until="domcontentloaded")
            await asyncio.sleep(random.uniform(1.0, 2.5))

            # Extract organic search result links
            results = await page.query_selector_all(".result")
            
            JOB_DOMAINS = [
                "linkedin.com", "indeed.com", "greenhouse.io", "lever.co",
                "ashby.com", "workday.com", "jobs.", "careers.", "myworkdayjobs",
                "jobvite.com", "smartrecruiters.com", "breezy.hr", "bamboohr.com",
                "recruiterbox.com", "icims.com", "taleo.net", "remoteok.com",
                "weworkremotely.com", "remotive.com", "wellfound.com", "angel.co",
            ]

            for result in results[:25]:
                try:
                    title_el = await result.query_selector(".result__title a")
                    snippet_el = await result.query_selector(".result__snippet")
                    
                    if not title_el:
                        continue
                    
                    href = await title_el.get_attribute("href") or ""
                    title_text = (await title_el.inner_text()).strip()
                    snippet = (await snippet_el.inner_text()).strip() if snippet_el else ""
                    
                    # Only keep real job-board URLs
                    if not any(d in href for d in JOB_DOMAINS):
                        continue
                    
                    # Skip obvious aggregator homepages
                    if href.endswith(("indeed.com/", "linkedin.com/", "glassdoor.com/")):
                        continue
                    
                    # Extract company from title patterns
                    company = ""
                    cmp = re.search(r"(?:at|@)\s+([^|\-–\n,]+)", title_text, re.I)
                    if cmp:
                        company = cmp.group(1).strip()

                    jobs.append(_normalize({
                        "title":        title_text,
                        "company_name": company,
                        "job_url":      href,
                        "description":  snippet,
                        "location":     location or "Remote",
                        "source":       "duckduckgo",
                    }))
                except Exception:
                    continue
            
            await browser.close()
        logger.info(f"[DuckDuckGo] Found {len(jobs)} job links for '{query}'")
    except Exception as e:
        logger.warning(f"[DuckDuckGo] Error: {e}")
    
    return jobs


# ─── MAIN AGGREGATOR ──────────────────────────────────────────────────────────

async def scrape_google_jobs(
    query: str,
    location: str = "",
    company_type: str = "",
    num_results: int = 20,
) -> list[dict]:
    """
    Primary entry point. Aggregates jobs from multiple reliable sources.
    Hard-capped at MAX_RESULTS_CAP (50) per call to prevent overloading.
    """
    # Enforce global cap — never return more than 50 jobs per call
    num_results = min(num_results, MAX_RESULTS_CAP)
    # Enrich query with company type hint
    enriched_query = query
    if company_type and company_type.lower() == "startup":
        enriched_query = f"{query}"   # APIs already focus on tech/startup roles

    all_jobs: list[dict] = []
    seen_urls: set[str] = set()

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as http:
        # Run all API sources concurrently
        tasks = [
            _fetch_remotive(enriched_query, location, http),
            _fetch_jobicy(enriched_query, location, http),
            _fetch_remoteok(enriched_query, http),
            _fetch_arbeitnow(enriched_query, location, http),
            _fetch_wwr_rss(enriched_query, http),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                for job in result:
                    url = job.get("job_url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_jobs.append(job)

    # If APIs gave enough results, skip the slower Playwright step
    if len(all_jobs) < num_results:
        try:
            ddg_jobs = await _fetch_duckduckgo_jobs(query, location)
            for job in ddg_jobs:
                url = job.get("job_url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_jobs.append(job)
        except Exception as e:
            logger.warning(f"[Aggregator] DuckDuckGo fallback failed: {e}")

    # Filter by location preference
    if location and location.lower() not in ("remote", ""):
        loc_lower = location.lower()
        location_matched = [
            j for j in all_jobs
            if loc_lower in j.get("location", "").lower()
            or "remote" in j.get("location", "").lower()
            or "worldwide" in j.get("location", "").lower()
            or "anywhere" in j.get("location", "").lower()
        ]
        # Fall back to all results if the filter is too aggressive
        all_jobs = location_matched if location_matched else all_jobs

    # Filter by company type keywords in title/description
    if company_type and company_type.lower() in ("startup", "faang", "enterprise"):
        type_keywords = {
            "startup": ["startup", "early-stage", "seed", "yc", "series", "scale-up", "venture"],
            "faang": ["google", "meta", "amazon", "apple", "netflix", "microsoft", "openai", "anthropic"],
            "enterprise": ["enterprise", "fortune", "global", "corporate"],
        }.get(company_type.lower(), [])
        
        if type_keywords:
            typed = [
                j for j in all_jobs
                if any(kw in (j.get("title","") + j.get("company_name","") + j.get("description","")).lower()
                       for kw in type_keywords)
            ]
            all_jobs = typed if typed else all_jobs   # don't wipe everything if no match

    logger.info(f"[Aggregator] Total unique jobs found: {len(all_jobs)} | Returning top {num_results}")
    return all_jobs[:num_results]


# ─── CONTACT FINDER HELPER ────────────────────────────────────────────────────

async def ddg_linkedin_search(company: str, role: str) -> list[dict]:
    """
    Stealth DuckDuckGo HTML search for LinkedIn employee profiles at a company.
    Used by contact_finder.py — much less bot-protected than Google.
    Returns a list of: { name, linkedin_url, title, confidence, source }
    """
    decision_roles = "Recruiter OR \"HR Manager\" OR \"Talent Acquisition\" OR Founder OR CTO OR CEO OR \"Engineering Manager\""
    query = f'site:linkedin.com/in "{company}" ({decision_roles} OR "{role}")'
    encoded = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    results = []
    try:
        from playwright.async_api import async_playwright
        from playwright_stealth import Stealth

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1366, "height": 768},
                locale="en-US",
            )
            page = await context.new_page()
            await Stealth().apply_stealth_async(page)
            await page.route("**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf}", lambda r: r.abort())

            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            await asyncio.sleep(random.uniform(1.0, 2.0))

            result_els = await page.query_selector_all(".result")
            for el in result_els[:10]:
                try:
                    title_el = await el.query_selector(".result__title a")
                    snippet_el = await el.query_selector(".result__snippet")
                    if not title_el:
                        continue
                    href = await title_el.get_attribute("href") or ""
                    title_text = (await title_el.inner_text()).strip()
                    snippet = (await snippet_el.inner_text()).strip() if snippet_el else ""

                    if "linkedin.com/in/" not in href:
                        continue

                    # Parse name from "Name - Title at Company | LinkedIn"
                    name_match = re.match(r"^([^|\-–·]+)", title_text)
                    name = name_match.group(1).strip() if name_match else title_text

                    results.append({
                        "name":         name,
                        "linkedin_url": href,
                        "title":        title_text,
                        "snippet":      snippet,
                        "confidence":   "medium",
                        "source":       "duckduckgo_linkedin",
                    })
                except Exception:
                    continue

            await browser.close()
        logger.info(f"[DDGLinkedIn] Found {len(results)} profiles for '{company}'")
    except Exception as e:
        logger.warning(f"[DDGLinkedIn] Error: {e}")

    return results

