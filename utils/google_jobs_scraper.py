"""
Google Jobs Scraper — uses Playwright to scrape Google's Jobs search panel.
No API key required. Works by triggering the &ibp=htl;jobs tab in Google Search.
"""
import re
import asyncio
import random
from typing import Optional
from playwright.async_api import async_playwright, Page

# Rotate user agents to avoid blocks
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

COMPANY_TYPE_HINTS = {
    "startup": ["startup", "seed", "series a", "early stage", "Y Combinator", "YC"],
    "midsize":  ["mid-size", "growing company", "scale-up"],
    "enterprise": ["enterprise", "fortune 500", "global", "multinational"],
    "faang": ["Google", "Meta", "Amazon", "Apple", "Netflix", "Microsoft"],
}


def _build_search_query(query: str, location: str, company_type: str) -> str:
    """Build an optimized Google Jobs search query."""
    parts = [query, "jobs"]
    if location and location.lower() not in ("remote", ""):
        parts.append(location)
    if company_type and company_type in COMPANY_TYPE_HINTS:
        hints = COMPANY_TYPE_HINTS[company_type][:2]
        parts.append(f"({' OR '.join(hints)})")
    return " ".join(parts)


def _extract_domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else ""


async def scrape_google_jobs(
    query: str,
    location: str = "",
    company_type: str = "",
    num_results: int = 10,
) -> list[dict]:
    """
    Scrape Google Jobs search results using Playwright.
    Returns a list of job dicts.
    """
    search_query = _build_search_query(query, location, company_type)
    # The &ibp=htl;jobs param triggers Google's dedicated Jobs tab
    encoded = search_query.replace(" ", "+")
    url = f"https://www.google.com/search?q={encoded}&ibp=htl;jobs&hl=en&gl=us"

    jobs = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            page = await context.new_page()

            # Block images/fonts to speed up scraping
            await page.route("**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf}", lambda r: r.abort())

            await page.goto(url, timeout=20000, wait_until="domcontentloaded")
            await asyncio.sleep(2)  # Allow JS to settle

            # ── Try parsing the Google Jobs structured panel ──────────────────
            jobs = await _parse_google_jobs_panel(page, query, location, num_results)

            # ── Fallback: parse organic results if jobs panel didn't render ──
            if not jobs:
                jobs = await _parse_organic_job_results(page, query, location)

            await browser.close()
    except Exception as e:
        print(f"[GoogleJobsScraper] Error: {e}")

    return jobs[:num_results]


async def _parse_google_jobs_panel(page: Page, query: str, location: str, limit: int) -> list[dict]:
    """Parse job cards from Google's structured Jobs panel (ibp=htl;jobs)."""
    jobs = []
    try:
        # Wait for job listing container
        await page.wait_for_selector('[data-hveid]', timeout=6000)

        # Job cards in the Google Jobs tab use specific selectors
        # Try multiple known selectors (Google changes these occasionally)
        card_selectors = [
            ".iFjolb",        # Job card wrapper (common)
            "[data-id]",      # Generic data-id cards
            ".gws-plugins-horizon-jobs__li-ed",
            ".VkpGBb",
        ]

        cards = []
        for sel in card_selectors:
            cards = await page.query_selector_all(sel)
            if cards:
                break

        for card in cards[:limit]:
            try:
                # Click card to load full description in side panel
                await card.click()
                await asyncio.sleep(0.8)

                title = await _safe_text(page, [
                    ".KLsYvd", ".sH3zDb", "h2.KLsYvd",
                    ".iFjolb .tJ9zfc", "[data-ved] h2"
                ])
                company = await _safe_text(page, [
                    ".nJlQNd", ".vNEEBe", ".sMzDkb", ".hiS8de"
                ])
                location_text = await _safe_text(page, [
                    ".Qk80Jf", ".location", ".vjbI0b"
                ])
                description = await _safe_text(page, [
                    ".HBvzbc", ".NGTcnd", ".job-description",
                    "[data-md]"
                ], max_len=600)
                apply_url = await _safe_href(page, [
                    "a.WpHeLc", "a[data-id]", ".pMhGee a", "a[href*='apply']"
                ]) or page.url

                if title:
                    jobs.append({
                        "title": title.strip(),
                        "company_name": (company or "Unknown").strip(),
                        "job_url": apply_url,
                        "description": (description or "").strip(),
                        "domain": _extract_domain(apply_url),
                        "location": (location_text or location or "Remote").strip(),
                        "source": "google_jobs_scrape",
                    })
            except Exception:
                continue

    except Exception as e:
        print(f"[GoogleJobsScraper] Panel parse error: {e}")

    return jobs


async def _parse_organic_job_results(page: Page, query: str, location: str) -> list[dict]:
    """
    Fallback: parse regular Google organic results that are job postings
    (from LinkedIn, Greenhouse, Lever, Ashby, etc.)
    """
    jobs = []
    try:
        results = await page.query_selector_all("div.g, div[data-sokoban-container]")
        for result in results[:15]:
            try:
                link_el = await result.query_selector("a")
                title_el = await result.query_selector("h3")
                snippet_el = await result.query_selector(".VwiC3b, .lyLwlc, span.st")

                if not link_el or not title_el:
                    continue

                href = await link_el.get_attribute("href") or ""
                title_text = await title_el.inner_text()
                snippet_text = await snippet_el.inner_text() if snippet_el else ""

                # Only keep job posting URLs
                job_domains = ["linkedin.com", "greenhouse.io", "lever.co", "ashby",
                               "workday.com", "indeed.com", "jobs."]
                if not any(d in href for d in job_domains):
                    continue

                # Extract company from title: "Senior Eng at Stripe | LinkedIn" → "Stripe"
                company_match = re.search(r"(?:at|@)\s+([^|–\-\n]+)", title_text, re.I)
                company = company_match.group(1).strip() if company_match else _extract_domain(href).split(".")[0].title()

                clean_title = re.sub(r"\s*[\|–\-].*$", "", title_text).strip()

                jobs.append({
                    "title": clean_title,
                    "company_name": company,
                    "job_url": href,
                    "description": snippet_text.strip(),
                    "domain": _extract_domain(href),
                    "location": location or "Remote",
                    "source": "google_organic_scrape",
                })
            except Exception:
                continue
    except Exception as e:
        print(f"[GoogleJobsScraper] Organic parse error: {e}")

    return jobs


async def _safe_text(page: Page, selectors: list[str], max_len: int = 200) -> str:
    """Try multiple CSS selectors and return first found text."""
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                text = await el.inner_text()
                return text[:max_len] if text else ""
        except Exception:
            continue
    return ""


async def _safe_href(page: Page, selectors: list[str]) -> str:
    """Try multiple CSS selectors and return first found href."""
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                href = await el.get_attribute("href")
                if href:
                    return href
        except Exception:
            continue
    return ""


# ── DuckDuckGo HTML scraper (for LinkedIn employee search — no JS/API needed) ─

async def ddg_linkedin_search(company: str, role: str) -> list[dict]:
    """
    Scrape DuckDuckGo HTML results (no JS) to find LinkedIn profiles.
    Query: site:linkedin.com/in "Company" "Role"
    """
    query = f'site:linkedin.com/in "{company}" "{role}"'
    encoded = query.replace(" ", "+").replace('"', "%22")
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    results = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-US",
            )
            page = await context.new_page()
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")

            # DuckDuckGo HTML results are in .result__body divs
            items = await page.query_selector_all(".result__body, .result")
            for item in items[:8]:
                try:
                    link_el = await item.query_selector("a.result__a, a.result__url")
                    title_el = await item.query_selector("a.result__a")
                    snippet_el = await item.query_selector(".result__snippet")

                    if not link_el:
                        continue

                    href = await link_el.get_attribute("href") or ""
                    title_text = await title_el.inner_text() if title_el else ""
                    snippet = await snippet_el.inner_text() if snippet_el else ""

                    # Only LinkedIn profile URLs
                    if "linkedin.com/in/" not in href:
                        continue

                    # Parse name from "Name - Title at Company | LinkedIn"
                    name_match = re.match(r"^([^-|·]+)", title_text)
                    name = name_match.group(1).strip() if name_match else ""

                    # Parse title from snippet or title string
                    title_match = re.search(r"[-–·]\s*([^|·\n]+)", title_text)
                    job_title = title_match.group(1).strip() if title_match else ""

                    results.append({
                        "name": name,
                        "linkedin_url": href,
                        "title": job_title,
                        "snippet": snippet,
                        "source": "ddg_linkedin_scrape",
                    })
                except Exception:
                    continue

            await browser.close()
    except Exception as e:
        print(f"[DDGLinkedIn] Error: {e}")

    return results
