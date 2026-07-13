"""
Contact Discovery Utility — No Apollo, No Google CSE required.
5-layer pipeline:
  1. DuckDuckGo HTML scrape → LinkedIn employee profiles
  2. Hunter.io domain search (free tier, optional)
  3. GitHub API org members (free, no auth)
  4. Playwright website scrape (/about /team /contact)
  5. Email pattern guesser + DNS MX validation
"""
import os
import re
import json
import asyncio
import httpx
import smtplib
import dns.resolver
from playwright.async_api import async_playwright
from typing import Optional
from utils.job_aggregator import ddg_linkedin_search
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

try:
    groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    groq_client = None

HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "")

EMAIL_PATTERNS = [
    "{first}@{domain}",
    "{first}.{last}@{domain}",
    "{f}{last}@{domain}",
    "{first}{last}@{domain}",
    "{first}_{last}@{domain}",
]


def _build_email_guesses(first: str, last: str, domain: str) -> list[str]:
    f = first[0].lower() if first else ""
    return [
        p.format(first=first.lower(), last=last.lower(), f=f, domain=domain)
        for p in EMAIL_PATTERNS
    ]


def _verify_email_smtp(email: str) -> str:
    """Live SMTP check. Returns 'valid', 'invalid', or 'unknown'"""
    try:
        domain = email.split('@')[1]
        records = dns.resolver.resolve(domain, 'MX')
        mxRecord = str(records[0].exchange)
        
        server = smtplib.SMTP(timeout=3)
        server.connect(mxRecord)
        server.helo(server.local_hostname)
        server.mail('hello@example.com')
        code, message = server.rcpt(email)
        server.quit()
        
        if code == 250:
            return "valid"
        elif code >= 500:
            return "invalid"
        return "unknown"
    except Exception:
        return "unknown"

async def _verify_email_async(email: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _verify_email_smtp, email)


async def _google_organic_linkedin_search(company: str, role: str) -> list[dict]:
    """Fallback Google organic search for LinkedIn profiles if DDG fails."""
    # Modified to look for decision makers
    search_terms = f'("{role}" OR "HR" OR "Recruiter" OR "Talent Acquisition" OR "Founder")'
    query = f'site:linkedin.com/in "{company}" {search_terms}'
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    results = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=10000)
            
            links = await page.query_selector_all("div.g")
            for link in links[:5]:
                try:
                    a = await link.query_selector("a")
                    h3 = await link.query_selector("h3")
                    if a and h3:
                        href = await a.get_attribute("href")
                        title = await h3.inner_text()
                        if href and "linkedin.com/in/" in href:
                            name_match = re.match(r"^([^-|·]+)", title)
                            name = name_match.group(1).strip() if name_match else ""
                            results.append({
                                "name": name,
                                "linkedin_url": href,
                                "title": title,
                                "confidence": "medium",
                                "source": "google_organic"
                            })
                except Exception:
                    continue
            await browser.close()
    except Exception as e:
        print(f"[ContactFinder] Google fallback error: {e}")
    return results

async def _ddg_linkedin_search_contacts(company: str, role: str) -> list[dict]:
    """Layer 1: DuckDuckGo HTML → LinkedIn profile URLs of employees."""
    try:
        raw = await ddg_linkedin_search(company, role)
        if not raw:
            print("[ContactFinder] DDG returned 0 results. Using Google Fallback...")
            raw = await _google_organic_linkedin_search(company, role)
        return raw
    except Exception as e:
        print(f"[ContactFinder] DDG/Google search error: {e}")
        return []


async def _hunter_domain_search(domain: str) -> dict:
    """Layer 2: Hunter.io domain search for email patterns + known emails."""
    if not HUNTER_API_KEY:
        return {}
    url = f"https://api.hunter.io/v2/domain-search"
    params = {"domain": domain, "api_key": HUNTER_API_KEY, "limit": 5}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            data = resp.json().get("data", {})
            return {
                "pattern": data.get("pattern"),
                "emails": [
                    {
                        "email": e.get("value"),
                        "name": f"{e.get('first_name', '')} {e.get('last_name', '')}".strip(),
                        "title": e.get("position", ""),
                        "confidence": "high" if e.get("confidence", 0) > 70 else "medium",
                    }
                    for e in data.get("emails", [])
                ],
            }
    except Exception as e:
        print(f"[ContactFinder] Hunter.io error: {e}")
        return {}


async def _github_org_search(company: str) -> list[dict]:
    """Layer 3: GitHub API — find org members with public emails (no auth required)."""
    try:
        # Step 1: find org handle by company name
        search_url = "https://api.github.com/search/users"
        params = {"q": f"{company} type:org", "per_page": 3}
        async with httpx.AsyncClient(timeout=10, headers={"Accept": "application/vnd.github+json"}) as client:
            resp = await client.get(search_url, params=params)
            orgs = resp.json().get("items", [])
            if not orgs:
                return []

            org_login = orgs[0].get("login", "")
            members_url = f"https://api.github.com/orgs/{org_login}/members?per_page=10"
            members_resp = await client.get(members_url)
            members = members_resp.json()

            contacts = []
            for member in members[:5]:
                user_resp = await client.get(f"https://api.github.com/users/{member['login']}")
                user = user_resp.json()
                if user.get("email"):
                    contacts.append({
                        "name": user.get("name") or user.get("login"),
                        "email": user.get("email"),
                        "github_url": user.get("html_url"),
                        "title": user.get("bio", ""),
                        "confidence": "medium",
                        "source": "github",
                    })
            return contacts
    except Exception as e:
        print(f"[ContactFinder] GitHub API error: {e}")
        return []


async def _scrape_company_website(domain: str) -> list[dict]:
    """Layer 4: Playwright scrape + AI Parsing for exact names and emails."""
    contacts = []
    paths = ["/about", "/team", "/about-us", "/people", "/contact"]
    
    combined_text = ""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            for path in paths[:3]:
                url = f"https://{domain}{path}"
                try:
                    await page.goto(url, timeout=10000, wait_until="domcontentloaded")
                    text = await page.inner_text("body")
                    combined_text += text + "\n\n"
                except Exception:
                    continue
            await browser.close()
            
        if groq_client and len(combined_text) > 100:
            prompt = f"""
            You are a data extraction AI. Read the text from {domain} below.
            Extract names, job titles, and emails of key personnel (Founders, Engineers, HR, Recruiters).
            Return ONLY a valid JSON object containing a "contacts" array. Each object in the array must have "name", "title", and "email". Leave email empty if not found.
            
            Text:
            {combined_text[:12000]}
            """
            
            try:
                res = await groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                data = json.loads(res.choices[0].message.content)
                for c in data.get("contacts", []):
                    if c.get("name") and len(c.get("name")) > 2:
                        contacts.append({
                            "name": c.get("name"),
                            "email": c.get("email", ""),
                            "title": c.get("title", ""),
                            "confidence": "high" if c.get("email") else "low",
                            "source": "ai_website_scrape"
                        })
            except Exception as e:
                print(f"[ContactFinder] Groq parsing error: {e}")
                
    except Exception as e:
        print(f"[ContactFinder] Website scrape error: {e}")

    return contacts


def _guess_emails(name: str, domain: str) -> list[dict]:
    """Layer 5: Generate email guesses from name + domain pattern."""
    parts = name.strip().split()
    if len(parts) < 2:
        return []
    first, last = parts[0], parts[-1]
    guesses = _build_email_guesses(first, last, domain)
    return [{"email": g, "confidence": "low", "source": "pattern_guess"} for g in guesses]


async def discover_contacts(
    company_name: str,
    domain: str,
    target_role: str,
) -> list[dict]:
    """
    Main entry point — runs all 5 layers and merges results.
    Returns a ranked list of contacts with confidence scores.
    """
    all_contacts: list[dict] = []
    seen_emails: set[str] = set()

    # Layer 1: DuckDuckGo → LinkedIn profiles → email pattern guessing
    # Expand target role to find decision makers
    expanded_role = f"{target_role} OR HR OR Recruiter OR Founder"
    ddg_results = await _ddg_linkedin_search_contacts(company_name, expanded_role)
    for r in ddg_results:
        if r.get("name") and domain:
            guesses = _guess_emails(r["name"], domain)
            for g in guesses:
                if g["email"] not in seen_emails:
                    seen_emails.add(g["email"])
                    all_contacts.append({
                        "name": r["name"],
                        "email": g["email"],
                        "title": r.get("title", ""),
                        "linkedin_url": r.get("linkedin_url"),
                        "confidence": "low",
                        "source": "ddg+pattern",
                    })

    # Layer 2: Hunter.io
    hunter_data = await _hunter_domain_search(domain)
    for email_obj in hunter_data.get("emails", []):
        if email_obj["email"] not in seen_emails:
            seen_emails.add(email_obj["email"])
            all_contacts.append({
                "name": email_obj.get("name", ""),
                "email": email_obj["email"],
                "title": email_obj.get("title", ""),
                "confidence": email_obj.get("confidence", "medium"),
                "source": "hunter_io",
            })

    # Layer 3: GitHub
    github_contacts = await _github_org_search(company_name)
    for c in github_contacts:
        if c.get("email") and c["email"] not in seen_emails:
            seen_emails.add(c["email"])
            all_contacts.append(c)

    # Layer 4: Website scrape (only if no high-confidence contacts yet)
    high_conf = [c for c in all_contacts if c.get("confidence") == "high"]
    if not high_conf and domain:
        scraped = await _scrape_company_website(domain)
        for c in scraped:
            if c["email"] not in seen_emails:
                seen_emails.add(c["email"])
                all_contacts.append(c)

    # Smart Filter: Re-order to prioritize AI-found and Live SMTP-verified emails
    verified_contacts = []
    for c in all_contacts:
        if c.get("email") and "@" in c["email"]:
            if c.get("confidence") == "low" or c.get("source") == "pattern_guess":
                # Verify guessed emails live via SMTP!
                status = await _verify_email_async(c["email"])
                if status == "valid":
                    c["confidence"] = "high"
                    c["source"] += "+smtp_verified"
                    verified_contacts.append(c)
                elif status == "unknown":
                    verified_contacts.append(c) # Keep it, but low confidence
            else:
                verified_contacts.append(c) # Already found via AI or API
        elif c.get("name") and not c.get("email"):
            verified_contacts.append(c) # Name only, LLM will write email on LinkedIn
            
    # Sort: high > medium > low
    order = {"high": 0, "medium": 1, "low": 2}
    verified_contacts.sort(key=lambda c: order.get(c.get("confidence", "low"), 3))

    return verified_contacts[:10]  # Return top 10
