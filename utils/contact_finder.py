"""
Modern Contact Discovery Utility
Concurrent 5-layer pipeline with AI ranking and Async SMTP validation.
"""
import os
import re
import json
import asyncio
import httpx
import aiosmtplib
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

# ─── ASYNC SMTP VERIFICATION WITH CATCH-ALL DETECTION ────────────────────────

async def _get_mx_record(domain: str) -> str:
    try:
        loop = asyncio.get_event_loop()
        records = await loop.run_in_executor(None, dns.resolver.resolve, domain, 'MX')
        return str(records[0].exchange)
    except Exception:
        return ""

async def _check_catch_all(domain: str, mx_record: str) -> bool:
    """Tests if a domain accepts random invalid emails (catch-all)."""
    fake_email = f"invalid-random-{os.urandom(4).hex()}@{domain}"
    try:
        smtp = aiosmtplib.SMTP(hostname=mx_record, port=25, timeout=5)
        await smtp.connect()
        await smtp.ehlo()
        await smtp.mail("hello@example.com")
        code, message = await smtp.rcpt(fake_email)
        await smtp.quit()
        return code == 250
    except Exception:
        return False

async def _verify_email_smtp_async(email: str, mx_record: str = "", is_catch_all: bool = False) -> str:
    if is_catch_all:
        return "unknown"  # Cannot reliably verify on catch-all domains

    domain = email.split('@')[1]
    if not mx_record:
        mx_record = await _get_mx_record(domain)
    if not mx_record:
        return "invalid"

    try:
        smtp = aiosmtplib.SMTP(hostname=mx_record, port=25, timeout=5)
        await smtp.connect()
        await smtp.ehlo()
        await smtp.mail("hello@example.com")
        code, message = await smtp.rcpt(email)
        await smtp.quit()
        if code == 250:
            return "valid"
        elif code >= 500:
            return "invalid"
        return "unknown"
    except Exception:
        return "unknown"

# ─── LAYER 1: LINKEDIN SEARCH ───────────────────────────────────────────────

async def _google_organic_linkedin_search(company: str, role: str) -> list[dict]:
    search_terms = f'("{role}" OR "HR" OR "Recruiter" OR "Talent Acquisition" OR "Founder" OR "CTO")'
    query = f'site:linkedin.com/in "{company}" {search_terms}'
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    results = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
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
                            results.append({"name": name, "linkedin_url": href, "title": title, "confidence": "medium", "source": "linkedin_organic"})
                except Exception:
                    continue
            await browser.close()
    except Exception:
        pass
    return results

async def _ddg_linkedin_search_contacts(company: str, role: str) -> list[dict]:
    try:
        raw = await ddg_linkedin_search(company, role)
        if not raw:
            raw = await _google_organic_linkedin_search(company, role)
        return raw
    except Exception:
        return []

# ─── LAYER 2: HUNTER.IO ─────────────────────────────────────────────────────

async def _hunter_domain_search(domain: str) -> dict:
    if not HUNTER_API_KEY or not domain:
        return {}
    url = f"https://api.hunter.io/v2/domain-search"
    params = {"domain": domain, "api_key": HUNTER_API_KEY, "limit": 10}
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
                        "source": "hunter"
                    }
                    for e in data.get("emails", [])
                ],
            }
    except Exception as e:
        print(f"[ContactFinder] Hunter error: {e}")
        return {}

# ─── LAYER 3: GITHUB ────────────────────────────────────────────────────────

async def _github_org_search(company: str) -> list[dict]:
    try:
        search_url = "https://api.github.com/search/users"
        params = {"q": f"{company} type:org", "per_page": 3}
        async with httpx.AsyncClient(timeout=10, headers={"Accept": "application/vnd.github+json"}) as client:
            resp = await client.get(search_url, params=params)
            orgs = resp.json().get("items", [])
            if not orgs: return []
            org_login = orgs[0].get("login", "")
            members_resp = await client.get(f"https://api.github.com/orgs/{org_login}/members?per_page=10")
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
    except Exception:
        return []

# ─── LAYER 4: WEBSITE SCRAPE ───────────────────────────────────────────────

async def _scrape_company_website(domain: str) -> list[dict]:
    contacts = []
    paths = ["/about", "/team", "/about-us", "/people", "/contact"]
    combined_text = ""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            for path in paths[:3]:
                try:
                    await page.goto(f"https://{domain}{path}", timeout=10000, wait_until="domcontentloaded")
                    combined_text += (await page.inner_text("body")) + "\n\n"
                except Exception:
                    continue
            await browser.close()
            
        if groq_client and len(combined_text) > 100:
            prompt = f"Extract names, job titles, and emails of key personnel from {domain}. Return JSON: {{'contacts': [{{'name', 'title', 'email'}}]}}.\n{combined_text[:12000]}"
            res = await groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="qwen/qwen3.8-27b",
                response_format={"type": "json_object"},
                temperature=0.1
            )
            data = json.loads(res.choices[0].message.content)
            for c in data.get("contacts", []):
                if c.get("name") and len(c.get("name")) > 2:
                    contacts.append({
                        "name": c.get("name"), "email": c.get("email", ""), "title": c.get("title", ""),
                        "confidence": "high" if c.get("email") else "low", "source": "ai_scrape"
                    })
    except Exception as e:
        print(f"[ContactFinder] Scrape Error: {e}")
    return contacts

# ─── DOMAIN FALLBACK ────────────────────────────────────────────────────────

async def _find_company_domain(company_name: str) -> str:
    """Fallback to find the company's official domain via Clearbit Autocomplete API."""
    import urllib.parse
    encoded_name = urllib.parse.quote(company_name)
    url = f"https://autocomplete.clearbit.com/v1/companies/suggest?query={encoded_name}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            data = resp.json()
            if data and isinstance(data, list):
                # Return the first matching domain
                return data[0].get("domain", "")
    except Exception as e:
        print(f"[ContactFinder] Clearbit domain fallback failed: {e}")
    return ""

# ─── GROQ AI RANKING ────────────────────────────────────────────────────────

async def _groq_rank_contacts(contacts: list[dict], target_role: str, company_name: str) -> list[dict]:
    if not groq_client or not contacts:
        return contacts
    
    contacts_str = json.dumps(contacts, indent=2)
    prompt = f"""
    You are an expert executive recruiter. We are looking for the best decision-maker at "{company_name}" matching the role: "{target_role}".
    Review the following discovered contacts. Rank them by relevance.
    Give higher preference to decision makers (Managers, Directors, VP, CTO, CEO, Founders, HR Heads, Lead Recruiters).
    If a contact has an email, they are more valuable.
    
    Return a JSON array "ranked_contacts" containing the contact objects in order of best match first. Only include people who could reasonably be relevant.
    
    Contacts:
    {contacts_str}
    """
    try:
        res = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen/qwen3.8-27b",
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(res.choices[0].message.content)
        return data.get("ranked_contacts", contacts)
    except Exception as e:
        print(f"[ContactFinder] Groq ranking failed: {e}")
        return contacts

# ─── MAIN ORCHESTRATOR ──────────────────────────────────────────────────────

async def discover_contacts(company_name: str, domain: str, target_role: str) -> list[dict]:
    # Ensure domain is perfectly clean (remove http://, https://, www., and any paths)
    if domain:
        import re
        d = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", domain)
        if d:
            domain = d.group(1).lower()

    if not domain:
        print(f"[ContactFinder] No domain provided for {company_name}, attempting fallback...")
        domain = await _find_company_domain(company_name)
    
    all_contacts = []
    seen_emails = set()

    # CONCURRENT DISCOVERY
    print(f"[ContactFinder] Running concurrent discovery for {company_name} ({domain})")
    ddg_task = asyncio.create_task(_ddg_linkedin_search_contacts(company_name, target_role))
    hunter_task = asyncio.create_task(_hunter_domain_search(domain))
    github_task = asyncio.create_task(_github_org_search(company_name))
    
    results = await asyncio.gather(ddg_task, hunter_task, github_task)
    ddg_results, hunter_data, github_results = results
    
    # Process Hunter
    for email_obj in hunter_data.get("emails", []):
        if email_obj["email"] not in seen_emails:
            seen_emails.add(email_obj["email"])
            all_contacts.append(email_obj)
            
    # Process GitHub
    for c in github_results:
        if c.get("email") and c["email"] not in seen_emails:
            seen_emails.add(c["email"])
            all_contacts.append(c)

    # Process DDG LinkedIn & Intelligent Pattern Matching
    hunter_pattern = hunter_data.get("pattern")
    for r in ddg_results:
        if r.get("name") and domain:
            parts = r["name"].strip().split()
            if len(parts) >= 2 and hunter_pattern:
                # Intelligent construction using Hunter pattern
                first, last = parts[0].lower(), parts[-1].lower()
                pattern_map = {"{first}": first, "{last}": last, "{f}": first[0], "{l}": last[0]}
                email_guess = hunter_pattern
                for k, v in pattern_map.items():
                    email_guess = email_guess.replace(k, v)
                
                if email_guess not in seen_emails:
                    seen_emails.add(email_guess)
                    r["email"] = email_guess
                    r["confidence"] = "high" # Verified pattern
                    r["source"] = "linkedin+hunter_pattern"
                    all_contacts.append(r)
            else:
                # Fallback to guessing if no pattern
                guesses = _build_email_guesses(parts[0], parts[-1] if len(parts)>1 else "", domain)
                for g in guesses:
                    if g not in seen_emails:
                        seen_emails.add(g)
                        new_c = dict(r)
                        new_c["email"] = g
                        new_c["confidence"] = "low"
                        new_c["source"] = "linkedin+guess"
                        all_contacts.append(new_c)

    # FALLBACK SCRAPE IF POOR RESULTS
    high_conf = [c for c in all_contacts if c.get("confidence") == "high"]
    if not high_conf and domain:
        print("[ContactFinder] Concurrent layers yielded 0 high-confidence emails. Initiating website scrape...")
        scraped = await _scrape_company_website(domain)
        for c in scraped:
            if c.get("email") and c["email"] not in seen_emails:
                seen_emails.add(c["email"])
                all_contacts.append(c)

    # TRUE ASYNC SMTP VERIFICATION & CATCH-ALL DETECTION
    if all_contacts and domain:
        print(f"[ContactFinder] Verifying emails for {domain}...")
        mx_record = await _get_mx_record(domain)
        is_catch_all = await _check_catch_all(domain, mx_record) if mx_record else False
        if is_catch_all:
            print(f"[ContactFinder] Domain {domain} is catch-all, skipping SMTP validation.")
            
        async def verify_contact(c):
            if c.get("email") and "@" in c["email"] and c.get("confidence") != "high":
                status = await _verify_email_smtp_async(c["email"], mx_record, is_catch_all)
                if status == "valid":
                    c["confidence"] = "high"
                elif status == "invalid":
                    c["email"] = "" # Drop invalid emails
            return c
            
        # Verify in parallel
        all_contacts = await asyncio.gather(*(verify_contact(c) for c in all_contacts))

    # Clean up empty emails
    all_contacts = [c for c in all_contacts if c.get("email")]

    # GROQ AI DECISION MAKER RANKING
    if all_contacts:
        print(f"[ContactFinder] Ranking {len(all_contacts)} contacts with Groq AI...")
        ranked = await _groq_rank_contacts(all_contacts, target_role, company_name)
        return ranked
    else:
        return []
