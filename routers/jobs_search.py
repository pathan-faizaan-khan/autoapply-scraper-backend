"""
Jobs Search + Contact Discovery + AI Email & Resume Generation Router
FastAPI endpoints called by the Next.js BFF layer.
All job discovery uses Playwright scraping — zero API keys required for search.
"""
import os
import re
import json
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from playwright.async_api import async_playwright

from utils.google_jobs_scraper import scrape_google_jobs
from utils.contact_finder import discover_contacts
from utils.matching import compute_match_score

router = APIRouter(prefix="/api", tags=["jobs-search"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ─── MODELS ──────────────────────────────────────────────────────────────────

class JobSearchRequest(BaseModel):
    query: str                        # e.g. "Senior Frontend Engineer"
    location: Optional[str] = ""
    company_type: Optional[str] = ""  # startup|midsize|enterprise|faang
    num_results: Optional[int] = 10

class FindContactsRequest(BaseModel):
    company_name: str
    domain: str
    target_role: str

class GenerateEmailRequest(BaseModel):
    contact_name: str
    contact_title: Optional[str] = ""
    company_name: str
    job_title: str
    job_description: str
    candidate_name: str
    candidate_skills: list[str]
    candidate_summary: Optional[str] = ""
    candidate_experience_summary: Optional[str] = ""

class TailorResumeRequest(BaseModel):
    job_title: str
    job_description: str
    resume_data: dict
    candidate_name: str

class MatchScoreRequest(BaseModel):
    resume_data: dict
    job_description: str
    job_title: Optional[str] = ""

class ScrapeDescriptionRequest(BaseModel):
    url: str


# ─── GOOGLE JOBS SCRAPE ──────────────────────────────────────────────────────

@router.post("/jobs/google-search")
def google_jobs_search(req: JobSearchRequest):
    """
    Scrape Google Jobs search using Playwright — no API key required.
    Tries the dedicated Google Jobs panel first, falls back to organic results.
    """
    try:
        jobs = asyncio.run(scrape_google_jobs(
            query=req.query,
            location=req.location or "",
            company_type=req.company_type or "",
            num_results=req.num_results or 10,
        ))

        # If scraping returns nothing (e.g., bot detection), fall back to mock
        if not jobs:
            print(f"[JobSearch] Scraper returned 0 results for '{req.query}', using mock fallback")
            return {"jobs": _mock_jobs(req.query, req.location or ""), "source": "mock"}

        return {"jobs": jobs, "source": jobs[0].get("source", "google_scrape")}

    except Exception as e:
        print(f"[JobSearch] Error: {e} — falling back to mock")
        return {"jobs": _mock_jobs(req.query, req.location or ""), "source": "mock"}


@router.post("/jobs/scrape-description")
async def scrape_job_description(req: ScrapeDescriptionRequest):
    """Scrape the text of a given URL (job description) using Playwright."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(req.url, timeout=15000, wait_until="domcontentloaded")
            # Get all text from body
            text = await page.inner_text("body")
            await browser.close()
            # Clean up the text: remove excessive newlines
            clean_text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
            return {"description": clean_text[:20000]} # Limit size to avoid too massive context
    except Exception as e:
        print(f"[Scraper] Error scraping description from {req.url}: {e}")
        raise HTTPException(status_code=500, detail="Failed to scrape job description")


# ─── CONTACT DISCOVERY ───────────────────────────────────────────────────────

@router.post("/jobs/find-contacts")
def find_contacts(req: FindContactsRequest):
    """
    Multi-layer contact discovery:
    1. DuckDuckGo HTML scrape → LinkedIn profiles
    2. Hunter.io domain search (if key set)
    3. GitHub API org members (free, no key)
    4. Playwright website scrape (/about /team /contact pages)
    5. Email pattern guessing + MX validation
    """
    try:
        contacts = asyncio.run(discover_contacts(
            company_name=req.company_name,
            domain=req.domain,
            target_role=req.target_role,
        ))
        return {"contacts": contacts, "total": len(contacts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contact discovery error: {str(e)}")


# ─── MATCH SCORE ─────────────────────────────────────────────────────────────

@router.post("/jobs/match-score")
async def get_match_score(req: MatchScoreRequest):
    """Compute TF-IDF match score (0-100) between a resume and job description."""
    score = compute_match_score(req.resume_data, req.job_description, req.job_title)
    return {"score": score}


# ─── AI EMAIL GENERATION ─────────────────────────────────────────────────────

@router.post("/email/generate")
async def generate_cold_email(req: GenerateEmailRequest):
    """Use Groq LLM to generate a personalized cold email."""
    if not groq_client:
        return _mock_email(req)

    skills_str = ", ".join(req.candidate_skills[:8])
    prompt = f"""Write a short, compelling cold email from a job seeker to a hiring contact at a company.

Candidate: {req.candidate_name}
Key Skills: {skills_str}
Summary: {req.candidate_summary or "Experienced software professional"}
Recent Experience: {req.candidate_experience_summary or ""}

Target Contact: {req.contact_name} ({req.contact_title or "Hiring Manager"})
Company: {req.company_name}
Role Applying For: {req.job_title}
Job Description Snippet: {req.job_description[:600]}

Instructions:
- Keep it under 200 words
- Sound human, not robotic
- Format the email into multiple short paragraphs (separated by blank lines). Do NOT use a single large block of text.
- Mention 1-2 specific skills relevant to the job description
- Include a clear call-to-action (request for a 15-min call)
- Do NOT use generic phrases like "I am writing to express my interest"
- Output format: JSON with keys "subject" and "body"
"""

    try:
        chat = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        result = json.loads(chat.choices[0].message.content)
        return {"subject": result.get("subject", ""), "body": result.get("body", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email generation error: {str(e)}")


# ─── RESUME TAILORING ────────────────────────────────────────────────────────

@router.post("/resume/tailor")
async def tailor_resume(req: TailorResumeRequest):
    """
    Use Groq LLM to tailor a resume JSON to a specific job description.
    Returns a NEW modified copy — never overwrites the original.
    """
    if not groq_client:
        return {"tailored_resume": req.resume_data, "changes": ["Mock mode — Groq key not configured"]}

    resume_json = json.dumps(req.resume_data, indent=2)

    prompt = f"""You are a professional resume optimizer. Given a candidate's resume data (JSON) and a job description,
rewrite the resume to better match the job.

Rules:
- Do NOT invent fake experience, companies, or skills
- Reorder skills to put the most relevant ones first
- Rewrite experience descriptions to emphasize keywords from the job description
- Strengthen the summary to align with the role
- Return ONLY a valid JSON object with the same structure as the input, plus a "changes" array listing what you changed

Job Title: {req.job_title}
Job Description: {req.job_description[:800]}

Original Resume JSON:
{resume_json}
"""

    try:
        chat = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        result = json.loads(chat.choices[0].message.content)
        changes = result.pop("changes", ["Summary rewritten", "Skills reordered"])
        return {"tailored_resume": result, "changes": changes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume tailoring error: {str(e)}")


# ─── MOCK HELPERS ────────────────────────────────────────────────────────────

def _mock_jobs(query: str, location: str = "") -> list[dict]:
    companies = [
        ("Stripe", "stripe.com"),
        ("Vercel", "vercel.com"),
        ("Linear", "linear.app"),
        ("Notion", "notion.so"),
        ("Figma", "figma.com"),
        ("Loom", "loom.com"),
        ("Clerk", "clerk.com"),
        ("PlanetScale", "planetscale.com"),
    ]
    return [
        {
            "title": query,
            "company_name": c[0],
            "job_url": f"https://{c[1]}/careers",
            "description": f"We're looking for a talented {query} to join the team at {c[0]}. "
                           f"You'll work on cutting-edge products used by millions.",
            "domain": c[1],
            "location": location or "Remote",
            "source": "mock",
        }
        for c in companies
    ]


def _mock_email(req: GenerateEmailRequest) -> dict:
    return {
        "subject": f"Excited about the {req.job_title} role at {req.company_name}",
        "body": f"""Hi {req.contact_name},

I came across the {req.job_title} opportunity at {req.company_name} and was immediately drawn to it.

I'm {req.candidate_name}, with strong expertise in {', '.join(req.candidate_skills[:3])}. I've been following {req.company_name}'s work closely and believe my background aligns well with what you're building.

Would you be open to a quick 15-minute call to explore whether there's a mutual fit?

Best,
{req.candidate_name}""",
    }
