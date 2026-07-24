"""
Opportunity Service

Responsible for:
  1. Serving cached opportunities (hackathons, Summer of Code, competitions, etc.)
     directly from the PostgreSQL opportunities table.
  2. Refreshing the cache by scraping / calling external opportunity sources.

Scraping logic is NOT implemented yet.  The skeleton exposes the full
method signatures so the router and repository layers compile cleanly.
"""

from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession

import repositories.career_repository as repo
from config.career_config import DEFAULT_PAGE_SIZE
from services.document_indexer import DocumentIndexer


# ─── STATIC SEED DATA (placeholder until scraper is implemented) ─────────────

_PLACEHOLDER_OPPORTUNITIES = [
    {
        "title":        "Google Summer of Code",
        "organization": "Google",
        "opp_type":     "summer_of_code",
        "country":      "Remote",
        "deadline":     None,
        "url":          "https://summerofcode.withgoogle.com/",
        "description":  "Annual program sponsoring contributors for open-source development.",
        "tags":         ["Python", "JavaScript", "Open Source", "Mentorship"],
    },
    {
        "title":        "MLH Fellowship",
        "organization": "Major League Hacking",
        "opp_type":     "internship",
        "country":      "Remote",
        "deadline":     None,
        "url":          "https://fellowship.mlh.io/",
        "description":  "A remote internship alternative for software engineers.",
        "tags":         ["JavaScript", "Python", "Open Source", "Internship"],
    },
    {
        "title":        "Devpost Global Hackathon",
        "organization": "Devpost",
        "opp_type":     "hackathon",
        "country":      "Remote",
        "deadline":     None,
        "url":          "https://devpost.com/hackathons",
        "description":  "Browse hundreds of online and in-person hackathons.",
        "tags":         ["AI", "Web", "Mobile", "Hackathon"],
    },
    {
        "title":        "Outreachy",
        "organization": "Software Freedom Conservancy",
        "opp_type":     "internship",
        "country":      "Remote",
        "deadline":     None,
        "url":          "https://www.outreachy.org/",
        "description":  "Paid remote internships in open source for underrepresented groups.",
        "tags":         ["Open Source", "Diversity", "Internship", "Python"],
    },
    {
        "title":        "LFX Mentorship",
        "organization": "Linux Foundation",
        "opp_type":     "summer_of_code",
        "country":      "Remote",
        "deadline":     None,
        "url":          "https://lfx.linuxfoundation.org/tools/mentorship",
        "description":  "Open source mentorship programme by the Linux Foundation.",
        "tags":         ["Linux", "Cloud", "Open Source", "Mentorship"],
    },
]


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

async def getCachedOpportunities(
    session: AsyncSession,
    *,
    opp_type: Optional[str]       = None,
    tags:     Optional[List[str]] = None,
    country:  Optional[str]       = None,
    limit:    int                  = DEFAULT_PAGE_SIZE,
    offset:   int                  = 0,
) -> dict:
    """
    Serve paginated opportunities from the database cache.

    If the cache is empty (e.g. first boot), automatically seeds it with
    the placeholder data defined in _PLACEHOLDER_OPPORTUNITIES so the
    endpoint is immediately usable.

    Future implementation steps:
      1. Replace placeholder seed with real scraped data from refreshOpportunities().
      2. Add deadline filtering (exclude opportunities past their deadline).
      3. Add AI-powered relevance ranking based on the user's skill profile.

    Args:
        session:  Async SQLAlchemy session (injected via FastAPI Depends).
        opp_type: Filter by opportunity type string.
        tags:     Filter by any matching skill tag.
        country:  Case-insensitive partial country match.
        limit:    Page size.
        offset:   Pagination offset.

    Returns:
        dict matching the OpportunitiesListResponse schema.
    """
    rows, total = await repo.getOpportunities(
        session,
        opp_type=opp_type,
        tags=tags,
        country=country,
        limit=limit,
        offset=offset,
    )

    # Auto-seed cache on first run so the endpoint is usable immediately.
    if total == 0:
        print("[OpportunityService] Cache empty — seeding placeholder data...")
        await _seedPlaceholders(session)
        rows, total = await repo.getOpportunities(
            session,
            opp_type=opp_type,
            tags=tags,
            country=country,
            limit=limit,
            offset=offset,
        )

    return {
        "total":         total,
        "limit":         limit,
        "offset":        offset,
        "opportunities": rows,
    }


async def refreshOpportunities(session: AsyncSession) -> dict:
    """
    Trigger a refresh of the opportunities cache by scraping external sources.

    Future implementation steps:
      1. Scrape Devpost, MLH, GSoC, Outreachy, LFX, and competition aggregators.
      2. For each discovered opportunity, call repo.saveOpportunity() to upsert.
      3. Mark stale opportunities (deadline passed) as is_active=False.
      4. Return a summary of added / updated / expired records.

    Args:
        session: Async SQLAlchemy session.

    Returns:
        dict with a status message and counts.
    """
    # ── PLACEHOLDER ── Remove when scraper is implemented ────────────────────
    seeded = await _seedPlaceholders(session)

    return {
        "status":  "placeholder",
        "message": (
            f"[PLACEHOLDER] Opportunity refresh triggered. "
            f"Seeded {seeded} records. "
            "Real scraping will be implemented in the next iteration."
        ),
        "seeded":   seeded,
        "scraped":  0,
        "expired":  0,
    }
    # ── END PLACEHOLDER ────────────────────────────────────────────────────────


# ─── INTERNAL HELPERS ─────────────────────────────────────────────────────────

async def _seedPlaceholders(session: AsyncSession) -> int:
    """
    Insert placeholder opportunities into the DB and index them for RAG search.
    Skips duplicates via ON CONFLICT.

    Args:
        session: Async SQLAlchemy session.

    Returns:
        Count of rows successfully saved and indexed.
    """
    saved = 0
    indexer = DocumentIndexer()
    for opp in _PLACEHOLDER_OPPORTUNITIES:
        result = await repo.saveOpportunity(
            session,
            title=opp["title"],
            organization=opp.get("organization"),
            opp_type=opp.get("opp_type"),
            country=opp.get("country"),
            deadline=opp.get("deadline"),
            url=opp.get("url"),
            description=opp.get("description"),
            tags=opp.get("tags"),
        )
        if result:
            saved += 1
            # Index the opportunity for RAG
            await indexer.index_opportunity(session, result)
            
    return saved
