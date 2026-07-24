"""
Career Guidance Router

Registers all /api/career/* endpoints and delegates exclusively to the
CareerAgent orchestrator service. This router contains NO business logic —
it only validates the request, calls the service, and returns the response.

Pattern follows routers/jobs_search.py and routers/ml_autofill.py:
  - APIRouter with prefix + tags
  - Pydantic request/response models from schemas/career_schema.py
  - HTTPException for error propagation
  - Async endpoint functions
  - FastAPI Depends() for DB session injection
"""

from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from schemas.career_schema import (
    CareerChatRequest,
    CareerChatResponse,
    RoadmapRequest,
    RoadmapResponse,
    SkillGapRequest,
    SkillGapResponse,
    ProgressUpdateRequest,
    ProgressResponse,
    UserProgressSummaryResponse,
    OpportunitiesListResponse,
)
import services.career_agent as agent
import services.opportunity_service as opportunity_svc
import services.roadmap_service as roadmap_svc
from config.career_config import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    DEFAULT_PATHS_PAGE_SIZE,
    MAX_PATHS_PAGE_SIZE,
    DEFAULT_ROADMAP_DURATION_DAYS,
    DEFAULT_EXPERIENCE_LEVEL,
)

router = APIRouter(prefix="/api/career", tags=["career-guidance"])


# ─── CHAT ─────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=CareerChatResponse,
    summary="AI Career Guidance Chat",
    description=(
        "Send a free-text message to the AI Career Guidance agent. "
        "The agent maintains conversation history per user_id and will "
        "route follow-up questions to the appropriate sub-service "
        "(roadmap, skill gap, opportunities)."
    ),
)
async def career_chat(
    req: CareerChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    POST /api/career/chat

    Input  : CareerChatRequest  (user_id, message, optional context)
    Output : CareerChatResponse (reply, suggestions, session_id)
    """
    try:
        result = await agent.handleCareerChat(
            db,
            user_id=req.user_id,
            message=req.message,
            resume_data=req.resume_data,
            target_role=req.target_role,
            career_path=req.career_path,
        )
        return CareerChatResponse(**result)
    except Exception as exc:
        print(f"[CareerRouter] /chat error: {exc}")
        raise HTTPException(status_code=500, detail=f"Career chat error: {str(exc)}")


# ─── ROADMAP ──────────────────────────────────────────────────────────────────

@router.post(
    "/generate-roadmap",
    response_model=RoadmapResponse,
    summary="Generate Personalised Career Roadmap",
    description=(
        "Generate a step-by-step career roadmap tailored to the user's "
        "target role, current skills, and preferred timeline."
    ),
)
async def generate_roadmap(
    req: RoadmapRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    POST /api/career/generate-roadmap

    Input  : RoadmapRequest  (user_id, target_role, current_skills, ...)
    Output : RoadmapResponse (steps list, estimated_days, career_path)
    """
    try:
        result = await agent.orchestrateRoadmap(
            db,
            user_id=req.user_id,
            target_role=req.target_role,
            current_skills=req.current_skills,
            experience_level=req.experience_level or DEFAULT_EXPERIENCE_LEVEL,
            preferred_duration=req.preferred_duration or DEFAULT_ROADMAP_DURATION_DAYS,
            learning_style=req.learning_style,
        )
        return RoadmapResponse(**result)
    except Exception as exc:
        print(f"[CareerRouter] /generate-roadmap error: {exc}")
        raise HTTPException(status_code=500, detail=f"Roadmap generation error: {str(exc)}")


@router.get(
    "/roadmap/{career_path_id}",
    response_model=RoadmapResponse,
    summary="Retrieve Stored Roadmap",
    description="Fetch a previously generated and persisted career roadmap by its ID.",
)
async def get_roadmap(
    career_path_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    GET /api/career/roadmap/{career_path_id}

    Input  : career_path_id (path parameter)
    Output : RoadmapResponse
    """
    try:
        result = await roadmap_svc.getRoadmap(db, career_path_id=career_path_id)
        return RoadmapResponse(**result)
    except Exception as exc:
        print(f"[CareerRouter] /roadmap/{career_path_id} error: {exc}")
        raise HTTPException(status_code=500, detail=f"Roadmap retrieval error: {str(exc)}")


# ─── SKILL GAP ────────────────────────────────────────────────────────────────

@router.post(
    "/skill-gap",
    response_model=SkillGapResponse,
    summary="Analyse Skill Gap",
    description=(
        "Compare the user's current skills against the requirements of a "
        "target role and return a structured gap analysis with a match score "
        "and personalised learning recommendations."
    ),
)
async def skill_gap_analysis(
    req: SkillGapRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    POST /api/career/skill-gap

    Input  : SkillGapRequest  (user_id, target_role, current_skills, ...)
    Output : SkillGapResponse (missing_skills, gap_score, recommendations)
    """
    try:
        result = await agent.orchestrateSkillGap(
            db,
            user_id=req.user_id,
            target_role=req.target_role,
            current_skills=req.current_skills,
            resume_data=req.resume_data,
            save_assessment=req.save_assessment if req.save_assessment is not None else True,
        )
        return SkillGapResponse(**result)
    except Exception as exc:
        print(f"[CareerRouter] /skill-gap error: {exc}")
        raise HTTPException(status_code=500, detail=f"Skill gap analysis error: {str(exc)}")


# ─── OPPORTUNITIES ────────────────────────────────────────────────────────────

@router.get(
    "/opportunities",
    response_model=OpportunitiesListResponse,
    summary="List Cached Opportunities",
    description=(
        "Return a paginated list of cached opportunities (hackathons, "
        "Summer of Code programmes, competitions, internships). "
        "The cache is auto-seeded with placeholder data on first access "
        "and will be populated by a real scraper in a future iteration."
    ),
)
async def list_opportunities(
    opp_type: Optional[str] = Query(None, alias="type", description="Filter: hackathon | summer_of_code | competition | internship"),
    country:  Optional[str] = Query(None, description="Country filter (partial match)"),
    tags:     Optional[str] = Query(None, description="Comma-separated skill tags to filter by"),
    limit:    int            = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset:   int            = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    GET /api/career/opportunities

    Query params : type (alias), country, tags (CSV), limit, offset
    Output       : OpportunitiesListResponse (total, limit, offset, opportunities[])
    """
    try:
        tags_list: Optional[List[str]] = (
            [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        )
        result = await agent.orchestrateOpportunities(
            db,
            opp_type=opp_type,
            tags=tags_list,
            country=country,
            limit=limit,
            offset=offset,
        )
        return OpportunitiesListResponse(**result)
    except Exception as exc:
        print(f"[CareerRouter] /opportunities error: {exc}")
        raise HTTPException(status_code=500, detail=f"Opportunities retrieval error: {str(exc)}")


@router.post(
    "/opportunities/refresh",
    summary="Refresh Opportunities Cache",
    description=(
        "Trigger a manual refresh of the opportunities cache. "
        "Currently seeds placeholder data and indexes them into the RAG vector store."
    ),
)
async def refresh_opportunities(db: AsyncSession = Depends(get_db)):
    """
    POST /api/career/opportunities/refresh

    Output : status message with counts
    """
    try:
        result = await opportunity_svc.refreshOpportunities(db)
        return result
    except Exception as exc:
        print(f"[CareerRouter] /opportunities/refresh error: {exc}")
        raise HTTPException(status_code=500, detail=f"Opportunity refresh error: {str(exc)}")


# ─── PROGRESS ─────────────────────────────────────────────────────────────────

@router.post(
    "/progress/update",
    response_model=ProgressResponse,
    summary="Update Step Progress",
    description="Record or update a user's completion state for a single roadmap step.",
)
async def update_progress(
    req: ProgressUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    POST /api/career/progress/update

    Input  : ProgressUpdateRequest (user_id, step_id, status, ...)
    Output : ProgressResponse
    """
    try:
        result = await agent.orchestrateProgressUpdate(
            db,
            user_id=req.user_id,
            step_id=req.step_id,
            status=req.status,
            completion_percentage=req.completion_percentage or 0.0,
            score=req.score,
            notes=req.notes,
        )
        if not result:
            raise HTTPException(status_code=500, detail="Failed to save progress")
        return ProgressResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[CareerRouter] /progress/update error: {exc}")
        raise HTTPException(status_code=500, detail=f"Progress update error: {str(exc)}")


@router.get(
    "/progress/{user_id}",
    response_model=UserProgressSummaryResponse,
    summary="Get User Progress Summary",
    description=(
        "Retrieve the aggregated progress summary for a user, "
        "optionally scoped to a single career path."
    ),
)
async def get_progress(
    user_id: str,
    career_path_id: Optional[int] = Query(None, description="Scope to a specific career path"),
    db: AsyncSession = Depends(get_db),
):
    """
    GET /api/career/progress/{user_id}

    Path param   : user_id
    Query params : career_path_id (optional)
    Output       : UserProgressSummaryResponse
    """
    try:
        result = await agent.orchestrateProgressSummary(
            db,
            user_id=user_id,
            career_path_id=career_path_id,
        )
        return UserProgressSummaryResponse(**result)
    except Exception as exc:
        print(f"[CareerRouter] /progress/{user_id} error: {exc}")
        raise HTTPException(status_code=500, detail=f"Progress retrieval error: {str(exc)}")


# ─── CAREER PATHS ─────────────────────────────────────────────────────────────

@router.get(
    "/paths",
    summary="List Career Paths",
    description="Return all available career path titles from the database.",
)
async def list_career_paths(
    limit:  int = Query(DEFAULT_PATHS_PAGE_SIZE, ge=1, le=MAX_PATHS_PAGE_SIZE),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    GET /api/career/paths

    Output : list of career path records
    """
    try:
        result = await agent.orchestrateCareerPaths(db, limit=limit, offset=offset)
        return result
    except Exception as exc:
        print(f"[CareerRouter] /paths error: {exc}")
        raise HTTPException(status_code=500, detail=f"Career paths retrieval error: {str(exc)}")
