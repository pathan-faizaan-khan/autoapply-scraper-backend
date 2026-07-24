"""
Roadmap Service — Skeleton

Responsible for:
  1. Generating a personalised career roadmap (steps) using an LLM.
  2. Retrieving a stored roadmap from the database.

Business logic and LLM integration are intentionally NOT implemented yet.
All public methods return structured placeholder data so the router layer
compiles and responds correctly while business logic is added incrementally.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession

import repositories.career_repository as repo
from config.career_config import DEFAULT_ROADMAP_DURATION_DAYS, DEFAULT_EXPERIENCE_LEVEL
from services.llm_service import LLMService
from services.prompt_loader import PromptLoader
from services.context_builder import ContextBuilder


class LLMRoadmapStep(BaseModel):
    step_order: int
    title: str
    description: str
    difficulty: str
    estimated_days: int
    category: str
    is_mandatory: bool
    resources: List[Dict[str, Any]] = []

class LLMRoadmapResponse(BaseModel):
    steps: List[LLMRoadmapStep]


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

async def generateRoadmap(
    db: AsyncSession,
    *,
    user_id:            str,
    target_role:        str,
    current_skills:     List[str],
    experience_level:   str           = DEFAULT_EXPERIENCE_LEVEL,
    preferred_duration: int           = DEFAULT_ROADMAP_DURATION_DAYS,
    learning_style:     Optional[str] = None,
) -> dict:
    """
    Generate a personalised career roadmap for the user.

    Future implementation steps:
      1. Call the Groq LLM with a structured prompt that includes
         target_role, current_skills, experience_level, and preferred_duration.
      2. Parse the JSON response into a list of RoadmapStep dicts.
      3. Upsert the CareerPath record via career_repository.upsertCareerPath().
      4. Bulk-insert steps via career_repository.bulkInsertRoadmapSteps().
      5. Return the structured roadmap dict for the router to serialise.

    Args:
        session:            Async SQLAlchemy session (injected by FastAPI Depends).
        user_id:            Application user identifier.
        target_role:        The job role the user is targeting.
        current_skills:     Skills the user already possesses.
        experience_level:   "junior" | "mid" | "senior".
        preferred_duration: Total target days to complete the roadmap.
        learning_style:     Preferred format — "video" | "reading" | "project" | "mixed".

    Returns:
        dict matching the RoadmapResponse schema.
    """
    # ── STEP 1: Build Context & Retrieve RAG ──────────────────────────────────
    skill_assessment = await repo.getLatestSkillAssessment(db, user_id=user_id)
    missing_skills = skill_assessment.get("missing_skills", []) if skill_assessment else []
    
    ctx_builder = ContextBuilder()
    context = await ctx_builder.build(
        db,
        user_id=user_id,
        query=f"Generate a {preferred_duration}-day roadmap for {target_role} ({experience_level})",
        target_role=target_role,
        existing_skills=current_skills,
        missing_skills=missing_skills,
        rag_category="roadmap",
        rag_top_k=3
    )

    # ── STEP 2: Render Prompt ─────────────────────────────────────────────────
    variables = ctx_builder.render_to_prompt_variables(context)
    # The roadmap prompt needs timeline_days and experience injected
    variables["timeline_days"] = str(preferred_duration)
    variables["experience"] = experience_level
    system_prompt = await PromptLoader.render_prompt("roadmap", variables)

    # ── STEP 3: Call LLM ──────────────────────────────────────────────────────
    llm = LLMService()
    try:
        llm_result = await llm.generate_json(
            system_prompt=system_prompt,
            response_model=LLMRoadmapResponse,
        )
        generated_steps = llm_result.steps
    except Exception as e:
        import logging
        logging.getLogger("roadmap_service").error("LLM roadmap generation failed: %s", e)
        return {
            "user_id":        user_id,
            "career_path":    target_role,
            "total_steps":    0,
            "estimated_days": 0,
            "steps":          [],
            "message":        "Failed to generate roadmap due to AI service error.",
        }

    # ── STEP 4: Persist to DB ─────────────────────────────────────────────────
    career_path_row = await repo.upsertCareerPath(
        db,
        title=target_role,
        description=f"Generated AI roadmap for {target_role}",
    )
    
    if not career_path_row:
        return {"message": "Database error while saving career path."}
        
    path_id = career_path_row["id"]
    
    step_dicts = []
    total_estimated = 0
    for s in generated_steps:
        step_dicts.append({
            "step_order": s.step_order,
            "title": s.title,
            "description": s.description,
            "difficulty": s.difficulty,
            "estimated_days": s.estimated_days,
            "category": s.category,
            "is_mandatory": s.is_mandatory,
            "resources": s.resources,
        })
        total_estimated += s.estimated_days

    await repo.bulkInsertRoadmapSteps(
        db,
        career_path_id=path_id,
        career_title=target_role,
        steps=step_dicts
    )

    # Fetch the newly inserted steps so they have database-assigned IDs
    saved_steps = await repo.getRoadmap(db, path_id)

    # ── STEP 5: Return Standardized Response ──────────────────────────────────
    return {
        "user_id":        user_id,
        "career_path_id": path_id,
        "career_path":    target_role,
        "total_steps":    len(saved_steps),
        "estimated_days": total_estimated,
        "steps":          saved_steps,
        "message":        f"Successfully generated a personalized {total_estimated}-day roadmap for {target_role}.",
    }


async def getRoadmap(
    db: AsyncSession,
    *,
    career_path_id: int,
) -> dict:
    """
    Retrieve a previously generated roadmap from the database.

    Future implementation steps:
      1. Call career_repository.getCareerPath() to validate the path exists.
      2. Call career_repository.getRoadmap() to fetch ordered steps.
      3. Return structured data.

    Args:
        session:        Async SQLAlchemy session.
        career_path_id: PK of the career_paths row.

    Returns:
        dict matching the RoadmapResponse schema, or an empty shell if not found.
    """
    # ── Skeleton DB call (repository layer ready, data may not exist yet) ──────
    steps = await repo.getRoadmap(db, career_path_id)
    career_path = await repo.getCareerPath(db, path_id=career_path_id)

    if not career_path:
        return {
            "career_path_id": career_path_id,
            "career_path":    None,
            "total_steps":    0,
            "estimated_days": None,
            "steps":          [],
            "message":        f"No career path found with id={career_path_id}.",
        }

    return {
        "career_path_id": career_path_id,
        "career_path":    career_path.get("title"),
        "total_steps":    len(steps),
        "estimated_days": sum(s.get("estimated_days") or 0 for s in steps) or None,
        "steps":          steps,
        "message":        "Roadmap retrieved from database.",
    }
