"""
Skill Gap Service — Skeleton

Responsible for comparing a user's current skills against the requirements
of a target role and returning a structured gap analysis with recommendations.

Business logic and LLM integration are intentionally NOT implemented yet.
The method signatures, docstrings, and placeholder return shapes are
production-ready so the router and repository layers compile cleanly.
"""

from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession

import repositories.career_repository as repo
from config.career_config import MAX_SKILL_RECOMMENDATIONS
from services.llm_service import LLMService
from services.prompt_loader import PromptLoader
from services.context_builder import ContextBuilder
from pydantic import BaseModel


class LLMSkillGap(BaseModel):
    required_skills: List[str]
    missing_skills: List[str]
    recommendations: List[str]


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

async def analyzeSkillGap(
    db: AsyncSession,
    *,
    user_id:         str,
    target_role:     str,
    current_skills:  List[str],
    resume_data:     Optional[dict] = None,
    save_assessment: bool = True,
) -> dict:
    """
    Analyse the gap between a user's current skills and the skills
    required for their target role.

    Future implementation steps:
      1. Replace _resolve_required_skills() with a Groq LLM call that
         returns a dynamic, role-specific skill list based on current
         job market data.
      2. Optionally feed resume_data into the LLM prompt to extract
         implicit skills from experience descriptions.
      3. Generate AI-powered personalised recommendations beyond the
         simple list diff.
      4. Persist assessment via career_repository.saveSkillAssessment()
         when save_assessment is True.

    Args:
        session:         Async SQLAlchemy session (injected via FastAPI Depends).
        user_id:         Application user identifier.
        target_role:     The job role the user is targeting.
        current_skills:  Skills the user already possesses (from their resume).
        resume_data:     Full parsed resume dict for deeper extraction (optional).
        save_assessment: Whether to persist the result to skill_assessments.

    Returns:
        dict matching the SkillGapResponse schema.
    """
    # Normalise inputs
    existing_normalised = [s.strip() for s in current_skills if s.strip()]

    # Augment with skills extracted from resume
    if resume_data:
        resume_skills = resume_data.get("skills", [])
        for skill in resume_skills:
            if skill not in existing_normalised:
                existing_normalised.append(skill)

    # ── STEP 1: Build Context & Retrieve RAG ──────────────────────────────────
    ctx_builder = ContextBuilder()
    context = await ctx_builder.build(
        db,
        user_id=user_id,
        query=f"Analyze skill gap for {target_role}",
        target_role=target_role,
        resume_data=resume_data,
        existing_skills=existing_normalised,
        rag_category="career",
        rag_top_k=2
    )

    # ── STEP 2: Render Prompt ─────────────────────────────────────────────────
    variables = ctx_builder.render_to_prompt_variables(context)
    system_prompt = await PromptLoader.render_prompt("skill_gap", variables)

    # ── STEP 3: Call LLM ──────────────────────────────────────────────────────
    llm = LLMService()
    try:
        llm_result = await llm.generate_json(
            system_prompt=system_prompt,
            response_model=LLMSkillGap,
        )
        required_skills = llm_result.required_skills
        missing_skills = llm_result.missing_skills
        recommendations = llm_result.recommendations
    except Exception as e:
        import logging
        logging.getLogger("skill_gap_service").error("LLM skill gap generation failed: %s", e)
        return {
            "user_id":         user_id,
            "target_role":     target_role,
            "existing_skills": existing_normalised,
            "required_skills": [],
            "missing_skills":  [],
            "gap_score":       0.0,
            "recommendations": [],
            "message":         "Failed to generate skill gap due to AI service error.",
        }

    # ── STEP 4: Compute Gap Score ─────────────────────────────────────────────
    existing_lower   = {s.lower() for s in existing_normalised}
    required_lower   = {s.lower() for s in required_skills}
    missing_lower    = {s.lower() for s in missing_skills}

    if required_skills:
        # Number of required skills the user already has
        matched = len(required_lower) - len(missing_lower)
        gap_score = round((matched / len(required_lower)) * 100, 1)
    else:
        gap_score = 0.0
        
    # Cap recommendations
    recommendations = recommendations[:MAX_SKILL_RECOMMENDATIONS]

    # ── STEP 5: Persist if requested ──────────────────────────────────────────
    assessment_id = None
    if save_assessment:
        saved = await repo.saveSkillAssessment(
            db,
            user_id=user_id,
            target_role=target_role,
            existing_skills=existing_normalised,
            required_skills=required_skills,
            missing_skills=missing_skills,
            gap_score=gap_score,
            recommendations=recommendations,
        )
        if saved:
            assessment_id = saved.get("id")

    return {
        "user_id":         user_id,
        "target_role":     target_role,
        "existing_skills": existing_normalised,
        "required_skills": required_skills,
        "missing_skills":  missing_skills,
        "gap_score":       gap_score,
        "assessment_id":   assessment_id,
        "recommendations": recommendations,
        "message": (
            f"Skill gap analysis complete. "
            f"You have {len(existing_lower & required_lower)}/{len(required_skills)} "
            f"required skills. Gap score: {gap_score}/100."
        ),
    }
