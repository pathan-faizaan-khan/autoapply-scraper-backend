"""
Resume Match Service

Evaluates a resume against a job description using a hybrid approach
(deterministic matching + LLM reasoning).
"""
import json
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from schemas.career_schema import ResumeMatchRequest, ResumeMatchResponse
from repositories import career_repository
from services.decision_engine import DecisionEngine

class ResumeMatchService:
    def __init__(self):
        self.decision_engine = DecisionEngine()

    def _calculate_deterministic_metrics(self, resume_data: dict, job_description: str) -> dict:
        """
        Calculates basic deterministic overlap.
        In a real scenario, this might use sentence-transformers for embedding_score
        and spaCy for keyword extraction. Here we use placeholders to simulate the hybrid flow.
        """
        # Placeholder for hybrid deterministic logic
        return {
            "embedding_score": 0.75,
            "skill_overlap": 60,
            "missing_keywords": ["Kubernetes", "AWS", "GraphQL"]
        }

    async def generate_resume_match(
        self,
        session: AsyncSession,
        request: ResumeMatchRequest
    ) -> ResumeMatchResponse:
        """
        Gathers deterministic scores and delegates to Decision Engine for final evaluation.
        """
        # 1. Deterministic calculation
        metrics = self._calculate_deterministic_metrics(request.resume_data or {}, request.job_description)

        # 2. Build kwargs for prompt
        kwargs = {
            "user_id": request.user_id,
            "job_description": request.job_description,
            "resume_data": json.dumps(request.resume_data) if request.resume_data else "No resume data provided.",
            "embedding_score": metrics["embedding_score"],
            "skill_overlap": metrics["skill_overlap"],
            "missing_keywords": ", ".join(metrics["missing_keywords"])
        }

        # 3. Call Decision Engine
        response = await self.decision_engine.generate_resume_match(kwargs)

        # 4. Persist to DB
        await career_repository.saveResumeMatch(
            session,
            user_id=response.user_id,
            job_description=request.job_description,
            match_score=response.match_score,
            missing_skills=response.missing_skills,
            strong_skills=response.strong_skills,
            weak_areas=response.weak_areas,
            suggestions=response.suggestions,
        )

        return response
