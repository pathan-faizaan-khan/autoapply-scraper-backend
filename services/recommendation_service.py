"""
Recommendation Service

Generates structured AI recommendations for career paths, skills, projects,
certifications, and opportunities.
"""
from typing import Dict, Any, List
import json
from sqlalchemy.ext.asyncio import AsyncSession

from schemas.career_schema import RecommendationRequest, RecommendationResponse
from repositories import career_repository
from services.decision_engine import DecisionEngine

class RecommendationService:
    def __init__(self):
        self.decision_engine = DecisionEngine()

    async def generate_recommendations(
        self,
        session: AsyncSession,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Gathers context from the DB and generates a set of personalized recommendations.
        """
        # 1. Fetch user context
        user_session = await career_repository.getSession(session, request.user_id)
        latest_assessment = await career_repository.getLatestSkillAssessment(session, request.user_id)
        progress = await career_repository.getProgress(session, request.user_id)
        
        # We can also fetch some general opportunities to feed into the prompt
        opps, _ = await career_repository.getOpportunities(session, limit=10)

        # 2. Build kwargs for the prompt
        kwargs = {
            "user_id": request.user_id,
            "career_path": user_session.get("metadata", {}).get("career_path", "Unknown") if user_session else "Unknown",
            "target_role": latest_assessment.get("target_role", "Unknown") if latest_assessment else "Unknown",
            "skill_context": json.dumps(latest_assessment.get("missing_skills", []) if latest_assessment else []),
            "progress_context": json.dumps([{"step": p["step_title"], "status": p["status"]} for p in progress[:5]]),
            "opportunities_context": json.dumps([{"id": o["id"], "title": o["title"]} for o in opps]),
        }

        # 3. Call Decision Engine
        response = await self.decision_engine.generate_recommendations(kwargs)

        # 4. Persist to DB
        await career_repository.saveRecommendation(
            session,
            user_id=response.user_id,
            career_matches=[cm.model_dump() for cm in response.career_matches],
            recommended_skills=[rs.model_dump() for rs in response.recommended_skills],
            recommended_projects=[rp.model_dump() for rp in response.recommended_projects],
            recommended_certifications=[rc.model_dump() for rc in response.recommended_certifications],
            recommended_opportunities=response.recommended_opportunities,
        )

        return response
