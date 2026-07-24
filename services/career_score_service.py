"""
Career Score Service

Calculates a readiness score using a hybrid of deterministic metrics
and AI qualitative evaluation.
"""
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from schemas.career_schema import CareerScoreResponse
from repositories import career_repository
from services.decision_engine import DecisionEngine

class CareerScoreService:
    def __init__(self):
        self.decision_engine = DecisionEngine()

    def _get_deterministic_metrics(self, progress: list) -> dict:
        """
        Calculates hard metrics based on user progress and other DB tables.
        """
        if not progress:
            return {
                "roadmap_completion_pct": 0,
                "assessments_passed": 0,
                "projects_built": 0,
                "applications_submitted": 0,
            }
        
        completed = sum(1 for p in progress if p.get("status") == "completed")
        roadmap_completion_pct = int((completed / len(progress)) * 100) if progress else 0

        # Placeholders for other deterministic signals
        return {
            "roadmap_completion_pct": roadmap_completion_pct,
            "assessments_passed": 2, 
            "projects_built": 1,
            "applications_submitted": 5,
        }

    async def get_or_generate_score(
        self,
        session: AsyncSession,
        user_id: str,
        force_recalculate: bool = False
    ) -> CareerScoreResponse:
        """
        Returns cached score if recent, else recalculates using DecisionEngine.
        """
        if not force_recalculate:
            latest = await career_repository.getLatestCareerScore(session, user_id)
            if latest:
                # Optionally check if it's too old, e.g., > 7 days
                return CareerScoreResponse(
                    user_id=latest["user_id"],
                    overall_score=latest["overall_score"],
                    technical_score=latest["technical_score"],
                    resume_score=latest["resume_score"],
                    interview_score=latest["interview_score"],
                    market_score=latest["market_score"],
                    confidence=latest["metadata"].get("confidence", 0.0),
                    reason=latest["metadata"].get("reason", "Cached deterministic score."),
                    last_updated=latest["created_at"]
                )

        # Recalculate
        user_session = await career_repository.getSession(session, user_id)
        latest_assessment = await career_repository.getLatestSkillAssessment(session, user_id)
        progress = await career_repository.getProgress(session, user_id)

        metrics = self._get_deterministic_metrics(progress)

        kwargs = {
            "user_id": user_id,
            "career_path": user_session.get("metadata", {}).get("career_path", "Unknown") if user_session else "Unknown",
            "target_role": latest_assessment.get("target_role", "Unknown") if latest_assessment else "Unknown",
            "roadmap_completion_pct": metrics["roadmap_completion_pct"],
            "assessments_passed": metrics["assessments_passed"],
            "projects_built": metrics["projects_built"],
            "applications_submitted": metrics["applications_submitted"],
            "chat_context": json.dumps([m for m in user_session.get("messages", [])[-3:]] if user_session else []),
            "skill_gap_context": json.dumps(latest_assessment.get("missing_skills", []) if latest_assessment else [])
        }

        # Delegate to AI for qualitative synthesis
        response = await self.decision_engine.generate_career_score(kwargs)

        # Persist hybrid score
        await career_repository.saveCareerScore(
            session,
            user_id=response.user_id,
            overall_score=response.overall_score,
            technical_score=response.technical_score,
            resume_score=response.resume_score,
            interview_score=response.interview_score,
            market_score=response.market_score,
            metadata={
                "confidence": response.confidence,
                "reason": response.reason,
            }
        )

        # Ensure datetime is attached for response
        response.last_updated = datetime.now(timezone.utc)
        return response
