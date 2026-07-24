"""
Learning Plan Service

Transforms static roadmap steps into an adaptive weekly study schedule.
"""
import json
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from schemas.career_schema import LearningPlanRequest, LearningPlanResponse
from repositories import career_repository
from services.decision_engine import DecisionEngine

class LearningPlanService:
    def __init__(self):
        self.decision_engine = DecisionEngine()

    async def generate_learning_plan(
        self,
        session: AsyncSession,
        request: LearningPlanRequest
    ) -> LearningPlanResponse:
        """
        Creates an adaptive learning plan by feeding uncompleted roadmap steps to the AI.
        """
        user_session = await career_repository.getSession(session, request.user_id)
        career_path_id = user_session.get("metadata", {}).get("career_path_id") if user_session else None
        
        target_role = user_session.get("metadata", {}).get("career_path", "Unknown") if user_session else "Unknown"
        
        # Default empty if no roadmap
        roadmap_steps_context = []
        
        if career_path_id:
            steps = await career_repository.getRoadmap(session, career_path_id)
            progress = await career_repository.getProgress(session, request.user_id, career_path_id)
            
            completed_step_ids = {p["step_id"] for p in progress if p["status"] == "completed"}
            uncompleted_steps = [s for s in steps if s["id"] not in completed_step_ids]
            
            # Format steps for the prompt
            roadmap_steps_context = [
                {
                    "title": s["title"],
                    "difficulty": s["difficulty"],
                    "estimated_days": s["estimated_days"]
                }
                for s in uncompleted_steps[:10]  # Take next 10 steps to keep prompt size manageable
            ]

        latest_assessment = await career_repository.getLatestSkillAssessment(session, request.user_id)
        current_skills = latest_assessment.get("existing_skills", []) if latest_assessment else []

        kwargs = {
            "user_id": request.user_id,
            "target_role": target_role,
            "weekly_study_hours": request.weekly_study_hours,
            "target_date": request.target_date.isoformat() if request.target_date else "Flexible",
            "roadmap_steps": json.dumps(roadmap_steps_context),
            "current_skills": json.dumps(current_skills),
        }

        response = await self.decision_engine.generate_learning_plan(kwargs)

        await career_repository.saveLearningPlan(
            session,
            user_id=response.user_id,
            career_path_id=career_path_id,
            target_date=request.target_date,
            weekly_study_hours=request.weekly_study_hours,
            plan_data=[w.model_dump() for w in response.weeks]
        )

        return response
