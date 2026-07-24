"""
Decision Engine Service

Coordinates all AI reasoning for the AI Career Copilot.
Acts as a facade between the specialized domain services (Recommendation, Resume, Score, Plan)
and the LLMService, ensuring prompts are loaded and responses are strictly validated.
"""
from typing import Dict, Any, Type
from pydantic import BaseModel

from services.llm_service import LLMService
from services.prompt_loader import PromptLoader
from schemas.career_schema import (
    RecommendationResponse,
    ResumeMatchResponse,
    CareerScoreResponse,
    LearningPlanResponse,
)

class DecisionEngine:
    def __init__(self):
        self.llm_service = LLMService()

    async def generate_recommendations(self, kwargs: Dict[str, Any]) -> RecommendationResponse:
        """Generates AI recommendations using the recommendation prompt."""
        system_prompt = await PromptLoader.render_prompt("recommendation", kwargs)
        # Using generate_json with structured output validation
        return await self.llm_service.generate_json(
            system_prompt=system_prompt,
            user_prompt="Please generate recommendations based on my profile.",
            response_model=RecommendationResponse
        )

    async def generate_resume_match(self, kwargs: Dict[str, Any]) -> ResumeMatchResponse:
        """Evaluates a resume against a job description using the resume_match prompt."""
        system_prompt = await PromptLoader.render_prompt("resume_match", kwargs)
        return await self.llm_service.generate_json(
            system_prompt=system_prompt,
            user_prompt="Evaluate my resume against the target job description.",
            response_model=ResumeMatchResponse
        )

    async def generate_career_score(self, kwargs: Dict[str, Any]) -> CareerScoreResponse:
        """Calculates a qualitative career score using the career_score prompt."""
        system_prompt = await PromptLoader.render_prompt("career_score", kwargs)
        return await self.llm_service.generate_json(
            system_prompt=system_prompt,
            user_prompt="Assess my career readiness.",
            response_model=CareerScoreResponse
        )

    async def generate_learning_plan(self, kwargs: Dict[str, Any]) -> LearningPlanResponse:
        """Creates an adaptive learning plan using the learning_plan prompt."""
        system_prompt = await PromptLoader.render_prompt("learning_plan", kwargs)
        return await self.llm_service.generate_json(
            system_prompt=system_prompt,
            user_prompt="Generate my weekly learning plan.",
            response_model=LearningPlanResponse
        )
