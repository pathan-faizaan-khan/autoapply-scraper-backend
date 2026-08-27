"""
Career Agent — Orchestrator Service

This is the top-level service that all career router endpoints call.
It wires together:
  - RoadmapService     → personalised learning roadmap generation
  - SkillGapService    → skill comparison and gap analysis
  - OpportunityService → hackathon / SoC / competition discovery

The agent also manages the conversation session (memory) stored in
the career_sessions table so the user's context is preserved
across multiple chat interactions.

Groq / LLM integration is intentionally left as a clearly marked
placeholder — the full agentic loop will be added in the next iteration.
"""

from typing import Optional, List
import json

from sqlalchemy.ext.asyncio import AsyncSession

import repositories.career_repository as repo
import services.roadmap_service     as roadmap_svc
import services.skill_gap_service   as skill_gap_svc
import services.opportunity_service as opportunity_svc
from services.llm_service import LLMService
from services.prompt_loader import PromptLoader
from services.context_builder import ContextBuilder
from config.career_config import (
    DEFAULT_ROADMAP_DURATION_DAYS,
    DEFAULT_EXPERIENCE_LEVEL,
    DEFAULT_CHAT_SUGGESTIONS,
    CHAT_CONTEXT_WINDOW,
    DEFAULT_PAGE_SIZE,
)


# ─── CHAT / CONVERSATION ──────────────────────────────────────────────────────

async def handleCareerChat(
    db: AsyncSession,
    *,
    user_id:     str,
    message:     str,
    resume_data: Optional[dict] = None,
    target_role: Optional[str]  = None,
    career_path: Optional[str]  = None,
    reset_session: bool         = False,
    voice_mode: bool            = False,
) -> dict:
    """
    Handle a single turn of the AI career guidance chat.

    Orchestration plan (fully implemented once LLM is wired up):
      1. Load existing conversation history from career_sessions via repo.getSession().
      2. Apply a sliding window of CHAT_CONTEXT_WINDOW messages to bound context size.
      3. Append the new user message to the history list.
      4. Build a system prompt that includes:
           - user's target role and career path
           - current skill snapshot from the latest SkillAssessment
           - the windowed message history
      5. Call Groq (qwen/qwen3.8-27b) with the assembled messages.
      6. Parse the assistant's reply and extract any structured action items
         (e.g. "generate roadmap", "show opportunities", "check skill gap").
      7. Delegate to the appropriate sub-service if an action is detected.
      8. Append the assistant reply to the session history.
      9. Persist the updated session via repo.upsertSession().
      10. Return the reply and any follow-up suggestions.

    Args:
        db:          Async SQLAlchemy session (injected via FastAPI Depends).
                     Named `db` to match the FastAPI convention and avoid
                     confusion with career_sessions DB rows.
        user_id:     Application user identifier.
        message:     The user's free-text input message.
        resume_data: Optional parsed resume for personalisation.
        target_role: User's stated target role.
        career_path: Currently selected career path title.
        reset_session: Wipe chat history before processing.

    Returns:
        dict matching the CareerChatResponse schema.
    """
    # ── STEP 1: Load session history ──────────────────────────────────────────
    existing = await repo.getSession(db, user_id) if not reset_session else None
    history: list = []

    if existing:
        raw_history = existing.get("messages") or []
        history = raw_history[-CHAT_CONTEXT_WINDOW:]

    # ── STEP 2: Build Context & Retrieve RAG ──────────────────────────────────
    # We do not append the new user message to history *yet* because the LLM
    # gets the current message via the user prompt. We just need context.
    # We fetch the latest skill assessment and roadmap steps if they exist.
    skill_assessment = await repo.getLatestSkillAssessment(db, user_id=user_id)
    
    # We need to know the active career path id to get the roadmap.
    # For now, let's just grab the active path from metadata or target_role.
    active_roadmap = []
    if skill_assessment and skill_assessment.get("career_path_id"):
        active_roadmap = await repo.getRoadmap(db, skill_assessment["career_path_id"])

    ctx_builder = ContextBuilder()
    context = await ctx_builder.build(
        db,
        user_id=user_id,
        query=message,
        target_role=target_role or "",
        career_path=career_path or "",
        resume_data=resume_data,
        skill_assessment=skill_assessment,
        roadmap_steps=active_roadmap,
        session_history=history,
    )

    # ── STEP 3: Load and Render Prompt ────────────────────────────────────────
    variables = ctx_builder.render_to_prompt_variables(context)
    system_prompt = await PromptLoader.render_prompt("career_chat", variables)
    # Convert session history to standard format.
    # LLMService._build_messages accepts an explicit messages list.
    messages = [{"role": "system", "content": system_prompt}]
    
    # Implement Sliding Window to avoid context overflow (keep last 20 messages max)
    MAX_CONTEXT = 20
    if len(history) > MAX_CONTEXT:
        history = history[-MAX_CONTEXT:]

    # Inject past conversation
    for msg in history:
        # Ensure we only have standard roles
        role = msg.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        messages.append({"role": role, "content": msg.get("content", "")})
        
    # Append the current user message
    messages.append({"role": "user", "content": message})

    # ── STEP 4: Call LLM ──────────────────────────────────────────────────────
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_roadmap",
                "description": "Generate a personalized career roadmap for the user. Call this when the user explicitly asks for a roadmap, learning path, or a step-by-step guide.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_role": {"type": "string", "description": "The target role (e.g., Software Engineer)"},
                        "preferred_duration": {"type": "integer", "description": "Duration in days (e.g., 90)"}
                    },
                    "required": ["target_role"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_user_resume",
                "description": "Fetch the user's parsed resume and skills from the database. Call this when you need context about the user's background.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_job_applications",
                "description": "Fetch the user's recent job applications from the database to see what they have applied for.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_opportunities",
                "description": "Fetch career opportunities such as hackathons, internships, or open source programs that match the user's skills or target role.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "opp_type": {"type": "string", "description": "Type of opportunity: hackathon, internship, summer_of_code, or competition"},
                        "tags": {"type": "string", "description": "Comma-separated list of skills or topics (e.g. 'react, node')"},
                        "country": {"type": "string", "description": "Country abbreviation or name"}
                    }
                }
            }
        }
    ]

    llm = LLMService()
    action_taken = None
    assistant_reply = ""
    
    try:
        max_tool_iterations = 3
        
        # Reduce output tokens as requested
        target_max_tokens = 150 if voice_mode else 300
        
        for i in range(max_tool_iterations):
            response = await llm.generate_text(messages=messages, tools=tools, max_tokens=target_max_tokens)
            assistant_reply = response.content or ""
            
            if not response.tool_calls:
                break
                
            tool_calls_dict = [tc.model_dump() for tc in response.tool_calls]
            messages.append({"role": "assistant", "content": assistant_reply, "tool_calls": tool_calls_dict})
            
            for tool_call in response.tool_calls:
                tool_name = getattr(tool_call.function, "name", "")
                tool_id = getattr(tool_call, "id", "")
                
                if tool_name == "generate_roadmap":
                    args_str = getattr(tool_call.function, "arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}

                    roadmap_role = args.get("target_role", target_role or "Software Engineer")
                    roadmap_duration = args.get("preferred_duration", DEFAULT_ROADMAP_DURATION_DAYS)

                    await orchestrateRoadmap(
                        db,
                        user_id=user_id,
                        target_role=roadmap_role,
                        current_skills=[], 
                        experience_level=DEFAULT_EXPERIENCE_LEVEL,
                        preferred_duration=roadmap_duration,
                    )
                    action_taken = "roadmap_generated"
                    messages.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": "Roadmap successfully generated in DB."})
                    
                elif tool_name == "fetch_user_resume":
                    profile = await repo.getUserProfile(db, user_id)
                    if profile:
                        content = f"Resume:\n{profile.get('resume_text', 'None')}\nSkills: {profile.get('skills', 'None')}"
                    else:
                        content = "User profile or resume not found."
                    messages.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": content})
                    
                elif tool_name == "fetch_job_applications":
                    apps = await repo.getJobApplications(db, user_id)
                    if apps:
                        content = "Recent applications:\n" + "\n".join([f"- {a['job_title']} at {a['company_name']} (Status: {a['status']})" for a in apps])
                    else:
                        content = "No recent job applications found."
                    messages.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": content})
                    
                elif tool_name == "fetch_opportunities":
                    args_str = getattr(tool_call.function, "arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}
                    
                    tags = args.get("tags")
                    tags_list = [t.strip() for t in tags.split(",")] if tags else None
                    opps = await orchestrateOpportunities(
                        db,
                        opp_type=args.get("opp_type"),
                        tags=tags_list,
                        country=args.get("country"),
                        limit=5,
                        offset=0
                    )
                    
                    if opps.get("opportunities"):
                        content = "Found opportunities:\n" + "\n".join([f"- {o['title']} ({o['type']}) at {o['organization']}" for o in opps["opportunities"]])
                    else:
                        content = "No matching opportunities found right now."
                    messages.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": content})
                
                else:
                    messages.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": "Unknown tool."})
    except Exception as e:
        # Graceful degradation on LLM failure
        import logging
        logging.getLogger("career_agent").error("LLM generation failed: %s", e)
        if "Rate limit" in str(e) or "429" in str(e):
            assistant_reply = f"I'm currently receiving too many requests (Rate Limit). Please wait a moment and try again. (Details: {e})"
        else:
            assistant_reply = f"I'm having trouble connecting to my AI brain right now. Details: {e}"

    # ── STEP 5: Append to History & Persist ───────────────────────────────────
    if not assistant_reply:
        assistant_reply = "I've checked my tools, but I need a moment to organize the information. What specifically would you like to know?"
        
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": assistant_reply})

    session_meta = {
        "career_path": career_path,
        "target_role": target_role,
    }
    saved = await repo.upsertSession(
        db,
        user_id=user_id,
        messages=history,
        metadata=session_meta,
    )
    session_id: Optional[int] = saved.get("id") if saved else None

    # TODO: In a future iteration, we will use a structured JSON call or tool calling 
    # to dynamically determine suggestions. For now, use the defaults.
    suggestions: list[str] = list(DEFAULT_CHAT_SUGGESTIONS)

    return {
        "user_id":     user_id,
        "reply":       assistant_reply,
        "suggestions": suggestions,
        "session_id":  session_id,
        "action_taken": action_taken,
    }


# ─── ROADMAP ORCHESTRATION ────────────────────────────────────────────────────

async def orchestrateRoadmap(
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
    Delegate to RoadmapService and return its result.

    Future extension:
      - After generating the roadmap, automatically run analyzeSkillGap()
        so the UI can show which steps directly address skill gaps.
      - Store the generated roadmap ID in the user's career_session metadata.

    Args:
        db:                 Async SQLAlchemy session.
        user_id:            Application user identifier.
        target_role:        Target job role string.
        current_skills:     List of skills the user already has.
        experience_level:   "junior" | "mid" | "senior".
        preferred_duration: Target completion time in days.
        learning_style:     User's preferred learning format.

    Returns:
        dict matching the RoadmapResponse schema.
    """
    return await roadmap_svc.generateRoadmap(
        db,
        user_id=user_id,
        target_role=target_role,
        current_skills=current_skills,
        experience_level=experience_level,
        preferred_duration=preferred_duration,
        learning_style=learning_style,
    )


# ─── SKILL GAP ORCHESTRATION ─────────────────────────────────────────────────

async def orchestrateSkillGap(
    db: AsyncSession,
    *,
    user_id:         str,
    target_role:     str,
    current_skills:  List[str],
    resume_data:     Optional[dict] = None,
    save_assessment: bool           = True,
) -> dict:
    """
    Delegate to SkillGapService and return its result.

    Future extension:
      - Cross-reference the gap analysis with the user's active roadmap
        and highlight which roadmap steps directly address missing skills.

    Args:
        db:              Async SQLAlchemy session.
        user_id:         Application user identifier.
        target_role:     Desired job role.
        current_skills:  User's existing skills.
        resume_data:     Full parsed resume for deeper extraction.
        save_assessment: Whether to persist the result.

    Returns:
        dict matching the SkillGapResponse schema.
    """
    return await skill_gap_svc.analyzeSkillGap(
        db,
        user_id=user_id,
        target_role=target_role,
        current_skills=current_skills,
        resume_data=resume_data,
        save_assessment=save_assessment,
    )


# ─── OPPORTUNITY ORCHESTRATION ────────────────────────────────────────────────

async def orchestrateOpportunities(
    db: AsyncSession,
    *,
    opp_type: Optional[str]       = None,
    tags:     Optional[List[str]] = None,
    country:  Optional[str]       = None,
    limit:    int                  = DEFAULT_PAGE_SIZE,
    offset:   int                  = 0,
) -> dict:
    """
    Delegate to OpportunityService and return paginated cached results.
    The parameter is named `opp_type` (not `type`) to avoid shadowing
    the Python builtin.

    Future extension:
      - Accept user_id and rank opportunities by skill gap relevance.
      - Trigger a background refresh if cache is older than 24 hours.

    Args:
        db:       Async SQLAlchemy session.
        opp_type: Filter by opportunity type.
        tags:     Filter by matching tags.
        country:  Filter by country.
        limit:    Page size.
        offset:   Pagination offset.

    Returns:
        dict matching the OpportunitiesListResponse schema.
    """
    return await opportunity_svc.getCachedOpportunities(
        db,
        opp_type=opp_type,
        tags=tags,
        country=country,
        limit=limit,
        offset=offset,
    )


# ─── PROGRESS ORCHESTRATION ───────────────────────────────────────────────────

async def orchestrateProgressUpdate(
    db: AsyncSession,
    *,
    user_id:               str,
    step_id:               int,
    status:                str,
    completion_percentage: float           = 0.0,
    score:                 Optional[float] = None,
    notes:                 Optional[str]   = None,
) -> Optional[dict]:
    """
    Persist a user's progress on a single roadmap step.

    Business rule: completion_percentage is forced to 100.0 when status
    is 'completed', ensuring the two fields stay consistent.

    Args:
        db:                    Async SQLAlchemy session.
        user_id:               Application user identifier.
        step_id:               FK to roadmap_steps.id.
        status:                Lifecycle status (see PROGRESS_STATUSES in config).
        completion_percentage: 0.0–100.0.
        score:                 Optional quiz/assessment result.
        notes:                 Free-text notes from the user.

    Returns:
        Saved row dict or None on failure.
    """
    # Enforce consistency: completing a step must set percentage to 100.
    if status == "completed":
        completion_percentage = 100.0

    return await repo.saveProgress(
        db,
        user_id=user_id,
        step_id=step_id,
        status=status,
        completion_percentage=completion_percentage,
        score=score,
        notes=notes,
    )


async def orchestrateProgressSummary(
    db: AsyncSession,
    *,
    user_id:        str,
    career_path_id: Optional[int] = None,
) -> dict:
    """
    Retrieve and aggregate a user's progress across a roadmap.

    Aggregation logic lives here (not in the router) because the rule
    "overall percentage = completed / total" is a business decision.

    Args:
        db:             Async SQLAlchemy session.
        user_id:        Application user identifier.
        career_path_id: When provided, scopes results to one career path.

    Returns:
        dict matching the UserProgressSummaryResponse schema.
    """
    rows = await repo.getProgress(db, user_id, career_path_id=career_path_id)

    total_steps       = len(rows)
    completed_steps   = sum(1 for r in rows if r.get("status") == "completed")
    in_progress_steps = sum(1 for r in rows if r.get("status") == "in_progress")
    overall_pct       = (
        round((completed_steps / total_steps) * 100, 1) if total_steps else 0.0
    )

    return {
        "user_id":           user_id,
        "career_path":       None,          # Populated by future career path lookup
        "total_steps":       total_steps,
        "completed_steps":   completed_steps,
        "in_progress_steps": in_progress_steps,
        "overall_percentage": overall_pct,
        "step_details":      rows,
    }


async def orchestrateCareerPaths(
    db: AsyncSession,
    *,
    limit:  int = 50,
    offset: int = 0,
) -> dict:
    """
    Retrieve a paginated list of available career paths.

    Args:
        db:     Async SQLAlchemy session.
        limit:  Page size.
        offset: Pagination offset.

    Returns:
        dict with total count and paths list.
    """
    paths = await repo.listCareerPaths(db, limit=limit, offset=offset)
    return {"total": len(paths), "paths": paths}


# ─── SPRINT 4: DECISION ENGINE ORCHESTRATION ──────────────────────────────────

from schemas.career_schema import (
    RecommendationRequest,
    ResumeMatchRequest,
    LearningPlanRequest,
)
from services.recommendation_service import RecommendationService
from services.resume_match_service import ResumeMatchService
from services.career_score_service import CareerScoreService
from services.learning_plan_service import LearningPlanService

async def orchestrateRecommendation(
    db: AsyncSession,
    request: RecommendationRequest
) -> dict:
    svc = RecommendationService()
    resp = await svc.generate_recommendations(db, request)
    return resp.model_dump()

async def orchestrateResumeMatch(
    db: AsyncSession,
    request: ResumeMatchRequest
) -> dict:
    svc = ResumeMatchService()
    resp = await svc.generate_resume_match(db, request)
    return resp.model_dump()

async def orchestrateCareerScore(
    db: AsyncSession,
    user_id: str
) -> dict:
    svc = CareerScoreService()
    # Returns cached if recent
    resp = await svc.get_or_generate_score(db, user_id, force_recalculate=False)
    return resp.model_dump()

async def orchestrateCareerScoreRecalculate(
    db: AsyncSession,
    user_id: str
) -> dict:
    svc = CareerScoreService()
    # Forces recalculation
    resp = await svc.get_or_generate_score(db, user_id, force_recalculate=True)
    return resp.model_dump()

async def orchestrateLearningPlan(
    db: AsyncSession,
    request: LearningPlanRequest
) -> dict:
    svc = LearningPlanService()
    resp = await svc.generate_learning_plan(db, request)
    return resp.model_dump()
