"""
Career Guidance Module — Pydantic Schemas (Request / Response)

Follows the same pattern as the existing Pydantic models in
routers/jobs_search.py and routers/ml_autofill.py.
All fields are typed; optional fields default to None or sensible defaults.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ─── SHARED / PRIMITIVE SCHEMAS ───────────────────────────────────────────────

class ResourceItem(BaseModel):
    """A single learning resource attached to a roadmap step."""
    title: str
    url:   str
    type:  Optional[str] = None   # article | video | course | book | tool


# ─── REQUEST SCHEMAS ──────────────────────────────────────────────────────────

class CareerChatRequest(BaseModel):
    """
    POST /api/career/chat

    The frontend sends the user's free-text message plus a minimal
    context payload so the agent can personalise its response without
    a vector store (context window approach).
    """
    user_id:      str                          = Field(..., description="Unique user identifier")
    message:      str                          = Field(..., description="User's chat message")
    # Optional lightweight context — avoids a DB round-trip on every message
    resume_data:  Optional[dict]               = Field(None,  description="Parsed resume JSON")
    target_role:  Optional[str]                = Field(None,  description="User's target role")
    career_path:  Optional[str]                = Field(None,  description="Currently selected career path title")


class RoadmapRequest(BaseModel):
    """
    POST /api/career/generate-roadmap

    Provides enough context for the AI to generate an ordered,
    personalised learning roadmap.
    """
    user_id:            str                    = Field(..., description="Unique user identifier")
    target_role:        str                    = Field(..., description="The role the user wants to reach (e.g. ML Engineer)")
    current_skills:     List[str]              = Field(default_factory=list, description="Skills the user already has")
    experience_level:   Optional[str]          = Field("junior", description="junior | mid | senior")
    preferred_duration: Optional[int]          = Field(90, description="Target completion in days")
    learning_style:     Optional[str]          = Field(None,  description="video | reading | project | mixed")


class SkillGapRequest(BaseModel):
    """
    POST /api/career/skill-gap

    Compares the user's current skills against the requirements
    of a target role and returns a structured gap analysis.
    """
    user_id:         str                       = Field(..., description="Unique user identifier")
    target_role:     str                       = Field(..., description="Desired job role")
    current_skills:  List[str]                 = Field(default_factory=list, description="User's existing skills")
    resume_data:     Optional[dict]            = Field(None,  description="Full parsed resume for deeper analysis")
    save_assessment: Optional[bool]            = Field(True,  description="Persist the result to skill_assessments table")



# ─── RESPONSE SCHEMAS ─────────────────────────────────────────────────────────

class CareerChatResponse(BaseModel):
    """
    Response envelope for POST /api/career/chat
    """
    user_id:    str
    reply:      str                                         # AI-generated reply
    suggestions: Optional[List[str]] = None                 # Follow-up action suggestions
    session_id: Optional[int]        = None                 # DB id of the career_session row


class RoadmapStepResponse(BaseModel):
    """A single step returned inside a RoadmapResponse."""
    id:              int
    step_order:      int
    title:           str
    description:     Optional[str]     = None
    difficulty:      Optional[str]     = None
    estimated_days:  Optional[int]     = None
    category:        Optional[str]     = None
    is_mandatory:    bool              = True
    resources:       Optional[List[ResourceItem]] = None

    class Config:
        from_attributes = True


class RoadmapResponse(BaseModel):
    """
    Response envelope for POST /api/career/generate-roadmap
    """
    user_id:         str
    career_path_id:  Optional[int]             = None
    career_path:     Optional[str]             = None      # title of the career path
    total_steps:     int                       = 0
    estimated_days:  Optional[int]             = None
    steps:           List[RoadmapStepResponse] = Field(default_factory=list)
    message:         Optional[str]             = None      # human-readable summary


class SkillGapResponse(BaseModel):
    """
    Response envelope for POST /api/career/skill-gap
    """
    user_id:            str
    target_role:        str
    existing_skills:    List[str]
    required_skills:    List[str]
    missing_skills:     List[str]
    gap_score:          float                  # 0-100, higher = better match
    assessment_id:      Optional[int]          = None
    recommendations:    Optional[List[str]]    = None
    message:            Optional[str]          = None


class OpportunityResponse(BaseModel):
    """
    A single cached opportunity returned by GET /api/career/opportunities
    """
    id:           int
    title:        str
    organization: Optional[str]      = None
    type:         Optional[str]      = None
    country:      Optional[str]      = None
    deadline:     Optional[datetime] = None
    url:          Optional[str]      = None
    description:  Optional[str]      = None
    tags:         Optional[List[str]] = None
    is_active:    bool               = True
    created_at:   Optional[datetime] = None

    class Config:
        from_attributes = True


class OpportunitiesListResponse(BaseModel):
    """
    Paginated wrapper around a list of OpportunityResponse items.
    """
    total:         int
    limit:         int
    offset:        int
    opportunities: List[OpportunityResponse] = Field(default_factory=list)


# ─── PROGRESS SCHEMAS ─────────────────────────────────────────────────────────

class ProgressUpdateRequest(BaseModel):
    """
    POST /api/career/progress/update

    Update a user's status for a specific roadmap step.
    """
    user_id:               str
    step_id:               int
    status:                str    = Field(..., description="not_started | in_progress | completed | skipped")
    completion_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    score:                 Optional[float] = Field(None, ge=0.0, le=100.0)
    notes:                 Optional[str]   = None


class ProgressResponse(BaseModel):
    """
    Response for a user's progress on a specific step.
    """
    id:                    int
    user_id:               str
    step_id:               int
    status:                str
    completion_percentage: float
    score:                 Optional[float] = None
    notes:                 Optional[str]   = None
    updated_at:            Optional[datetime] = None

    class Config:
        from_attributes = True


class UserProgressSummaryResponse(BaseModel):
    """
    Aggregated progress across an entire roadmap for a user.
    """
    user_id:               str
    career_path:           Optional[str]  = None
    total_steps:           int
    completed_steps:       int
    in_progress_steps:     int
    overall_percentage:    float
    step_details:          List[ProgressResponse] = Field(default_factory=list)
