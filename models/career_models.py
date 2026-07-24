"""
Career Guidance Module — SQLAlchemy ORM Models

Tables:
  - career_paths       : Master catalogue of career tracks
  - roadmap_steps      : Ordered steps belonging to a career path
  - user_progress      : Per-user completion state for each roadmap step
  - opportunities      : Cached hackathons, SoC programmes, competitions, etc.
  - career_sessions    : Persistent LLM conversation memory per user
  - skill_assessments  : Snapshot of a user's self-reported / AI-assessed skills
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ─── career_paths ─────────────────────────────────────────────────────────────

class CareerPath(Base):
    """
    Master catalogue of career tracks (e.g. "Full-Stack Engineer",
    "ML Engineer", "DevOps Engineer").
    AI-generated or manually seeded.
    """
    __tablename__ = "career_paths"

    id            = Column(Integer, primary_key=True, index=True)
    title         = Column(String(255), nullable=False, unique=True, index=True)
    description   = Column(Text, nullable=True)
    avg_salary    = Column(String(100), nullable=True)   # e.g. "$120k – $160k"
    future_scope  = Column(Text, nullable=True)          # AI-generated market outlook
    is_active     = Column(Boolean, default=True)        # soft-delete flag
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    steps = relationship("RoadmapStep", back_populates="career_path",
                         cascade="all, delete-orphan", lazy="select")

    def __repr__(self) -> str:
        return f"<CareerPath id={self.id} title='{self.title}'>"


# ─── roadmap_steps ────────────────────────────────────────────────────────────

class RoadmapStep(Base):
    """
    Individual ordered steps inside a career path roadmap.
    Describes what the candidate must learn / do to progress.
    """
    __tablename__ = "roadmap_steps"

    id              = Column(Integer, primary_key=True, index=True)
    career_path_id  = Column(Integer, ForeignKey("career_paths.id",
                             ondelete="CASCADE"), nullable=False, index=True)

    # Denormalised title for convenience queries (avoids JOIN for display)
    career_title    = Column(String(255), nullable=True)

    step_order      = Column(Integer, nullable=False)       # 1-based ordering
    title           = Column(String(255), nullable=False)
    description     = Column(Text, nullable=True)
    difficulty      = Column(String(50), nullable=True)     # beginner|intermediate|advanced
    estimated_days  = Column(Integer, nullable=True)        # suggested completion window
    category        = Column(String(100), nullable=True)    # skill|project|certification|soft-skill
    is_mandatory    = Column(Boolean, default=True)         # optional vs required steps
    resources       = Column(JSON, nullable=True)           # [{title, url, type}]
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Relationships
    career_path = relationship("CareerPath", back_populates="steps")
    progress    = relationship("UserProgress", back_populates="step",
                               cascade="all, delete-orphan", lazy="select")

    def __repr__(self) -> str:
        return (f"<RoadmapStep id={self.id} career_path_id={self.career_path_id} "
                f"order={self.step_order} title='{self.title}'>")


# ─── user_progress ────────────────────────────────────────────────────────────

class UserProgress(Base):
    """
    Tracks each user's completion state against individual roadmap steps.
    One row per (user_id, step_id) pair.
    """
    __tablename__ = "user_progress"

    id                    = Column(Integer, primary_key=True, index=True)
    user_id               = Column(String(255), nullable=False, index=True)  # UUID string
    step_id               = Column(Integer, ForeignKey("roadmap_steps.id",
                                   ondelete="CASCADE"), nullable=False, index=True)
    status                = Column(String(50), default="not_started")
    # not_started | in_progress | completed | skipped
    completion_percentage = Column(Float, default=0.0)    # 0.0 – 100.0
    score                 = Column(Float, nullable=True)  # optional quiz / assessment score
    notes                 = Column(Text, nullable=True)   # user notes on the step
    updated_at            = Column(DateTime, default=datetime.utcnow,
                                   onupdate=datetime.utcnow)

    # Relationships
    step = relationship("RoadmapStep", back_populates="progress")

    def __repr__(self) -> str:
        return (f"<UserProgress id={self.id} user_id='{self.user_id}' "
                f"step_id={self.step_id} status='{self.status}'>")


# ─── opportunities ────────────────────────────────────────────────────────────

class Opportunity(Base):
    """
    Cached external opportunities: hackathons, GSoC/MLH/competitions,
    open-source programmes, internships.
    Populated by opportunity_service.py scraper.
    """
    __tablename__ = "opportunities"

    id           = Column(Integer, primary_key=True, index=True)
    title        = Column(String(255), nullable=False)
    organization = Column(String(255), nullable=True)
    type         = Column(String(100), nullable=True, index=True)
    # hackathon | summer_of_code | competition | grant | internship | open_source
    country      = Column(String(100), nullable=True)
    deadline     = Column(DateTime, nullable=True)
    url          = Column(String(555), nullable=True, unique=True)
    description  = Column(Text, nullable=True)
    tags         = Column(JSON, nullable=True)     # ["Python", "ML", "Remote"]
    is_active    = Column(Boolean, default=True)   # expired / removed flag
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Opportunity id={self.id} title='{self.title}' type='{self.type}'>"


# ─── career_sessions ──────────────────────────────────────────────────────────

class CareerSession(Base):
    """
    Persistent chat/memory store for the AI Career Guidance agent.
    Stores conversation history (messages) as JSON so the LLM
    can maintain context across requests without a vector store.
    """
    __tablename__ = "career_sessions"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(String(255), nullable=False, index=True, unique=True)
    messages     = Column(JSON, nullable=True, default=list)
    # [{role: "user"|"assistant", content: "..."}]
    metadata     = Column(JSON, nullable=True, default=dict)
    # {career_path, current_step, skill_snapshot, ...}
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CareerSession id={self.id} user_id='{self.user_id}'>"


# ─── skill_assessments ────────────────────────────────────────────────────────

class SkillAssessment(Base):
    """
    Point-in-time snapshot of a user's skill proficiency levels.
    Created each time the user runs a skill gap analysis.
    Used to track improvement over time.
    """
    __tablename__ = "skill_assessments"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(String(255), nullable=False, index=True)
    career_path_id  = Column(Integer, ForeignKey("career_paths.id",
                             ondelete="SET NULL"), nullable=True)
    target_role     = Column(String(255), nullable=True)
    existing_skills = Column(JSON, nullable=True)   # ["Python", "React", ...]
    required_skills = Column(JSON, nullable=True)   # skills needed for target role
    missing_skills  = Column(JSON, nullable=True)   # gap = required - existing
    gap_score       = Column(Float, nullable=True)  # 0-100, 100 = perfect match
    recommendations = Column(JSON, nullable=True)   # AI-suggested next steps
    created_at      = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (f"<SkillAssessment id={self.id} user_id='{self.user_id}' "
                f"gap_score={self.gap_score}>")
