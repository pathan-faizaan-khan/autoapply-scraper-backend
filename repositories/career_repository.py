"""
Career Guidance Module — Repository Layer

Provides a clean data-access interface over PostgreSQL for all
career-related tables.  Follows the same raw-SQLAlchemy-text approach
used in scraper/playwright_scraper.py so the connection pool and session
factory are re-used without introducing a second engine.

All public methods are async and accept a SQLAlchemy AsyncSession,
which is created by the session factory in db/session.py.
"""

import json
from datetime import datetime
from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


# ─── CAREER PATHS ─────────────────────────────────────────────────────────────

async def getCareerPath(
    session: AsyncSession,
    *,
    path_id:   Optional[int]  = None,
    title:     Optional[str]  = None,
    active_only: bool = True,
) -> Optional[dict]:
    """
    Fetch a single career path by id or (exact) title.
    Returns a plain dict or None when not found.

    Args:
        session:     Async DB session.
        path_id:     PK of the career_paths row.
        title:       Exact title match (case-insensitive).
        active_only: When True, exclude soft-deleted rows.
    """
    where_clauses = []
    params: dict = {}

    if path_id is not None:
        where_clauses.append("id = :path_id")
        params["path_id"] = path_id

    if title is not None:
        where_clauses.append("LOWER(title) = LOWER(:title)")
        params["title"] = title

    if active_only:
        where_clauses.append("is_active = TRUE")

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    sql = text(f"SELECT * FROM career_paths WHERE {where_sql} LIMIT 1")

    try:
        result = await session.execute(sql, params)
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[CareerRepository] getCareerPath error: {exc}")
        return None


async def listCareerPaths(
    session: AsyncSession,
    *,
    active_only: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> List[dict]:
    """
    Return all career paths, optionally filtering soft-deleted rows.
    """
    active_filter = "WHERE is_active = TRUE" if active_only else ""
    sql = text(
        f"SELECT * FROM career_paths {active_filter} "
        f"ORDER BY title ASC LIMIT :limit OFFSET :offset"
    )
    try:
        result = await session.execute(sql, {"limit": limit, "offset": offset})
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        print(f"[CareerRepository] listCareerPaths error: {exc}")
        return []


async def upsertCareerPath(
    session: AsyncSession,
    *,
    title: str,
    description: Optional[str] = None,
    avg_salary:  Optional[str] = None,
    future_scope: Optional[str] = None,
) -> Optional[dict]:
    """
    Insert a new career path or update description/salary/scope when the
    title already exists.  Returns the full row after upsert.
    """
    sql = text("""
        INSERT INTO career_paths (title, description, avg_salary, future_scope,
                                  is_active, created_at, updated_at)
        VALUES (:title, :description, :avg_salary, :future_scope,
                TRUE, NOW(), NOW())
        ON CONFLICT (title) DO UPDATE
            SET description  = EXCLUDED.description,
                avg_salary   = EXCLUDED.avg_salary,
                future_scope = EXCLUDED.future_scope,
                updated_at   = NOW()
        RETURNING *
    """)
    try:
        result = await session.execute(sql, {
            "title":        title,
            "description":  description,
            "avg_salary":   avg_salary,
            "future_scope": future_scope,
        })
        await session.commit()
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] upsertCareerPath error: {exc}")
        return None


# ─── ROADMAP STEPS ────────────────────────────────────────────────────────────

async def getRoadmap(
    session: AsyncSession,
    career_path_id: int,
) -> List[dict]:
    """
    Fetch all roadmap steps for a given career path, ordered by step_order.

    Args:
        session:        Async DB session.
        career_path_id: FK reference to career_paths.id.

    Returns:
        List of step dicts ordered by step_order ASC.
    """
    sql = text("""
        SELECT * FROM roadmap_steps
        WHERE career_path_id = :career_path_id
        ORDER BY step_order ASC
    """)
    try:
        result = await session.execute(sql, {"career_path_id": career_path_id})
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        print(f"[CareerRepository] getRoadmap error: {exc}")
        return []


async def bulkInsertRoadmapSteps(
    session: AsyncSession,
    career_path_id: int,
    career_title: str,
    steps: List[dict],
) -> bool:
    """
    Bulk-insert roadmap steps generated by the AI service.
    Clears existing steps for the same career_path_id before inserting
    to allow full regeneration.

    Args:
        session:        Async DB session.
        career_path_id: FK reference to career_paths.id.
        career_title:   Denormalised title (stored on each step row).
        steps:          List of step dicts matching roadmap_steps columns.

    Returns:
        True on success, False on failure.
    """
    try:
        # Clear old steps for this path before inserting new ones
        await session.execute(
            text("DELETE FROM roadmap_steps WHERE career_path_id = :cpid"),
            {"cpid": career_path_id},
        )

        insert_sql = text("""
            INSERT INTO roadmap_steps
                (career_path_id, career_title, step_order, title, description,
                 difficulty, estimated_days, category, is_mandatory, resources, created_at)
            VALUES
                (:career_path_id, :career_title, :step_order, :title, :description,
                 :difficulty, :estimated_days, :category, :is_mandatory, :resources, NOW())
        """)

        for step in steps:
            await session.execute(insert_sql, {
                "career_path_id": career_path_id,
                "career_title":   career_title,
                "step_order":     step.get("step_order", 1),
                "title":          step.get("title", ""),
                "description":    step.get("description"),
                "difficulty":     step.get("difficulty"),
                "estimated_days": step.get("estimated_days"),
                "category":       step.get("category"),
                "is_mandatory":   step.get("is_mandatory", True),
                "resources":      json.dumps(step.get("resources") or []),
            })

        await session.commit()
        return True
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] bulkInsertRoadmapSteps error: {exc}")
        return False


# ─── USER PROGRESS ────────────────────────────────────────────────────────────

async def saveProgress(
    session: AsyncSession,
    *,
    user_id:               str,
    step_id:               int,
    status:                str,
    completion_percentage: float = 0.0,
    score:                 Optional[float] = None,
    notes:                 Optional[str] = None,
) -> Optional[dict]:
    """
    Upsert a user's progress record for a single roadmap step.

    Args:
        session:               Async DB session.
        user_id:               Application-level user identifier (UUID string).
        step_id:               FK to roadmap_steps.id.
        status:                Lifecycle status string.
        completion_percentage: 0.0 – 100.0.
        score:                 Optional quiz/assessment score.
        notes:                 User's free-text notes.

    Returns:
        The saved row as a dict, or None on failure.
    """
    sql = text("""
        INSERT INTO user_progress
            (user_id, step_id, status, completion_percentage, score, notes, updated_at)
        VALUES
            (:user_id, :step_id, :status, :completion_percentage, :score, :notes, NOW())
        ON CONFLICT (user_id, step_id) DO UPDATE
            SET status                = EXCLUDED.status,
                completion_percentage = EXCLUDED.completion_percentage,
                score                 = EXCLUDED.score,
                notes                 = EXCLUDED.notes,
                updated_at            = NOW()
        RETURNING *
    """)
    try:
        result = await session.execute(sql, {
            "user_id":               user_id,
            "step_id":               step_id,
            "status":                status,
            "completion_percentage": completion_percentage,
            "score":                 score,
            "notes":                 notes,
        })
        await session.commit()
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] saveProgress error: {exc}")
        return None


async def getProgress(
    session: AsyncSession,
    user_id: str,
    career_path_id: Optional[int] = None,
) -> List[dict]:
    """
    Retrieve all progress rows for a user, optionally scoped to one career path.

    Args:
        session:        Async DB session.
        user_id:        Application-level user identifier.
        career_path_id: When supplied, only returns steps belonging to that path.

    Returns:
        List of progress row dicts joined with step metadata.
    """
    if career_path_id is not None:
        sql = text("""
            SELECT up.*, rs.title as step_title, rs.step_order, rs.difficulty,
                   rs.estimated_days, rs.career_path_id
            FROM user_progress up
            JOIN roadmap_steps rs ON rs.id = up.step_id
            WHERE up.user_id = :user_id
              AND rs.career_path_id = :career_path_id
            ORDER BY rs.step_order ASC
        """)
        params = {"user_id": user_id, "career_path_id": career_path_id}
    else:
        sql = text("""
            SELECT up.*, rs.title as step_title, rs.step_order, rs.difficulty,
                   rs.estimated_days, rs.career_path_id
            FROM user_progress up
            JOIN roadmap_steps rs ON rs.id = up.step_id
            WHERE up.user_id = :user_id
            ORDER BY rs.career_path_id ASC, rs.step_order ASC
        """)
        params = {"user_id": user_id}

    try:
        result = await session.execute(sql, params)
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        print(f"[CareerRepository] getProgress error: {exc}")
        return []


# ─── OPPORTUNITIES ────────────────────────────────────────────────────────────

async def saveOpportunity(
    session: AsyncSession,
    *,
    title:        str,
    organization: Optional[str]      = None,
    opp_type:     Optional[str]      = None,
    country:      Optional[str]      = None,
    deadline:     Optional[datetime] = None,
    url:          Optional[str]      = None,
    description:  Optional[str]      = None,
    tags:         Optional[list]     = None,
) -> Optional[dict]:
    """
    Upsert a single opportunity (deduplicated by URL).
    The parameter is named `opp_type` (not `type`) to avoid shadowing
    the Python builtin.

    Returns:
        The saved/updated row as a dict, or None on failure.
    """
    sql = text("""
        INSERT INTO opportunities
            (title, organization, type, country, deadline, url,
             description, tags, is_active, created_at, updated_at)
        VALUES
            (:title, :organization, :opp_type, :country, :deadline, :url,
             :description, :tags, TRUE, NOW(), NOW())
        ON CONFLICT (url) DO UPDATE
            SET title        = EXCLUDED.title,
                organization = EXCLUDED.organization,
                type         = EXCLUDED.type,
                country      = EXCLUDED.country,
                deadline     = EXCLUDED.deadline,
                description  = EXCLUDED.description,
                tags         = EXCLUDED.tags,
                is_active    = TRUE,
                updated_at   = NOW()
        RETURNING *
    """)
    try:
        result = await session.execute(sql, {
            "title":        title,
            "organization": organization,
            "opp_type":     opp_type,
            "country":      country,
            "deadline":     deadline,
            "url":          url,
            "description":  description,
            "tags":         json.dumps(tags or []),
        })
        await session.commit()
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] saveOpportunity error: {exc}")
        return None


async def getOpportunities(
    session: AsyncSession,
    *,
    opp_type:    Optional[str]       = None,
    tags:        Optional[List[str]] = None,
    country:     Optional[str]       = None,
    active_only: bool                = True,
    limit:       int                 = 20,
    offset:      int                 = 0,
) -> tuple[List[dict], int]:
    """
    Fetch paginated opportunities from the cache.
    The parameter is named `opp_type` (not `type`) to avoid shadowing
    the Python builtin.

    Args:
        session:     Async DB session.
        opp_type:    Filter by opportunity type.
        tags:        Filter by any matching tag (PostgreSQL JSONB overlap).
        country:     Filter by country (case-insensitive contains).
        active_only: Exclude expired / removed opportunities.
        limit:       Page size.
        offset:      Pagination offset.

    Returns:
        (rows, total_count) tuple.
    """
    where_clauses = []
    params: dict = {"limit": limit, "offset": offset}

    if active_only:
        where_clauses.append("is_active = TRUE")

    if opp_type:
        where_clauses.append("type = :opp_type")
        params["opp_type"] = opp_type

    if country:
        where_clauses.append("LOWER(country) LIKE LOWER(:country)")
        params["country"] = f"%{country}%"

    # JSONB tag overlap — use jsonb_exists_any() which works correctly with
    # asyncpg's parameter binding (the ?| operator requires special escaping).
    if tags:
        where_clauses.append("jsonb_exists_any(tags, :tags)")
        params["tags"] = tags  # asyncpg accepts a Python list for text[] params

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    count_sql = text(f"SELECT COUNT(*) FROM opportunities WHERE {where_sql}")
    data_sql  = text(
        f"SELECT * FROM opportunities WHERE {where_sql} "
        f"ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
    )

    try:
        count_result = await session.execute(count_sql, params)
        total = count_result.scalar() or 0

        data_result = await session.execute(data_sql, params)
        rows = [dict(r) for r in data_result.mappings().all()]
        return rows, total
    except Exception as exc:
        print(f"[CareerRepository] getOpportunities error: {exc}")
        return [], 0


# ─── CAREER SESSIONS (MEMORY) ─────────────────────────────────────────────────

async def getSession(
    session: AsyncSession,
    user_id: str,
) -> Optional[dict]:
    """Retrieve the conversation session for a user."""
    sql = text("SELECT * FROM career_sessions WHERE user_id = :user_id LIMIT 1")
    try:
        result = await session.execute(sql, {"user_id": user_id})
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[CareerRepository] getSession error: {exc}")
        return None


async def upsertSession(
    session: AsyncSession,
    *,
    user_id:   str,
    messages:  list,
    metadata:  Optional[dict] = None,
) -> Optional[dict]:
    """
    Upsert conversation history for a user.
    Existing message list is REPLACED (caller manages appending).
    """
    sql = text("""
        INSERT INTO career_sessions (user_id, messages, metadata, created_at, updated_at)
        VALUES (:user_id, :messages, :metadata, NOW(), NOW())
        ON CONFLICT (user_id) DO UPDATE
            SET messages   = EXCLUDED.messages,
                metadata   = EXCLUDED.metadata,
                updated_at = NOW()
        RETURNING *
    """)
    try:
        result = await session.execute(sql, {
            "user_id":  user_id,
            "messages": json.dumps(messages),
            "metadata": json.dumps(metadata or {}),
        })
        await session.commit()
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] upsertSession error: {exc}")
        return None


# ─── SKILL ASSESSMENTS ────────────────────────────────────────────────────────

async def saveSkillAssessment(
    session: AsyncSession,
    *,
    user_id:         str,
    target_role:     str,
    career_path_id:  Optional[int]   = None,
    existing_skills: Optional[list]  = None,
    required_skills: Optional[list]  = None,
    missing_skills:  Optional[list]  = None,
    gap_score:       Optional[float] = None,
    recommendations: Optional[list]  = None,
) -> Optional[dict]:
    """
    Persist the result of a skill gap analysis.
    A new row is always inserted (history is preserved).
    """
    sql = text("""
        INSERT INTO skill_assessments
            (user_id, career_path_id, target_role, existing_skills, required_skills,
             missing_skills, gap_score, recommendations, created_at)
        VALUES
            (:user_id, :career_path_id, :target_role, :existing_skills, :required_skills,
             :missing_skills, :gap_score, :recommendations, NOW())
        RETURNING *
    """)
    try:
        result = await session.execute(sql, {
            "user_id":         user_id,
            "career_path_id":  career_path_id,
            "target_role":     target_role,
            "existing_skills": json.dumps(existing_skills or []),
            "required_skills": json.dumps(required_skills or []),
            "missing_skills":  json.dumps(missing_skills or []),
            "gap_score":       gap_score,
            "recommendations": json.dumps(recommendations or []),
        })
        await session.commit()
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] saveSkillAssessment error: {exc}")
        return None


async def getLatestSkillAssessment(
    session: AsyncSession,
    user_id: str,
    target_role: Optional[str] = None,
) -> Optional[dict]:
    """
    Retrieve the most recent skill assessment for a user,
    optionally filtered by target_role.
    """
    if target_role:
        sql = text("""
            SELECT * FROM skill_assessments
            WHERE user_id = :user_id AND target_role = :target_role
            ORDER BY created_at DESC LIMIT 1
        """)
        params = {"user_id": user_id, "target_role": target_role}
    else:
        sql = text("""
            SELECT * FROM skill_assessments
            WHERE user_id = :user_id
            ORDER BY created_at DESC LIMIT 1
        """)
        params = {"user_id": user_id}

    try:
        result = await session.execute(sql, params)
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[CareerRepository] getLatestSkillAssessment error: {exc}")
        return None


# ─── SPRINT 4: DECISION ENGINE TABLES ─────────────────────────────────────────

async def saveCareerScore(
    session: AsyncSession,
    *,
    user_id:         str,
    overall_score:   float,
    technical_score: float,
    resume_score:    float,
    interview_score: float,
    market_score:    float,
    metadata:        Optional[dict] = None,
) -> Optional[dict]:
    sql = text("""
        INSERT INTO career_scores
            (user_id, overall_score, technical_score, resume_score, interview_score, market_score, metadata, created_at)
        VALUES
            (:user_id, :overall_score, :technical_score, :resume_score, :interview_score, :market_score, :metadata, NOW())
        RETURNING *
    """)
    try:
        result = await session.execute(sql, {
            "user_id": user_id,
            "overall_score": overall_score,
            "technical_score": technical_score,
            "resume_score": resume_score,
            "interview_score": interview_score,
            "market_score": market_score,
            "metadata": json.dumps(metadata or {}),
        })
        await session.commit()
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] saveCareerScore error: {exc}")
        return None

async def getLatestCareerScore(session: AsyncSession, user_id: str) -> Optional[dict]:
    sql = text("SELECT * FROM career_scores WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 1")
    try:
        result = await session.execute(sql, {"user_id": user_id})
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[CareerRepository] getLatestCareerScore error: {exc}")
        return None


async def saveRecommendation(
    session: AsyncSession,
    *,
    user_id:                    str,
    career_matches:             list,
    recommended_skills:         list,
    recommended_projects:       list,
    recommended_certifications: list,
    recommended_opportunities:  list,
) -> Optional[dict]:
    sql = text("""
        INSERT INTO recommendations
            (user_id, career_matches, recommended_skills, recommended_projects, recommended_certifications, recommended_opportunities, created_at)
        VALUES
            (:user_id, :cm, :rs, :rp, :rc, :ro, NOW())
        RETURNING *
    """)
    try:
        result = await session.execute(sql, {
            "user_id": user_id,
            "cm": json.dumps(career_matches),
            "rs": json.dumps(recommended_skills),
            "rp": json.dumps(recommended_projects),
            "rc": json.dumps(recommended_certifications),
            "ro": json.dumps(recommended_opportunities),
        })
        await session.commit()
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] saveRecommendation error: {exc}")
        return None

async def getLatestRecommendation(session: AsyncSession, user_id: str) -> Optional[dict]:
    sql = text("SELECT * FROM recommendations WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 1")
    try:
        result = await session.execute(sql, {"user_id": user_id})
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[CareerRepository] getLatestRecommendation error: {exc}")
        return None


async def saveLearningPlan(
    session: AsyncSession,
    *,
    user_id:            str,
    career_path_id:     Optional[int],
    target_date:        Optional[datetime],
    weekly_study_hours: int,
    plan_data:          list,
) -> Optional[dict]:
    sql = text("""
        INSERT INTO learning_plans
            (user_id, career_path_id, target_date, weekly_study_hours, plan_data, created_at, updated_at)
        VALUES
            (:user_id, :career_path_id, :target_date, :weekly_study_hours, :plan_data, NOW(), NOW())
        RETURNING *
    """)
    try:
        result = await session.execute(sql, {
            "user_id": user_id,
            "career_path_id": career_path_id,
            "target_date": target_date,
            "weekly_study_hours": weekly_study_hours,
            "plan_data": json.dumps(plan_data),
        })
        await session.commit()
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] saveLearningPlan error: {exc}")
        return None

async def getLatestLearningPlan(session: AsyncSession, user_id: str) -> Optional[dict]:
    sql = text("SELECT * FROM learning_plans WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 1")
    try:
        result = await session.execute(sql, {"user_id": user_id})
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[CareerRepository] getLatestLearningPlan error: {exc}")
        return None


async def saveResumeMatch(
    session: AsyncSession,
    *,
    user_id:         str,
    job_description: str,
    match_score:     float,
    missing_skills:  list,
    strong_skills:   list,
    weak_areas:      list,
    suggestions:     list,
) -> Optional[dict]:
    sql = text("""
        INSERT INTO resume_matches
            (user_id, job_description, match_score, missing_skills, strong_skills, weak_areas, suggestions, created_at)
        VALUES
            (:user_id, :job_description, :match_score, :missing_skills, :strong_skills, :weak_areas, :suggestions, NOW())
        RETURNING *
    """)
    try:
        result = await session.execute(sql, {
            "user_id": user_id,
            "job_description": job_description,
            "match_score": match_score,
            "missing_skills": json.dumps(missing_skills),
            "strong_skills": json.dumps(strong_skills),
            "weak_areas": json.dumps(weak_areas),
            "suggestions": json.dumps(suggestions),
        })
        await session.commit()
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        await session.rollback()
        print(f"[CareerRepository] saveResumeMatch error: {exc}")
        return None

async def getLatestResumeMatch(session: AsyncSession, user_id: str) -> Optional[dict]:
    sql = text("SELECT * FROM resume_matches WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 1")
    try:
        result = await session.execute(sql, {"user_id": user_id})
        row = result.mappings().first()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[CareerRepository] getLatestResumeMatch error: {exc}")
        return None
