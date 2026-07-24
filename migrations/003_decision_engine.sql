-- =============================================================================
-- Migration: Decision Engine (Sprint 4)
-- Adds tables for proactive career scoring, resume matching, adaptive learning,
-- and structured recommendations.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. career_scores
--    Stores hybrid (deterministic + AI qualitative) readiness scores.
--    History is preserved to track progress over time.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS career_scores (
    id                 SERIAL PRIMARY KEY,
    user_id            VARCHAR(255)  NOT NULL,
    overall_score      FLOAT         NOT NULL,
    technical_score    FLOAT         NOT NULL,
    resume_score       FLOAT         NOT NULL,
    interview_score    FLOAT         NOT NULL,
    market_score       FLOAT         NOT NULL,
    metadata           JSONB         NOT NULL DEFAULT '{}'::JSONB,
    created_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_career_scores_user_id ON career_scores (user_id);


-- ---------------------------------------------------------------------------
-- 2. recommendations
--    Stores structured AI recommendations (skills, projects, etc.)
--    Includes confidence and reasoning for explainability.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommendations (
    id                         SERIAL PRIMARY KEY,
    user_id                    VARCHAR(255)  NOT NULL,
    career_matches             JSONB         NOT NULL DEFAULT '[]'::JSONB,
    recommended_skills         JSONB         NOT NULL DEFAULT '[]'::JSONB,
    recommended_projects       JSONB         NOT NULL DEFAULT '[]'::JSONB,
    recommended_certifications JSONB         NOT NULL DEFAULT '[]'::JSONB,
    recommended_opportunities  JSONB         NOT NULL DEFAULT '[]'::JSONB,
    created_at                 TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations (user_id);


-- ---------------------------------------------------------------------------
-- 3. learning_plans
--    Adaptive weekly schedules derived from static roadmaps.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS learning_plans (
    id                 SERIAL PRIMARY KEY,
    user_id            VARCHAR(255)  NOT NULL,
    career_path_id     INTEGER       REFERENCES career_paths(id) ON DELETE CASCADE,
    target_date        TIMESTAMPTZ,
    weekly_study_hours INTEGER,
    plan_data          JSONB         NOT NULL, -- The week-by-week schedule
    created_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_learning_plans_user_id ON learning_plans (user_id);


-- ---------------------------------------------------------------------------
-- 4. resume_matches
--    Semantic match results between a resume and a job description.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS resume_matches (
    id                 SERIAL PRIMARY KEY,
    user_id            VARCHAR(255)  NOT NULL,
    job_description    TEXT          NOT NULL,
    match_score        FLOAT         NOT NULL,
    missing_skills     JSONB         NOT NULL DEFAULT '[]'::JSONB,
    strong_skills      JSONB         NOT NULL DEFAULT '[]'::JSONB,
    weak_areas         JSONB         NOT NULL DEFAULT '[]'::JSONB,
    suggestions        JSONB         NOT NULL DEFAULT '[]'::JSONB,
    created_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_resume_matches_user_id ON resume_matches (user_id);
