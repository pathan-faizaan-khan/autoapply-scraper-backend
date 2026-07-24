-- =============================================================================
-- Migration: Career Guidance Module
-- Applies cleanly on top of the existing AutoApply PostgreSQL schema.
-- Run order: this file must be run ONCE after deploying the new backend code.
-- All statements are idempotent (IF NOT EXISTS / ON CONFLICT).
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. career_paths
--    Master catalogue of career tracks.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS career_paths (
    id            SERIAL PRIMARY KEY,
    title         VARCHAR(255)  NOT NULL UNIQUE,
    description   TEXT,
    avg_salary    VARCHAR(100),                          -- e.g. "$120k – $160k"
    future_scope  TEXT,                                  -- AI-generated market outlook
    is_active     BOOLEAN       NOT NULL DEFAULT TRUE,   -- soft-delete flag
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_career_paths_title    ON career_paths (LOWER(title));
CREATE INDEX IF NOT EXISTS idx_career_paths_active   ON career_paths (is_active);


-- ---------------------------------------------------------------------------
-- 2. roadmap_steps
--    Ordered steps inside a career path.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS roadmap_steps (
    id              SERIAL PRIMARY KEY,
    career_path_id  INTEGER       NOT NULL REFERENCES career_paths(id) ON DELETE CASCADE,
    career_title    VARCHAR(255),                         -- denormalised for convenience
    step_order      INTEGER       NOT NULL,
    title           VARCHAR(255)  NOT NULL,
    description     TEXT,
    difficulty      VARCHAR(50),                          -- beginner|intermediate|advanced
    estimated_days  INTEGER,                              -- suggested completion window
    category        VARCHAR(100),                         -- skill|project|certification|soft-skill
    is_mandatory    BOOLEAN       NOT NULL DEFAULT TRUE,
    resources       JSONB,                                -- [{title, url, type}]
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_roadmap_steps_career_path  ON roadmap_steps (career_path_id);
CREATE INDEX IF NOT EXISTS idx_roadmap_steps_order        ON roadmap_steps (career_path_id, step_order);


-- ---------------------------------------------------------------------------
-- 3. user_progress
--    Per-user completion state for each roadmap step.
--    Unique on (user_id, step_id) to allow upsert.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_progress (
    id                    SERIAL PRIMARY KEY,
    user_id               VARCHAR(255)  NOT NULL,
    step_id               INTEGER       NOT NULL REFERENCES roadmap_steps(id) ON DELETE CASCADE,
    status                VARCHAR(50)   NOT NULL DEFAULT 'not_started',
    -- Allowed values: not_started | in_progress | completed | skipped
    completion_percentage FLOAT         NOT NULL DEFAULT 0.0,
    score                 FLOAT,                          -- optional quiz/assessment result
    notes                 TEXT,
    updated_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_user_progress_user_step UNIQUE (user_id, step_id)
);

CREATE INDEX IF NOT EXISTS idx_user_progress_user_id  ON user_progress (user_id);
CREATE INDEX IF NOT EXISTS idx_user_progress_step_id  ON user_progress (step_id);


-- ---------------------------------------------------------------------------
-- 4. opportunities
--    Cached external opportunities: hackathons, SoC, competitions, etc.
--    Unique on url to allow upsert-based refresh.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS opportunities (
    id           SERIAL PRIMARY KEY,
    title        VARCHAR(255)  NOT NULL,
    organization VARCHAR(255),
    type         VARCHAR(100),                            -- hackathon|summer_of_code|competition|internship
    country      VARCHAR(100),
    deadline     TIMESTAMPTZ,
    url          VARCHAR(555)  UNIQUE,
    description  TEXT,
    tags         JSONB,                                   -- ["Python","ML","Remote"]
    is_active    BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_opportunities_type      ON opportunities (type);
CREATE INDEX IF NOT EXISTS idx_opportunities_active    ON opportunities (is_active);
CREATE INDEX IF NOT EXISTS idx_opportunities_deadline  ON opportunities (deadline);
-- GIN index on JSONB tags for fast tag-overlap queries (?|)
CREATE INDEX IF NOT EXISTS idx_opportunities_tags      ON opportunities USING GIN (tags);


-- ---------------------------------------------------------------------------
-- 5. career_sessions
--    Persistent LLM conversation memory per user.
--    Unique on user_id — one active session per user.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS career_sessions (
    id         SERIAL PRIMARY KEY,
    user_id    VARCHAR(255)  NOT NULL UNIQUE,
    messages   JSONB         NOT NULL DEFAULT '[]'::JSONB,
    -- [{role:"user"|"assistant", content:"..."}]
    metadata   JSONB         NOT NULL DEFAULT '{}'::JSONB,
    -- {career_path, target_role, skill_snapshot, ...}
    created_at TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_career_sessions_user_id ON career_sessions (user_id);


-- ---------------------------------------------------------------------------
-- 6. skill_assessments
--    Point-in-time snapshot of a user's skill gap analysis.
--    Multiple rows per user — history is preserved for trend tracking.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skill_assessments (
    id              SERIAL PRIMARY KEY,
    user_id         VARCHAR(255)  NOT NULL,
    career_path_id  INTEGER       REFERENCES career_paths(id) ON DELETE SET NULL,
    target_role     VARCHAR(255),
    existing_skills JSONB,                               -- ["Python","React",...]
    required_skills JSONB,                               -- skills needed for the role
    missing_skills  JSONB,                               -- gap = required - existing
    gap_score       FLOAT,                               -- 0-100, higher = better match
    recommendations JSONB,                               -- ["Learn Docker","..."]
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_skill_assessments_user_id  ON skill_assessments (user_id);
CREATE INDEX IF NOT EXISTS idx_skill_assessments_role     ON skill_assessments (target_role);
-- GIN indexes for JSONB skill arrays
CREATE INDEX IF NOT EXISTS idx_skill_assessments_existing ON skill_assessments USING GIN (existing_skills);
CREATE INDEX IF NOT EXISTS idx_skill_assessments_missing  ON skill_assessments USING GIN (missing_skills);


-- =============================================================================
-- VERIFICATION QUERY
-- Run after migration to confirm all tables were created:
--
-- SELECT table_name
-- FROM information_schema.tables
-- WHERE table_schema = 'public'
--   AND table_name IN (
--       'career_paths', 'roadmap_steps', 'user_progress',
--       'opportunities', 'career_sessions', 'skill_assessments'
--   )
-- ORDER BY table_name;
-- =============================================================================
