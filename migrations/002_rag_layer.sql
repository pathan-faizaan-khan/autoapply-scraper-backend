-- =============================================================================
-- Migration 002: RAG Layer — pgvector + Career Documents
-- Run order: AFTER 001_career_guidance.sql
-- All statements are idempotent (CREATE IF NOT EXISTS, DO $$ blocks).
-- =============================================================================

-- Enable the pgvector extension (idempotent).
-- Supabase enables this by default; this is a no-op if already present.
CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- career_documents
--
-- Stores chunked text documents used by the RAG retrieval pipeline.
-- Supports multiple categories so the same table serves all AI features
-- (career guidance, resume advice, interview prep, etc.).
--
-- Category values (non-exhaustive, enforced by the application layer):
--   career | roadmap | resume | interview | learning | opportunities |
--   certifications | ai_ml | software_engineering | cold_email
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS career_documents (
    id          BIGSERIAL     PRIMARY KEY,
    title       VARCHAR(512)  NOT NULL,
    content     TEXT          NOT NULL,
    source      VARCHAR(512),                           -- URL or filepath of origin
    category    VARCHAR(100)  NOT NULL DEFAULT 'career',
    metadata    JSONB         NOT NULL DEFAULT '{}',    -- arbitrary key-value tags
    embedding   VECTOR(384),                            -- 384-dim: sentence-transformers all-MiniLM-L6-v2
                                                        -- Change to VECTOR(1536) for OpenAI ada-002 / text-embedding-3-small
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_career_documents_title_source UNIQUE (title, source)
);

-- ── Indexes ────────────────────────────────────────────────────────────────

-- Index for fast category + recency queries.
CREATE INDEX IF NOT EXISTS idx_career_documents_category
    ON career_documents (category, created_at DESC);

-- GIN index on metadata JSONB for arbitrary metadata filtering.
CREATE INDEX IF NOT EXISTS idx_career_documents_metadata
    ON career_documents USING GIN (metadata);

-- Full-text search index on content (complement to vector similarity).
CREATE INDEX IF NOT EXISTS idx_career_documents_content_fts
    ON career_documents USING GIN (to_tsvector('english', content));

-- IVFFlat vector index for ANN (approximate nearest-neighbour) search.
-- lists=100 is appropriate for up to ~1M rows; increase for larger datasets.
-- Only created when the table has data (index requires at least 1 row).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'career_documents'
          AND indexname  = 'idx_career_documents_embedding'
    ) THEN
        -- Use cosine distance, which matches the normalised embeddings.
        EXECUTE '
            CREATE INDEX idx_career_documents_embedding
            ON career_documents
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        ';
    END IF;
EXCEPTION
    WHEN undefined_object THEN
        -- vector extension not yet installed; skip gracefully.
        NULL;
END $$;


-- =============================================================================
-- VERIFICATION
--
-- SELECT table_name, column_name, data_type
-- FROM information_schema.columns
-- WHERE table_name = 'career_documents'
-- ORDER BY ordinal_position;
-- =============================================================================
