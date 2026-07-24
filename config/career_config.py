"""
Career Guidance Module — Central Configuration

All tunable constants for the Career Guidance module live here.
Importing from a single location makes future changes trivially easy
and prevents magic numbers from scattering across the codebase.

Environment variables override defaults where applicable.
"""

import os


# ─── PAGINATION ───────────────────────────────────────────────────────────────

# Default and maximum number of items returned in paginated list endpoints.
DEFAULT_PAGE_SIZE: int     = 20
MAX_PAGE_SIZE: int         = 100
DEFAULT_PATHS_PAGE_SIZE: int = 50
MAX_PATHS_PAGE_SIZE: int   = 200


# ─── ROADMAP ──────────────────────────────────────────────────────────────────

# Default roadmap completion target when the user does not specify one.
DEFAULT_ROADMAP_DURATION_DAYS: int = 90

# Default experience level assumed when the user does not specify one.
DEFAULT_EXPERIENCE_LEVEL: str = "junior"

# Maximum number of AI-generated roadmap steps stored per career path.
MAX_ROADMAP_STEPS: int = 20


# ─── SKILL GAP ────────────────────────────────────────────────────────────────

# Maximum number of skill recommendations returned to the user.
MAX_SKILL_RECOMMENDATIONS: int = 5

# Minimum gap_score (0-100) to be considered "well-matched" for a role.
GOOD_GAP_SCORE_THRESHOLD: float = 70.0


# ─── CHAT / SESSION ───────────────────────────────────────────────────────────

# Number of most-recent messages passed to the LLM as context.
# Older messages beyond this window are dropped before sending to the API.
CHAT_CONTEXT_WINDOW: int = 20

# Default follow-up suggestions shown to the user when no LLM is wired.
DEFAULT_CHAT_SUGGESTIONS: list[str] = [
    "Generate my career roadmap",
    "Analyse my skill gap",
    "Show me hackathons to join",
]


# ─── OPPORTUNITIES ────────────────────────────────────────────────────────────

# Valid opportunity type values — used for validation and documentation.
OPPORTUNITY_TYPES: list[str] = [
    "hackathon",
    "summer_of_code",
    "competition",
    "internship",
    "grant",
    "open_source",
]

# Valid status values for user_progress rows.
PROGRESS_STATUSES: list[str] = [
    "not_started",
    "in_progress",
    "completed",
    "skipped",
]


# ─── DATABASE ─────────────────────────────────────────────────────────────────

# Connection pool configuration (shared with db/session.py).
DB_POOL_SIZE: int     = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW: int  = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_ECHO: bool         = os.getenv("DB_ECHO", "false").lower() == "true"
