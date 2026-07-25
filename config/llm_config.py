"""
LLM Configuration

Centralised configuration for the reusable LLM layer.
All secrets MUST be loaded from environment variables.

Note on Groq parameters:
  Groq's chat completion API supports: temperature, max_tokens, top_p, stop.
  Parameters like top_k are NOT exposed — they are omitted here to avoid
  false expectations that they are being sent to the provider.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API CONFIGURATION ────────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ─── MODEL SETTINGS ───────────────────────────────────────────────────────────

MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
TOP_P: float = float(os.getenv("LLM_TOP_P", "1.0"))

# ─── EXECUTION SETTINGS ───────────────────────────────────────────────────────

REQUEST_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "30.0"))
MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
JSON_REPAIR_RETRY_COUNT: int = int(os.getenv("LLM_JSON_REPAIR_RETRY_COUNT", "1"))

# ─── DEFAULT PROMPTS ──────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful, intelligent AI assistant. "
    "Provide clear, concise, and accurate answers."
)

# ─── RAG SETTINGS ─────────────────────────────────────────────────────────────

# Number of documents returned by a single retrieve() call.
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

# Minimum cosine similarity score (0.0–1.0) to include a document in results.
# Lower values return more (less relevant) results.
RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))

# Maximum number of characters from a retrieved document included in the LLM context.
RAG_CHUNK_MAX_CHARS: int = int(os.getenv("RAG_CHUNK_MAX_CHARS", "1500"))

# ─── EMBEDDING SETTINGS ───────────────────────────────────────────────────────

# Vector dimension — must match the VECTOR(N) column in the migration.
# Default 384 → sentence-transformers/all-MiniLM-L6-v2
# Use 1536 when switching to OpenAI text-embedding-3-small / ada-002.
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
