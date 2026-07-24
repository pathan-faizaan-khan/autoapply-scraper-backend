"""
Embedding Service

Provides a clean, provider-agnostic interface for generating text embeddings.

Current implementation: Groq does not offer an embeddings API.
We default to a lightweight sentence-transformers model running locally via
the `sentence-transformers` library (all-MiniLM-L6-v2, 384 dimensions).

The EmbeddingProvider protocol is intentionally designed so that swapping to
OpenAI, Cohere, or any other provider requires only a new Provider class —
no changes to RAGService, ContextBuilder, or any caller.

Dimension configuration:
  EMBEDDING_DIMENSIONS in config/llm_config.py controls the vector size.
  The migration uses VECTOR(1536) which covers OpenAI ada-002.
  sentence-transformers/all-MiniLM-L6-v2 produces 384-dim vectors.
  If you switch providers, update the VECTOR(...) column and re-index.
"""

import logging
import os
from typing import List, Protocol, runtime_checkable

logger = logging.getLogger("embedding_service")


# ─── PROVIDER PROTOCOL ────────────────────────────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Abstract contract for any embedding provider.

    Implementing this protocol (via duck typing) allows the EmbeddingService
    to swap providers at runtime without changing any downstream code.
    """

    async def embed(self, text: str) -> List[float]:
        """Generate a single embedding vector for `text`."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of texts."""
        ...

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the output vectors."""
        ...


# ─── LOCAL PROVIDER (sentence-transformers) ───────────────────────────────────

class LocalEmbeddingProvider:
    """
    Embedding provider using sentence-transformers running locally.

    Model: all-MiniLM-L6-v2 (384 dimensions, ~80 MB, fast CPU inference).
    No API key required.

    Lazy-loads the model on first use to avoid slowing down startup.
    """

    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    _model = None  # Lazy singleton

    def _get_model(self):
        """Lazily load the sentence-transformers model."""
        if LocalEmbeddingProvider._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                LocalEmbeddingProvider._model = SentenceTransformer(self.MODEL_NAME)
                logger.info(
                    "Loaded sentence-transformers model: %s (%d dims)",
                    self.MODEL_NAME,
                    LocalEmbeddingProvider._model.get_sentence_embedding_dimension(),
                )
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
        return LocalEmbeddingProvider._model

    @property
    def dimensions(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    async def embed(self, text: str) -> List[float]:
        """
        Encode a single text string.

        sentence-transformers encode() is synchronous (CPU/GPU bound).
        We call it directly — for production workloads with many concurrent
        requests, wrap in asyncio.run_in_executor for true async behaviour.
        """
        model = self._get_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts in a single forward pass."""
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]


# ─── GEMINI-COMPATIBLE PROVIDER (optional, activated via env var) ─────────────

class GeminiEmbeddingProvider:
    """
    Embedding provider using the Gemini Embeddings API (models/text-embedding-004).

    Activated when GEMINI_API_KEY is set in the environment.
    Requires: pip install google-genai
    """

    MODEL_NAME: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    _client = None

    @classmethod
    def _ensure_configured(cls):
        if cls._client is None:
            try:
                from google import genai
                api_key = os.getenv("GEMINI_API_KEY", "")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY is not set.")
                cls._client = genai.Client(api_key=api_key)
            except ImportError as exc:
                raise ImportError(
                    "google-genai package is required for Gemini embeddings. "
                    "Install it with: pip install google-genai"
                ) from exc

    @property
    def dimensions(self) -> int:
        return 768

    async def embed(self, text: str) -> List[float]:
        self._ensure_configured()
        from google.genai import types
        result = self._client.models.embed_content(
            model=self.MODEL_NAME,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return result.embeddings[0].values

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_configured()
        from google.genai import types
        result = self._client.models.embed_content(
            model=self.MODEL_NAME,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return [e.values for e in result.embeddings]


# ─── SERVICE ──────────────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Unified embedding interface that delegates to the configured provider.

    Provider selection (priority order):
      1. If GEMINI_API_KEY is present → GeminiEmbeddingProvider (768 dims)
      2. Otherwise → LocalEmbeddingProvider (sentence-transformers, 384 dims)

    To force a specific provider, inject it via the constructor:
        svc = EmbeddingService(provider=MyCustomProvider())
    """

    def __init__(self, provider: EmbeddingProvider = None) -> None:
        if provider is not None:
            self._provider = provider
        elif os.getenv("GEMINI_API_KEY"):
            self._provider = GeminiEmbeddingProvider()
            logger.info("EmbeddingService: using Gemini provider (768 dims).")
        else:
            self._provider = LocalEmbeddingProvider()
            logger.info("EmbeddingService: using local sentence-transformers provider.")

    @property
    def dimensions(self) -> int:
        """Number of dimensions produced by the current provider."""
        return self._provider.dimensions

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding vector for the given text.

        Args:
            text: The input string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")
        return await self._provider.embed(text.strip())

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of texts in a single batch call.

        Args:
            texts: List of non-empty input strings.

        Returns:
            List of embedding vectors in the same order as `texts`.

        Raises:
            ValueError: If any text in the list is empty.
        """
        if not texts:
            return []

        cleaned = [t.strip() for t in texts]
        empties = [i for i, t in enumerate(cleaned) if not t]
        if empties:
            raise ValueError(f"Empty text at index(es): {empties}")

        return await self._provider.embed_batch(cleaned)
