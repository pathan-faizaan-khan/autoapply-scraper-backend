"""
LLM Service

A reusable infrastructure layer for communicating with LLMs (e.g. Groq).
Provides structured responses, robust error handling, automated JSON repair,
and unified streaming support.
"""

import json
import time
import uuid
import logging
import asyncio
from typing import Optional, List, Dict, Any, Type, AsyncGenerator, AsyncIterator

from pydantic import BaseModel
from groq import AsyncGroq
import groq

from config.llm_config import (
    GROQ_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_P,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    JSON_REPAIR_RETRY_COUNT,
    DEFAULT_SYSTEM_PROMPT,
)
from services.llm_exceptions import (
    LLMError,
    MissingAPIKeyError,
    AuthenticationError,
    InvalidResponseError,
    InvalidJSONError,
    RateLimitError,
    TimeoutError,
    NetworkError,
)

# Configure structured logging
logger = logging.getLogger("llm_service")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# ─── STANDARD RESPONSE OBJECT ─────────────────────────────────────────────────

class LLMResponse(BaseModel):
    """Standardized response object returned by all LLMService methods."""
    content: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    tool_calls: Optional[List[Any]] = None
    raw_response: Optional[Dict[str, Any]] = None


# ─── SERVICE ──────────────────────────────────────────────────────────────────

class LLMService:
    """
    Production-grade async LLM Service backed by Groq.

    Design decisions:
    - Manual retry logic (not groq's built-in) so we can log every attempt.
    - TOP_K is omitted: Groq's chat completion API does not expose it.
    - stream_text() does NOT use _execute_with_retries because a streaming
      response is consumed incrementally; mid-stream retries are unsound.
    - health_check() calls `models.list()` (zero tokens) instead of a real
      completion to keep the probe lightweight.
    """

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            logger.error("Missing GROQ_API_KEY — service cannot be initialised.")
            raise MissingAPIKeyError("GROQ_API_KEY is not configured.")

        self.client = AsyncGroq(
            api_key=GROQ_API_KEY,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,  # Retries are managed manually for per-attempt logging
        )

    # ── INTERNAL HELPERS ──────────────────────────────────────────────────────

    async def _execute_with_retries(
        self,
        req_id: str,
        coro_func,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Execute a single Groq API call with transient-error retry logic.

        Retries:
          - RateLimitError    (exponential back-off: 1s, 2s, 4s, …)
          - APITimeoutError   (fixed 1s back-off)
          - APIConnectionError (fixed 1s back-off)

        Does NOT retry:
          - AuthenticationError  — the key is wrong; retrying achieves nothing.
          - APIError (other)     — treat as fatal; surface to caller.
        """
        for attempt in range(MAX_RETRIES + 1):
            try:
                start = time.monotonic()
                response = await coro_func(*args, **kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                return response, latency_ms

            except groq.AuthenticationError as exc:
                # Never retry auth failures.
                logger.error("[%s] Authentication failed — check GROQ_API_KEY.", req_id)
                raise AuthenticationError(f"LLM authentication failed: {exc}") from exc

            except groq.RateLimitError as exc:
                logger.warning(
                    "[%s] Rate-limit hit (attempt %d/%d).",
                    req_id, attempt + 1, MAX_RETRIES + 1,
                )
                if attempt == MAX_RETRIES:
                    raise RateLimitError(
                        f"Rate limit exceeded after {MAX_RETRIES + 1} attempts."
                    ) from exc
                await asyncio.sleep(2 ** attempt)

            except groq.APITimeoutError as exc:
                logger.warning(
                    "[%s] Request timed out (attempt %d/%d).",
                    req_id, attempt + 1, MAX_RETRIES + 1,
                )
                if attempt == MAX_RETRIES:
                    raise TimeoutError(
                        f"LLM request timed out after {MAX_RETRIES + 1} attempts."
                    ) from exc
                await asyncio.sleep(1)

            except groq.APIConnectionError as exc:
                logger.warning(
                    "[%s] Network error (attempt %d/%d).",
                    req_id, attempt + 1, MAX_RETRIES + 1,
                )
                if attempt == MAX_RETRIES:
                    raise NetworkError(
                        f"LLM network error after {MAX_RETRIES + 1} attempts."
                    ) from exc
                await asyncio.sleep(1)

            except groq.APIError as exc:
                # All other Groq API errors are not transient — surface immediately.
                logger.error("[%s] Non-transient API error: %s", req_id, exc)
                raise LLMError(f"LLM API error: {exc}") from exc

            except Exception as exc:
                logger.error("[%s] Unexpected error: %s", req_id, exc, exc_info=True)
                raise LLMError(f"Unexpected LLM error: {exc}") from exc

    def _build_messages(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Normalise the two supported input forms into the Groq messages format.

        Priority:
          1. If `messages` is provided directly, use it as-is (advanced callers).
          2. Otherwise, build from (system_prompt, user_prompt) with fallback to
             DEFAULT_SYSTEM_PROMPT.
        """
        if messages:
            return list(messages)  # Defensive copy — never mutate caller's list.

        built: List[Dict[str, str]] = []

        sys = system_prompt or DEFAULT_SYSTEM_PROMPT
        if sys:
            built.append({"role": "system", "content": sys})

        if user_prompt:
            built.append({"role": "user", "content": user_prompt})

        return built

    def _extract_response(self, response) -> tuple[str, str, Optional[List[Any]]]:
        """Extract (content, finish_reason, tool_calls) from a Groq response object."""
        if not response.choices:
            raise InvalidResponseError("LLM returned an empty choices array.")
        choice = response.choices[0]
        return (choice.message.content or ""), (choice.finish_reason or ""), choice.message.tool_calls

    # ── PUBLIC METHODS ────────────────────────────────────────────────────────

    async def generate_text(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> LLMResponse:
        """
        Generate plain text from the LLM.

        Supports two input styles:
          A) system_prompt + user_prompt  (convenience)
          B) messages=[...]               (full control)

        Returns:
            LLMResponse with content, token counts, latency, and finish reason.
        """
        req_id = str(uuid.uuid4())
        built_messages = self._build_messages(system_prompt, user_prompt, messages)

        logger.info(
            "[%s] generate_text — model=%s temperature=%.2f max_tokens=%d",
            req_id, MODEL_NAME, temperature, max_tokens,
        )

        kwargs = {
            "messages": built_messages,
            "model": MODEL_NAME,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": TOP_P,
        }
        if tools:
            kwargs["tools"] = tools

        response, latency_ms = await self._execute_with_retries(
            req_id,
            self.client.chat.completions.create,
            **kwargs,
        )

        content, finish_reason, tool_calls = self._extract_response(response)

        logger.info(
            "[%s] generate_text — done in %.2fms | tokens=%d finish=%s",
            req_id, latency_ms, response.usage.total_tokens, finish_reason,
        )

        return LLMResponse(
            content=content,
            model=response.model,
            latency_ms=latency_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw_response=response.model_dump(),
        )

    async def generate_json(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        response_model: Optional[Type[BaseModel]] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> Any:
        """
        Generate structured JSON output from the LLM.

        Workflow:
          1. Instruct the model to output JSON (via response_format + system hint).
          2. Call Groq.
          3. Parse the response.
          4. If JSON is invalid, append the error and retry once (repair loop).
          5. If a response_model is provided, validate with Pydantic; retry on
             validation failure.
          6. Return the validated Pydantic model or raw dict.

        Raises:
            InvalidJSONError: After all repair retries are exhausted.
        """
        # Build messages once — mutations happen inside the retry loop.
        built_messages = self._build_messages(system_prompt, user_prompt, messages)

        # Ensure a JSON instruction exists in the system prompt.
        if response_model is not None:
            schema = response_model.model_json_schema()
            schema_str = json.dumps(schema, indent=2)
            schema_instruction = f"\n\nYou MUST reply in valid JSON format only. Your JSON response MUST perfectly match this JSON Schema:\n{schema_str}"
            if built_messages and built_messages[0]["role"] == "system":
                built_messages[0]["content"] += schema_instruction
            else:
                built_messages.insert(0, {"role": "system", "content": schema_instruction})
        else:
            if built_messages and built_messages[0]["role"] == "system":
                if "json" not in built_messages[0]["content"].lower():
                    built_messages[0]["content"] += "\nYou must reply in valid JSON format only."
            else:
                built_messages.insert(
                    0,
                    {"role": "system", "content": "You must reply in valid JSON format only."},
                )

        # Single req_id per generate_json call — preserved across repair retries.
        req_id = str(uuid.uuid4())

        for attempt in range(JSON_REPAIR_RETRY_COUNT + 1):
            logger.info(
                "[%s] generate_json — attempt %d/%d model=%s",
                req_id, attempt + 1, JSON_REPAIR_RETRY_COUNT + 1, MODEL_NAME,
            )

            response, latency_ms = await self._execute_with_retries(
                req_id,
                self.client.chat.completions.create,
                messages=built_messages,
                model=MODEL_NAME,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=TOP_P,
                response_format={"type": "json_object"},
            )

            content, _, _ = self._extract_response(response)

            logger.info(
                "[%s] generate_json — raw response in %.2fms | tokens=%d",
                req_id, latency_ms, response.usage.total_tokens,
            )

            # ── Parse ──────────────────────────────────────────────────────────
            try:
                parsed: Dict[str, Any] = json.loads(content)
            except json.JSONDecodeError as parse_err:
                logger.warning(
                    "[%s] JSON parse error on attempt %d: %s",
                    req_id, attempt + 1, parse_err,
                )
                if attempt < JSON_REPAIR_RETRY_COUNT:
                    # Append the bad output + error message for the repair turn.
                    built_messages.append({"role": "assistant", "content": content})
                    built_messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous response was not valid JSON. "
                            f"Error: {parse_err}. "
                            "Please return only valid JSON with no markdown fences."
                        ),
                    })
                    continue
                raise InvalidJSONError(
                    f"JSON parse failed after {attempt + 1} attempts: {parse_err}"
                ) from parse_err

            # ── Validate against schema ────────────────────────────────────────
            if response_model is not None:
                try:
                    return response_model(**parsed)
                except Exception as val_err:
                    logger.warning(
                        "[%s] Pydantic validation failed on attempt %d: %s",
                        req_id, attempt + 1, val_err,
                    )
                    if attempt < JSON_REPAIR_RETRY_COUNT:
                        built_messages.append({"role": "assistant", "content": content})
                        built_messages.append({
                            "role": "user",
                            "content": (
                                f"Your JSON did not match the required schema. "
                                f"Validation errors: {val_err}. "
                                "Please fix and return only valid JSON."
                            ),
                        })
                        continue
                    raise InvalidJSONError(
                        f"Schema validation failed after {attempt + 1} attempts: {val_err}"
                    ) from val_err

            return parsed

        # Should never reach here — loop always returns or raises.
        raise InvalidJSONError("generate_json exhausted all attempts without returning.")

    async def stream_text(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> AsyncIterator[str]:
        """
        Stream text chunks from the LLM asynchronously.

        Note: Retries are NOT applied here. Retrying mid-stream is unsound
        because partial content may already have been yielded to the caller.
        The initial connection uses the standard REQUEST_TIMEOUT.

        Yields:
            str — individual text delta chunks as they arrive.
        """
        req_id = str(uuid.uuid4())
        built_messages = self._build_messages(system_prompt, user_prompt, messages)

        logger.info("[%s] stream_text — model=%s", req_id, MODEL_NAME)

        try:
            stream = await self.client.chat.completions.create(
                messages=built_messages,
                model=MODEL_NAME,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=TOP_P,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

            logger.info("[%s] stream_text — stream complete.", req_id)

        except groq.AuthenticationError as exc:
            raise AuthenticationError(f"Streaming auth failed: {exc}") from exc
        except groq.APITimeoutError as exc:
            raise TimeoutError(f"Streaming timed out: {exc}") from exc
        except Exception as exc:
            logger.error("[%s] stream_text error: %s", req_id, exc, exc_info=True)
            raise LLMError(f"Streaming failed: {exc}") from exc

    async def health_check(self) -> dict:
        """
        Verify that the API key is present and the Groq client can reach the API.

        Uses models.list() (zero token cost) rather than a real completion so
        the health probe is lightweight and does not consume quota.

        Returns:
            dict with keys: status, model, provider, configured, error.
        """
        status: Dict[str, Any] = {
            "status": "unhealthy",
            "model": MODEL_NAME,
            "provider": "groq",
            "configured": bool(GROQ_API_KEY),
            "error": None,
        }

        if not status["configured"]:
            status["error"] = "GROQ_API_KEY is missing."
            return status

        try:
            await self.client.models.list()
            status["status"] = "healthy"
        except groq.AuthenticationError as exc:
            status["error"] = f"Authentication failed: {exc}"
        except groq.APITimeoutError as exc:
            status["error"] = f"Connection timed out: {exc}"
        except Exception as exc:
            status["error"] = str(exc)

        return status
