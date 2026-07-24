"""
Prompt Loader Service

Responsible for loading and rendering prompt templates from the file system.
Implements an in-memory cache to prevent redundant disk I/O on each request.

Security note:
  render_prompt() strips prompt-injection attempts by rejecting variable values
  that contain the template delimiter `{{` (preventing nested injection).
  Path traversal is blocked by resolving paths against PROMPTS_DIR and verifying
  the result is still a child of that directory.
"""

import os
import asyncio
from typing import Dict, Any

# Absolute path to the prompts/ directory, resolved relative to this module's
# parent package (i.e. the project root).
PROMPTS_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts",
)

# Module-level in-memory cache.  Populated lazily on first access per key.
# Protected by an asyncio.Lock to prevent duplicate I/O during concurrent
# first-reads (e.g. multiple requests arriving before the cache is warm).
_prompt_cache: Dict[str, str] = {}
_cache_lock = asyncio.Lock()


class PromptLoader:
    """
    Service for loading and rendering file-based prompt templates.

    All methods are static — no instance state is needed.
    The cache is module-level (shared across all callers) by design.
    """

    @staticmethod
    def _safe_path(name: str) -> str:
        """
        Resolve and validate the prompt file path.

        Ensures the resolved path is a child of PROMPTS_DIR to prevent
        path-traversal attacks (e.g. name='../../etc/passwd').

        Args:
            name: Raw file name, with or without .txt extension.

        Returns:
            Absolute, validated file path string.

        Raises:
            ValueError:       If the resolved path escapes PROMPTS_DIR.
            FileNotFoundError: If the file does not exist.
        """
        if not name.endswith(".txt"):
            name = name + ".txt"

        resolved = os.path.realpath(os.path.join(PROMPTS_DIR, name))
        prompts_root = os.path.realpath(PROMPTS_DIR)

        if not resolved.startswith(prompts_root + os.sep) and resolved != prompts_root:
            raise ValueError(
                f"Prompt name '{name}' resolves outside the prompts directory."
            )

        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Prompt file not found: {resolved}")

        return resolved

    @staticmethod
    async def load_prompt(name: str) -> str:
        """
        Load a prompt template by name, with async-safe in-memory caching.

        The cache lock ensures that two coroutines racing on a cold cache
        only read the file once.

        Args:
            name: Prompt file name, e.g. 'career_chat' or 'career_chat.txt'.

        Returns:
            The raw prompt template string.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
            ValueError:        If the path resolves outside PROMPTS_DIR.
        """
        cache_key = name if name.endswith(".txt") else name + ".txt"

        # Fast path — no lock needed if already cached.
        if cache_key in _prompt_cache:
            return _prompt_cache[cache_key]

        async with _cache_lock:
            # Re-check under lock (another coroutine may have loaded it).
            if cache_key in _prompt_cache:
                return _prompt_cache[cache_key]

            file_path = PromptLoader._safe_path(name)

            with open(file_path, "r", encoding="utf-8") as fh:
                template = fh.read()

            _prompt_cache[cache_key] = template
            return template

    @staticmethod
    def load_prompt_sync(name: str) -> str:
        """
        Synchronous (blocking) variant for use in non-async contexts
        (e.g. startup scripts, testing, or background threads).

        The cache is shared with load_prompt() — if the async version has
        already populated the entry, this returns immediately.

        Args:
            name: Prompt file name.

        Returns:
            The raw prompt template string.
        """
        cache_key = name if name.endswith(".txt") else name + ".txt"

        if cache_key in _prompt_cache:
            return _prompt_cache[cache_key]

        file_path = PromptLoader._safe_path(name)
        with open(file_path, "r", encoding="utf-8") as fh:
            template = fh.read()

        _prompt_cache[cache_key] = template
        return template

    @staticmethod
    async def render_prompt(name: str, variables: Dict[str, Any]) -> str:
        """
        Load a prompt template and substitute {{variable}} placeholders.

        Security:
          Variable values are converted to strings and checked for the
          template delimiter '{{'. If a value contains '{{', it is rejected
          to prevent prompt injection via user-controlled input.

        Args:
            name:      Prompt file name.
            variables: Mapping of placeholder names to substitution values.

        Returns:
            The fully rendered prompt string.

        Raises:
            ValueError: If a variable value contains the '{{' delimiter.
        """
        template = await PromptLoader.load_prompt(name)

        rendered = template
        for key, value in variables.items():
            str_value = str(value)
            if "{{" in str_value:
                raise ValueError(
                    f"Variable '{key}' contains '{{{{' which is the template "
                    "delimiter. This may be a prompt-injection attempt."
                )
            rendered = rendered.replace(f"{{{{{key}}}}}", str_value)

        return rendered

    @staticmethod
    def render_prompt_sync(name: str, variables: Dict[str, Any]) -> str:
        """
        Synchronous variant of render_prompt().

        Suitable for use in non-async contexts.
        """
        template = PromptLoader.load_prompt_sync(name)

        rendered = template
        for key, value in variables.items():
            str_value = str(value)
            if "{{" in str_value:
                raise ValueError(
                    f"Variable '{key}' contains '{{{{' — possible prompt injection."
                )
            rendered = rendered.replace(f"{{{{{key}}}}}", str_value)

        return rendered

    @staticmethod
    def clear_cache() -> None:
        """
        Evict all entries from the in-memory prompt cache.

        Useful in testing to force templates to be re-read from disk.
        Should not be called in production code.
        """
        _prompt_cache.clear()
