"""
Custom Exceptions for the LLM Layer.
"""

class LLMError(Exception):
    """Base exception for all LLM-related errors."""
    pass


class MissingAPIKeyError(LLMError):
    """Raised when the API key is not configured."""
    pass


class AuthenticationError(LLMError):
    """Raised when the LLM provider rejects the API key."""
    pass


class InvalidResponseError(LLMError):
    """Raised when the LLM returns an unexpected or empty response format."""
    pass


class InvalidJSONError(LLMError):
    """Raised when generate_json() fails to parse or validate the LLM output."""
    pass


class RateLimitError(LLMError):
    """Raised when the LLM provider rate limits the request."""
    pass


class TimeoutError(LLMError):
    """Raised when the LLM provider request times out."""
    pass


class NetworkError(LLMError):
    """Raised when there is a transient network failure communicating with the LLM provider."""
    pass
