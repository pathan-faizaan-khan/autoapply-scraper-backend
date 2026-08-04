"""
utils/auth.py — JWT Authentication Utilities for the Scraper Backend

Provides FastAPI dependency functions to:
  1. Decode and verify JWTs signed by the Node.js backend.
  2. Identify and block guest users from mutating endpoints.

The JWT_SECRET must match the one used in the Node.js backend.
Set it via the JWT_SECRET environment variable.

Usage in routers:
    from utils.auth import get_current_user, require_non_guest

    @router.get("/protected")
    async def read_only_route(user=Depends(get_current_user)):
        return {"userId": user["userId"]}

    @router.post("/mutations-only")
    async def mutation_route(user=Depends(require_non_guest)):
        return {"message": "Real users only"}
"""
import os
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

try:
    from jose import JWTError, jwt as jose_jwt
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False

JWT_SECRET = os.getenv("JWT_SECRET", "fallback_secret")
JWT_ALGORITHM = "HS256"

# HTTPBearer scheme — reads the "Authorization: Bearer <token>" header
_bearer_scheme = HTTPBearer(auto_error=False)


def _decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT token.
    Returns the payload dict on success, raises HTTPException on failure.
    """
    if not JOSE_AVAILABLE:
        # Fallback: basic base64 decode without verification (dev only)
        import base64
        import json
        try:
            parts = token.split(".")
            if len(parts) != 3:
                raise ValueError("Invalid token structure")
            # Pad base64 and decode
            padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            return payload
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or malformed token. Install python-jose for proper verification.",
            )

    try:
        payload = jose_jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid or expired token: {exc}",
        )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """
    FastAPI dependency — extracts and validates the Bearer JWT.
    Returns the decoded payload dict (includes userId, email, isGuest).
    Raises HTTP 401 if no token is present.
    Raises HTTP 403 if the token is invalid or expired.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _decode_token(credentials.credentials)


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency — like get_current_user but returns None if no token
    is provided. Useful for endpoints that are public but token-aware.
    """
    if credentials is None:
        return None
    try:
        return _decode_token(credentials.credentials)
    except HTTPException:
        return None


def require_non_guest(
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    FastAPI dependency — blocks guest users (isGuest == True) from the endpoint.
    Real authenticated users are passed through.
    Raises HTTP 403 for guest sessions.
    """
    if user.get("isGuest") is True:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Guest access restricted",
                "message": (
                    "This action is not available for guest sessions. "
                    "Please sign up for a free account to unlock full access."
                ),
                "isGuestBlock": True,
            },
        )
    return user
