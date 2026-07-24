"""
Shared async SQLAlchemy session factory.

Re-uses the same DATABASE_URL environment variable and asyncpg driver
as scraper/playwright_scraper.py, but exposes it as a dependency-injectable
async context manager and a FastAPI Depends() helper so routers
don't have to manage sessions manually.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from config.career_config import DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_ECHO

load_dotenv()

# ─── Engine setup (lazy singleton, mirrors playwright_scraper.py pattern) ────

_engine = None
_AsyncSessionLocal = None


def _get_session_factory() -> sessionmaker:
    """
    Build (or return cached) async SQLAlchemy session factory.
    Converts any legacy postgres:// / postgresql:// scheme to the asyncpg driver.

    Returns:
        Configured SQLAlchemy sessionmaker bound to the async engine.

    Raises:
        RuntimeError: When DATABASE_URL is missing from the environment.
    """
    global _engine, _AsyncSessionLocal

    if _engine is None:
        db_url = os.environ.get("DATABASE_URL", "")

        if not db_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Add it to your .env file or Render dashboard."
            )

        # Normalise scheme to asyncpg
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        _engine = create_async_engine(
            db_url,
            echo=DB_ECHO,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            connect_args={"statement_cache_size": 0},  # required for asyncpg + pgbouncer
        )

        _AsyncSessionLocal = sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    return _AsyncSessionLocal


# ─── Context manager (for use inside services / background tasks) ─────────────

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a database session.
    Intended for use inside background tasks or services that run
    outside of a FastAPI request context.

    Usage:
        async with get_db_session() as session:
            result = await some_repository_function(session, ...)
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# ─── FastAPI Depends() helper ─────────────────────────────────────────────────

async def get_db() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency that provides a per-request database session.
    Use with FastAPI's Depends() mechanism in route function signatures.

    Usage:
        @router.get("/example")
        async def example_endpoint(db: AsyncSession = Depends(get_db)):
            ...

    Yields:
        AsyncSession: A live database session scoped to this request.
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
