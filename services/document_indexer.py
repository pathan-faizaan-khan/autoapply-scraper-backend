"""
Document Indexer

A utility service for bulk-ingesting documents into the RAG vector store.

Supports indexing:
  - Career roadmaps and guides
  - Learning resources (articles, tutorials)
  - Interview preparation notes
  - Resume writing advice
  - AI/ML career guides
  - Software engineering best-practices
  - Opportunity descriptions (normalized from scraper output)

This is deliberately a utility (not a FastAPI router) — it is called by:
  - Startup seed scripts
  - Admin/maintenance endpoints
  - Background jobs triggered by the scheduler in main.py

The DocumentIndexer itself contains NO business logic:
it calls RAGService.index_documents() and reports results.
"""

import logging
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from services.rag_service import RAGService, DocumentChunk

logger = logging.getLogger("document_indexer")


# ─── VALID CATEGORIES ─────────────────────────────────────────────────────────

VALID_CATEGORIES = frozenset({
    "career",
    "roadmap",
    "resume",
    "interview",
    "learning",
    "opportunities",
    "certifications",
    "ai_ml",
    "software_engineering",
    "cold_email",
})


# ─── INDEXER ──────────────────────────────────────────────────────────────────

class DocumentIndexer:
    """
    Utility for ingesting documents into the RAG vector index.

    All methods are async and accept an AsyncSession so they compose
    cleanly with FastAPI background tasks or the APScheduler jobs
    already present in main.py.
    """

    def __init__(self, rag_service: Optional[RAGService] = None) -> None:
        self._rag = rag_service or RAGService()

    async def index_raw(
        self,
        db: AsyncSession,
        *,
        title: str,
        content: str,
        source: str = "",
        category: str = "career",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """
        Index a single raw text document.

        Args:
            db:       Async DB session.
            title:    Short, descriptive title for the document.
            content:  Full text content to embed and store.
            source:   Origin URL or filepath (used for deduplication).
            category: One of VALID_CATEGORIES.
            metadata: Arbitrary key-value metadata tags (e.g. {"level": "junior"}).

        Returns:
            Database row ID on success, None on failure.
        """
        if category not in VALID_CATEGORIES:
            logger.warning(
                "Unknown category '%s' — allowed values: %s",
                category, sorted(VALID_CATEGORIES),
            )

        doc = DocumentChunk(
            title=title,
            content=content,
            source=source,
            category=category,
            metadata=metadata or {},
        )
        return await self._rag.index_document(db, doc)

    async def index_batch(
        self,
        db: AsyncSession,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Batch-index a list of document dicts.

        Each dict should have:
          title    (str, required)
          content  (str, required)
          source   (str, optional)
          category (str, optional, default 'career')
          metadata (dict, optional)

        Args:
            db:        Async DB session.
            documents: List of document attribute dicts.

        Returns:
            {"indexed": N, "failed": M, "skipped": K}
        """
        chunks: List[DocumentChunk] = []
        skipped = 0

        for raw in documents:
            title   = raw.get("title", "").strip()
            content = raw.get("content", "").strip()

            if not title or not content:
                logger.warning("Skipping document with missing title/content: %r", raw)
                skipped += 1
                continue

            category = raw.get("category", "career")
            if category not in VALID_CATEGORIES:
                logger.warning("Coercing unknown category '%s' → 'career'", category)
                category = "career"

            chunks.append(DocumentChunk(
                title=title,
                content=content,
                source=raw.get("source", ""),
                category=category,
                metadata=raw.get("metadata") or {},
            ))

        if not chunks:
            return {"indexed": 0, "failed": 0, "skipped": skipped}

        result = await self._rag.index_documents(db, chunks)
        result["skipped"] = skipped
        return result

    async def index_opportunity(
        self,
        db: AsyncSession,
        opportunity: Dict[str, Any],
    ) -> Optional[int]:
        """
        Convert a normalized opportunity dict (from the scraper or DB) into
        a RAG document and index it.

        The opportunity's description and tags become searchable via vector search,
        enabling the LLM to surface relevant opportunities from natural-language queries.

        Args:
            db:          Async DB session.
            opportunity: Opportunity dict (title, description, type, tags, url, ...).

        Returns:
            Database row ID, or None on failure.
        """
        title = opportunity.get("title", "Untitled Opportunity")
        description = opportunity.get("description", "")
        tags = opportunity.get("tags") or []
        if isinstance(tags, str):
            import json
            try:
                tags = json.loads(tags)
            except Exception:
                tags = [tags]

        content_parts = [f"Opportunity: {title}"]
        if description:
            content_parts.append(description)
        if tags:
            content_parts.append(f"Skills: {', '.join(str(t) for t in tags)}")
        if opportunity.get("organization"):
            content_parts.append(f"Organization: {opportunity['organization']}")
        if opportunity.get("country"):
            content_parts.append(f"Location: {opportunity['country']}")

        return await self.index_raw(
            db,
            title=title,
            content="\n".join(content_parts),
            source=opportunity.get("url", ""),
            category="opportunities",
            metadata={
                "type":         opportunity.get("type"),
                "organization": opportunity.get("organization"),
                "country":      opportunity.get("country"),
                "tags":         tags,
            },
        )

    # ── SEED DATA ─────────────────────────────────────────────────────────────

    async def seed_default_documents(self, db: AsyncSession) -> Dict[str, int]:
        """
        Seed the vector store with a curated set of baseline career documents.

        Called on first boot when the career_documents table is empty.
        This ensures RAG retrieval works out of the box without requiring
        an external data pipeline.
        """
        seed_docs = [
            {
                "title": "How to Break Into Software Engineering",
                "content": (
                    "Breaking into software engineering requires a combination of "
                    "technical skills, portfolio projects, and networking. Start by "
                    "mastering a core language (Python or JavaScript), build 2-3 real "
                    "projects, contribute to open source, and apply consistently. "
                    "Bootcamps, CS degrees, and self-learning all lead to the same outcome."
                ),
                "category": "career",
                "source": "autoapply_seed",
                "metadata": {"level": "beginner"},
            },
            {
                "title": "Backend Engineering Career Roadmap",
                "content": (
                    "A typical backend engineering progression: "
                    "1. Learn Python or Node.js fundamentals. "
                    "2. Understand HTTP, REST APIs, and databases (SQL + NoSQL). "
                    "3. Learn Docker and basic DevOps. "
                    "4. Study system design: load balancers, caching, message queues. "
                    "5. Gain cloud experience (AWS, GCP, or Azure). "
                    "6. Contribute to distributed systems or microservices projects."
                ),
                "category": "roadmap",
                "source": "autoapply_seed",
                "metadata": {"role": "backend_engineer", "level": "all"},
            },
            {
                "title": "Machine Learning Engineer Career Roadmap",
                "content": (
                    "ML Engineer progression: "
                    "1. Master Python (NumPy, Pandas, Matplotlib). "
                    "2. Study ML fundamentals: linear algebra, statistics, gradient descent. "
                    "3. Learn scikit-learn, then TensorFlow or PyTorch. "
                    "4. Practice on Kaggle competitions. "
                    "5. Build end-to-end ML pipelines (data → training → deployment). "
                    "6. Learn MLOps tools (MLflow, DVC, Kubeflow). "
                    "7. Study LLMs and transformer architectures."
                ),
                "category": "roadmap",
                "source": "autoapply_seed",
                "metadata": {"role": "ml_engineer", "level": "all"},
            },
            {
                "title": "Interview Preparation: Data Structures and Algorithms",
                "content": (
                    "For DS&A interviews, focus on: arrays, strings, hash maps, trees, "
                    "graphs, dynamic programming, and sorting algorithms. "
                    "Practice LeetCode (medium difficulty). "
                    "Aim for 100-150 problems across all categories. "
                    "Key patterns: sliding window, two pointers, BFS/DFS, backtracking, "
                    "and greedy algorithms. Practice explaining your approach aloud."
                ),
                "category": "interview",
                "source": "autoapply_seed",
                "metadata": {"type": "technical"},
            },
            {
                "title": "Resume Writing Tips for Software Engineers",
                "content": (
                    "A strong software engineering resume: "
                    "1. Lead with impact metrics (e.g. 'Reduced latency by 40%'). "
                    "2. Use the STAR format for project descriptions. "
                    "3. Keep it to 1 page for <5 years experience. "
                    "4. List technologies relevant to the target role. "
                    "5. Include links to GitHub and deployed projects. "
                    "6. ATS-optimise by mirroring job description keywords."
                ),
                "category": "resume",
                "source": "autoapply_seed",
                "metadata": {"type": "general"},
            },
            {
                "title": "Google Summer of Code — Student Guide",
                "content": (
                    "GSoC is a global program offering paid remote internships with "
                    "open-source organisations. Students work for 3 months on a coding "
                    "project. To be accepted: contribute small patches before the deadline, "
                    "write a strong proposal, contact mentors early, and pick a project "
                    "matching your skills. Stipends range from $1,500 to $6,600 depending "
                    "on country."
                ),
                "category": "opportunities",
                "source": "autoapply_seed",
                "metadata": {"type": "summer_of_code", "organization": "Google"},
            },
        ]

        result = await self.index_batch(db, seed_docs)
        logger.info("Seeded %d documents into RAG index.", result.get("indexed", 0))
        return result
