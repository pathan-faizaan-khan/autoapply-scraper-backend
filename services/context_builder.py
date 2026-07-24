"""
Context Builder

Collects context from multiple sources and merges them into a single
structured object ready to be injected into the LLM prompt.

The LLM should NEVER query repositories or databases directly.
All data retrieval happens here, and the result is a plain dict
that the career_agent passes to LLMService.

Context sources:
  - Resume data           (passed in from the request)
  - Skill gap assessment  (from DB via repository)
  - Roadmap steps         (from DB via repository)
  - RAG documents         (via RAGService similarity search)
  - Opportunity list      (from DB via repository)

The ContextBuilder is pure orchestration — no business logic.
"""

import logging
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from services.rag_service import RAGService, RetrievedDocument
from config.llm_config import RAG_CHUNK_MAX_CHARS

logger = logging.getLogger("context_builder")


class CareerContext:
    """
    Structured context object consumed by the LLM prompt renderer.

    All fields are optional — callers only populate what is available
    for the current request to avoid unnecessary DB queries.
    """

    __slots__ = (
        "user_id",
        "query",
        "target_role",
        "career_path",
        "resume_snippet",
        "existing_skills",
        "missing_skills",
        "gap_score",
        "roadmap_steps",
        "retrieved_docs",
        "opportunities",
        "session_history",
    )

    def __init__(
        self,
        user_id: str,
        query: str,
        target_role: str = "",
        career_path: str = "",
        resume_snippet: str = "",
        existing_skills: Optional[List[str]] = None,
        missing_skills: Optional[List[str]] = None,
        gap_score: Optional[float] = None,
        roadmap_steps: Optional[List[dict]] = None,
        retrieved_docs: Optional[List[RetrievedDocument]] = None,
        opportunities: Optional[List[dict]] = None,
        session_history: Optional[List[dict]] = None,
    ) -> None:
        self.user_id = user_id
        self.query = query
        self.target_role = target_role
        self.career_path = career_path
        self.resume_snippet = resume_snippet
        self.existing_skills = existing_skills or []
        self.missing_skills = missing_skills or []
        self.gap_score = gap_score
        self.roadmap_steps = roadmap_steps or []
        self.retrieved_docs = retrieved_docs or []
        self.opportunities = opportunities or []
        self.session_history = session_history or []

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for logging or debugging."""
        return {
            "user_id":        self.user_id,
            "query":          self.query,
            "target_role":    self.target_role,
            "career_path":    self.career_path,
            "existing_skills": self.existing_skills,
            "missing_skills": self.missing_skills,
            "gap_score":      self.gap_score,
            "roadmap_steps":  len(self.roadmap_steps),
            "retrieved_docs": len(self.retrieved_docs),
            "opportunities":  len(self.opportunities),
        }


class ContextBuilder:
    """
    Assembles a CareerContext from multiple async data sources.

    Usage:
        builder = ContextBuilder(rag_service)
        ctx = await builder.build(
            db=db,
            user_id="...",
            query="How do I become a backend engineer?",
            resume_data={...},
            skill_assessment={...},
            roadmap_steps=[...],
            opportunities=[...],
            session_history=[...],
        )
        prompt = render_prompt(ctx)
    """

    def __init__(self, rag_service: Optional[RAGService] = None) -> None:
        self._rag = rag_service or RAGService()

    async def build(
        self,
        db: AsyncSession,
        *,
        user_id: str,
        query: str,
        target_role: str = "",
        career_path: str = "",
        resume_data: Optional[Dict[str, Any]] = None,
        skill_assessment: Optional[Dict[str, Any]] = None,
        roadmap_steps: Optional[List[dict]] = None,
        opportunities: Optional[List[dict]] = None,
        session_history: Optional[List[dict]] = None,
        rag_category: Optional[str] = None,
        rag_top_k: int = 5,
    ) -> CareerContext:
        """
        Build a fully-populated CareerContext for a single request.

        Performs the RAG retrieval and merges all available context.
        All parameters except db, user_id, and query are optional —
        callers pass only what they have.

        Args:
            db:               Async DB session.
            user_id:          Application user identifier.
            query:            The user's natural-language question.
            target_role:      Job role the user is targeting.
            career_path:      Currently selected career path title.
            resume_data:      Parsed resume dict (from ML autofill or upload).
            skill_assessment: Latest skill gap analysis dict from DB.
            roadmap_steps:    Fetched roadmap step dicts.
            opportunities:    Fetched opportunity dicts.
            session_history:  Recent conversation messages.
            rag_category:     Optional category filter for RAG retrieval.
            rag_top_k:        Max documents to retrieve.

        Returns:
            CareerContext ready to be rendered into an LLM prompt.
        """
        # ── Extract resume snippet ─────────────────────────────────────────────
        resume_snippet = self._extract_resume_snippet(resume_data)

        # ── Extract skill gap data ─────────────────────────────────────────────
        existing_skills: List[str] = []
        missing_skills: List[str] = []
        gap_score: Optional[float] = None

        if skill_assessment:
            existing_skills = skill_assessment.get("existing_skills") or []
            missing_skills  = skill_assessment.get("missing_skills") or []
            gap_score       = skill_assessment.get("gap_score")

        # ── RAG retrieval ──────────────────────────────────────────────────────
        rag_query = self._build_rag_query(query, target_role, missing_skills)
        retrieved_docs: List[RetrievedDocument] = []

        if db is not None:
            try:
                retrieved_docs = await self._rag.retrieve(
                    db,
                    query=rag_query,
                    top_k=rag_top_k,
                    category=rag_category,
                )
                logger.info(
                    "[ContextBuilder] RAG retrieved %d docs for user=%s",
                    len(retrieved_docs), user_id,
                )
            except Exception as exc:
                # RAG failure is non-fatal — degrade gracefully.
                logger.warning("[ContextBuilder] RAG retrieval failed: %s", exc)

        return CareerContext(
            user_id=user_id,
            query=query,
            target_role=target_role,
            career_path=career_path,
            resume_snippet=resume_snippet,
            existing_skills=existing_skills,
            missing_skills=missing_skills,
            gap_score=gap_score,
            roadmap_steps=roadmap_steps or [],
            retrieved_docs=retrieved_docs,
            opportunities=opportunities or [],
            session_history=session_history or [],
        )

    def render_to_prompt_variables(self, ctx: CareerContext) -> Dict[str, str]:
        """
        Convert a CareerContext into a flat dict of prompt template variables.

        These variables are passed directly to PromptLoader.render_prompt().
        Every value is a string so the template engine can substitute cleanly.
        """
        rag_context = self._format_rag_docs(ctx.retrieved_docs)
        opportunities_text = self._format_opportunities(ctx.opportunities)
        roadmap_text = self._format_roadmap(ctx.roadmap_steps)

        return {
            "user_id":         ctx.user_id,
            "query":           ctx.query,
            "target_role":     ctx.target_role or "Not specified",
            "career_path":     ctx.career_path or "Not specified",
            "resume":          ctx.resume_snippet or "No resume provided.",
            "skills":          ", ".join(ctx.existing_skills) or "Not specified",
            "missing_skills":  ", ".join(ctx.missing_skills) or "None identified",
            "gap_score":       str(ctx.gap_score) if ctx.gap_score is not None else "N/A",
            "roadmap":         roadmap_text,
            "rag_context":     rag_context,
            "opportunities":   opportunities_text,
            "experience":      ctx.resume_snippet or "Not provided",
        }

    # ── PRIVATE FORMATTERS ────────────────────────────────────────────────────

    @staticmethod
    def _extract_resume_snippet(resume_data: Optional[Dict[str, Any]]) -> str:
        """
        Pull a concise text snippet from a parsed resume dict.
        Gracefully handles None or unexpected structure.
        """
        if not resume_data:
            return ""
        parts = []
        if resume_data.get("summary"):
            parts.append(f"Summary: {resume_data['summary']}")
        if resume_data.get("skills"):
            skills = resume_data["skills"]
            if isinstance(skills, list):
                parts.append(f"Skills: {', '.join(str(s) for s in skills[:20])}")
            elif isinstance(skills, str):
                parts.append(f"Skills: {skills}")
        if resume_data.get("experience"):
            exp = resume_data["experience"]
            if isinstance(exp, list) and exp:
                first = exp[0]
                if isinstance(first, dict):
                    title = first.get("title", "")
                    company = first.get("company", "")
                    if title or company:
                        parts.append(f"Most recent role: {title} at {company}")
        return "\n".join(parts)

    @staticmethod
    def _build_rag_query(
        query: str,
        target_role: str,
        missing_skills: List[str],
    ) -> str:
        """
        Enrich the raw query with role and skill context for better retrieval.
        Longer, richer queries improve cosine similarity against chunk embeddings.
        """
        parts = [query]
        if target_role:
            parts.append(f"Target role: {target_role}")
        if missing_skills:
            parts.append(f"Skills to learn: {', '.join(missing_skills[:5])}")
        return " | ".join(parts)

    @staticmethod
    def _format_rag_docs(docs: List[RetrievedDocument]) -> str:
        """Format retrieved documents into a numbered context block."""
        if not docs:
            return "No relevant resources found."
        lines = []
        for i, doc in enumerate(docs, 1):
            snippet = doc.content[:RAG_CHUNK_MAX_CHARS]
            if len(doc.content) > RAG_CHUNK_MAX_CHARS:
                snippet += "…"
            lines.append(
                f"[{i}] {doc.title} (category={doc.category}, "
                f"relevance={doc.similarity:.2f})\n{snippet}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _format_opportunities(opportunities: List[dict]) -> str:
        """Format opportunities into a compact numbered list."""
        if not opportunities:
            return "No opportunities available."
        lines = []
        for i, opp in enumerate(opportunities[:5], 1):
            lines.append(
                f"{i}. {opp.get('title', 'Untitled')} "
                f"({opp.get('type', 'opportunity')}) — {opp.get('organization', '')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_roadmap(steps: List[dict]) -> str:
        """Format roadmap steps into an ordered list."""
        if not steps:
            return "No roadmap generated yet."
        lines = []
        for step in steps[:10]:
            status = step.get("status", "not_started")
            lines.append(
                f"Step {step.get('step_order', '?')}: {step.get('title', '?')} "
                f"[{status}]"
            )
        return "\n".join(lines)
