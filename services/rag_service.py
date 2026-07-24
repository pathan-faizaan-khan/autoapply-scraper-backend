"""
RAG Service — Retrieval-Augmented Generation

Provides document indexing and similarity-based retrieval using
pgvector stored in the existing Supabase/PostgreSQL instance.

This service is intentionally category-agnostic: the same index,
retrieve, and delete methods work for career, resume, interview,
learning, opportunities, and any future document category.

Responsibilities:
  - index_document()   — embed a single document and store in DB.
  - index_documents()  — batch index multiple documents.
  - retrieve()         — embed query, run cosine similarity search, return top-K.
  - delete_document()  — remove a document from the index by ID.

The service does NOT query business repositories and does NOT make
LLM calls. Those responsibilities belong to the career_agent orchestrator.
"""

import json
import logging
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from services.embedding_service import EmbeddingService
from config.llm_config import RAG_TOP_K, RAG_SIMILARITY_THRESHOLD

logger = logging.getLogger("rag_service")


# ─── DOMAIN OBJECTS ───────────────────────────────────────────────────────────

class DocumentChunk:
    """Represents a document to be indexed into the vector store."""

    __slots__ = ("title", "content", "source", "category", "metadata")

    def __init__(
        self,
        title: str,
        content: str,
        source: str = "",
        category: str = "career",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.title = title
        self.content = content
        self.source = source
        self.category = category
        self.metadata = metadata or {}


class RetrievedDocument:
    """A document returned from similarity search, with its relevance score."""

    __slots__ = ("id", "title", "content", "source", "category", "metadata", "similarity")

    def __init__(
        self,
        id: int,
        title: str,
        content: str,
        source: str,
        category: str,
        metadata: Dict[str, Any],
        similarity: float,
    ) -> None:
        self.id = id
        self.title = title
        self.content = content
        self.source = source
        self.category = category
        self.metadata = metadata
        self.similarity = similarity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":         self.id,
            "title":      self.title,
            "content":    self.content,
            "source":     self.source,
            "category":   self.category,
            "metadata":   self.metadata,
            "similarity": self.similarity,
        }


# ─── SERVICE ──────────────────────────────────────────────────────────────────

class RAGService:
    """
    Vector store interface over PostgreSQL / pgvector.

    Design:
      - No singleton. Callers create an instance per request or share one
        at the app level — EmbeddingService caches its model lazily.
      - All DB interactions use sqlalchemy.text() (matching existing project style).
      - Cosine distance (<=> operator) is used because embeddings are
        L2-normalised by the EmbeddingService.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None) -> None:
        self._embedder = embedding_service or EmbeddingService()

    # ── INDEXING ──────────────────────────────────────────────────────────────

    async def index_document(
        self,
        db: AsyncSession,
        document: DocumentChunk,
        user_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Embed a single document and upsert it into career_documents.

        De-duplication strategy: documents with the same (title, source) pair
        are updated rather than duplicated.  Pass an empty source if you
        intentionally want multiple documents with the same title.

        Args:
            db:       Async DB session.
            document: DocumentChunk to embed and store.

        Returns:
            The database row ID, or None on failure.
        """
        try:
            embedding = await self._embedder.generate_embedding(document.content)
            embedding_str = f"[{','.join(map(str, embedding))}]"

            sql = text("""
                INSERT INTO career_documents
                    (title, content, source, category, metadata, embedding,
                     created_at, updated_at)
                VALUES
                    (:title, :content, :source, :category, :metadata,
                     CAST(:embedding AS vector), NOW(), NOW())
                ON CONFLICT (title, source) DO UPDATE
                    SET content    = EXCLUDED.content,
                        category   = EXCLUDED.category,
                        metadata   = EXCLUDED.metadata,
                        embedding  = EXCLUDED.embedding,
                        updated_at = NOW()
                RETURNING id
            """)

            meta = document.metadata.copy()
            if user_id:
                meta["user_id"] = user_id

            result = await db.execute(sql, {
                "title":     document.title,
                "content":   document.content,
                "source":    document.source,
                "category":  document.category,
                "metadata":  json.dumps(meta),
                "embedding": embedding_str,
            })
            await db.commit()
            row = result.fetchone()
            doc_id = row[0] if row else None
            logger.info(
                "Indexed document id=%s title=%r category=%s",
                doc_id, document.title, document.category,
            )
            return doc_id

        except Exception as exc:
            await db.rollback()
            logger.error("index_document error: %s", exc, exc_info=True)
            return None

    async def index_documents(
        self,
        db: AsyncSession,
        documents: List[DocumentChunk],
        user_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Batch-index multiple documents, embedding them in a single provider call.

        Args:
            db:        Async DB session.
            documents: List of DocumentChunk objects to index.

        Returns:
            dict with 'indexed' (success count) and 'failed' (error count).
        """
        if not documents:
            return {"indexed": 0, "failed": 0}

        contents = [d.content for d in documents]
        try:
            embeddings = await self._embedder.generate_embeddings(contents)
        except Exception as exc:
            logger.error("batch embedding failed: %s", exc, exc_info=True)
            return {"indexed": 0, "failed": len(documents)}

        indexed, failed = 0, 0
        sql = text("""
            INSERT INTO career_documents
                (title, content, source, category, metadata, embedding,
                 created_at, updated_at)
            VALUES
                (:title, :content, :source, :category, :metadata,
                 CAST(:embedding AS vector), NOW(), NOW())
            ON CONFLICT (title, source) DO UPDATE
                SET content    = EXCLUDED.content,
                    category   = EXCLUDED.category,
                    metadata   = EXCLUDED.metadata,
                    embedding  = EXCLUDED.embedding,
                    updated_at = NOW()
        """)

        for doc, emb in zip(documents, embeddings):
            try:
                emb_str = f"[{','.join(map(str, emb))}]"
                meta = doc.metadata.copy()
                if user_id:
                    meta["user_id"] = user_id
                
                await db.execute(sql, {
                    "title":     doc.title,
                    "content":   doc.content,
                    "source":    doc.source,
                    "category":  doc.category,
                    "metadata":  json.dumps(meta),
                    "embedding": emb_str,
                })
                indexed += 1
            except Exception as exc:
                logger.warning("Failed to index '%s': %s", doc.title, exc)
                failed += 1

        try:
            await db.commit()
        except Exception as exc:
            await db.rollback()
            logger.error("batch commit failed: %s", exc, exc_info=True)
            return {"indexed": 0, "failed": len(documents)}

        logger.info("Batch index complete: %d indexed, %d failed.", indexed, failed)
        return {"indexed": indexed, "failed": failed}

    # ── RETRIEVAL ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        user_id: str,
        *,
        top_k: int = RAG_TOP_K,
        category: Optional[str] = None,
        min_similarity: float = RAG_SIMILARITY_THRESHOLD,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieve the top-K most semantically similar documents to `query`.

        Workflow:
          1. Generate embedding for the query string.
          2. Run pgvector cosine similarity search (1 - distance = similarity).
          3. Filter by category and minimum similarity threshold.
          4. Return RetrievedDocument objects ordered by similarity DESC.

        Args:
            db:              Async DB session.
            query:           Natural-language query string.
            top_k:           Maximum number of documents to return.
            category:        Optional category filter (e.g. 'career', 'resume').
            min_similarity:  Minimum cosine similarity score (0.0–1.0).
            metadata_filter: Optional exact-match JSONB filter (e.g. {"level": "junior"}).

        Returns:
            List of RetrievedDocument, ordered by similarity descending.
        """
        if not query.strip():
            return []

        try:
            query_embedding = await self._embedder.generate_embedding(query)
            emb_str = f"[{','.join(map(str, query_embedding))}]"
        except Exception as exc:
            logger.error("retrieve: failed to embed query: %s", exc)
            return []

        # Build WHERE clauses
        where_parts = ["(1 - (embedding <=> CAST(:embedding AS vector))) >= :min_similarity"]
        params: Dict[str, Any] = {
            "embedding":      emb_str,
            "min_similarity": min_similarity,
            "top_k":          top_k,
        }

        if category:
            where_parts.append("category = :category")
            params["category"] = category

        meta_filter = metadata_filter or {}
        
        # User isolation: match this user's docs OR global docs (no user_id)
        where_parts.append("(metadata @> CAST(:user_id_json AS jsonb) OR metadata->'user_id' IS NULL)")
        params["user_id_json"] = json.dumps({"user_id": user_id})
        
        if meta_filter:
            where_parts.append("metadata @> CAST(:meta_filter AS jsonb)")
            params["meta_filter"] = json.dumps(meta_filter)

        where_sql = " AND ".join(where_parts)

        sql = text(f"""
            SELECT
                id,
                title,
                content,
                source,
                category,
                metadata,
                (1 - (embedding <=> CAST(:embedding AS vector))) AS similarity
            FROM career_documents
            WHERE {where_sql}
            ORDER BY similarity DESC
            LIMIT :top_k
        """)

        try:
            result = await db.execute(sql, params)
            rows = result.mappings().all()

            docs = []
            for row in rows:
                meta = row["metadata"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                docs.append(RetrievedDocument(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    source=row["source"] or "",
                    category=row["category"],
                    metadata=meta,
                    similarity=float(row["similarity"]),
                ))

            logger.info(
                "retrieve: query=%r category=%s → %d docs returned (top_k=%d)",
                query[:60], category, len(docs), top_k,
            )
            return docs

        except Exception as exc:
            logger.error("retrieve error: %s", exc, exc_info=True)
            return []

    # ── DELETION ──────────────────────────────────────────────────────────────

    async def delete_document(self, db: AsyncSession, document_id: int) -> bool:
        """
        Remove a document from the vector index by its primary key.

        Args:
            db:          Async DB session.
            document_id: PK of the career_documents row to delete.

        Returns:
            True if the row was deleted; False otherwise.
        """
        sql = text("DELETE FROM career_documents WHERE id = :id RETURNING id")
        try:
            result = await db.execute(sql, {"id": document_id})
            await db.commit()
            deleted = result.fetchone() is not None
            if deleted:
                logger.info("Deleted document id=%d", document_id)
            else:
                logger.warning("delete_document: id=%d not found.", document_id)
            return deleted
        except Exception as exc:
            await db.rollback()
            logger.error("delete_document error: %s", exc, exc_info=True)
            return False
