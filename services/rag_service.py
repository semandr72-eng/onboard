import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import async_sessionmaker

from database.models import KnowledgeDocument
from database.repository import KnowledgeChunkRepository
from services.embedder import Embedder
from services.knowledge_ingest import parse_document

logger = logging.getLogger(__name__)

RRF_K = 60


@dataclass
class RankedHit:
    chunk_id: int
    title: str
    content: str
    score: float


def reciprocal_rank_fusion(rank_lists: list[list[tuple[int, str, str]]], top_n: int, k: int = RRF_K) -> list[RankedHit]:
    """Склеивает ранжированные списки (векторный, FTS, ...) через RRF.

    Каждый список — упорядоченные по убыванию релевантности кортежи (chunk_id, title, content).
    """
    scores: dict[int, float] = {}
    meta: dict[int, tuple[str, str]] = {}
    for rank_list in rank_lists:
        for rank, (chunk_id, title, content) in enumerate(rank_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            meta[chunk_id] = (title, content)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [
        RankedHit(chunk_id=chunk_id, title=meta[chunk_id][0], content=meta[chunk_id][1], score=round(score, 6))
        for chunk_id, score in ordered
    ]


class RAGService:
    """Гибридный поиск по базе знаний: pgvector (семантика) + FTS (ключевые слова).

    Эмбеддинги считаются локально (fastembed), наружу ничего не уходит.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker,
        embedder: Embedder,
        top_k: int = 4,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._top_k = top_k

    async def ingest(self, source: str, text: str) -> int:
        """Загружает документ в базу знаний. Возвращает id документа."""
        parsed = parse_document(text)
        if not parsed.chunks:
            raise ValueError("Документ не содержит текста для загрузки.")

        vectors = self._embedder.embed_passages([chunk.content for chunk in parsed.chunks])

        async with self._session_factory() as session:
            repository = KnowledgeChunkRepository(session)
            document = await repository.create_document(
                source=source,
                metadata=parsed.metadata,
                chunk_count=len(parsed.chunks),
            )
            for ordinal, (chunk, vector) in enumerate(zip(parsed.chunks, vectors), start=1):  # noqa: B905
                await repository.add_chunk(
                    document_id=document.id,
                    ordinal=ordinal,
                    title=chunk.title,
                    content=chunk.content,
                    embedding=vector,
                )
        logger.info("Ingested document %s (%s chunks)", source, len(parsed.chunks))
        return document.id

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RankedHit]:
        limit = top_k or self._top_k
        if not query.strip():
            return []
        query_vector = self._embedder.embed_query(query)
        async with self._session_factory() as session:
            repository = KnowledgeChunkRepository(session)
            if not await repository.has_any_chunk():
                return []
            vector_hits = await repository.vector_search(query_vector, limit=limit * 2)
            fts_hits = await repository.fts_search(query, limit=limit * 2)
        return reciprocal_rank_fusion([vector_hits, fts_hits], top_n=limit)

    def build_context(self, hits: list[RankedHit]) -> str:
        parts = [f"### {hit.title}\n{hit.content}" for hit in hits]
        return "\n\n---\n\n".join(parts)

    async def attribute_chunk(self, text: str) -> str | None:
        """Относит текст (вопрос теста) к ближайшему чанку базы знаний — для аналитики пробелов."""
        if not text.strip():
            return None
        vector = self._embedder.embed_query(text)
        async with self._session_factory() as session:
            repository = KnowledgeChunkRepository(session)
            if not await repository.has_any_chunk():
                return None
            hits = await repository.vector_search(vector, limit=1)
        return hits[0][1] if hits else None

    async def list_documents(self) -> list[KnowledgeDocument]:
        async with self._session_factory() as session:
            return await KnowledgeChunkRepository(session).list_documents()

    async def section_titles(self) -> list[str]:
        """Названия разделов базы знаний в порядке прохождения (маршрут обучения)."""
        async with self._session_factory() as session:
            return await KnowledgeChunkRepository(session).list_titles()

    async def delete_document(self, document_id: int) -> int:
        async with self._session_factory() as session:
            return await KnowledgeChunkRepository(session).delete_document(document_id)
