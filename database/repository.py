import json
from datetime import UTC, datetime, timedelta
from typing import cast

from sqlalchemy import case, delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    AuditLog,
    Consent,
    InviteCode,
    KnowledgeChunk,
    KnowledgeDocument,
    QualityCheck,
    TrainingAnswer,
    TrainingResult,
)
from schemas import TrainingResultCreate


class TrainingResultRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, result_in: TrainingResultCreate) -> TrainingResult:
        result = TrainingResult(**result_in.model_dump())
        self._session.add(result)
        await self._session.commit()
        await self._session.refresh(result)
        return result

    async def get_by_id(self, result_id: int) -> TrainingResult | None:
        statement = select(TrainingResult).where(TrainingResult.id == result_id)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_subject_id(self, subject_id: str) -> list[TrainingResult]:
        statement = (
            select(TrainingResult)
            .where(TrainingResult.subject_id == subject_id)
            .order_by(TrainingResult.created_at.desc())
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def delete_by_subject_id(self, subject_id: str) -> int:
        statement = delete(TrainingResult).where(TrainingResult.subject_id == subject_id)
        result = cast("CursorResult[None]", await self._session.execute(statement))
        await self._session.commit()
        return result.rowcount or 0

    async def delete_older_than(self, days: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        statement = delete(TrainingResult).where(TrainingResult.created_at < cutoff)
        result = cast("CursorResult[None]", await self._session.execute(statement))
        await self._session.commit()
        return result.rowcount or 0

    async def list_filtered(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        channel: str | None = None,
        min_score: int | None = None,
    ) -> list[TrainingResult]:
        statement = select(TrainingResult)
        if date_from is not None:
            statement = statement.where(TrainingResult.created_at >= date_from)
        if date_to is not None:
            statement = statement.where(TrainingResult.created_at <= date_to)
        if channel is not None:
            statement = statement.where(TrainingResult.channel == channel)
        if min_score is not None:
            statement = statement.where(TrainingResult.score_percent >= min_score)
        statement = statement.order_by(TrainingResult.created_at.desc())
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def stats_summary(self) -> dict:
        statement = select(
            func.count().label("total"),
            func.coalesce(func.avg(TrainingResult.score_percent), 0).label("avg_score"),
        )
        row = (await self._session.execute(statement)).one()
        return {"total_sessions": int(row.total), "avg_score": round(float(row.avg_score), 1)}

    async def stats_by_channel(self) -> list[dict]:
        statement = (
            select(
                TrainingResult.channel,
                func.count().label("total"),
                func.coalesce(func.avg(TrainingResult.score_percent), 0).label("avg_score"),
            )
            .group_by(TrainingResult.channel)
            .order_by(TrainingResult.channel)
        )
        rows = (await self._session.execute(statement)).all()
        return [
            {"channel": row.channel, "total": int(row.total), "avg_score": round(float(row.avg_score), 1)}
            for row in rows
        ]

    async def stats_by_day(self, days: int = 30) -> list[dict]:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        day = func.date(TrainingResult.created_at)
        statement = (
            select(
                day.label("day"),
                func.count().label("total"),
                func.coalesce(func.avg(TrainingResult.score_percent), 0).label("avg_score"),
            )
            .where(TrainingResult.created_at >= cutoff)
            .group_by(day)
            .order_by(day)
        )
        rows = (await self._session.execute(statement)).all()
        return [
            {"day": str(row.day), "total": int(row.total), "avg_score": round(float(row.avg_score), 1)}
            for row in rows
        ]


class ConsentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_latest(self, subject_id: str) -> Consent | None:
        statement = (
            select(Consent)
            .where(Consent.subject_id == subject_id)
            .order_by(Consent.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_by_subject_id(self, subject_id: str) -> list[Consent]:
        statement = (
            select(Consent)
            .where(Consent.subject_id == subject_id)
            .order_by(Consent.created_at.desc())
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def add(self, subject_id: str, version: str, text_sha256: str) -> Consent:
        consent = Consent(
            subject_id=subject_id,
            version=version,
            text_sha256=text_sha256,
        )
        self._session.add(consent)
        await self._session.commit()
        await self._session.refresh(consent)
        return consent

    async def revoke_latest(self, subject_id: str) -> Consent | None:
        consent = await self.get_latest(subject_id)
        if consent is not None and consent.status == "granted":
            consent.status = "revoked"
            consent.revoked_at = datetime.now(UTC)
            await self._session.commit()
            await self._session.refresh(consent)
        return consent


class AuditLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        action: str,
        status: str = "ok",
        subject_id: str | None = None,
        meta: dict | None = None,
    ) -> AuditLog:
        record = AuditLog(
            action=action,
            status=status,
            subject_id=subject_id,
            meta_json=json.dumps(meta or {}, ensure_ascii=False),
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record


class InviteCodeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        code: str,
        max_uses: int = 1,
        expires_at: datetime | None = None,
    ) -> InviteCode:
        invite = InviteCode(code=code, max_uses=max_uses, expires_at=expires_at)
        self._session.add(invite)
        await self._session.commit()
        await self._session.refresh(invite)
        return invite

    async def consume(self, code: str) -> InviteCode | None:
        """Атомарно расходует одно использование кода. Возвращает None, если код нельзя использовать."""
        now = datetime.now(UTC)
        statement = (
            update(InviteCode)
            .where(
                InviteCode.code == code,
                InviteCode.active.is_(True),
                InviteCode.used_count < InviteCode.max_uses,
                (InviteCode.expires_at.is_(None)) | (InviteCode.expires_at > now),
            )
            .values(used_count=InviteCode.used_count + 1)
        )
        result = cast("CursorResult[None]", await self._session.execute(statement))
        await self._session.commit()
        if result.rowcount == 0:
            return None
        return await self.get(code)

    async def get(self, code: str) -> InviteCode | None:
        statement = select(InviteCode).where(InviteCode.code == code)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_all(self) -> list[InviteCode]:
        statement = select(InviteCode).order_by(InviteCode.created_at.desc())
        result = await self._session.execute(statement)
        return list(result.scalars().all())


class KnowledgeChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_document(self, source: str, metadata: dict, chunk_count: int) -> KnowledgeDocument:
        document = KnowledgeDocument(
            source=source,
            meta_json=json.dumps(metadata, ensure_ascii=False),
            chunk_count=chunk_count,
        )
        self._session.add(document)
        await self._session.commit()
        await self._session.refresh(document)
        return document

    async def add_chunk(
        self,
        document_id: int,
        ordinal: int,
        title: str,
        content: str,
        embedding: list[float],
    ) -> KnowledgeChunk:
        chunk = KnowledgeChunk(
            document_id=document_id,
            ordinal=ordinal,
            title=title,
            content=content,
            embedding=embedding,
        )
        self._session.add(chunk)
        await self._session.commit()
        return chunk

    async def has_any_chunk(self) -> bool:
        statement = select(KnowledgeChunk.id).limit(1)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none() is not None

    async def vector_search(self, embedding: list[float], limit: int) -> list[tuple[int, str, str]]:
        statement = (
            select(KnowledgeChunk.id, KnowledgeChunk.title, KnowledgeChunk.content)
            .order_by(KnowledgeChunk.embedding.cosine_distance(embedding))
            .limit(limit)
        )
        result = await self._session.execute(statement)
        return [(row.id, row.title, row.content) for row in result.all()]

    async def fts_search(self, query: str, limit: int) -> list[tuple[int, str, str]]:
        ts_query = func.plainto_tsquery("russian", query)
        ts_vector = func.to_tsvector("russian", KnowledgeChunk.content)
        statement = (
            select(KnowledgeChunk.id, KnowledgeChunk.title, KnowledgeChunk.content)
            .where(KnowledgeChunk.content.op("@@")(ts_query))
            .order_by(func.ts_rank_cd(ts_vector, ts_query).desc())
            .limit(limit)
        )
        result = await self._session.execute(statement)
        return [(row.id, row.title, row.content) for row in result.all()]

    async def list_documents(self) -> list[KnowledgeDocument]:
        statement = select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def delete_document(self, document_id: int) -> int:
        statement = delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id)
        result = cast("CursorResult[None]", await self._session.execute(statement))
        document = await self._session.get(KnowledgeDocument, document_id)
        if document is not None:
            await self._session.delete(document)
        await self._session.commit()
        return result.rowcount or 0


class TrainingAnswerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_many(self, result_id: int, answers: list) -> int:
        rows = [
            TrainingAnswer(
                result_id=result_id,
                question=answer.question,
                is_correct=answer.is_correct,
                feedback=answer.feedback,
                chunk_title=answer.chunk_title,
            )
            for answer in answers
        ]
        self._session.add_all(rows)
        await self._session.commit()
        return len(rows)

    async def gaps(self, limit: int = 20) -> list[dict]:
        """Вопросы с самой низкой долей верных ответов — 'пробелы знаний'."""
        correct_rate = func.avg(case((TrainingAnswer.is_correct.is_(True), 1.0), else_=0.0))
        statement = (
            select(
                TrainingAnswer.question,
                TrainingAnswer.chunk_title,
                func.count().label("times_asked"),
                correct_rate.label("correct_rate"),
            )
            .group_by(TrainingAnswer.question, TrainingAnswer.chunk_title)
            .order_by(correct_rate.asc(), func.count().desc())
            .limit(limit)
        )
        rows = (await self._session.execute(statement)).all()
        return [
            {
                "question": row.question,
                "chunk_title": row.chunk_title,
                "times_asked": int(row.times_asked),
                "correct_rate": round(float(row.correct_rate), 3),
            }
            for row in rows
        ]


class QualityCheckRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        kind: str,
        passed: bool,
        subject_id: str | None = None,
        detail: dict | None = None,
    ) -> QualityCheck:
        record = QualityCheck(
            kind=kind,
            passed=passed,
            subject_id=subject_id,
            detail_json=json.dumps(detail or {}, ensure_ascii=False),
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def stats(self) -> list[dict]:
        pass_rate = func.avg(case((QualityCheck.passed.is_(True), 1.0), else_=0.0))
        statement = (
            select(
                QualityCheck.kind,
                func.count().label("total"),
                func.coalesce(pass_rate, 0).label("pass_rate"),
            )
            .group_by(QualityCheck.kind)
            .order_by(QualityCheck.kind)
        )
        rows = (await self._session.execute(statement)).all()
        return [
            {"kind": row.kind, "total": int(row.total), "pass_rate": round(float(row.pass_rate), 3)}
            for row in rows
        ]

    async def recent_failures(self, limit: int = 20) -> list[QualityCheck]:
        statement = (
            select(QualityCheck)
            .where(QualityCheck.passed.is_(False))
            .order_by(QualityCheck.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())
