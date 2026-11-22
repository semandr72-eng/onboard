from datetime import UTC, datetime, timedelta

from database.repository import (
    AuditLogRepository,
    ConsentRepository,
    InviteCodeRepository,
    QualityCheckRepository,
    TrainingAnswerRepository,
    TrainingResultRepository,
)
from schemas import EvaluatedAnswer, TrainingResultCreate
from services.personal_data_service import ConsentService, PersonalDataService


def _result_in(**overrides) -> TrainingResultCreate:
    defaults = {
        "employee_name": "Иван Петров",
        "subject_id": "web:test-1",
        "channel": "web",
        "topic": "Регламент",
        "total_questions": 5,
        "correct_answers": 4,
        "score_percent": 80,
    }
    defaults.update(overrides)
    return TrainingResultCreate(**defaults)


async def test_result_crud_and_filters(session_factory):
    async with session_factory() as session:
        repository = TrainingResultRepository(session)
        await repository.create(_result_in(subject_id="tg:1", channel="telegram", score_percent=60))
        await repository.create(_result_in(subject_id="web:1", channel="web", score_percent=90))

        assert len(await repository.list_filtered()) == 2
        assert len(await repository.list_filtered(channel="web")) == 1
        assert len(await repository.list_filtered(min_score=80)) == 1
        assert (await repository.list_filtered(min_score=80))[0].score_percent == 90
        assert len(await repository.list_by_subject_id("tg:1")) == 1


async def test_result_stats(session_factory):
    async with session_factory() as session:
        repository = TrainingResultRepository(session)
        await repository.create(_result_in(channel="web", score_percent=100))
        await repository.create(_result_in(channel="telegram", score_percent=50))

        summary = await repository.stats_summary()
        assert summary["total_sessions"] == 2
        assert summary["avg_score"] == 75.0

        by_channel = await repository.stats_by_channel()
        assert {row["channel"] for row in by_channel} == {"web", "telegram"}

        by_day = await repository.stats_by_day(days=30)
        assert len(by_day) == 1
        assert by_day[0]["total"] == 2


async def test_retention_deletes_old_results(session_factory):
    async with session_factory() as session:
        repository = TrainingResultRepository(session)
        result = await repository.create(_result_in())
        # Эмулируем старую запись: backdate created_at напрямую.
        result.created_at = datetime.now(UTC) - timedelta(days=200)
        await session.commit()

        deleted = await TrainingResultRepository(session).delete_older_than(days=180)
        assert deleted == 1
        assert await repository.list_filtered() == []


async def test_answers_saved_and_gaps_reported(session_factory):
    async with session_factory() as session:
        results = TrainingResultRepository(session)
        answers = TrainingAnswerRepository(session)
        result = await results.create(_result_in())
        await answers.create_many(
            result.id,
            [
                EvaluatedAnswer(question="Вопрос А?", is_correct=False, chunk_title="Упаковка"),
                EvaluatedAnswer(question="Вопрос А?", is_correct=True, chunk_title="Упаковка"),
                EvaluatedAnswer(question="Вопрос Б?", is_correct=True, chunk_title="Смена"),
            ],
        )

        gaps = await answers.gaps(limit=5)
        assert len(gaps) == 2
        assert gaps[0]["question"] == "Вопрос А?"  # 0.5 < 1.0
        assert gaps[0]["correct_rate"] == 0.5
        assert gaps[0]["chunk_title"] == "Упаковка"


async def test_answers_cascade_deleted_with_result(session_factory):
    async with session_factory() as session:
        results = TrainingResultRepository(session)
        answers = TrainingAnswerRepository(session)
        result = await results.create(_result_in())
        await answers.create_many(result.id, [EvaluatedAnswer(question="Вопрос?", is_correct=True)])

        await results.delete_by_subject_id("web:test-1")
        assert await answers.gaps(limit=10) == []


async def test_invite_code_lifecycle(session_factory):
    async with session_factory() as session:
        invites = InviteCodeRepository(session)
        await invites.create(code="ABC123", max_uses=2)

        consumed = await invites.consume("ABC123")
        assert consumed is not None
        assert consumed.used_count == 1

        await invites.consume("ABC123")
        assert await invites.consume("ABC123") is None  # лимит исчерпан


async def test_invite_code_expired_and_unknown(session_factory):
    async with session_factory() as session:
        invites = InviteCodeRepository(session)
        await invites.create(code="OLD", max_uses=1, expires_at=datetime.now(UTC) - timedelta(days=1))

        assert await invites.consume("OLD") is None
        assert await invites.consume("NOPE") is None


async def test_quality_check_stats(session_factory):
    async with session_factory() as session:
        checks = QualityCheckRepository(session)
        await checks.create(kind="faithfulness", passed=True)
        await checks.create(kind="faithfulness", passed=False)
        await checks.create(kind="grade", passed=True)

        stats = {row["kind"]: row for row in await checks.stats()}
        assert stats["faithfulness"]["total"] == 2
        assert stats["faithfulness"]["pass_rate"] == 0.5
        assert stats["grade"]["pass_rate"] == 1.0

        failures = await checks.recent_failures(limit=10)
        assert len(failures) == 1
        assert failures[0].kind == "faithfulness"


async def test_consent_grant_versioning_and_export_delete(session_factory):
    async with session_factory() as session:
        consents = ConsentRepository(session)
        audits = AuditLogRepository(session)
        results = TrainingResultRepository(session)
        consent_service = ConsentService(notice_version="1.0")
        personal_data = PersonalDataService()
        subject = "tg:42"

        assert await consent_service.has_valid_consent(consents, subject) is False

        await consent_service.grant(consents, audits, subject)
        assert await consent_service.has_valid_consent(consents, subject) is True

        # Смена версии уведомления → старое согласие недействительно.
        newer_service = ConsentService(notice_version="2.0")
        assert await newer_service.has_valid_consent(consents, subject) is False

        await results.create(_result_in(subject_id=subject, channel="telegram"))

        exported = await personal_data.export_data(results, consents, audits, subject)
        assert exported["subject_id"] == subject
        assert len(exported["training_results"]) == 1
        assert len(exported["consents"]) == 1

        deleted = await personal_data.delete_data(results, consents, audits, subject)
        assert deleted["deleted_results"] == 1
        assert deleted["consent_revoked"] is True
        assert await results.list_by_subject_id(subject) == []

        # Аудит зафиксировал выдачу согласия, выгрузку и удаление.
        async with session_factory() as audit_session:
            from sqlalchemy import select

            from database.models import AuditLog

            rows = (await audit_session.execute(select(AuditLog).where(AuditLog.subject_id == subject))).scalars().all()
            actions = {row.action for row in rows}
            assert {"consent.granted", "data.exported", "data.deleted"} <= actions
