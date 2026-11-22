import hashlib
from typing import Any

from database.models import Consent
from database.repository import AuditLogRepository, ConsentRepository, TrainingResultRepository


class ConsentService:
    """Согласие сотрудника на обработку персональных данных (152-ФЗ).

    Текст уведомления версионируется: при смене текста или версии
    согласие запрашивается заново. Факт выдачи/отзыва фиксируется в аудите.
    """

    PRIVACY_NOTICE = (
        "Наставник обучит и проверит знания тестом. Результат увидит HR.\n\n"
        "Для этого мы используем твоё имя и результаты тестов.\n"
        "Имя никому не передаём — перед анализом текста мы его скрываем.\n\n"
        "Нажимая «Согласен», ты даёшь согласие на обработку персональных данных (152-ФЗ)."
    )

    def __init__(self, notice_version: str) -> None:
        self._version = notice_version
        self._text_sha256 = hashlib.sha256(self.PRIVACY_NOTICE.encode("utf-8")).hexdigest()

    @property
    def notice_version(self) -> str:
        return self._version

    async def has_valid_consent(
        self,
        repository: ConsentRepository,
        subject_id: str,
    ) -> bool:
        consent = await repository.get_latest(subject_id)
        return (
            consent is not None
            and consent.status == "granted"
            and consent.version == self._version
        )

    async def grant(
        self,
        consent_repository: ConsentRepository,
        audit_repository: AuditLogRepository,
        subject_id: str,
    ) -> Consent:
        consent = await consent_repository.add(subject_id, self._version, self._text_sha256)
        await audit_repository.create(
            action="consent.granted",
            status="ok",
            subject_id=subject_id,
            meta={"version": self._version, "text_sha256": self._text_sha256, "channel": subject_id.split(":", 1)[0]},
        )
        return consent


class PersonalDataService:
    """Права субъекта персональных данных: выгрузка и удаление."""

    async def export_data(
        self,
        results_repository: TrainingResultRepository,
        consent_repository: ConsentRepository,
        audit_repository: AuditLogRepository,
        subject_id: str,
    ) -> dict[str, Any]:
        results = await results_repository.list_by_subject_id(subject_id)
        consents = await consent_repository.list_by_subject_id(subject_id)
        await audit_repository.create(
            action="data.exported",
            status="ok",
            subject_id=subject_id,
            meta={"results": len(results), "consents": len(consents)},
        )
        return {
            "subject_id": subject_id,
            "training_results": [
                {
                    "employee_name": row.employee_name,
                    "channel": row.channel,
                    "topic": row.topic,
                    "total_questions": row.total_questions,
                    "correct_answers": row.correct_answers,
                    "score_percent": row.score_percent,
                    "final_summary": row.final_summary,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in results
            ],
            "consents": [
                {
                    "version": row.version,
                    "status": row.status,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
                }
                for row in consents
            ],
        }

    async def delete_data(
        self,
        results_repository: TrainingResultRepository,
        consent_repository: ConsentRepository,
        audit_repository: AuditLogRepository,
        subject_id: str,
    ) -> dict[str, Any]:
        deleted_results = await results_repository.delete_by_subject_id(subject_id)
        consent = await consent_repository.revoke_latest(subject_id)
        revoked = consent is not None and consent.status == "revoked"
        await audit_repository.create(
            action="data.deleted",
            status="ok",
            subject_id=subject_id,
            meta={"deleted_results": deleted_results, "consent_revoked": revoked},
        )
        return {"deleted_results": deleted_results, "consent_revoked": revoked}
