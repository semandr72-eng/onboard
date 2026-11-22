import asyncio
import contextlib
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

from bot.handlers import router
from bot.middlewares import LoggingMiddleware
from config import get_settings
from database import (
    AuditLogRepository,
    QualityCheckRepository,
    TrainingResultRepository,
    create_engine,
    create_session_factory,
    init_db,
)
from services import (
    AITrainingService,
    ConsentService,
    FastEmbedE5,
    LLMGateway,
    PersonalDataService,
    QualityJudge,
    RAGService,
    TrainingService,
)

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    import logging.config

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": level.upper(),
                }
            },
            "root": {
                "handlers": ["console"],
                "level": level.upper(),
            },
        }
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiogram").setLevel(logging.INFO)


async def run_bot() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    storage = MemoryStorage()
    bot = Bot(token=settings.bot_token, default=DefaultBotProperties())
    dp = Dispatcher(storage=storage)

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    training_service = TrainingService()
    llm_gateway = LLMGateway(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    async def audit_sink(action: str, subject_id: str | None, status: str, meta: dict) -> None:
        async with session_factory() as session:
            repository = AuditLogRepository(session)
            await repository.create(
                action=action,
                status=status,
                subject_id=subject_id,
                meta=meta,
            )

    async def quality_sink(kind: str, passed: bool, subject_id: str | None, detail: dict) -> None:
        async with session_factory() as session:
            await QualityCheckRepository(session).create(
                kind=kind,
                passed=passed,
                subject_id=subject_id,
                detail=detail,
            )
            if not passed:
                await AuditLogRepository(session).create(
                    action="quality.warn",
                    status="warn",
                    subject_id=subject_id,
                    meta={"kind": kind, "detail": detail},
                )

    embedder = FastEmbedE5(model_name=settings.embedding_model)
    rag_service = RAGService(session_factory=session_factory, embedder=embedder, top_k=settings.knowledge_top_k)
    quality_judge = QualityJudge(llm_gateway) if settings.quality_control else None

    ai_training_service = AITrainingService(
        settings=settings,
        llm_gateway=llm_gateway,
        audit_sink=audit_sink,
        rag_service=rag_service,
        quality_judge=quality_judge,
        quality_sink=quality_sink,
    )
    consent_service = ConsentService(notice_version=settings.privacy_notice_version)
    personal_data_service = PersonalDataService()

    await init_db(engine)

    dp.message.middleware(LoggingMiddleware())
    dp.include_router(router)
    dp["settings"] = settings
    dp["session_factory"] = session_factory
    dp["training_service"] = training_service
    dp["ai_training_service"] = ai_training_service
    dp["consent_service"] = consent_service
    dp["personal_data_service"] = personal_data_service

    async def retention_worker() -> None:
        while True:
            try:
                async with session_factory() as session:
                    deleted = await TrainingResultRepository(session).delete_older_than(settings.retention_days)
                if deleted:
                    logger.info("Retention: удалено %s результатов старше %s дней", deleted, settings.retention_days)
            except Exception:
                logger.exception("Retention worker failed")
            await asyncio.sleep(24 * 3600)

    retention_task = asyncio.create_task(retention_worker())

    logger.info("Starting AI training bot")
    try:
        await dp.start_polling(bot)
    finally:
        retention_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await retention_task
        await llm_gateway.close()
        await engine.dispose()
        await bot.session.close()
