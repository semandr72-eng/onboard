"""Запуск Alembic-миграций из кода приложения (на старте бота и веб-канала)."""

import asyncio
import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logger = logging.getLogger(__name__)

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


def run_migrations(database_url: str) -> None:
    """Применить миграции Postgres до head.

    Синхронная обёртка: вызывать через asyncio.to_thread, чтобы не создавать
    вложенный event loop (alembic сам делает asyncio.run внутри).
    """
    asyncio.run(_migrate(database_url))


async def _migrate(database_url: str) -> None:
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", database_url)

    # Базы, созданные до Alembic через create_all: таблицы на месте, alembic_version — нет.
    # Фиксируем их текущее состояние как head, чтобы baseline-миграция не пыталась
    # пересоздать существующие таблицы, а последующие миграции применялись как обычно.
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            version_table = await conn.scalar(text("SELECT to_regclass('public.alembic_version')"))
            if version_table is None:
                has_app_tables = await conn.scalar(text("SELECT to_regclass('public.training_results')"))
                if has_app_tables is not None:
                    logger.warning(
                        "Обнаружена база, созданная до Alembic (create_all). "
                        "Фиксирую версию схемы как head без изменений (alembic stamp head)."
                    )
                    command.stamp(cfg, "head")
    finally:
        await engine.dispose()

    command.upgrade(cfg, "head")
