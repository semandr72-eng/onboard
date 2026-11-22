"""Alembic-окружение: async-движок, URL из настроек приложения (DATABASE_URL)."""

import asyncio
import threading
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from config import get_settings
from database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Приоритет URL: то, что передано в Config (database/migrations.py или CLI -x),
# иначе настройки приложения (переменная DATABASE_URL, затем .env проекта).
DATABASE_URL = config.get_main_option("sqlalchemy.url") or get_settings().database_url
config.set_main_option("sqlalchemy.url", DATABASE_URL)


def run_migrations_offline() -> None:
    """Режим --sql: SQL-скрипт без подключения к БД (для asyncpg не работает)."""
    raise SystemExit(
        "Офлайн-режим (alembic --sql) не поддерживается с asyncpg. "
        "Применяйте миграции к живой БД: alembic upgrade head"
    )


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = DATABASE_URL
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(run_async_migrations())
        return

    # Вызов из уже запущенного event loop (например, init_db приложения):
    # вложенный asyncio.run невозможен, миграция исполняется в отдельном потоке.
    error: list[BaseException] = []

    def worker() -> None:
        try:
            asyncio.run(run_async_migrations())
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if error:
        raise error[0]


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
