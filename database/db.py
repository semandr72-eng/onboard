import asyncio

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from database.models import Base


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, future=True, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    if engine.dialect.name == "postgresql":
        # Postgres: схема управляется Alembic-миграциями (alembic/versions/).
        # Базы, созданные раньше через create_all, автоматически фиксируются на head.
        from database.migrations import run_migrations

        await asyncio.to_thread(run_migrations, engine.url.render_as_string(hide_password=False))
        return

    # SQLite (локальные тесты): pgvector-таблицы только для PostgreSQL.
    async with engine.begin() as connection:
        await connection.run_sync(
            lambda sync_conn: Base.metadata.create_all(
                sync_conn,
                tables=[
                    table
                    for name, table in Base.metadata.tables.items()
                    if not name.startswith("knowledge_")
                ],
            )
        )
