"""Интеграционный тест гибридного поиска (pgvector + FTS) против реального Postgres.

Поднимает контейнер pgvector/pgvector:pg16 на отдельном порту. Если docker
недоступен — тест пропускается, чтобы набор не падал в чужих окружениях.
"""

import asyncio
import shutil
import subprocess
import time
import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.migrations import run_migrations
from services.rag_service import RAGService

CONTAINER_PORT = "55432"
IMAGE = "pgvector/pgvector:pg16"

DOCUMENT = """---
document_type: Регламент
---
# ЧАНК 1. Температурный режим

Заморозка может находиться вне холодильника не более 20 минут в термопакете.

# ЧАНК 2. Упаковка

Бытовая химия упаковывается в отдельный пакет, отдельно от продуктов.
"""


def _docker(*args: str) -> subprocess.CompletedProcess:
    # Таймаут с запасом: первый запуск может включать скачивание образа (~650 МБ).
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=300)


@pytest.fixture
async def pgvector_url():
    if shutil.which("docker") is None:
        pytest.skip("docker не установлен")

    name = f"onboard-pgvector-test-{uuid.uuid4().hex[:8]}"
    try:
        started = _docker(
            "run", "-d", "--rm", "--name", name,
            "-e", "POSTGRES_DB=onboarding",
            "-e", "POSTGRES_USER=postgres",
            "-e", "POSTGRES_PASSWORD=postgres",
            "-p", f"{CONTAINER_PORT}:5432",
            IMAGE,
        )
        if started.returncode != 0:
            pytest.skip(f"не удалось поднять контейнер pgvector: {started.stderr.strip()}")

        url = f"postgresql+asyncpg://postgres:postgres@localhost:{CONTAINER_PORT}/onboarding"
        for _ in range(60):
            ready = _docker("exec", name, "pg_isready", "-U", "postgres", "-d", "onboarding")
            if ready.returncode == 0:
                break
            time.sleep(2)
        else:
            pytest.skip("postgres в контейнере не стал готов за отведённое время")

        await asyncio.to_thread(run_migrations, url)
        yield url
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"docker недоступен в этом окружении: {exc}")
    finally:
        _docker("rm", "-f", name)


async def test_hybrid_search_over_pgvector(pgvector_url, fixed_embedder):
    engine = create_async_engine(pgvector_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    rag = RAGService(session_factory=session_factory, embedder=fixed_embedder, top_k=2)

    document_id = await rag.ingest(source="reglament.md", text=DOCUMENT)
    documents = await rag.list_documents()
    assert len(documents) == 1
    assert documents[0].id == document_id
    assert documents[0].chunk_count == 2

    # Пустой запрос не должен падать и ничего не возвращать.
    assert await rag.retrieve("") == []

    # Запрос по ключевому слову: FTS-ветка находит нужный чанк.
    hits = await rag.retrieve("заморозка", top_k=2)
    assert len(hits) == 2
    titles = {hit.title for hit in hits}
    assert "Температурный режим" in titles
    assert "Упаковка" in titles

    # Семантическая атрибуция вопроса к чанку (для аналитики пробелов).
    chunk_title = await rag.attribute_chunk("Куда упаковывать бытовую химию?")
    assert chunk_title in titles

    deleted = await rag.delete_document(document_id)
    assert deleted == 2
    assert await rag.retrieve("заморозка") == []

    await engine.dispose()
