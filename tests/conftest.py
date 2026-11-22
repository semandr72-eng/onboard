import hashlib

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from database.db import init_db
from services.llm_gateway import LLMJsonResponse


@pytest.fixture
async def session_factory():
    """SQLite в памяти со схемой приложения (без pgvector-таблиц)."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    from sqlalchemy import event

    def _fk_on(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    event.listen(engine.sync_engine, "connect", _fk_on)

    await init_db(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


class FakeLLMGateway:
    """Имитация LLM Gateway: отдаёт подготовленные payload'ы, запоминает запросы."""

    def __init__(self, payloads: list[dict] | None = None) -> None:
        self.model = "fake-llm"
        self._payloads = list(payloads or [])
        self.requests: list[dict] = []

    async def generate_json(self, messages: list[dict], json_schema: dict) -> LLMJsonResponse:
        self.requests.append({"messages": messages, "json_schema": json_schema})
        payload = self._payloads.pop(0) if self._payloads else {"reply": "Ок.", "phase": "learning"}
        return LLMJsonResponse(payload=payload, model=self.model)

    async def close(self) -> None:
        return None


class FixedEmbedder:
    """Детерминированный эмбеддер заданной размерности для RAG-тестов."""

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        buckets = [0.0] * self.dimension
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        for index in range(self.dimension):
            buckets[index] = digest[index % len(digest)] / 255.0
        norm = sum(v * v for v in buckets) ** 0.5 or 1.0
        return [round(v / norm, 6) for v in buckets]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def make_gateway():
    """Фабрика FakeLLMGateway с заданной очередью payload'ов."""

    def _make(payloads: list[dict] | None = None) -> FakeLLMGateway:
        return FakeLLMGateway(payloads)

    return _make


@pytest.fixture
def fixed_embedder() -> FixedEmbedder:
    return FixedEmbedder(dimension=1024)
