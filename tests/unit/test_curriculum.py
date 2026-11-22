import pytest

from config import Settings
from schemas import TrainingSessionDraft
from services.ai_training_service import AITrainingService
from services.llm_gateway import LLMJsonResponse


def _settings() -> Settings:
    return Settings(
        BOT_TOKEN="test-token",
        DATABASE_URL="sqlite+aiosqlite://",
        LLM_API_KEY="test-key",
        QUALITY_CONTROL=False,
    )


class StubRAG:
    def __init__(self, titles: list[str]) -> None:
        self._titles = titles

    async def section_titles(self) -> list[str]:
        return self._titles

    async def retrieve(self, query: str):
        return []

    def build_context(self, hits) -> str:
        return ""


class StubGateway:
    model = "stub-llm"

    async def generate_json(self, messages: list[dict], json_schema: dict) -> LLMJsonResponse:
        return LLMJsonResponse(
            payload={"reply": "Удачи на тесте!", "phase": "testing", "next_question": "Вопрос 1?"},
            model=self.model,
        )


async def test_incomplete_curriculum_blocks_testing():
    service = AITrainingService(
        settings=_settings(),
        llm_gateway=StubGateway(),
        rag_service=StubRAG(["ЧАНК 1. Смена", "ЧАНК 2. Упаковка"]),
    )
    draft = TrainingSessionDraft(employee_name="Иван Петров", phase="learning", covered_sections=["ЧАНК 1. Смена"])

    assert await service.is_curriculum_complete(draft) is False
    assert await service.curriculum_status(draft) == (1, 2)
    with pytest.raises(RuntimeError, match="Обучение не завершено"):
        await service.start_testing(draft)


async def test_complete_curriculum_allows_testing():
    service = AITrainingService(
        settings=_settings(),
        llm_gateway=StubGateway(),
        rag_service=StubRAG(["ЧАНК 1. Смена", "ЧАНК 2. Упаковка"]),
    )
    draft = TrainingSessionDraft(
        employee_name="Иван Петров",
        phase="learning",
        covered_sections=["ЧАНК 1. Смена", "ЧАНК 2. Упаковка"],
    )

    assert await service.is_curriculum_complete(draft) is True
    turn = await service.start_testing(draft)
    assert turn.phase == "testing"
    assert turn.next_question == "Вопрос 1?"


async def test_empty_curriculum_means_no_gate():
    """Без базы знаний (нет разделов) маршрут не задан — тест доступен как раньше."""
    service = AITrainingService(settings=_settings(), llm_gateway=StubGateway(), rag_service=StubRAG([]))
    draft = TrainingSessionDraft(employee_name="Иван Петров", phase="learning")

    assert await service.curriculum_status(draft) == (0, 0)
    assert await service.is_curriculum_complete(draft) is True


def test_match_section_normalizes_case_and_spaces():
    matched = AITrainingService._match_section(" чанк 1. смена ", ["ЧАНК 1. Смена", "ЧАНК 2. Упаковка"])
    assert matched == "ЧАНК 1. Смена"


def test_match_section_rejects_unknown_title():
    assert AITrainingService._match_section("Придуманный раздел", ["ЧАНК 1. Смена"]) is None
    assert AITrainingService._match_section(None, ["ЧАНК 1. Смена"]) is None
    assert AITrainingService._match_section("   ", ["ЧАНК 1. Смена"]) is None


def test_match_section_without_route_accepts_any():
    assert AITrainingService._match_section("Свободная тема", []) == "Свободная тема"
