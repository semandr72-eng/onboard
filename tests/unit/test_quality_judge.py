from services.llm_gateway import LLMJsonResponse
from services.quality_service import QualityJudge


class ScriptedGateway:
    """Gateway, отдающий заготовленные ответы и записывающий ушедшие сообщения."""

    def __init__(self, *payloads: dict) -> None:
        self.model = "fake-judge"
        self._payloads = list(payloads)
        self.sent_messages: list[list[dict]] = []

    async def generate_json(self, messages: list[dict], json_schema: dict) -> LLMJsonResponse:
        self.sent_messages.append(messages)
        return LLMJsonResponse(payload=self._payloads.pop(0), model=self.model)


async def test_grade_answer_returns_verdict():
    gateway = ScriptedGateway({"is_correct": True, "justification": "Смысл совпадает с материалом."})
    judge = QualityJudge(gateway)

    is_correct, justification = await judge.grade_answer(
        question="Куда класть бытовую химию?",
        answer="В отдельный пакет.",
        context="### Упаковка\nБытовая химия упаковывается отдельно.",
    )

    assert is_correct is True
    assert "Смысл" in justification


async def test_grade_answer_false_verdict():
    gateway = ScriptedGateway({"is_correct": False, "justification": "Противоречит материалу."})
    judge = QualityJudge(gateway)
    is_correct, _ = await judge.grade_answer("Вопрос?", "Неверный ответ.", "Материал")
    assert is_correct is False


async def test_grade_sends_clean_triplet_without_dialogue_history():
    gateway = ScriptedGateway({"is_correct": True, "justification": "ок"})
    judge = QualityJudge(gateway)
    await judge.grade_answer("Вопрос?", "Ответ.", "Материал")

    user_content = gateway.sent_messages[0][1]["content"]
    assert "Вопрос?" in user_content
    assert "Ответ." in user_content
    assert "Материал" in user_content
    # Судья получает только тройку «вопрос + ответ + материал» — без истории диалога.
    assert len(gateway.sent_messages[0]) == 2


async def test_check_faithfulness_collects_unsupported_claims():
    gateway = ScriptedGateway(
        {
            "grounded": False,
            "unsupported_claims": ["Холодильник необязателен"],
            "note": "Одно утверждение не подтверждено.",
        }
    )
    judge = QualityJudge(gateway)
    result = await judge.check_faithfulness("Ответ наставника.", "Материал")

    assert result["grounded"] is False
    assert result["unsupported_claims"] == ["Холодильник необязателен"]
    assert result["note"]
