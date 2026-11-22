import logging
from typing import Any

from services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

GRADE_VERDICT_SCHEMA: dict[str, Any] = {
    "name": "grade_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_correct": {"type": "boolean"},
            "justification": {"type": "string"},
        },
        "required": ["is_correct", "justification"],
        "additionalProperties": False,
    },
}

FAITHFULNESS_SCHEMA: dict[str, Any] = {
    "name": "faithfulness_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "grounded": {"type": "boolean"},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string"},
        },
        "required": ["grounded", "unsupported_claims", "note"],
        "additionalProperties": False,
    },
}

_GRADE_SYSTEM = (
    "Ты — независимый эксперт-оценщик знаний сотрудника. Оценивай смысл, а не формулировку. "
    "Ответ считается верным, если он по смыслу совпадает с материалом, даже при других словах "
    "или нехватке второстепенных деталей. Неверным считай только ответ, противоречащий материалу, "
    "не относящийся к вопросу или пропускающий обязательное ключевое условие вопроса. "
    "Отвечай строго по схеме: is_correct и короткая justification на русском."
)

_FAITHFULNESS_SYSTEM = (
    "Ты — контролёр качества ответов ИИ-наставника. Проверь, подтверждается ли каждое значимое "
    "утверждение ответа наставника предоставленным материалом. Если есть утверждения, которых нет "
    "в материале (вымысел, неточности), перечисли их в unsupported_claims. Формулировки своими "
    "словами, повторяющие смысл материала, считай подтверждёнными. Отвечай строго по схеме."
)


class QualityJudge:
    """Независимый судья качества: оценка ответов сотрудников и достоверность реплик наставника.

    Работает через тот же LLM Gateway; на вход получает только обезличенный текст
    (ничего не покидает периметр сверх того, что уже ушло к LLM).
    """

    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    async def grade_answer(self, question: str, answer: str, context: str) -> tuple[bool, str]:
        """Независимый вердикт: вопрос + ответ + материал."""
        response = await self._gateway.generate_json(
            messages=[
                {"role": "system", "content": _GRADE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Материал:\n{context}\n\nВопрос наставника: {question}\n"
                        f"Ответ сотрудника: {answer}\n\nОцени ответ."
                    ),
                },
            ],
            json_schema=GRADE_VERDICT_SCHEMA,
        )
        payload = response.payload
        return bool(payload["is_correct"]), str(payload["justification"])

    async def check_faithfulness(self, reply: str, context: str) -> dict[str, Any]:
        """Проверка, что реплика наставника опирается на материал."""
        response = await self._gateway.generate_json(
            messages=[
                {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                {
                    "role": "user",
                    "content": f"Материал:\n{context}\n\nОтвет наставника:\n{reply}\n\nПроверь достоверность.",
                },
            ],
            json_schema=FAITHFULNESS_SCHEMA,
        )
        payload = response.payload
        return {
            "grounded": bool(payload["grounded"]),
            "unsupported_claims": [str(item) for item in payload.get("unsupported_claims", [])],
            "note": str(payload.get("note", "")),
        }
