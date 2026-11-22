import json
import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMJsonResponse(BaseModel):
    payload: dict[str, Any]
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMGateway:
    """Клиент к OpenAI-совместимому LLM API (DeepSeek) с гарантированным JSON-ответом.

    - Сначала используется ``response_format=json_schema``; если API отклоняет
      такой формат (400/422), навсегда переключаемся на ``json_object`` и
      передаем схему текстом в системном сообщении.
    - Если модель вернула невалидный JSON, делается один повторный запрос
      с просьбой исправить ответ.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0) -> None:
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self._json_schema_supported: bool | None = None

    async def generate_json(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
    ) -> LLMJsonResponse:
        prepared = self._prepare_messages(messages, json_schema)

        body = await self._post(prepared, json_schema)
        payload = self._parse_json(body["choices"][0]["message"]["content"])

        if payload is None:
            logger.warning("LLM returned invalid JSON, requesting a fix")
            repaired = self._repair_messages(prepared, json_schema, body["choices"][0]["message"]["content"])
            body = await self._post(repaired, json_schema)
            payload = self._parse_json(body["choices"][0]["message"]["content"])
            if payload is None:
                raise ValueError("LLM returned invalid JSON and the repair attempt failed")

        usage = body.get("usage") or {}
        return LLMJsonResponse(
            payload=payload,
            model=body.get("model") or self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _prepare_messages(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
    ) -> list[dict[str, str]]:
        if self._json_schema_supported is not False:
            return list(messages)
        schema_hint = {
            "role": "system",
            "content": (
                "Отвечай строго одним валидным JSON-объектом без пояснений. "
                f"Схема ответа: {json.dumps(json_schema, ensure_ascii=False)}"
            ),
        }
        return [schema_hint, *messages]

    async def _post(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.2,
            "messages": messages,
            "response_format": self._response_format(json_schema),
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if self._json_schema_supported is None and exc.response.status_code in (400, 422):
                self._json_schema_supported = False
                logger.warning("LLM API rejected response_format=json_schema, falling back to json_object")
                payload["response_format"] = self._response_format(json_schema)
                payload["messages"] = self._prepare_messages(messages, json_schema)
                response = await self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
            else:
                raise
        if self._json_schema_supported is None:
            self._json_schema_supported = True
        return response.json()

    def _response_format(self, json_schema: dict[str, Any]) -> dict[str, Any]:
        if self._json_schema_supported is False:
            return {"type": "json_object"}
        return {"type": "json_schema", "json_schema": json_schema}

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any] | None:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _repair_messages(
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        broken_content: str,
    ) -> list[dict[str, str]]:
        return [
            *messages,
            {"role": "assistant", "content": broken_content},
            {
                "role": "user",
                "content": (
                    "Предыдущий ответ не является валидным JSON. Исправь его, "
                    "не меняя содержания: верни только валидный JSON по схеме "
                    f"{json.dumps(json_schema, ensure_ascii=False)}."
                ),
            },
        ]
