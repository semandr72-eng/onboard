import httpx
import pytest
from asgi_lifespan import LifespanManager

from config import Settings
from web.app import create_app

ADMIN_TOKEN = "admin-token"

# Сценарий ответов фейковой модели по порядку вызовов generate_turn:
# 1) старт обучения после имени, 2) переход к тесту, 3) оценка финального ответа.
TURN_PAYLOADS = [
    {
        "reply": "Привет! 👋 Разберём регламент по шагам.",
        "phase": "learning",
        "quiz": {
            "question": "Куда кладём заморозку после сборки?",
            "options": ["В термопакет", "В подсобку"],
            "correct_index": 0,
        },
    },
    {
        "reply": "Удачи на тесте! 🍀",
        "phase": "testing",
        "next_question": "Куда упаковывать бытовую химию?",
    },
    {
        "reply": "Верно! ✅",
        "phase": "completed",
        "latest_answer_evaluated": True,
        "answer_is_correct": True,
        "answer_feedback": "Верно, химия — в отдельный пакет.",
        "next_question": None,
        "final_summary": "Итог: отличный результат, всё засчитано.",
    },
]


def _settings() -> Settings:
    return Settings(
        BOT_TOKEN="test-token",
        DATABASE_URL="sqlite+aiosqlite://",
        LLM_API_KEY="test-key",
        LLM_MODEL="fake",
        LLM_BASE_URL="http://localhost",
        ADMIN_API_TOKEN=ADMIN_TOKEN,
        QUALITY_CONTROL=False,
        TRAINING_MATERIAL="Бытовая химия упаковывается отдельно от продуктов.",
        TRAINING_TOPIC="Регламент сборки",
        QUIZ_QUESTION_COUNT=5,
    )


@pytest.fixture
async def client(make_gateway):
    gateway = make_gateway(payloads=list(TURN_PAYLOADS))
    app = create_app(settings=_settings(), llm_gateway=gateway)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def test_admin_endpoints_require_token(client):
    response = await client.get("/api/admin/codes")
    assert response.status_code == 401

    response = await client.get("/api/admin/codes", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert response.status_code == 200


async def test_full_web_flow_from_invite_to_result(client):
    # Вход без кода и с чужим кодом отклоняются.
    assert (await client.post("/api/session", json={})).status_code == 409
    assert (await client.post("/api/session", json={"invite_code": "WRONG"})).status_code == 409

    # HR создаёт одноразовый код приглашения.
    created = await client.post(
        "/api/admin/codes",
        json={"max_uses": 1},
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert created.status_code == 200
    code = created.json()["code"]

    # Сотрудник входит по коду и даёт согласие.
    started = await client.post("/api/session", json={"invite_code": code})
    assert started.status_code == 200
    assert started.json()["state"] == "awaiting_consent"

    consented = await client.post("/api/consent", json={"agree": True})
    assert consented.status_code == 200
    assert consented.json()["state"] == "awaiting_name"

    # Имя запускает обучение (первый вызов модели).
    named = await client.post("/api/name", json={"name": "Иван Петров"})
    assert named.status_code == 200
    assert named.json()["state"] == "active"
    assert "шагам" in named.json()["turn"]["reply"]

    # Переход к тесту командой /start_test (второй вызов модели).
    testing = await client.post("/api/message", json={"text": "/start_test"})
    assert testing.status_code == 200
    turn = testing.json()["turn"]
    assert turn["phase"] == "testing"
    assert "химию" in turn["next_question"]
    # Без базы знаний маршрут не задан (total=0) — ограничения на вход нет.
    assert testing.json()["coverage"] == {"covered": 0, "total": 0}

    # Ответ на вопрос завершает тест (третий вызов модели).
    finished = await client.post("/api/message", json={"text": "В отдельный пакет, отдельно от продуктов"})
    assert finished.status_code == 200
    body = finished.json()
    assert body["state"] == "completed"
    assert body["turn"]["phase"] == "completed"
    assert body["turn"]["answer_is_correct"] is True
    assert body["turn"]["final_summary"]

    # Результат сохранился и виден HR: 5 вопросов, 1 верный = 20%.
    results = await client.get("/api/admin/results", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert results.status_code == 200
    rows = results.json()["results"]
    assert len(rows) == 1
    assert rows[0]["employee_name"] == "Иван Петров"
    assert rows[0]["channel"] == "web"
    assert rows[0]["total_questions"] == 5
    assert rows[0]["correct_answers"] == 1
    assert rows[0]["score_percent"] == 20

    # Код приглашения израсходован: после выхода повторный вход невозможен.
    await client.post("/api/logout")
    again = await client.post("/api/session", json={"invite_code": code})
    assert again.status_code == 409


async def test_consent_refusal_stops_flow(client):
    created = await client.post(
        "/api/admin/codes",
        json={"max_uses": 1},
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    code = created.json()["code"]
    await client.post("/api/session", json={"invite_code": code})

    declined = await client.post("/api/consent", json={"agree": False})
    assert declined.status_code == 200
    assert declined.json()["state"] == "declined"

    # Без согласия диалог недоступен.
    blocked = await client.post("/api/name", json={"name": "Иван Петров"})
    assert blocked.status_code == 409


async def test_message_requires_active_session(client):
    response = await client.post("/api/message", json={"text": "Привет"})
    assert response.status_code == 401
