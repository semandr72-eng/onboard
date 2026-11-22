import csv
import io
import json
import logging
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import Settings, get_settings
from database import (
    AuditLogRepository,
    ConsentRepository,
    InviteCodeRepository,
    QualityCheckRepository,
    TrainingAnswerRepository,
    TrainingResultRepository,
    create_engine,
    create_session_factory,
    init_db,
)
from schemas import ChatMessage, TrainingAssistantTurn
from services import (
    AITrainingService,
    ConsentService,
    FastEmbedE5,
    LLMGateway,
    QualityJudge,
    RAGService,
    TrainingService,
)
from services.moderation import Moderator

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "onboard_sid"
COOKIE_MAX_AGE = 24 * 3600
HISTORY_LIMIT = 12  # сообщений (6 реплик наставника + 6 сотрудника)

FIRST_MESSAGE_TEXT = (
    "Привет! 👋 Давай изучим регламент по шагам: 1) начало смены и получение заказа, "
    "2) поиск товаров и замены, 3) упаковка и температурный режим, 4) завершение сборки. "
    "Начни с первого раздела маршрута: кратко объясни его суть простыми словами, с примерами. "
    "На мои вопросы отвечай по регламенту. К следующему разделу перейдёшь, когда я нажму «Понятно»; "
    "сам тест не начинай, пока я не нажму кнопку «Начать тест»."
)


class SessionStartIn(BaseModel):
    invite_code: str | None = Field(default=None, max_length=64)


class ConsentIn(BaseModel):
    agree: bool


class NameIn(BaseModel):
    name: str = Field(min_length=2, max_length=255)


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class AdminCodeIn(BaseModel):
    max_uses: int = Field(default=1, ge=1, le=1000)
    expires_in_days: int | None = Field(default=None, ge=1)


class KnowledgeIn(BaseModel):
    source: str = Field(default="manual", max_length=255)
    content: str = Field(min_length=1)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Некорректная дата: {value}") from exc


def create_app(settings: Settings | None = None, llm_gateway: LLMGateway | None = None) -> FastAPI:
    """Фабрика веб-приложения. Параметры нужны для тестов, в проде вызывается без аргументов."""
    settings = settings or get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    gateway = llm_gateway or LLMGateway(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    async def audit_sink(action: str, subject_id: str | None, status: str, meta: dict) -> None:
        async with session_factory() as session:
            await AuditLogRepository(session).create(action=action, status=status, subject_id=subject_id, meta=meta)

    async def quality_sink(kind: str, passed: bool, subject_id: str | None, detail: dict) -> None:
        async with session_factory() as session:
            await QualityCheckRepository(session).create(kind=kind, passed=passed, subject_id=subject_id, detail=detail)
            if not passed:
                await AuditLogRepository(session).create(
                    action="quality.warn",
                    status="warn",
                    subject_id=subject_id,
                    meta={"kind": kind, "detail": detail},
                )

    embedder = FastEmbedE5(model_name=settings.embedding_model)
    rag_service = RAGService(session_factory=session_factory, embedder=embedder, top_k=settings.knowledge_top_k)
    quality_judge = QualityJudge(gateway) if settings.quality_control else None
    ai_training_service = AITrainingService(
        settings=settings,
        llm_gateway=gateway,
        audit_sink=audit_sink,
        rag_service=rag_service,
        quality_judge=quality_judge,
        quality_sink=quality_sink,
    )
    training_service = TrainingService()
    consent_service = ConsentService(notice_version=settings.privacy_notice_version)
    moderator = Moderator()

    from web.sessions import WebSessionStore

    store = WebSessionStore()
    curriculum = {"total": 0}  # число разделов маршрута; 0 — маршрут не задан
    sections_state: dict[str, list[str]] = {"titles": []}  # названия разделов в порядке маршрута

    app = FastAPI(title="AI-наставник", docs_url=None, openapi_url=None)

    async def refresh_curriculum() -> None:
        try:
            titles = await rag_service.section_titles()
            sections_state["titles"] = titles
            curriculum["total"] = len(titles)
        except Exception:
            logger.exception("Failed to compute curriculum total")
            curriculum["total"] = 0
            sections_state["titles"] = []

    @app.on_event("startup")
    async def startup() -> None:
        await init_db(engine)
        await refresh_curriculum()

    def current_session(request: Request):
        session = store.get(request.cookies.get(SESSION_COOKIE))
        if session is None:
            raise HTTPException(status_code=401, detail="Сессия не найдена или истекла. Начните с кодом приглашения.")
        return session

    def require_admin(request: Request) -> None:
        if not settings.admin_api_token:
            raise HTTPException(status_code=503, detail="Админ-доступ выключен: задайте ADMIN_API_TOKEN.")
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {settings.admin_api_token}":
            raise HTTPException(status_code=401, detail="Неверный токен администратора.")

    def session_payload(session) -> dict:
        return {
            "state": session.state,
            "draft": session.draft.model_dump(),
            "consent_text": ConsentService.PRIVACY_NOTICE if session.state == "awaiting_consent" else None,
            "coverage": {
                "covered": len(session.draft.covered_sections),
                "total": curriculum["total"],
            },
        }

    def history_payload(session) -> list[ChatMessage]:
        return [ChatMessage.model_validate(message) for message in session.history]

    def append_history(session, user_text: str, assistant_reply: str) -> None:
        session.history.append({"role": "user", "content": user_text})
        session.history.append({"role": "assistant", "content": assistant_reply})
        if len(session.history) > HISTORY_LIMIT:
            session.history = session.history[-HISTORY_LIMIT:]

    async def persist_result(session) -> None:
        async with session_factory() as session_db:
            await training_service.create_result(
                repository=TrainingResultRepository(session_db),
                draft=session.draft,
                topic=settings.training_topic,
                subject_id=session.subject_id,
                channel="web",
                answer_repository=TrainingAnswerRepository(session_db),
            )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    async def get_state(request: Request) -> dict:
        return session_payload(current_session(request))

    @app.post("/api/session")
    async def start_session(payload: SessionStartIn, request: Request, response: Response) -> dict:
        existing = store.get(request.cookies.get(SESSION_COOKIE))
        if existing is not None:
            return session_payload(existing)

        if not payload.invite_code:
            raise HTTPException(status_code=409, detail="Нужен код приглашения.")

        async with session_factory() as session_db:
            invite = await InviteCodeRepository(session_db).consume(payload.invite_code.strip())
        if invite is None:
            raise HTTPException(status_code=409, detail="Код приглашения недействителен или уже использован.")

        web_session = store.create(total_questions=settings.quiz_question_count)
        response.set_cookie(
            SESSION_COOKIE,
            web_session.sid,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
        return session_payload(web_session)

    @app.post("/api/consent")
    async def give_consent(payload: ConsentIn, request: Request) -> dict:
        web_session = current_session(request)
        if web_session.state != "awaiting_consent":
            raise HTTPException(status_code=409, detail="Согласие сейчас не требуется.")

        if not payload.agree:
            web_session.state = "declined"
            return session_payload(web_session)

        async with session_factory() as session_db:
            await consent_service.grant(
                ConsentRepository(session_db),
                AuditLogRepository(session_db),
                web_session.subject_id,
            )
        web_session.state = "awaiting_name"
        return session_payload(web_session)

    @app.post("/api/name")
    async def set_name(payload: NameIn, request: Request) -> dict:
        web_session = current_session(request)
        if web_session.state != "awaiting_name":
            raise HTTPException(status_code=409, detail="Имя сейчас не запрашивается.")

        try:
            web_session.draft = training_service.register_employee_name(web_session.draft, payload.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            turn = await ai_training_service.generate_turn(
                draft=web_session.draft,
                user_message=FIRST_MESSAGE_TEXT,
                is_new_dialogue=True,
                subject_id=web_session.subject_id,
                history=history_payload(web_session),
            )
        except Exception as exc:
            logger.warning("AI turn failed on /api/name: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="ИИ-наставник временно недоступен. Попробуйте ещё раз или обратитесь к HR.",
            ) from exc
        if not moderator.is_clean(turn.reply):
            async with session_factory() as session_db:
                await AuditLogRepository(session_db).create(
                    action="moderation.warn",
                    status="warn",
                    subject_id=web_session.subject_id,
                    meta={"channel": "web", "direction": "out"},
                )
            turn = turn.model_copy(
                update={"reply": "Извини, не могу ответить в таком тоне. Давай вернёмся к регламенту 🙂"}
            )
        web_session.draft = training_service.apply_ai_turn(web_session.draft, turn)
        web_session.name_collected = True
        web_session.state = "active"
        return await _apply_turn(web_session, turn, FIRST_MESSAGE_TEXT)

    @app.post("/api/message")
    async def send_message(payload: MessageIn, request: Request) -> dict:
        web_session = current_session(request)
        if web_session.state != "active":
            raise HTTPException(status_code=409, detail="Сессия обучения не активна.")

        if not moderator.is_clean(payload.text):
            async with session_factory() as session_db:
                await AuditLogRepository(session_db).create(
                    action="moderation.warn",
                    status="warn",
                    subject_id=web_session.subject_id,
                    meta={"channel": "web", "direction": "in"},
                )
            raise HTTPException(status_code=409, detail="Давай без грубостей 🙂 Вернёмся к регламенту.")

        wants_test = payload.text.strip() == "/start_test"
        if wants_test and not await ai_training_service.is_curriculum_complete(web_session.draft):
            covered, total = await ai_training_service.curriculum_status(web_session.draft)
            raise HTTPException(
                status_code=409,
                detail=f"Сначала пройдите обучение: разделов пройдено {covered} из {total}.",
            )

        # «Понятно»: переход к следующему разделу ведёт код, не модель.
        titles = sections_state["titles"]
        display_text = payload.text
        llm_text = payload.text
        wants_next = payload.text.strip() == "/next_section" and web_session.draft.phase == "learning"
        if wants_next and titles:
            covered_count = len(web_session.draft.covered_sections)
            display_text = "Понятно 👌"
            if covered_count >= len(titles):
                # Все разделы уже пройдены: без обращения к ИИ — сразу отправляем к тесту.
                turn = TrainingAssistantTurn(
                    reply="Все разделы уже пройдены ✅ Жми «Начать тест» 🚀",
                    phase="learning",
                )
                web_session.draft = training_service.apply_ai_turn(web_session.draft, turn)
                return await _apply_turn(web_session, turn, display_text)
            web_session.draft = training_service.mark_section_covered(
                web_session.draft, titles[covered_count]
            )
            covered_count += 1
            if covered_count >= len(titles):
                llm_text = (
                    "[служебная пометка: сотрудник нажал «Понятно». Все разделы пройдены. "
                    "Кратко подведи итог обучения и предложи нажать «Начать тест».]"
                )
            else:
                llm_text = (
                    f"[служебная пометка: сотрудник нажал «Понятно». Переходи к разделу "
                    f"«{titles[covered_count]}»: кратко объясни его суть простыми словами, с примерами "
                    "из практики сборки, и предложи задать вопросы.]"
                )

        try:
            if wants_test:
                turn = await ai_training_service.start_testing(
                    draft=web_session.draft,
                    subject_id=web_session.subject_id,
                    history=history_payload(web_session),
                )
            else:
                turn = await ai_training_service.generate_turn(
                    draft=web_session.draft,
                    user_message=llm_text,
                    is_new_dialogue=False,
                    subject_id=web_session.subject_id,
                    history=history_payload(web_session),
                )
        except Exception as exc:
            logger.warning("AI turn failed on /api/message: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="ИИ-наставник временно недоступен. Попробуйте ещё раз или обратитесь к HR.",
            ) from exc

        if not moderator.is_clean(turn.reply):
            async with session_factory() as session_db:
                await AuditLogRepository(session_db).create(
                    action="moderation.warn",
                    status="warn",
                    subject_id=web_session.subject_id,
                    meta={"channel": "web", "direction": "out"},
                )
            turn = turn.model_copy(
                update={"reply": "Извини, не могу ответить в таком тоне. Давай вернёмся к регламенту 🙂"}
            )

        web_session.draft = training_service.apply_ai_turn(web_session.draft, turn)
        return await _apply_turn(web_session, turn, display_text)

    async def _apply_turn(web_session, turn: TrainingAssistantTurn, user_text: str) -> dict:
        append_history(web_session, user_text, turn.reply)
        body = session_payload(web_session)
        body["turn"] = {
            "reply": turn.reply,
            "phase": turn.phase,
            "answer_is_correct": turn.answer_is_correct,
            "answer_feedback": turn.answer_feedback,
            "next_question": turn.next_question,
            "final_summary": turn.final_summary,
        }
        if turn.phase == "completed":
            await persist_result(web_session)
            web_session.state = "completed"
            body["state"] = "completed"
        return body

    @app.post("/api/logout")
    async def logout(request: Request, response: Response) -> dict:
        sid = request.cookies.get(SESSION_COOKIE)
        if sid:
            store.drop(sid)
        response.delete_cookie(SESSION_COOKIE)
        return {"ok": True}

    @app.post("/api/admin/codes")
    async def create_code(payload: AdminCodeIn, request: Request) -> dict:
        require_admin(request)
        code = secrets.token_urlsafe(6).replace("-", "").replace("_", "")[:10].upper()
        expires_at = None
        if payload.expires_in_days is not None:
            expires_at = datetime.now(UTC) + timedelta(days=payload.expires_in_days)
        async with session_factory() as session_db:
            invite = await InviteCodeRepository(session_db).create(
                code=code, max_uses=payload.max_uses, expires_at=expires_at
            )
        return {
            "code": invite.code,
            "max_uses": invite.max_uses,
            "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
        }

    @app.get("/api/admin/codes")
    async def list_codes(request: Request) -> dict:
        require_admin(request)
        async with session_factory() as session_db:
            invites = await InviteCodeRepository(session_db).list_all()
        return {
            "codes": [
                {
                    "code": invite.code,
                    "max_uses": invite.max_uses,
                    "used_count": invite.used_count,
                    "active": invite.active,
                    "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
                    "created_at": invite.created_at.isoformat() if invite.created_at else None,
                }
                for invite in invites
            ]
        }

    @app.post("/api/admin/knowledge")
    async def upload_knowledge(payload: KnowledgeIn, request: Request) -> dict:
        require_admin(request)
        try:
            document_id = await rag_service.ingest(source=payload.source, text=payload.content)
            await refresh_curriculum()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"document_id": document_id}

    @app.get("/api/admin/knowledge")
    async def list_knowledge(request: Request) -> dict:
        require_admin(request)
        documents = await rag_service.list_documents()
        return {
            "documents": [
                {
                    "id": document.id,
                    "source": document.source,
                    "chunk_count": document.chunk_count,
                    "meta": json.loads(document.meta_json or "{}"),
                    "created_at": document.created_at.isoformat() if document.created_at else None,
                }
                for document in documents
            ]
        }

    @app.delete("/api/admin/knowledge/{document_id}")
    async def delete_knowledge(document_id: int, request: Request) -> dict:
        require_admin(request)
        deleted = await rag_service.delete_document(document_id)
        await refresh_curriculum()
        return {"deleted_chunks": deleted}

    @app.get("/api/admin/results")
    async def admin_results(
        request: Request,
        date_from: str | None = None,
        date_to: str | None = None,
        channel: str | None = None,
        min_score: int | None = None,
    ) -> dict:
        require_admin(request)
        async with session_factory() as session_db:
            rows = await TrainingResultRepository(session_db).list_filtered(
                date_from=_parse_datetime(date_from),
                date_to=_parse_datetime(date_to),
                channel=channel,
                min_score=min_score,
            )
        return {
            "results": [
                {
                    "id": row.id,
                    "employee_name": row.employee_name,
                    "channel": row.channel,
                    "topic": row.topic,
                    "total_questions": row.total_questions,
                    "correct_answers": row.correct_answers,
                    "score_percent": row.score_percent,
                    "final_summary": row.final_summary,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        }

    @app.get("/api/admin/stats")
    async def admin_stats(request: Request) -> dict:
        require_admin(request)
        async with session_factory() as session_db:
            repository = TrainingResultRepository(session_db)
            summary = await repository.stats_summary()
            by_channel = await repository.stats_by_channel()
            by_day = await repository.stats_by_day(days=30)
        return {"summary": summary, "by_channel": by_channel, "by_day": by_day}

    @app.get("/api/admin/gaps")
    async def admin_gaps(request: Request, limit: int = 20) -> dict:
        require_admin(request)
        async with session_factory() as session_db:
            gaps = await TrainingAnswerRepository(session_db).gaps(limit=limit)
        return {"gaps": gaps}

    @app.get("/api/admin/quality")
    async def admin_quality(request: Request) -> dict:
        require_admin(request)
        async with session_factory() as session_db:
            repository = QualityCheckRepository(session_db)
            stats = await repository.stats()
            failures = await repository.recent_failures(limit=20)
        return {
            "stats": stats,
            "recent_failures": [
                {
                    "id": row.id,
                    "kind": row.kind,
                    "subject_id": row.subject_id,
                    "detail": json.loads(row.detail_json or "{}"),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in failures
            ],
        }

    @app.get("/api/admin/export/results.csv")
    async def export_results(request: Request) -> Response:
        require_admin(request)
        async with session_factory() as session_db:
            rows = await TrainingResultRepository(session_db).list_filtered()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "id",
                "created_at",
                "channel",
                "employee_name",
                "topic",
                "total_questions",
                "correct_answers",
                "score_percent",
            ]
        )
        for row in rows:
            writer.writerow([
                row.id,
                row.created_at.isoformat() if row.created_at else "",
                row.channel,
                row.employee_name,
                row.topic,
                row.total_questions,
                row.correct_answers,
                row.score_percent,
            ])
        return Response(
            buffer.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=training_results.csv"},
        )

    return app
