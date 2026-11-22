import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from config import Settings
from schemas import ChatMessage, TrainingAssistantTurn, TrainingSessionDraft
from services.ai_training_prompts import AI_TRAINING_RESPONSE_SCHEMA, build_training_system_prompt
from services.anonymizer import Anonymizer
from services.llm_gateway import LLMGateway
from services.quality_service import QualityJudge
from services.rag_service import RAGService

logger = logging.getLogger(__name__)

AuditSink = Callable[[str, str | None, str, dict[str, Any]], Awaitable[None]]
QualitySink = Callable[[str, bool, str | None, dict[str, Any]], Awaitable[None]]


class AITrainingService:
    """Сценарий обучения: RAG -> промпт -> обезличивание -> LLM Gateway -> обратная подстановка -> аудит.

    Контекст для ответа берется из базы знаний (RAGService): по сообщению сотрудника
    и текущему вопросу теста подбираются релевантные фрагменты. Если база знаний
    пуста или недоступна — используется статический материал из настроек.
    Персональные данные (ФИО сотрудника и любые ПДн в тексте) заменяются на
    плейсхолдеры до отправки во внешний LLM; обезличенный текст покидает
    периметр, исходные значения остаются только локально в маппинге анонимайзера.
    Аудит фиксирует факт обращения к LLM: хэш ушедшего промпта, число
    замененных ПДн по типам, токены и латентность — без самих значений.
    """

    def __init__(
        self,
        settings: Settings,
        llm_gateway: LLMGateway,
        audit_sink: AuditSink | None = None,
        rag_service: RAGService | None = None,
        quality_judge: QualityJudge | None = None,
        quality_sink: QualitySink | None = None,
    ) -> None:
        self._settings = settings
        self._training_material = settings.get_training_material()
        self._llm = llm_gateway
        self._audit_sink = audit_sink
        self._rag = rag_service
        self._quality_judge = quality_judge
        self._quality_sink = quality_sink

    async def start_testing(
        self,
        draft: TrainingSessionDraft,
        subject_id: str | None = None,
        history: list[ChatMessage] | None = None,
    ) -> TrainingAssistantTurn:
        """Детерминированный переход к тесту: состоянием управляет код, модель даёт только содержимое.

        Черновик принудительно переводится в testing до вызова модели; если модель
        всё равно вернула learning или не задала вопрос — выполняется один повтор
        с жёсткой инструкцией, после чего phase выставляется принудительно.
        """
        testing_draft = draft.model_copy(update={"phase": "testing"})
        transition_message = (
            "Я прошёл обучение и нажимаю «Начать тест». Начни тестирование: "
            "phase=testing, в next_question задай первый вопрос по материалу, "
            "в reply — одна короткая фраза с пожеланием удачи и одним эмодзи (например 🍀), без пересказа материала."
        )
        turn = await self.generate_turn(
            draft=testing_draft,
            user_message=transition_message,
            is_new_dialogue=False,
            subject_id=subject_id,
            history=history,
        )
        if turn.phase != "testing" or not turn.next_question:
            turn = await self.generate_turn(
                draft=testing_draft,
                user_message="Задай первый вопрос теста по материалу. Ответь строго по схеме: phase=testing, next_question — текст первого вопроса, reply — короткое приветствие.",
                is_new_dialogue=False,
                subject_id=subject_id,
                history=history,
            )
        turn.phase = "testing"
        return turn

    async def generate_turn(
        self,
        draft: TrainingSessionDraft,
        user_message: str,
        is_new_dialogue: bool,
        subject_id: str | None = None,
        history: list[ChatMessage] | None = None,
    ) -> TrainingAssistantTurn:
        hits = []
        if self._rag is not None:
            query = self._build_retrieval_query(draft, user_message)
            try:
                hits = await self._rag.retrieve(query)
            except Exception:
                logger.exception("RAG retrieval failed, falling back to static material")

        material = self._training_material
        if hits and self._rag is not None:
            material = self._rag.build_context(hits)
        messages = self._build_messages(draft, user_message, is_new_dialogue, material, history)

        anonymizer = Anonymizer(known_names=[draft.employee_name] if draft.employee_name else None)
        for message in messages:
            message["content"] = anonymizer.anonymize(message["content"])

        prompt_fingerprint = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        started_at = time.monotonic()
        try:
            response = await self._llm.generate_json(
                messages=messages,
                json_schema=AI_TRAINING_RESPONSE_SCHEMA,
            )
            turn = TrainingAssistantTurn.model_validate(response.payload)
        except Exception as exc:
            await self._write_audit(
                subject_id=subject_id,
                status="error",
                meta={
                    "model": self._llm.model,
                    "prompt_sha256": prompt_fingerprint,
                    "pii": anonymizer.stats,
                    "rag_chunks": len(hits),
                    "error": type(exc).__name__,
                },
            )
            raise
        latency_ms = round((time.monotonic() - started_at) * 1000)

        if self._settings.debug_log_dialogues:
            if history:
                logger.info(
                    "[dialogue] history: %s",
                    json.dumps([message.model_dump() for message in history], ensure_ascii=False),
                )
            logger.info("[dialogue] user: %s", user_message)
            logger.info("[dialogue] assistant: %s", turn.reply)

        quality: dict[str, Any] = {}
        if self._quality_judge is not None:
            # Faithfulness: реплика наставника (еще обезличенная) против материала.
            try:
                faith = await self._quality_judge.check_faithfulness(turn.reply, material)
                quality["faithfulness"] = faith
                if self._quality_sink is not None:
                    await self._quality_sink("faithfulness", faith["grounded"], subject_id, faith)
            except Exception:
                logger.exception("Faithfulness check failed")
            # Аудит оценки: независимый вердикт судьи; при расхождении — пересмотр.
            if (
                turn.latest_answer_evaluated
                and draft.phase == "testing"
                and draft.current_question
                and turn.answer_is_correct is not None
            ):
                try:
                    anonymized_answer = anonymizer.anonymize(user_message)
                    judge_correct, judge_justification = await self._quality_judge.grade_answer(
                        question=draft.current_question,
                        answer=anonymized_answer,
                        context=material,
                    )
                    quality["judge"] = {"is_correct": judge_correct, "justification": judge_justification}
                    quality["verdict_match"] = judge_correct == turn.answer_is_correct
                    if judge_correct != turn.answer_is_correct:
                        # Судья видит чистую тройку «вопрос-ответ-материал», наставник — всю историю
                        # диалога и склонен к «поддакиванию» или избыточной строгости. Вердикт судьи приоритетен.
                        turn.answer_is_correct = judge_correct
                        if judge_correct:
                            turn.answer_feedback = f"Ответ засчитан после пересмотра: {judge_justification}"
                        quality["verdict_overridden"] = True
                        logger.info("Verdict overridden by judge: %s", judge_justification)
                    if self._quality_sink is not None:
                        await self._quality_sink(
                            "grade",
                            judge_correct,
                            subject_id,
                            {**quality["judge"], "question": draft.current_question},
                        )
                except Exception:
                    logger.exception("Grade audit failed")

        turn.reply = anonymizer.deanonymize(turn.reply)
        if turn.answer_feedback:
            turn.answer_feedback = anonymizer.deanonymize(turn.answer_feedback)
        if turn.next_question:
            turn.next_question = anonymizer.deanonymize(turn.next_question)
        if turn.final_summary:
            turn.final_summary = anonymizer.deanonymize(turn.final_summary)

        if (
            self._rag is not None
            and turn.latest_answer_evaluated
            and draft.phase == "testing"
            and draft.current_question
            and turn.answer_is_correct is not None
        ):
            try:
                turn.question_chunk_title = await self._rag.attribute_chunk(draft.current_question)
            except Exception:
                logger.exception("Chunk attribution failed")

        await self._write_audit(
            subject_id=subject_id,
            status="ok",
            meta={
                "model": response.model,
                "prompt_sha256": prompt_fingerprint,
                "pii": anonymizer.stats,
                "rag_chunks": len(hits),
                "latency_ms": latency_ms,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "quality": quality or None,
            },
        )
        return turn

    async def _write_audit(self, subject_id: str | None, status: str, meta: dict[str, Any]) -> None:
        if self._audit_sink is None:
            return
        try:
            await self._audit_sink(
                "llm.request",
                subject_id,
                status,
                meta,
            )
        except Exception:
            logger.exception("Failed to write audit record")

    @staticmethod
    def _build_retrieval_query(draft: TrainingSessionDraft, user_message: str) -> str:
        """Запрос к базе знаний: по умолчанию от сообщения сотрудника.

        Во время теста короткие сообщения — это ответы на текущий вопрос,
        поэтому ищем контекст по вопросу (для оценки). Длинные сообщения во время
        теста — уточнения и вопросы, ищем по самому сообщению.
        """
        message = user_message.strip()
        if draft.phase == "testing" and draft.current_question and 0 < len(message) <= 40:
            return " ".join(part for part in (draft.current_question, message) if part)
        return message or draft.current_question or ""

    def _build_messages(
        self,
        draft: TrainingSessionDraft,
        user_message: str,
        is_new_dialogue: bool,
        material: str,
        history: list[ChatMessage] | None = None,
    ) -> list[dict[str, str]]:
        serialized_draft = json.dumps(
            draft.model_dump(exclude={"employee_name"}),
            ensure_ascii=False,
            indent=2,
        )
        history_messages = [
            {"role": message.role, "content": message.content}
            for message in (history or [])
        ]
        return [
            {
                "role": "system",
                "content": build_training_system_prompt(
                    topic=self._settings.training_topic,
                    material=material,
                    total_questions=draft.total_questions,
                ),
            },
            *history_messages,
            {
                "role": "user",
                "content": (
                    f"Новая сессия: {str(is_new_dialogue).lower()}\n"
                    f"Текущее состояние сессии:\n{serialized_draft}\n\n"
                    f"Последнее сообщение сотрудника:\n{user_message}\n\n"
                    "Важно:\n"
                    "- на первом ходе новой сессии (Новая сессия: true) всегда начинай с обучения: кратко и понятно объясни материал своими словами, не начинай тест сразу;\n"
                    "- если phase сейчас learning, сначала обучай и только потом переводи в testing;\n"
                    "- если сотрудник явно просит начать тест («готов к тесту», «давай тест», «начинай тест») и phase сейчас learning, "
                    "переходи к тесту: phase=testing и первый вопрос в next_question, без повторного пересказа материала;\n"
                    "- «понятно», «ок», «ясно», «угу» — подтверждение понимания, а НЕ готовность к тесту: продолжай обучение "
                    "или спроси, какой раздел разобрать подробнее;\n"
                    "- если phase сейчас testing, но сообщение сотрудника — вопрос или просьба объяснить, а не ответ на current_question, "
                    "не оценивай его (latest_answer_evaluated = false), ответь на вопрос по материалу и вернись к текущему вопросу теста;\n"
                    "- если phase сейчас testing и current_question заполнен, оцени именно ответ на current_question;\n"
                    "- оценивай смысл ответа, а не дословное совпадение с материалом: если по смыслу ответ совпадает с материалом, ставь answer_is_correct=true, даже при другой формулировке; "
                    "answer_is_correct=false только при противоречии материалу или пропуске обязательного ключевого условия вопроса; "
                    "если не хватает второстепенной детали — ставь true и добавь уточнение в обратную связь;\n"
                    "- формулировка вердикта в reply и флаг answer_is_correct должны совпадать: «Верно» — только при true; при false пиши «Неверно» или «Не совсем» и не противоречь флагу;\n"
                    "- questions_answered уже содержит число проверенных ответов;\n"
                    "- когда проверенных ответов станет столько же, сколько total_questions, заверши сессию через phase=completed;\n"
                    "- после оценки ответа в тесте задавай следующий новый вопрос, отличный от текущего; повторять уже заданный вопрос запрещено."
                ),
            },
        ]
