import json
import logging

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bot.keyboards import cancel_keyboard, confirm_delete_keyboard, consent_keyboard, remove_keyboard
from config import Settings
from database import AuditLogRepository, ConsentRepository, TrainingAnswerRepository, TrainingResultRepository
from schemas import ChatMessage, TrainingSessionDraft
from services import AITrainingService, ConsentService, PersonalDataService, TrainingService

logger = logging.getLogger(__name__)
router = Router()


class TrainingStates(StatesGroup):
    awaiting_consent = State()
    active = State()


def _subject_id(message: Message) -> str | None:
    return f"tg:{message.from_user.id}" if message.from_user else None


def _callback_message(callback: CallbackQuery) -> Message | None:
    """Сообщение колбэка, если оно ещё доступно для редактирования/ответа."""
    return callback.message if isinstance(callback.message, Message) else None


async def _ask_employee_name(
    message: Message, state: FSMContext, settings: Settings, training_service: TrainingService
) -> None:
    await state.set_state(TrainingStates.active)
    await state.update_data(
        draft=training_service.start_session(settings.quiz_question_count).model_dump(),
        result_id=None,
        name_collected=False,
    )
    await message.answer(
        "Спасибо! Теперь напишите имя сотрудника, которого нужно обучить.",
        reply_markup=cancel_keyboard(),
    )


@router.message(Command("start"))
async def handle_start(
    message: Message,
    state: FSMContext,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    consent_service: ConsentService,
    training_service: TrainingService,
) -> None:
    await state.clear()
    subject_id = _subject_id(message)

    has_consent = False
    if subject_id is not None:
        async with session_factory() as session:
            has_consent = await consent_service.has_valid_consent(ConsentRepository(session), subject_id)

    if not has_consent:
        await state.set_state(TrainingStates.awaiting_consent)
        await message.answer(consent_service.PRIVACY_NOTICE, reply_markup=consent_keyboard())
        return

    await message.answer(
        "Здравствуйте! Я помогу изучить новый материал, а затем проведу тестирование.",
        reply_markup=cancel_keyboard(),
    )
    await _ask_employee_name(message, state, settings, training_service)


@router.callback_query(StateFilter(TrainingStates.awaiting_consent), F.data == "consent:agree")
async def handle_consent_agree(
    callback: CallbackQuery,
    state: FSMContext,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    consent_service: ConsentService,
    training_service: TrainingService,
) -> None:
    if callback.from_user is None or callback.message is None:
        await callback.answer()
        return
    async with session_factory() as session:
        await consent_service.grant(
            ConsentRepository(session),
            AuditLogRepository(session),
            f"tg:{callback.from_user.id}",
        )
    await callback.answer("Спасибо за согласие!")
    message = _callback_message(callback)
    if message is None:
        await callback.answer("Напишите /start, чтобы продолжить.", show_alert=True)
        return
    await message.edit_reply_markup(reply_markup=None)
    await _ask_employee_name(message, state, settings, training_service)


@router.callback_query(StateFilter(TrainingStates.awaiting_consent), F.data == "consent:decline")
async def handle_consent_decline(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    message = _callback_message(callback)
    if message is not None:
        await message.edit_text(
            "Без согласия на обработку персональных данных я не смогу провести обучение. "
            "Если передумаете — отправьте /start."
        )
    await callback.answer()


@router.message(Command("export"))
async def handle_export(
    message: Message,
    session_factory: async_sessionmaker[AsyncSession],
    personal_data_service: PersonalDataService,
) -> None:
    subject_id = _subject_id(message)
    if subject_id is None:
        await message.answer("Не удалось определить ваш идентификатор.")
        return

    async with session_factory() as session:
        data = await personal_data_service.export_data(
            TrainingResultRepository(session),
            ConsentRepository(session),
            AuditLogRepository(session),
            subject_id,
        )
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    await message.answer_document(
        BufferedInputFile(payload, filename=f"personal_data_{subject_id.replace(':', '_')}.json"),
        caption="Выгрузка ваших персональных данных.",
    )


@router.message(Command("delete"))
async def handle_delete(message: Message, state: FSMContext) -> None:
    await state.update_data(pending_delete=True)
    await message.answer(
        "Это удалит все ваши результаты обучения и отзовет согласие на обработку данных. Продолжить?",
        reply_markup=confirm_delete_keyboard(),
    )


@router.callback_query(F.data == "delete:confirm")
async def handle_delete_confirm(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    personal_data_service: PersonalDataService,
) -> None:
    state_data = await state.get_data()
    if not state_data.get("pending_delete"):
        await callback.answer("Этот запрос уже обработан.", show_alert=True)
        return
    if callback.from_user is None:
        await callback.answer()
        return

    async with session_factory() as session:
        outcome = await personal_data_service.delete_data(
            TrainingResultRepository(session),
            ConsentRepository(session),
            AuditLogRepository(session),
            f"tg:{callback.from_user.id}",
        )
    await state.update_data(pending_delete=False)
    await callback.answer("Данные удалены")
    message = _callback_message(callback)
    if message is not None:
        await message.edit_text(
            f"Готово. Удалено результатов обучения: {outcome['deleted_results']}. "
            "Согласие отозвано — для нового обучения нужно будет согласиться заново."
        )


@router.callback_query(F.data == "delete:cancel")
async def handle_delete_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await state.update_data(pending_delete=False)
    await callback.answer("Удаление отменено")
    message = _callback_message(callback)
    if message is not None:
        await message.edit_text("Удаление отменено. Ваши данные сохранены.")


@router.message(Command("test"), TrainingStates.active)
async def handle_start_test(
    message: Message,
    state: FSMContext,
    training_service: TrainingService,
    ai_training_service: AITrainingService,
) -> None:
    """Явный переход к тесту по команде сотрудника (только из обучения)."""
    state_data = await state.get_data()
    draft = TrainingSessionDraft.model_validate(state_data.get("draft", {}))
    if draft.phase != "learning":
        await message.answer(
            "Сейчас не идёт обучение — тест уже начат или сессия завершена.",
            reply_markup=cancel_keyboard(),
        )
        return

    if not await ai_training_service.is_curriculum_complete(draft):
        covered, total = await ai_training_service.curriculum_status(draft)
        await message.answer(
            f"Тест откроется после прохождения всего обучения: разделов пройдено {covered} из {total}. "
            "Продолжим — просто напиши сообщение.",
            reply_markup=cancel_keyboard(),
        )
        return

    history = [ChatMessage.model_validate(item) for item in state_data.get("history", [])]
    subject_id = _subject_id(message)
    try:
        ai_turn = await ai_training_service.start_testing(
            draft=draft,
            subject_id=subject_id,
            history=history,
        )
        updated_draft = training_service.apply_ai_turn(draft, ai_turn)
        await state.update_data(draft=updated_draft.model_dump())
        history.append(ChatMessage(role="user", content="Я готов — начинаем тест."))
        history.append(ChatMessage(role="assistant", content=ai_turn.reply))
        await state.update_data(history=[item.model_dump() for item in history[-12:]])
        await message.answer(ai_turn.reply, reply_markup=cancel_keyboard())
        if ai_turn.next_question and ai_turn.next_question not in ai_turn.reply:
            await message.answer(f"❓ {ai_turn.next_question}", reply_markup=cancel_keyboard())
    except Exception:
        logger.exception("Failed to start testing")
        await message.answer(
            "Не удалось начать тест. Попробуй ещё раз или отправь /cancel.",
            reply_markup=cancel_keyboard(),
        )


@router.message(Command("cancel"))
async def handle_cancel(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Сейчас нет активной сессии обучения.", reply_markup=remove_keyboard())
        return

    await state.clear()
    await message.answer(
        "Сессия обучения отменена. Чтобы начать заново, отправьте /start.",
        reply_markup=remove_keyboard(),
    )


@router.message(TrainingStates.active, F.text)
async def process_ai_training(
    message: Message,
    state: FSMContext,
    settings: Settings,
    training_service: TrainingService,
    ai_training_service: AITrainingService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    state_data = await state.get_data()
    draft = TrainingSessionDraft.model_validate(state_data.get("draft", {}))
    name_collected = bool(state_data.get("name_collected"))
    user_text = message.text or ""
    subject_id = _subject_id(message)
    raw_history = state_data.get("history", [])
    history = [ChatMessage.model_validate(item) for item in raw_history]

    try:
        if not name_collected:
            updated_draft = training_service.register_employee_name(draft=draft, employee_name=user_text)
            await state.update_data(draft=updated_draft.model_dump(), name_collected=True)
            first_message = (
                "Привет! 👋 Давай изучим регламент по шагам: 1) начало смены и получение заказа, "
                "2) поиск товаров и замены, 3) упаковка и температурный режим, 4) завершение сборки. "
                "Объясняй простыми словами, приводи примеры и задавай мне проверочные вопросы по ходу, "
                "чтобы убедиться, что я понимаю. Когда скажу, что всё ясно — предложи разобрать что-то ещё "
                "или дождись, пока я нажму кнопку «Начать тест». Сам тест пока не начинай."
            )
            ai_turn = await ai_training_service.generate_turn(
                draft=updated_draft,
                user_message=first_message,
                is_new_dialogue=True,
                subject_id=subject_id,
                history=history,
            )
            updated_draft = training_service.apply_ai_turn(updated_draft, ai_turn)
            await state.update_data(draft=updated_draft.model_dump())
            history.append(ChatMessage(role="user", content=first_message))
        else:
            ai_turn = await ai_training_service.generate_turn(
                draft=draft,
                user_message=user_text,
                is_new_dialogue=False,
                subject_id=subject_id,
                history=history,
            )
            updated_draft = training_service.apply_ai_turn(draft, ai_turn)
            await state.update_data(draft=updated_draft.model_dump())
            history.append(ChatMessage(role="user", content=user_text))

        history.append(ChatMessage(role="assistant", content=ai_turn.reply))
        if len(history) > 12:
            history = history[-12:]
        await state.update_data(history=[item.model_dump() for item in history])

        if ai_turn.phase == "completed":
            async with session_factory() as session:
                repository = TrainingResultRepository(session)
                await training_service.create_result(
                    repository=repository,
                    draft=updated_draft,
                    topic=settings.training_topic,
                    subject_id=subject_id or "tg:unknown",
                    channel="telegram",
                    answer_repository=TrainingAnswerRepository(session),
                )

            await state.clear()
            await message.answer(
                f"{ai_turn.reply}\n\n"
                f"Результат сохранен в Postgres.\n"
                f"Итог: {updated_draft.correct_answers}/{updated_draft.total_questions} "
                f"({updated_draft.score_percent()}%).",
                reply_markup=remove_keyboard(),
            )
            return

        await message.answer(ai_turn.reply, reply_markup=cancel_keyboard())
    except ValueError as exc:
        await message.answer(str(exc), reply_markup=cancel_keyboard())
    except Exception:
        logger.exception("Failed to process AI training")
        await message.answer(
            "Не удалось обработать сообщение. Попробуйте еще раз или отправьте /cancel.",
            reply_markup=cancel_keyboard(),
        )


@router.message(TrainingStates.active)
async def handle_invalid_collecting_input(message: Message) -> None:
    await message.answer("Пожалуйста, отправьте ответ текстом.", reply_markup=cancel_keyboard())


@router.message(F.text)
async def handle_text_without_flow(message: Message) -> None:
    await message.answer("Чтобы начать обучение и тестирование, отправьте /start.")


@router.message()
async def handle_unsupported_input(message: Message) -> None:
    await message.answer("Пожалуйста, используйте текстовые сообщения или команду /start.")
