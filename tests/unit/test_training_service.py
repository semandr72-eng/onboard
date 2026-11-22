import pytest
from pydantic import ValidationError

from schemas import TrainingAssistantTurn, TrainingSessionDraft
from schemas.training import Quiz
from services.training_service import TrainingService


def _turn(**overrides) -> TrainingAssistantTurn:
    defaults = {
        "reply": "Принято.",
        "phase": "testing",
        "latest_answer_evaluated": True,
        "answer_is_correct": True,
        "answer_feedback": "Верно!",
        "next_question": "Вопрос 2?",
    }
    defaults.update(overrides)
    return TrainingAssistantTurn(**defaults)


def _draft(**overrides) -> TrainingSessionDraft:
    defaults = {
        "employee_name": "Иван Петров",
        "phase": "testing",
        "total_questions": 3,
        "questions_answered": 1,
        "correct_answers": 1,
        "current_question": "Вопрос 1?",
    }
    defaults.update(overrides)
    return TrainingSessionDraft(**defaults)


def test_validate_employee_name_rejects_short_and_empty():
    with pytest.raises(ValueError):
        TrainingService.validate_employee_name("Я")
    with pytest.raises(ValueError):
        TrainingService.validate_employee_name("   ")


def test_validate_employee_name_normalizes_whitespace():
    assert TrainingService.validate_employee_name("  Иван   Петров  ") == "Иван Петров"


def test_register_employee_name_moves_to_learning():
    service = TrainingService()
    draft = service.register_employee_name(TrainingSessionDraft(), "Иван Петров")
    assert draft.employee_name == "Иван Петров"
    assert draft.phase == "learning"


def test_apply_ai_turn_appends_evaluated_answer_and_updates_score():
    service = TrainingService()
    draft = service.apply_ai_turn(_draft(), _turn())
    assert draft.questions_answered == 2
    assert draft.correct_answers == 2
    assert len(draft.evaluated_answers) == 1
    assert draft.evaluated_answers[0].question == "Вопрос 1?"
    assert draft.evaluated_answers[0].is_correct is True
    assert draft.current_question == "Вопрос 2?"


def test_repeated_question_does_not_burn_question_budget():
    """Повтор того же вопроса = повторная попытка: попытка фиксируется, бюджет не тратится."""
    service = TrainingService()
    turn = _turn(answer_is_correct=False, answer_feedback="Неверно, попробуй ещё раз.", next_question="Вопрос 1?")
    draft = service.apply_ai_turn(_draft(), turn)
    assert draft.questions_answered == 1  # не увеличился
    assert draft.correct_answers == 1  # не увеличился
    assert len(draft.evaluated_answers) == 1  # попытка зафиксирована
    assert draft.evaluated_answers[0].is_correct is False
    assert draft.current_question == "Вопрос 1?"


def test_wrong_answer_decrements_correct_count():
    service = TrainingService()
    draft = service.apply_ai_turn(_draft(correct_answers=1), _turn(answer_is_correct=False))
    assert draft.questions_answered == 2
    assert draft.correct_answers == 1


def test_completed_turn_sets_summary_and_phase():
    service = TrainingService()
    turn = _turn(
        phase="completed",
        next_question=None,
        final_summary="Итог: 2 из 3. Рекомендую повторить упаковку.",
    )
    draft = service.apply_ai_turn(_draft(), turn)
    assert draft.phase == "completed"
    assert draft.final_summary is not None
    assert "2 из 3" in draft.final_summary


def test_learning_turn_does_not_evaluate_answers():
    service = TrainingService()
    draft = service.apply_ai_turn(
        _draft(phase="learning", current_question=None),
        _turn(phase="learning", latest_answer_evaluated=False, answer_is_correct=None, next_question=None),
    )
    assert draft.evaluated_answers == []
    assert draft.questions_answered == 1


def test_apply_ai_turn_collects_completed_section():
    service = TrainingService()
    turn = _turn(
        phase="learning",
        latest_answer_evaluated=False,
        answer_is_correct=None,
        next_question=None,
        section_completed="ЧАНК 1. Смена",
    )
    draft = service.apply_ai_turn(_draft(phase="learning", current_question=None), turn)
    assert draft.covered_sections == ["ЧАНК 1. Смена"]


def test_apply_ai_turn_deduplicates_completed_sections():
    service = TrainingService()
    turn = _turn(
        phase="learning",
        latest_answer_evaluated=False,
        answer_is_correct=None,
        next_question=None,
        section_completed="ЧАНК 1. Смена",
    )
    draft = service.apply_ai_turn(
        _draft(phase="learning", current_question=None, covered_sections=["ЧАНК 1. Смена"]),
        turn,
    )
    assert draft.covered_sections == ["ЧАНК 1. Смена"]


def test_quiz_validates_options_count():
    quiz = Quiz(question="Куда кладём заморозку?", options=["В термопакет", "В подсобку"])
    assert quiz.options == ["В термопакет", "В подсобку"]

    with pytest.raises(ValidationError):
        Quiz(question="Куда?", options=["Только один вариант"])

    turn = _turn(
        phase="learning",
        latest_answer_evaluated=False,
        answer_is_correct=None,
        next_question=None,
        quiz={"question": "Куда кладём заморозку?", "options": ["В термопакет", "В подсобку"]},
    )
    assert turn.quiz is not None
    assert turn.quiz.question == "Куда кладём заморозку?"
