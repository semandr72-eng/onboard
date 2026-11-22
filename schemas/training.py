from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


class ChatMessage(BaseModel):
    """Сообщение диалога для передачи истории в контекст модели."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class EvaluatedAnswer(BaseModel):
    """Оцененный ответ на вопрос теста. Копится в черновике сессии, сохраняется вместе с результатом."""

    question: str = Field(min_length=1, max_length=2000)
    is_correct: bool
    feedback: str | None = Field(default=None, max_length=2000)
    chunk_title: str | None = Field(default=None, max_length=512)

    @field_validator("question", "feedback", "chunk_title")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_text(value)


class TrainingResultCreate(BaseModel):
    employee_name: str = Field(min_length=2, max_length=255)
    subject_id: str = Field(min_length=2, max_length=64)
    channel: Literal["telegram", "web"] = "telegram"
    topic: str = Field(min_length=2, max_length=255)
    total_questions: int = Field(ge=1, le=20)
    correct_answers: int = Field(ge=0, le=20)
    score_percent: int = Field(ge=0, le=100)
    final_summary: str | None = Field(default=None, max_length=4000)

    @field_validator("employee_name", "topic", "final_summary")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_text(value)


class TrainingResultRead(TrainingResultCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingSessionDraft(BaseModel):
    employee_name: str | None = None
    phase: Literal["collecting_name", "learning", "testing", "completed"] = "collecting_name"
    total_questions: int = Field(default=5, ge=1, le=20)
    questions_answered: int = Field(default=0, ge=0, le=20)
    correct_answers: int = Field(default=0, ge=0, le=20)
    current_question: str | None = None
    last_answer_feedback: str | None = None
    final_summary: str | None = None
    evaluated_answers: list[EvaluatedAnswer] = Field(default_factory=list)
    # Названия разделов регламента, пройденных в обучении (маршрут: все разделы -> тест).
    covered_sections: list[str] = Field(default_factory=list)
    # План теста: разделы, выбранные сервером случайно на старте теста (пусто — свободный режим).
    test_plan: list[str] = Field(default_factory=list)

    @field_validator("employee_name", "current_question", "last_answer_feedback", "final_summary")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _normalize_text(value)

    def remaining_questions(self) -> int:
        return max(self.total_questions - self.questions_answered, 0)

    def score_percent(self) -> int:
        if self.total_questions == 0:
            return 0
        return round((self.correct_answers / self.total_questions) * 100)


class TrainingAssistantTurn(BaseModel):
    reply: str = Field(min_length=1)
    phase: Literal["learning", "testing", "completed"]
    latest_answer_evaluated: bool = False
    answer_is_correct: bool | None = None
    answer_feedback: str | None = None
    next_question: str | None = None
    final_summary: str | None = None
    # Устарело: маршрут ведёт код (кнопка «Понятно»). Поле оставлено для совместимости, не используется.
    section_completed: str | None = None
    # Заполняется сервисом (не моделью): план теста — разделы для вопросов, выбранные на старте теста.
    test_plan: list[str] = Field(default_factory=list)
    # Заполняется сервисом (не моделью): к какому чанку базы знаний отнесен оцененный вопрос.
    question_chunk_title: str | None = None

    @field_validator("answer_feedback", "next_question", "final_summary")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _normalize_text(value)
