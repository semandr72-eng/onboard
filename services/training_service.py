from database.models import TrainingResult
from database.repository import TrainingAnswerRepository, TrainingResultRepository
from schemas import EvaluatedAnswer, TrainingAssistantTurn, TrainingResultCreate, TrainingSessionDraft


class TrainingService:
    @staticmethod
    def validate_employee_name(value: str) -> str:
        cleaned = " ".join(value.split()).strip()
        if len(cleaned) < 2:
            raise ValueError("Укажите имя сотрудника хотя бы из двух символов.")
        return cleaned

    def start_session(self, total_questions: int) -> TrainingSessionDraft:
        return TrainingSessionDraft(total_questions=total_questions)

    def register_employee_name(self, draft: TrainingSessionDraft, employee_name: str) -> TrainingSessionDraft:
        updated = TrainingSessionDraft.model_validate(draft.model_dump())
        updated.employee_name = self.validate_employee_name(employee_name)
        updated.phase = "learning"
        return updated

    def apply_ai_turn(
        self,
        current: TrainingSessionDraft,
        ai_turn: TrainingAssistantTurn,
    ) -> TrainingSessionDraft:
        updated = TrainingSessionDraft.model_validate(current.model_dump())
        updated.phase = ai_turn.phase

        if ai_turn.latest_answer_evaluated and current.phase == "testing" and current.current_question:
            if ai_turn.answer_is_correct is not None:
                updated.evaluated_answers.append(
                    EvaluatedAnswer(
                        question=current.current_question,
                        is_correct=ai_turn.answer_is_correct,
                        feedback=ai_turn.answer_feedback,
                        chunk_title=ai_turn.question_chunk_title,
                    )
                )
            # Повтор того же вопроса = повторная попытка: фиксируем попытку в evaluated_answers,
            # но не сжигаем бюджет total_questions и не меняем счётчик верных.
            question_repeated = (
                ai_turn.phase != "completed" and ai_turn.next_question == current.current_question
            )
            if not question_repeated:
                updated.questions_answered += 1
                if ai_turn.answer_is_correct:
                    updated.correct_answers += 1

        if ai_turn.answer_feedback is not None:
            updated.last_answer_feedback = ai_turn.answer_feedback

        updated.current_question = ai_turn.next_question

        if ai_turn.final_summary is not None:
            updated.final_summary = ai_turn.final_summary

        return updated

    async def create_result(
        self,
        repository: TrainingResultRepository,
        draft: TrainingSessionDraft,
        topic: str,
        subject_id: str,
        channel: str,
        answer_repository: TrainingAnswerRepository | None = None,
    ) -> TrainingResult:
        result_in = TrainingResultCreate(
            employee_name=draft.employee_name or "Неизвестный сотрудник",
            subject_id=subject_id,
            channel=channel,
            topic=topic,
            total_questions=draft.total_questions,
            correct_answers=draft.correct_answers,
            score_percent=draft.score_percent(),
            final_summary=draft.final_summary,
        )
        result = await repository.create(result_in)
        if answer_repository is not None and draft.evaluated_answers:
            await answer_repository.create_many(result.id, draft.evaluated_answers)
        return result
