import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from schemas import TrainingSessionDraft


@dataclass
class WebSession:
    """Сессия обучения в веб-канале. Живет в памяти процесса, TTL — 24 часа."""

    sid: str
    subject_id: str  # "web:<uuid>"
    state: str  # awaiting_consent | awaiting_name | active | completed | declined
    draft: TrainingSessionDraft
    expires_at: datetime
    name_collected: bool = False
    history: list[dict] = field(default_factory=list)


@dataclass
class WebSessionStore:
    ttl: timedelta = timedelta(hours=24)
    _sessions: dict[str, WebSession] = field(default_factory=dict)

    def create(self, total_questions: int) -> WebSession:
        self._purge_expired()
        sid = secrets.token_urlsafe(32)
        session = WebSession(
            sid=sid,
            subject_id=f"web:{secrets.token_urlsafe(16)}",
            state="awaiting_consent",
            draft=TrainingSessionDraft(total_questions=total_questions),
            expires_at=datetime.now(UTC) + self.ttl,
        )
        self._sessions[sid] = session
        return session

    def get(self, sid: str | None) -> WebSession | None:
        if not sid:
            return None
        session = self._sessions.get(sid)
        if session is None:
            return None
        if session.expires_at < datetime.now(UTC):
            self._sessions.pop(sid, None)
            return None
        return session

    def drop(self, sid: str) -> None:
        self._sessions.pop(sid, None)

    def _purge_expired(self) -> None:
        now = datetime.now(UTC)
        expired = [sid for sid, s in self._sessions.items() if s.expires_at < now]
        for sid in expired:
            self._sessions.pop(sid, None)
