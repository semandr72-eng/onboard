import re


class Anonymizer:
    """Обезличивание персональных данных перед отправкой во внешний LLM.

    Заменяет известные ФИО, ФИО с отчеством, телефоны, e-mail и telegram-ники
    на плейсхолдеры вида ``[ФИО_1]``, ``[ТЕЛЕФОН_1]``, ``[EMAIL_1]``, ``[TELEGRAM_1]``.
    Маппинг «плейсхолдер -> исходное значение» живет только в памяти процесса:
    во внешний API уходят только обезличенные тексты, обратная подстановка
    выполняется локально перед показом ответа сотруднику.

    Ограничение: произвольные пары «Имя Фамилия» без отчества не распознаются —
    для этого нужен NER (например, natasha), это следующий шаг.
    """

    _EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
    _PHONE_RE = re.compile(r"(?<!\d)(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)")
    _TELEGRAM_RE = re.compile(r"(?<![\w@])@[A-Za-z][A-Za-z0-9_]{3,31}\b")
    _NAME_STOPWORDS = "(?:Это|Он|Она|Оно|Они|Мы|Вы|Ты|Его|Её|Их|Тем|Так|Уже|Еще|Ещё|Все|Всё)"
    _FULL_NAME_MIDDLE_PATRONYMIC_RE = re.compile(
        rf"\b(?!{_NAME_STOPWORDS}\s)"
        r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:ович|овна|евич|евна|ична|инична)\s+[А-ЯЁ][а-яё]+\b"
    )
    _FULL_NAME_LAST_PATRONYMIC_RE = re.compile(
        rf"\b(?!{_NAME_STOPWORDS}\s)"
        r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:ович|овна|евич|евна|ична|инична)\b"
    )

    def __init__(self, known_names: list[str] | None = None) -> None:
        self._known_names = sorted(
            {name.strip() for name in (known_names or []) if name and len(name.strip()) >= 2},
            key=len,
            reverse=True,
        )
        self._counters: dict[str, int] = {}
        self._mapping: dict[str, str] = {}

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._mapping)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._counters)

    def anonymize(self, text: str) -> str:
        for name in self._known_names:
            if name.lower() in text.lower():
                text = re.sub(
                    re.escape(name),
                    lambda match: self._placeholder("ФИО", match.group(0)),
                    text,
                    flags=re.IGNORECASE,
                )
        for kind, pattern in (
            ("EMAIL", self._EMAIL_RE),
            ("ТЕЛЕФОН", self._PHONE_RE),
            ("TELEGRAM", self._TELEGRAM_RE),
            ("ФИО", self._FULL_NAME_MIDDLE_PATRONYMIC_RE),
            ("ФИО", self._FULL_NAME_LAST_PATRONYMIC_RE),
        ):
            text = self._mask(pattern, kind, text)
        return text

    def _mask(self, pattern: re.Pattern, kind: str, text: str) -> str:
        return pattern.sub(lambda match: self._placeholder(kind, match.group(0)), text)

    def deanonymize(self, text: str) -> str:
        for token, original in self._mapping.items():
            text = text.replace(token, original)
        return text

    def _placeholder(self, kind: str, original: str) -> str:
        self._counters[kind] = self._counters.get(kind, 0) + 1
        token = f"[{kind}_{self._counters[kind]}]"
        self._mapping[token] = original
        return token
