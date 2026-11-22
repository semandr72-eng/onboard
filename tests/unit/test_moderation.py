import pytest

from services.moderation import Moderator


@pytest.fixture
def moderator() -> Moderator:
    return Moderator()


def test_clean_text_passes(moderator):
    assert moderator.is_clean("Куда кладём заморозку после сборки?") is True
    assert moderator.is_clean("Хлеб и молоко — в один пакет.") is True


def test_profanity_detected(moderator):
    assert moderator.is_clean("Идиот, где мой заказ?") is False
    assert moderator.is_clean("Это полный бред и хуйня") is False
    assert moderator.is_clean("сука где терминал") is False


def test_masked_profanity_detected(moderator):
    assert moderator.is_clean("пИздец как долго") is False
    assert moderator.is_clean("БЛЯТЬ опять замена") is False


def test_innocent_words_not_flagged(moderator):
    # «хлеб» содержит опасную пару букв — ложных срабатываний быть не должно.
    assert moderator.is_clean("Хлебозавод рядом") is True
    assert moderator.is_clean("Молоко, хлеб, сметана") is True


def test_substring_traps_not_flagged(moderator):
    """Короткие корни не должны ловить невинные слова-подстроки."""
    assert moderator.is_clean("Регламент нужен всей команде сборки") is True
    assert moderator.is_clean("Команда работает по единым правилам") is True
    assert moderator.is_clean("Это небанальный подход к сборке") is True
    assert moderator.is_clean("Ступица тележки скрипит") is True
