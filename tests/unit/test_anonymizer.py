from services.anonymizer import Anonymizer


def test_full_name_with_patronymic_middle_is_masked():
    anonymizer = Anonymizer()
    text = anonymizer.anonymize("Меня зовут Иван Сергеевич Петров, я сборщик.")
    assert "Иван Сергеевич Петров" not in text
    assert "[ФИО_1]" in text
    assert anonymizer.mapping["[ФИО_1]"] == "Иван Сергеевич Петров"


def test_full_name_with_patronymic_last_is_masked():
    anonymizer = Anonymizer()
    text = anonymizer.anonymize("Ответственная — Сидорова Мария Ивановна.")
    assert "[ФИО_1]" in text
    assert "Сидорова" not in text


def test_phone_formats_are_masked():
    anonymizer = Anonymizer()
    text = anonymizer.anonymize("Позвони мне: +7 (912) 345-67-89 или 8 912 3456789.")
    assert "+7 (912) 345-67-89" not in text
    assert "8 912 3456789" not in text
    assert "[ТЕЛЕФОН_1]" in text
    assert "[ТЕЛЕФОН_2]" in text


def test_email_is_masked():
    anonymizer = Anonymizer()
    text = anonymizer.anonymize("Почта ivan.petrov+work@example-company.ru принимает письма.")
    assert "ivan.petrov+work@example-company.ru" not in text
    assert "[EMAIL_1]" in text


def test_telegram_nick_is_masked():
    anonymizer = Anonymizer()
    text = anonymizer.anonymize("Напиши @ivan_sidorov123, он в курсе.")
    assert "@ivan_sidorov123" not in text
    assert "[TELEGRAM_1]" in text


def test_known_name_without_patronymic_is_masked():
    anonymizer = Anonymizer(known_names=["Иван Петров"])
    text = anonymizer.anonymize("Иван Петров просил перезвонить Ивану Петрову.")
    assert "[ФИО_1]" in text
    assert "Иван Петров" not in text


def test_stopwords_are_not_masked():
    anonymizer = Anonymizer()
    text = anonymizer.anonymize("Это уже сделано, но оно ещё не готово.")
    assert "[" not in text
    assert anonymizer.stats == {}


def test_clean_text_stays_untouched():
    anonymizer = Anonymizer()
    text = "Заморозка вне холодильника — не более 20 минут в термопакете."
    assert anonymizer.anonymize(text) == text


def test_deanonymize_restores_original_values():
    anonymizer = Anonymizer(known_names=["Иван Петров"])
    original = "Иван Петров, телефон +79123456789"
    masked = anonymizer.anonymize(original)
    assert anonymizer.deanonymize(masked) == original


def test_stats_count_by_kind():
    anonymizer = Anonymizer()
    anonymizer.anonymize("Иван Сергеевич Петров, ivan@example.com, +79998887766")
    assert anonymizer.stats == {"ФИО": 1, "EMAIL": 1, "ТЕЛЕФОН": 1}
