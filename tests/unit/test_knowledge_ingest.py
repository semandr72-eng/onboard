from services.knowledge_ingest import parse_document

DOC_WITH_MARKERS = """---
document_type: Регламент
role: Сборщик заказов
version: "2.1"
---
# ЧАНК 1. Температурный режим

Заморозка не более 20 минут вне холодильника.
[reference:12]

# ЧАНК 2. Упаковка

Бытовая химия упаковывается отдельно от продуктов.
"""

DOC_WITH_HEADINGS = """## Начало смены

Сотрудник получает терминал и назначение.

## Завершение сборки

Подтвердите сборку в терминале.
"""

DOC_PLAIN = "Просто текст без структуры, но с полезным содержимым."


def test_frontmatter_metadata_parsed():
    parsed = parse_document(DOC_WITH_MARKERS)
    assert parsed.metadata == {"document_type": "Регламент", "role": "Сборщик заказов", "version": "2.1"}


def test_chunk_prefix_not_in_title():
    """Служебный префикс «ЧАНК N.» не должен доходить до пользователя."""
    parsed = parse_document(DOC_WITH_MARKERS)
    assert parsed.chunks[0].title == "Температурный режим"
    assert "ЧАНК" not in parsed.chunks[0].title


def test_chunk_markers_split_and_title_included():
    parsed = parse_document(DOC_WITH_MARKERS)
    assert len(parsed.chunks) == 2
    first = parsed.chunks[0]
    assert first.title == "Температурный режим"
    assert "Температурный режим" in first.content
    assert "20 минут" in first.content
    # Служебные [reference:N] вычищены.
    assert "[reference" not in first.content


def test_headings_fallback_when_no_markers():
    parsed = parse_document(DOC_WITH_HEADINGS)
    assert len(parsed.chunks) == 2
    assert parsed.chunks[0].title == "Начало смены"
    assert parsed.chunks[1].title == "Завершение сборки"


def test_plain_document_becomes_single_chunk():
    parsed = parse_document(DOC_PLAIN)
    assert len(parsed.chunks) == 1
    assert parsed.chunks[0].content == DOC_PLAIN
    assert parsed.chunks[0].title == "Документ"


def test_frontmatter_document_type_used_as_title_for_plain_doc():
    parsed = parse_document("---\ndocument_type: Инструкция\n---\n" + DOC_PLAIN)
    assert parsed.chunks[0].title == "Инструкция"


def test_horizontal_rules_removed_from_content():
    text = "---\ndocument_type: Тест\n---\n## Раздел\n\nТекст.\n\n---\n\nЕщё текст."
    parsed = parse_document(text)
    assert len(parsed.chunks) == 1
    assert "---" not in parsed.chunks[0].content
