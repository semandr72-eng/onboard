import re
from dataclasses import dataclass, field

import yaml

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
CHUNK_MARKER_RE = re.compile(r"^#\s*ЧАНК\s+\d+[^\n]*$", re.MULTILINE)
HEADING_RE = re.compile(r"^##\s+.+$", re.MULTILINE)
REFERENCE_RE = re.compile(r"\[reference:\d+\]")
SEPARATOR_RE = re.compile(r"^\s*---\s*$", re.MULTILINE)


@dataclass
class ChunkDraft:
    title: str
    content: str


@dataclass
class ParsedDocument:
    metadata: dict = field(default_factory=dict)
    chunks: list[ChunkDraft] = field(default_factory=list)


def _clean(text: str) -> str:
    text = REFERENCE_RE.sub("", text)
    text = SEPARATOR_RE.sub("", text)
    return text.strip()


def _split_by_markers(body: str, marker_re: re.Pattern) -> list[ChunkDraft]:
    markers = list(marker_re.finditer(body))
    chunks: list[ChunkDraft] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
        title = marker.group(0).lstrip("#").strip()
        content = _clean(body[start:end])
        if content:
            chunks.append(ChunkDraft(title=title, content=f"{title}\n\n{content}"))
    return chunks


def parse_document(text: str) -> ParsedDocument:
    """Парсит документ базы знаний: YAML-frontmatter + чанки.

    Поддерживает два формата разбивки:
    - явные маркеры «# ЧАНК N. ...» (регламенты, подготовленные для RAG);
    - заголовки второго уровня «## ...» как fallback для обычных markdown-документов.
    """
    metadata: dict = {}
    body = text
    frontmatter = FRONTMATTER_RE.match(text)
    if frontmatter:
        loaded = yaml.safe_load(frontmatter.group(1))
        metadata = loaded if isinstance(loaded, dict) else {}
        body = text[frontmatter.end():]

    chunks = _split_by_markers(body, CHUNK_MARKER_RE)
    if not chunks:
        chunks = _split_by_markers(body, HEADING_RE)
    if not chunks:
        cleaned = _clean(body)
        if cleaned:
            title = str(metadata.get("document_type") or "Документ")
            chunks = [ChunkDraft(title=title, content=cleaned)]

    return ParsedDocument(metadata=metadata, chunks=chunks)
