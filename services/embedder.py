from typing import Protocol


class Embedder(Protocol):
    """Локальный эмбеддер: ничего не покидает периметр."""

    dimension: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedE5:
    """multilingual-e5 (ONNX, fastembed) — сильная модель для русского языка.

    E5 требует префиксов: «passage: » для фрагментов базы знаний,
    «query: » для запросов.
    """

    dimension = 1024

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        return [vector.tolist() for vector in model.embed([f"passage: {text}" for text in texts])]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_loaded()
        return next(iter(model.embed([f"query: {text}"]))).tolist()


class FakeEmbedder:
    """Детерминированный эмбеддер для тестов без модели."""

    dimension = 8

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        buckets = [0.0] * FakeEmbedder.dimension
        for position, char in enumerate(text.lower()):
            buckets[(ord(char) + position) % FakeEmbedder.dimension] += 1.0
        norm = sum(v * v for v in buckets) ** 0.5 or 1.0
        return [round(v / norm, 6) for v in buckets]
