"""Local embeddings with BAAI/bge-small-en-v1.5 (not routed through src/llm.py, per contract)."""

from __future__ import annotations

from typing import Protocol

from src.schemas import EMBEDDING_MODEL

# bge retrieval models expect this prefix on queries only (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


class BgeEmbedder:
    """sentence-transformers wrapper; the model loads lazily on first use."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._get().encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]

    def embed_query(self, query: str) -> list[float]:
        vec = self._get().encode(QUERY_PREFIX + query, normalize_embeddings=True)
        return vec.tolist()
