"""Public retrieval interface: `retrieve(query, k) -> list[Chunk]` (CONTRACT.md section 2)."""

from __future__ import annotations

from src.schemas import Chunk

from .embed import BgeEmbedder, Embedder
from .index import query_index
from .paths import index_dir

_embedder: Embedder | None = None


def set_embedder(embedder: Embedder | None) -> None:
    """Override the query embedder (tests use a fake; None restores the default bge model)."""
    global _embedder
    _embedder = embedder


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = BgeEmbedder()
    return _embedder


def retrieve_with_scores(query: str, k: int = 5) -> list[tuple[Chunk, float]]:
    """Top-k chunks with cosine distance (0 = identical, 2 = opposite), nearest first.

    Extra to the contract: lets the agent apply its own relevance cutoff.
    """
    if not query.strip():
        return []
    return query_index(_get_embedder().embed_query(query), k, index_dir())


def retrieve(query: str, k: int = 5) -> list[Chunk]:
    """Top-k chunks for `query`, nearest first. Always returns up to k; no relevance cutoff."""
    return [c for c, _ in retrieve_with_scores(query, k)]
