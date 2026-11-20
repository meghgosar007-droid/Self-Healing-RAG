"""Retrieval seam: the real retriever when Track 1 has landed, else a stub over the fixtures."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src.schemas import Chunk

K_DEFAULT = 5
_FIXTURES = Path(__file__).resolve().parents[2] / "contract_examples" / "chunks.jsonl"
_STOP = frozenset(
    "a an and are as at be by does do for from how in is it of on or that the this to was what "
    "which with why".split()
)


@lru_cache(maxsize=1)
def _fixture_chunks() -> tuple[Chunk, ...]:
    lines = _FIXTURES.read_text(encoding="utf-8").splitlines()
    return tuple(Chunk.model_validate_json(line) for line in lines if line.strip())


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def stub_retrieve(query: str, k: int = K_DEFAULT) -> list[Chunk]:
    """Keyword-overlap retrieval over contract_examples/chunks.jsonl.

    Returns [] when nothing overlaps, so the "nothing relevant" refusal path is reachable.
    """
    q = _terms(query)
    scored = [(len(q & _terms(f"{c.section} {c.text}")), c) for c in _fixture_chunks()]
    ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])
    return [c for _, c in ranked[:k]]


def retrieve(query: str, k: int = K_DEFAULT) -> list[Chunk]:
    """`src.ingestion.retriever.retrieve` if it exists, otherwise the stub."""
    try:
        from src.ingestion.retriever import retrieve as real
    except ModuleNotFoundError as exc:
        if exc.name not in ("src.ingestion", "src.ingestion.retriever"):
            raise  # a real import error inside the retriever, not its absence
        return stub_retrieve(query, k)
    return real(query, k)
