"""Section-aware chunking and deterministic ids (CONTRACT.md section 1)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.schemas import EMBEDDING_MODEL, Chunk

from .parse import Block

TARGET_CHARS = 1000  # ~250 tokens; well inside bge-small's 512-token window
MAX_CHARS = 1400
MIN_CHARS = 150  # tiny trailing fragments merge into the previous chunk

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def slugify(section: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
    return slug[:60].strip("-") or "unknown"


def _split_long(text: str) -> list[str]:
    """Split an oversized block on sentence boundaries, hard-wrapping as a last resort."""
    parts: list[str] = []
    cur = ""
    for sent in _SENT_SPLIT.split(text):
        while len(sent) > MAX_CHARS:  # no sentence boundary: wrap on spaces
            cut = sent.rfind(" ", 0, MAX_CHARS)
            cut = cut if cut > 0 else MAX_CHARS
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + 1 + len(sent) > TARGET_CHARS:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}" if cur else sent
    if cur:
        parts.append(cur)
    return parts


def chunk_blocks(paper: str, blocks: list[Block]) -> list[Chunk]:
    """Pack consecutive blocks of one section into ~TARGET_CHARS chunks.

    A chunk never spans two sections; its page is the page of its first block.
    """
    pieces: list[list] = []  # [section, page, text]
    cur: list | None = None
    for b in blocks:
        for part in _split_long(b.text) if len(b.text) > MAX_CHARS else [b.text]:
            if cur and (cur[0] != b.section or len(cur[2]) + 1 + len(part) > TARGET_CHARS):
                pieces.append(cur)
                cur = None
            if cur:
                cur[2] = f"{cur[2]} {part}"
            else:
                cur = [b.section, b.page, part]
    if cur:
        pieces.append(cur)

    merged: list[list] = []
    for piece in pieces:  # fold tiny fragments into the previous piece of the same section
        if merged and len(piece[2]) < MIN_CHARS and merged[-1][0] == piece[0]:
            merged[-1][2] = f"{merged[-1][2]} {piece[2]}"
        else:
            merged.append(piece)

    counters: dict[tuple[str, int], int] = {}
    chunks: list[Chunk] = []
    for sec, page, text in merged:
        slug = slugify(sec)
        n = counters.get((slug, page), 0)
        counters[(slug, page)] = n + 1
        chunks.append(
            Chunk(
                id=f"{paper}:{slug}:p{page}:{n}",
                paper=paper,
                section=sec,
                page=page,
                text=text,
                embedding_model=EMBEDDING_MODEL,
            )
        )
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingestion.ingest` first.")
    with path.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]
