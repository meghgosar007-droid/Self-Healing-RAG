"""corpus_version and data/manifest.json (CONTRACT.md section 1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.schemas import EMBEDDING_MODEL, Chunk


def compute_corpus_version(chunks: list[Chunk]) -> str:
    """First 12 hex chars of SHA-256 over the sorted (id, text) pairs.

    Serialization: compact JSON (ensure_ascii=False) of the sorted list of [id, text] pairs,
    UTF-8 encoded.
    """
    pairs = sorted([c.id, c.text] for c in chunks)
    payload = json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def build_manifest(chunks: list[Chunk]) -> dict:
    return {
        "corpus_version": compute_corpus_version(chunks),
        "chunk_count": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
    }


def write_manifest(chunks: list[Chunk], path: Path) -> dict:
    manifest = build_manifest(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingestion.ingest` first.")
    return json.loads(path.read_text(encoding="utf-8"))
