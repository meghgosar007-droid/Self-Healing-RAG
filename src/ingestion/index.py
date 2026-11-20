"""Chroma vector store at data/index/, derived from data/chunks.jsonl."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.schemas import Chunk

from .embed import Embedder

COLLECTION = "chunks"


def _client(index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(index_dir), settings=Settings(anonymized_telemetry=False)
    )


def build_index(chunks: list[Chunk], embedder: Embedder, index_dir: Path, batch: int = 128) -> None:
    """(Re)create the collection from scratch so the index always mirrors the chunk list."""
    client = _client(index_dir)
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # collection did not exist yet
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        col.add(
            ids=[c.id for c in part],
            documents=[c.text for c in part],
            embeddings=embedder.embed_documents([c.text for c in part]),
            metadatas=[
                {
                    "paper": c.paper,
                    "section": c.section,
                    "page": c.page,
                    "embedding_model": c.embedding_model,
                }
                for c in part
            ],
        )


def query_index(query_embedding: list[float], k: int, index_dir: Path) -> list[tuple[Chunk, float]]:
    """Return up to k (chunk, cosine_distance) pairs, nearest first."""
    if not index_dir.is_dir():
        raise FileNotFoundError(
            f"No index at {index_dir}. Run `python -m src.ingestion.ingest --rebuild-index`."
        )
    col = _client(index_dir).get_collection(COLLECTION)
    res = col.query(query_embeddings=[query_embedding], n_results=max(1, min(k, col.count() or 1)))
    out: list[tuple[Chunk, float]] = []
    for id_, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
    ):
        out.append(
            (
                Chunk(
                    id=id_,
                    paper=meta["paper"],
                    section=meta["section"],
                    page=meta["page"],
                    text=doc,
                    embedding_model=meta["embedding_model"],
                ),
                float(dist),
            )
        )
    return out
