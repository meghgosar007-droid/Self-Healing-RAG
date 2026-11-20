"""Ingestion CLI.

python -m src.ingestion.ingest                  # PDFs -> chunks.jsonl, manifest.json, index
python -m src.ingestion.ingest --rebuild-index  # rebuild data/index/ from existing chunks.jsonl
python -m src.ingestion.ingest --no-index       # chunks + manifest only (no embedding model needed)
"""

from __future__ import annotations

import argparse
import sys

from src.schemas import Chunk

from . import paths
from .chunk import chunk_blocks, read_chunks, write_chunks
from .embed import BgeEmbedder, Embedder
from .index import build_index
from .manifest import write_manifest
from .parse import parse_pdf


def build_chunks() -> list[Chunk]:
    """Parse all three PDFs (fails clearly if any is missing) and chunk them."""
    pdfs = {p: paths.papers_dir() / f"{p}.pdf" for p in paths.PAPERS}
    missing = [str(f) for f in pdfs.values() if not f.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing source PDF(s): " + ", ".join(missing) + ". Nothing is downloaded; "
            "place them in data/papers/."
        )
    chunks: list[Chunk] = []
    for paper, pdf in pdfs.items():
        chunks.extend(chunk_blocks(paper, parse_pdf(pdf)))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate chunk ids produced; chunking is not deterministic")
    return chunks


def run(
    *, rebuild_index_only: bool = False, index: bool = True, embedder: Embedder | None = None
) -> dict:
    if rebuild_index_only:
        chunks = read_chunks(paths.chunks_path())
    else:
        chunks = build_chunks()
        write_chunks(chunks, paths.chunks_path())
    manifest = write_manifest(chunks, paths.manifest_path())
    if index or rebuild_index_only:
        build_index(chunks, embedder or BgeEmbedder(), paths.index_dir())
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rebuild-index",
        action="store_true",
        help="rebuild only the Chroma index from chunks.jsonl",
    )
    ap.add_argument("--no-index", action="store_true", help="skip embedding/indexing")
    args = ap.parse_args(argv)
    try:
        manifest = run(rebuild_index_only=args.rebuild_index, index=not args.no_index)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"ok: {manifest['chunk_count']} chunks, corpus_version={manifest['corpus_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
