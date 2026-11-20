"""Filesystem layout for ingestion (CONTRACT.md section 1). Overridable for tests via env."""

from __future__ import annotations

import os
from pathlib import Path

PAPERS = ("deepseek-r1", "deepseek-math", "dr-grpo")


def data_dir() -> Path:
    """`data/` at the repo root, or INGESTION_DATA_DIR if set (used by tests)."""
    env = os.environ.get("INGESTION_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data"


def papers_dir() -> Path:
    return data_dir() / "papers"


def chunks_path() -> Path:
    return data_dir() / "chunks.jsonl"


def manifest_path() -> Path:
    return data_dir() / "manifest.json"


def index_dir() -> Path:
    return data_dir() / "index"
