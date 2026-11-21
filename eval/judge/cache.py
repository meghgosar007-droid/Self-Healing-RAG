"""On-disk cache for judge LLM output.

One JSON file per entry, named by the SHA-256 of everything that can change the answer (stage,
prompt text, prompt version, model, input). Writes are atomic, so an interrupted run never leaves
a half-written entry and a re-run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def cache_key(**parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.misses += 1
            return None  # missing or corrupt entry: treat as a miss and recompute
        if not isinstance(entry, dict):
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, key: str, entry: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)


class NullCache:
    """Caching disabled (``--no-cache``)."""

    hits = 0
    misses = 0

    def get(self, key: str) -> dict[str, Any] | None:  # noqa: ARG002
        return None

    def put(self, key: str, entry: dict[str, Any]) -> None:  # noqa: ARG002
        return None
