"""PDF parsing: turn a PDF into ordered text blocks tagged with page and section heading."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Numbered headings: "3.2 GRPO", "3.2. Training", "A Policy Gradient", "A.1 Setup".
_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*\.?)\s+[A-Z0-9]")
_NAMED_HEADINGS = {"abstract", "references", "acknowledgements", "acknowledgments"}
_SKIP_SECTIONS = {"references"}  # bibliography is noise for QA
_MIN_HEADING_SIZE = 9.5  # excludes bold list items in small print
_MAX_HEADING_CHARS = 120
_MARGIN = 0.06  # top/bottom fraction of the page treated as header/footer
_BOLD = 16  # pymupdf span flag


@dataclass(frozen=True)
class Block:
    page: int  # 1-indexed
    section: str  # heading text, or "unknown"
    text: str  # whitespace-normalized


def normalize(text: str) -> str:
    """NFKC (folds ligatures), drop control chars, collapse all whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c in "\n\t " or unicodedata.category(c)[0] != "C")
    return " ".join(text.split())


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines, repairing end-of-line hyphenation ("optimi- zation")."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return out


def is_heading(text: str, spans: list[dict]) -> bool:
    if not spans or len(text) > _MAX_HEADING_CHARS:
        return False
    if not all(s["flags"] & _BOLD for s in spans):
        return False
    if spans[0]["size"] < _MIN_HEADING_SIZE:
        return False
    return bool(_HEADING_RE.match(text)) or text.lower() in _NAMED_HEADINGS


def clean_heading(text: str) -> str:
    """'3.2. Training Details' -> '3.2 Training Details' (matches the contract's style)."""
    return re.sub(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.(?=\s)", r"\1", normalize(text))


def parse_pdf(path: Path) -> list[Block]:
    """Extract text blocks in reading order. Raises FileNotFoundError if the PDF is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing PDF: {path}. Place the source PDFs in data/papers/ (nothing is downloaded)."
        )
    blocks: list[Block] = []
    section = "unknown"
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            height = page.rect.height
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0 or not b.get("lines"):
                    continue
                y0, y1 = b["bbox"][1], b["bbox"][3]
                if y1 < height * _MARGIN or y0 > height * (1 - _MARGIN):
                    continue  # running header/footer, page numbers
                spans = [s for ln in b["lines"] for s in ln["spans"] if s["text"].strip()]
                raw = _join_lines(["".join(s["text"] for s in ln["spans"]) for ln in b["lines"]])
                text = normalize(raw)
                if not text:
                    continue
                if is_heading(text, spans):
                    section = clean_heading(text)
                    continue
                if section.lower() in _SKIP_SECTIONS:
                    continue
                blocks.append(Block(page=page_no, section=section, text=text))
    return blocks
