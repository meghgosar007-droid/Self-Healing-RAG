"""Offline ingestion tests: synthetic PDFs, a fake bag-of-words embedder, no network, no model."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf
import pytest

from src.ingestion import ingest, paths, retriever
from src.ingestion.chunk import chunk_blocks, read_chunks, slugify
from src.ingestion.manifest import compute_corpus_version, read_manifest
from src.ingestion.parse import Block, parse_pdf
from src.schemas import EMBEDDING_MODEL, Chunk

DIM = 256


class FakeEmbedder:
    """Hashed bag-of-words: deterministic, offline, and similar texts land close together."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, query):
        return self._vec(query)


GRPO_TEXT = (
    "Group relative policy optimization samples a group of outputs per question and uses the "
    "group reward statistics as the baseline, so no value model is required. "
)
CORPUS_TEXT = (
    "The math corpus is collected from common crawl with a fasttext classifier that recalls "
    "math related web pages over several iterations. "
)
RL_TEXT = (
    "The zero model is trained with large scale reinforcement learning directly on the base "
    "model without any supervised fine tuning data. "
)


def make_pdf(path: Path, pages: list[list[tuple[str, bool, float]]]) -> None:
    """pages: per page, a list of (text, bold, size) paragraphs. Adds a footer page number."""
    doc = pymupdf.open()
    for i, paras in enumerate(pages, start=1):
        page = doc.new_page(width=612, height=792)
        y = 90.0
        for text, bold, size in paras:
            font = "hebo" if bold else "helv"
            rect = pymupdf.Rect(72, y, 540, y + 200)
            used = page.insert_textbox(rect, text, fontname=font, fontsize=size)
            assert used >= 0
            y += 200 - used + 14
        page.insert_text((300, 775), str(i), fontsize=9)  # footer
    doc.save(path)
    doc.close()


def make_papers(papers_dir: Path) -> None:
    papers_dir.mkdir(parents=True, exist_ok=True)
    make_pdf(
        papers_dir / "deepseek-math.pdf",
        [
            [
                ("1 Introduction", True, 13),
                ("We introduce a model. " + CORPUS_TEXT * 2, False, 10),
                ("2.1 GRPO Method", True, 11),
                (GRPO_TEXT * 3, False, 10),
            ],
            [
                (GRPO_TEXT * 2, False, 10),
                ("References", True, 13),
                ("Smith et al. 2020.", False, 10),
            ],
        ],
    )
    make_pdf(
        papers_dir / "deepseek-r1.pdf",
        [[("2. Zero Training", True, 13), (RL_TEXT * 3, False, 10)]],
    )
    make_pdf(
        papers_dir / "dr-grpo.pdf",
        [[("Some text before any heading. " * 6, False, 10)]],
    )


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("INGESTION_DATA_DIR", str(tmp_path))
    make_papers(tmp_path / "papers")
    retriever.set_embedder(None)
    yield tmp_path
    retriever.set_embedder(None)


# --- parse ---------------------------------------------------------------------------------


def test_parse_detects_headings_skips_footer_and_references(data):
    blocks = parse_pdf(paths.papers_dir() / "deepseek-math.pdf")
    sections = [b.section for b in blocks]
    assert sections[0] == "1 Introduction"
    assert "2.1 GRPO Method" in sections
    assert not any(b.text in {"1", "2"} for b in blocks)  # page-number footers dropped
    assert not any("Smith et al." in b.text for b in blocks)  # references dropped
    assert not any(b.text == "1 Introduction" for b in blocks)  # headings are not body text
    assert blocks[-1].page == 2


def test_parse_text_before_heading_is_unknown(data):
    blocks = parse_pdf(paths.papers_dir() / "dr-grpo.pdf")
    assert blocks and all(b.section == "unknown" for b in blocks)


def test_parse_missing_pdf_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing PDF"):
        parse_pdf(tmp_path / "nope.pdf")


# --- chunking ------------------------------------------------------------------------------


def _blocks() -> list[Block]:
    return [
        Block(1, "1 Intro", "Alpha sentence one. " * 20),
        Block(1, "1 Intro", "Beta sentence two. " * 20),
        Block(2, "1 Intro", "Gamma sentence three. " * 20),
        Block(3, "2.1 GRPO", "Delta sentence four. " * 30),
    ]


def test_chunks_validate_and_never_span_sections():
    chunks = chunk_blocks(
        "deepseek-math", [Block(b.page, b.section, " ".join(b.text.split())) for b in _blocks()]
    )
    for c in chunks:
        Chunk.model_validate(c.model_dump())
        assert c.embedding_model == EMBEDDING_MODEL
        assert ("Delta" in c.text) == (c.section == "2.1 GRPO")
        assert len(c.text) <= 1500
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_page_is_start_page_and_ids_deterministic():
    blocks = [Block(b.page, b.section, " ".join(b.text.split())) for b in _blocks()]
    a, b = chunk_blocks("deepseek-math", blocks), chunk_blocks("deepseek-math", blocks)
    assert [c.id for c in a] == [c.id for c in b]
    first = a[0]
    assert first.page == 1 and first.id == "deepseek-math:1-intro:p1:0"
    delta = next(c for c in a if "Delta" in c.text)
    assert delta.page == 3 and delta.id.startswith("deepseek-math:2-1-grpo:p3:")


def test_oversized_block_without_sentence_breaks_is_split():
    chunks = chunk_blocks("dr-grpo", [Block(1, "unknown", "x" * 5000)])
    assert len(chunks) >= 4 and all(len(c.text) <= 1400 for c in chunks)
    assert all(c.id.startswith("dr-grpo:unknown:p1:") for c in chunks)


def test_slugify():
    assert slugify("3.2 GRPO") == "3-2-grpo"
    assert slugify("???") == "unknown"


# --- manifest ------------------------------------------------------------------------------


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(
        id=f"dr-grpo:s:p1:{i}",
        paper="dr-grpo",
        section="S",
        page=1,
        text=text,
        embedding_model=EMBEDDING_MODEL,
    )


def test_corpus_version_is_order_independent_and_content_sensitive():
    a, b = _chunk(0, "one"), _chunk(1, "two")
    v = compute_corpus_version([a, b])
    assert re.fullmatch(r"[0-9a-f]{12}", v)
    assert v == compute_corpus_version([b, a])
    assert v != compute_corpus_version([a, _chunk(1, "changed")])


# --- end to end ----------------------------------------------------------------------------


def test_ingest_end_to_end_and_stable_across_reruns(data):
    m1 = ingest.run(embedder=FakeEmbedder())
    chunks1 = read_chunks(paths.chunks_path())
    raw1 = paths.chunks_path().read_bytes()
    m2 = ingest.run(embedder=FakeEmbedder())
    assert m1 == m2 and paths.chunks_path().read_bytes() == raw1  # stable ids and text
    assert m1["chunk_count"] == len(chunks1)
    assert set(m1) == {"corpus_version", "chunk_count", "embedding_model"}
    assert m1["embedding_model"] == EMBEDDING_MODEL
    assert read_manifest(paths.manifest_path())["corpus_version"] == compute_corpus_version(chunks1)
    assert {c.paper for c in chunks1} == {"deepseek-r1", "deepseek-math", "dr-grpo"}
    for line in raw1.decode().splitlines():
        assert set(json.loads(line)) == {
            "id",
            "paper",
            "section",
            "page",
            "text",
            "embedding_model",
        }


def test_ingest_fails_clearly_when_pdf_missing(data):
    (paths.papers_dir() / "dr-grpo.pdf").unlink()
    with pytest.raises(FileNotFoundError, match="dr-grpo.pdf"):
        ingest.run(embedder=FakeEmbedder())
    assert ingest.main(["--no-index"]) == 1


def test_known_passages_in_top5(data):
    ingest.run(embedder=FakeEmbedder())
    retriever.set_embedder(FakeEmbedder())
    top = retriever.retrieve(
        "how does group relative policy optimization get a baseline without a value model", k=5
    )
    assert any(c.paper == "deepseek-math" and c.section == "2.1 GRPO Method" for c in top)
    top = retriever.retrieve("fasttext classifier math web pages common crawl", k=5)
    assert top[0].section == "1 Introduction"
    top = retriever.retrieve("reinforcement learning without supervised fine tuning", k=5)
    assert top[0].paper == "deepseek-r1"
    assert all(isinstance(c, Chunk) for c in top) and len(top) <= 5


def test_retrieve_k_larger_than_corpus_and_blank_query(data):
    ingest.run(embedder=FakeEmbedder())
    retriever.set_embedder(FakeEmbedder())
    total = read_manifest(paths.manifest_path())["chunk_count"]
    assert len(retriever.retrieve("policy", k=1000)) == total
    assert retriever.retrieve("   ") == []


def test_rebuild_index_from_chunks_jsonl(data):
    ingest.run(index=False, embedder=FakeEmbedder())
    assert not paths.index_dir().exists()
    with pytest.raises(FileNotFoundError, match="rebuild-index"):
        retriever.set_embedder(FakeEmbedder())
        retriever.retrieve("policy")
    ingest.run(rebuild_index_only=True, embedder=FakeEmbedder())
    assert retriever.retrieve("policy", k=3)


# --- committed corpus ----------------------------------------------------------------------


def test_committed_chunks_validate_and_match_manifest():
    root = Path(__file__).resolve().parents[2] / "data"
    if not (root / "chunks.jsonl").exists():
        pytest.skip("data/chunks.jsonl not generated")
    chunks = read_chunks(root / "chunks.jsonl")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == len(chunks)
    assert manifest["corpus_version"] == compute_corpus_version(chunks)
    assert len({c.id for c in chunks}) == len(chunks)
