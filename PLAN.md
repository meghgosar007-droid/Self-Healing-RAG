# PLAN.md — Self-Healing RAG

**Status:** PROPOSED — pending approval. Interfaces referenced here are
defined in `CONTRACT.md`; this file only covers who does what and in which
order.

## Track 0 — Foundation (shared files, lands first)

Not one of the four tracks: a single `contract/foundation` PR that creates the
shared files everyone else builds on, so no track needs to touch them.

- `CONTRACT.md`, `PLAN.md`, `.claude/agents/reviewer.md`
- `pyproject.toml` (Python 3.11+, deps: `langgraph`, `langchain-core`,
  `langchain-google-genai`, `python-dotenv`, `pydantic>=2`,
  `sentence-transformers`, `numpy`, `pymupdf`, `pyyaml`, `pytest`,
  `pytest-asyncio`), `.gitignore` (`.env`, `data/`, `eval/results/`),
  `.env.example`
- `src/contracts.py` — `Chunk`, `ScoredChunk`, `Claim`, `CriticVerdict`,
  `GraphState`, `AgentResult`, `LLMUsage`, `EvalRun` and nested models
- `src/llm.py` — role-based `ainvoke`, semaphore + RPM cap, backoff/retry,
  usage accounting, judge≠generator/critic startup check
- `src/embeddings.py` — bge-small-en-v1.5 wrapper (`embed_passages`,
  `embed_query`)
- `config/models.yaml`, `config/pricing.yaml`, `config/corpus.yaml`,
  `config/gate.yaml`
- Unit tests for `src/llm.py` with a fake chat model (429 → backoff → success,
  concurrency never exceeds cap, 400 not retried)

## The four tracks

### 1. Ingestion
- **Owns:** `src/ingestion/`, `tests/ingestion/`, `scripts/ingest/`
- **Builds:**
  - PDF fetch by pinned arXiv version + SHA-256 check (from `config/corpus.yaml`)
  - Text extraction (PyMuPDF), header/footer stripping, section-heading
    detection, page tracking
  - Chunker (proposed: ~350 bge tokens, 50 overlap, never crosses a section
    boundary)
  - Index build → `data/index/{chunks.jsonl, embeddings.npy, manifest.json}`
  - `Retriever.search()` (cosine over normalised numpy matrix — corpus is
    small, no vector DB needed)
  - CLI: `python -m src.ingestion build`
- **Depends on:** Foundation (`Chunk`, `ScoredChunk`, `src/embeddings.py`,
  `config/corpus.yaml`). No LLM calls.
- **Done when:** deterministic ids across two builds; every chunk validates
  against `Chunk`; a handful of known-answer retrieval tests pass (e.g. "Dr.
  GRPO length normalization bias" retrieves a `dr-grpo` chunk in top 3).

### 2. Agent graph
- **Owns:** `src/agent/` (incl. `src/agent/prompts/` + `VERSION`), `tests/agent/`
- **Builds:** LangGraph nodes `plan`, `retrieve`, `generate`,
  `extract_claims`, `critic`, `rewrite`, `finalise`; routing per CONTRACT §3.1;
  `answer(question) -> AgentResult`; prompt versioning.
- **Depends on:** Foundation (`GraphState`, `src/llm.py`); Ingestion's
  `Retriever` interface. Until Ingestion merges, develop against a fixture
  retriever in `tests/agent/` returning hand-written `ScoredChunk`s.
- **Done when:** graph tests with a fake LLM cover all routes (accept, retry →
  accept, retry exhausted → refused, direct refuse); `final_status` is always
  set; `retry_count ≤ MAX_RETRIES`.

### 3. Eval suite
- **Owns:** `eval/` (`dataset/`, `prompts/`, `baselines/`, runner, judge,
  report), `tests/eval/`
- **Builds:**
  - `eval/dataset/questions.jsonl`: `{qid, question, label, paper?, notes}`;
    proposed 40 questions (30 answerable spread across the 3 papers + 10
    unanswerable: out-of-corpus, false-premise, cross-paper traps)
  - Judge: claim re-extraction + per-claim verdict against retrieved chunks,
    using `role="judge"` and prompts in `eval/prompts/`
  - Runner: `python -m eval.run --split {smoke,full}` → `EvalRun` JSON;
    aggregates per CONTRACT §4–§5
  - Human-readable markdown summary (for the CI job summary)
- **Depends on:** Foundation (`EvalRun`, `src/llm.py`); Agent's `answer()`.
  Develop against a stub `answer()` that returns canned `AgentResult`s, so
  metric code is unit-tested without any API calls.
- **Done when:** metric functions unit-tested on hand-built fixtures
  (including edge cases: zero claims, all refused, errors); one real run on the
  full set produces a valid `EvalRun`, which becomes `eval/baselines/main.json`.

### 4. CI
- **Owns:** `.github/workflows/`, `ci/`, `tests/ci/`
- **Builds:**
  - `ci.yml` on every PR: lint (ruff), unit tests (no network, no key),
    ownership check (changed files ⊆ branch's track folders, or
    `contract/*` branch), secret scan (gitleaks or regex for `AIza…`, `.env`)
  - `eval.yml`: build index (cache `data/index` + HF model by
    `corpus_version`), run eval, `ci/check_gate.py` applies `config/gate.yaml`
    vs `eval/baselines/main.json`, writes the `gate` block, posts a job
    summary, uploads the result JSON as an artifact, fails on any failed check
- **Depends on:** Foundation (`config/gate.yaml`, `EvalRun`); Eval runner CLI.
  `check_gate.py` is developed against fixture `EvalRun` JSONs (pass, fail,
  no-baseline) in `tests/ci/`.
- **Done when:** gate script unit-tested; a deliberately bad fixture fails the
  job; the workflow runs green end-to-end on a PR.

## Dependency graph

```
            Foundation (contract/foundation)
           /        |          |          \
     Ingestion   Agent*      Eval*       CI*
           \        |          |          |
            └──► Agent ──► Eval ──► CI (gate live)
   * can start in parallel against fixtures/stubs
```

## Merge order

1. `contract/foundation` — shared files + `src/llm.py` tests.
2. `ingestion/*` — real index and retriever.
3. `agent/*` — swaps fixture retriever for the real one; graph tests pass.
4. `eval/*` — dataset + judge + runner; first full run committed as baseline.
5. `ci/*` — unit/ownership/secret jobs first; the eval gate turns required
   once a baseline exists.

All four tracks can be developed in parallel after step 1; only the merge
order is sequential. Every PR is checked with the `reviewer` subagent before
merge. Any needed change to a shared file pauses the track and goes through a
`contract/*` PR (CONTRACT §9).
