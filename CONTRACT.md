# CONTRACT.md  (contract_version: 2)

Single source of truth for all tracks. Code that disagrees with this file is wrong.
Corpus: DeepSeek-R1, DeepSeekMath, Dr. GRPO papers. Source PDFs are supplied by the repo owner in `data/papers/`; nothing is downloaded by code.

## 1. Chunk schema

Ingestion reads local files only: `data/papers/deepseek-r1.pdf`, `data/papers/deepseek-math.pdf`, `data/papers/dr-grpo.pdf` (filename stem = `paper` enum value). No network access during ingestion; it fails with a clear error if a PDF is missing. `data/papers/` is committed so CI can rebuild the index. The vector store is Chroma, persisted at `data/index/` (derived, gitignored, rebuilt from `data/chunks.jsonl`).

Emitted by ingestion as `data/chunks.jsonl` (one JSON object per line). Pydantic model lives in `src/schemas.py`. Extra fields are forbidden.

| field | type | rule |
|---|---|---|
| id | str | deterministic: `{paper}:{section_slug}:p{page}:{n}`; stable across re-ingestion of the same PDF |
| paper | enum | `deepseek-r1` \| `deepseek-math` \| `dr-grpo` |
| section | str | nearest heading, e.g. `"3.2 GRPO"`; `"unknown"` if none detected |
| page | int | 1-indexed PDF page where the chunk starts |
| text | str | non-empty, whitespace-normalized |
| embedding_model | str | must be `BAAI/bge-small-en-v1.5` (local) |

`corpus_version` = first 12 hex chars of SHA-256 over the sorted `(id, text)` pairs. Written to `data/manifest.json` with chunk count and embedding model.

## 2. LangGraph state

`TypedDict` in `src/schemas.py`:

| key | type | notes |
|---|---|---|
| question | str | input, never mutated |
| sub_queries | list[str] | 2-3 retrieval queries (one parallel retriever each); rewritten on retry |
| chunks | list[Chunk] | retrieved for the current attempt (replaced, not appended, on retry) |
| answer | str \| None | generator output; every sentence cites chunk ids like `[id]` |
| claims | list[Claim] | `Claim = {text: str, chunk_ids: list[str]}` extracted from the answer |
| critic_verdict | `{verdict: "pass"\|"fail", unsupported_claims: list[str], reason: str}` \| None | |
| retry_count | int | starts 0; +1 per heal loop; `MAX_RETRIES = 2` |
| final_status | `"answered"` \| `"refused"` \| None | None until terminal |

Self-healing flow (nodes: planner, retrievers, merge, generator, critic, rewriter, refuse): planner -> parallel retrievers (one per sub-query, LangGraph fan-out) -> merge (fan-in, dedupe chunks by id) -> generator -> critic. The critic node extracts the claims and checks each against the chunks. `pass` -> answered. `fail` and `retry_count < MAX_RETRIES` -> rewriter produces new sub_queries, retry_count += 1, and the graph fans out to all retrievers again. Otherwise -> refused. Also refused straight from merge when retrieval returns nothing relevant. Refusal text is a fixed string; refusals contain no claims.

Plain baseline flow (for comparison only): one `retrieve(question, k)` -> generator. No planner, critic, or retry; it never refuses on its own except when retrieval returns nothing relevant. It shares the retriever, the generator prompt and the generator model with the self-healing system, so the comparison is fair.

Public entry points in `src/agent/graph.py`: `run(question: str) -> State` (self-healing, terminal state) and `run_plain(question: str) -> State` (plain baseline; same State shape, `claims` extracted only by the judge, `retry_count` always 0).
Public retrieval: `src/ingestion/retriever.py::retrieve(query: str, k: int = 5) -> list[Chunk]`.

## 3. Eval result JSON

Written by the eval runner to `eval/results/<run_id>.json`. Dataset rows in `eval/dataset/questions.jsonl`: `{id, question, answerable: bool, smoke: bool, reference_notes: str}`. `smoke: true` marks the smoke subset: 15-20 questions, a mix of answerable and unanswerable (at least 6 of each).

Runner modes: `--mode smoke` runs only `smoke: true` rows; `--mode full` runs every row. Every selected question is run through both systems: `plain` (`run_plain`) and `self_healing` (`run`). The judge scores both with the same prompt.

```json
{
  "meta": {
    "contract_version": 2,
    "eval_mode": "smoke|full",
    "prompt_version": "agent-1.0.0",
    "judge_prompt_version": "judge-1.0.0",
    "corpus_version": "a1b2c3d4e5f6",
    "generator_model": "...", "critic_model": "...", "judge_model": "...",
    "git_sha": "...", "timestamp": "ISO-8601"
  },
  "results": [{
    "id": "q001", "system": "plain|self_healing", "question": "...", "answerable": true,
    "final_status": "answered", "answer": "...", "retry_count": 1,
    "retrieved_chunk_ids": ["..."],
    "claims": [{"text": "...", "chunk_ids": ["..."], "verdict": "supported|unsupported|contradicted"}],
    "outcome": "answered_ok|hallucinated|correct_refusal|false_refusal",
    "latency_ms": 0, "llm_wait_ms": 0, "tokens_in": 0, "tokens_out": 0
  }],
  "aggregates": {
    "plain":        { "<aggregate block below>" },
    "self_healing": {
      "n_questions": 0, "n_answerable": 0, "n_unanswerable": 0, "n_answered": 0, "n_refused": 0,
      "hallucination_rate": 0.0, "faithfulness": 0.0,
      "correct_refusals": {"count": 0, "rate": 0.0},
      "false_refusals": {"count": 0, "rate": 0.0},
      "latency_ms": {"p50": 0, "p95": 0},
      "cost_usd_estimate": 0.0, "tokens_in": 0, "tokens_out": 0
    }
  }
}
```

- `aggregates` has one block per system with identical keys, computed only over that system's results. Results hold two rows per question (one per system).
- `generator_model`, `critic_model`, `judge_model` record the models actually used, after any fallback.
- `prompt_version` is `PROMPT_VERSION` in `src/agent/prompts/__init__.py`; bump on any prompt edit.
- `latency_ms` is end-to-end wall clock per question. `llm_wait_ms` is time spent in the concurrency queue and retry backoff (reported by `src/llm.py`). The gate uses `latency_ms - llm_wait_ms` so free-tier throttling does not cause false regressions.
- `cost_usd_estimate` is notional: tokens x list prices in `eval/pricing.json`, even though the free tier bills $0. It includes generator, critic and judge calls.

## 4. Definitions

- **Claim**: one atomic, self-contained factual assertion in the answer that can be checked against the corpus (a number, method property, result, comparison or causal statement). Excludes hedges, restatements of the question and citation markers. Extracted by the agent for the critic and re-extracted independently by the judge for scoring.
- **Hallucination**: an *answered* response with at least one claim the judge labels `unsupported` (not entailed by the cited or retrieved chunks) or `contradicted`. An answer to an `answerable: false` question is always a hallucination.
- **Correct refusal**: `answerable: false` and `final_status = refused`.
- **False refusal**: `answerable: true` and `final_status = refused`.

Aggregates (computed per system):
- `hallucination_rate` = hallucinated responses / answered responses.
- `faithfulness` = mean over answered responses of (supported claims / total claims).
- `correct_refusals.rate` = correct refusals / unanswerable questions.
- `false_refusals.rate` = false refusals / answerable questions.
- Percentiles use nearest-rank on `latency_ms - llm_wait_ms`; reported `latency_ms.p50/p95` use that adjusted value.

## 5. Gate rules (starting points)

Thresholds live in `ci/gate_config.json` and must equal these values. CI fails if any rule fails. Gate rules read `aggregates.self_healing` only; `aggregates.plain` is reported (PR comment, eval summary) but never gated. A run missing either system block is a failure.

**These values are starting points. They are calibrated after the first baseline eval run** (full mode), then updated here and in `ci/gate_config.json` through a `contract-change` PR. On the smoke subset the sample is small (about 10 answered questions), so `hallucination_rate <= 0.05` means effectively zero tolerance there; calibration decides whether smoke uses a looser value.

| rule | starting value |
|---|---|
| hallucination_rate | <= 0.05 |
| false_refusals.rate | <= 0.10 |
| p95 latency regression vs the mode's baseline | <= +25% |
| p50 latency regression vs baseline | <= +25% |

Also fails if `corpus_version` or `contract_version` differs from the baseline without a baseline update in the same PR. The baseline changes only via PR.

Baselines are per mode: `ci/baseline_smoke.json` and `ci/baseline_full.json`. The gate compares a result only with the baseline whose mode equals `meta.eval_mode`; a mode mismatch is a failure.

When it runs: pull requests run `--mode smoke`; pushes to `main` and a nightly schedule run `--mode full`. Both are gated.

## 6. LLM and model rules

- Every LLM call goes through `src/llm.py` (Gemini via `langchain-google-genai`). No other file imports `langchain_google_genai` or `google.generativeai`.
- `src/llm.py` API: `get_llm(role: "generator"|"critic"|"judge")`. It reads `GEMINI_API_KEY` from `.env` (via python-dotenv). Fails fast if missing. The key is never logged.
- Concurrency cap: global semaphore, `LLM_MAX_CONCURRENCY` (default 2).
- Retry: exponential backoff with jitter on 429 / `ResourceExhausted` / 503, honoring server retry-delay hints, max 6 attempts, cap 60 s per wait. Wait time accumulates into `llm_wait_ms`; token usage is recorded per call.
- Models are set by env vars, never hard-coded outside the defaults below. Each role has a primary and a fallback:

| role | env var | default | fallback env var | fallback default |
|---|---|---|---|---|
| generator | `GEN_MODEL` | `gemini-2.5-flash` | `GEN_MODEL_FALLBACK` | `gemini-2.5-flash-lite` |
| critic | `CRITIC_MODEL` | `gemini-2.5-flash` | `CRITIC_MODEL_FALLBACK` | `gemini-2.5-flash-lite` |
| judge | `JUDGE_MODEL` | `gemini-2.5-pro` | `JUDGE_MODEL_FALLBACK` | `gemini-2.5-flash-lite` |

  Defaults are placeholders until free-tier access is confirmed; the owner overrides them in `.env`. The fallback is used when the primary is unavailable (model not found, no access, or quota exhausted after retries). `src/llm.py` logs the switch (model names only) and reports the model actually used, which goes into `meta`.
- The judge must differ from the generator in **model and prompt**. After fallback resolution the runner asserts `judge_model != generator_model` and fails the run on a collision rather than degrading silently. Judge prompts live only under `eval/judge/` and are never imported from `src/agent`.
- Embeddings are local `BAAI/bge-small-en-v1.5` (inside `src/ingestion/`, not via `llm.py`).
- `.env` is gitignored; `.env.example` (no real values; lists `GEMINI_API_KEY`, `LLM_MAX_CONCURRENCY` and all six model vars) is committed.

## 7. Ownership

| owner | may edit |
|---|---|
| ingestion | `src/ingestion/`, `data/`, `tests/ingestion/` |
| agent | `src/agent/`, `tests/agent/` |
| eval | `eval/`, `tests/eval/` |
| ci | `ci/`, `.github/`, `tests/ci/` |
| shared | `CONTRACT.md`, `PLAN.md`, `src/schemas.py`, `src/llm.py`, `contract_examples/`, `tests/shared/`, `pyproject.toml`, `.env.example`, `.gitignore`, `.claude/` |

Rules:
1. A track edits only its own folders. Branches are named `track/<name>/<topic>`. The one-time Step 0 foundation branch `track/shared/foundation` creates the shared files and is exempt from the `contract_version` bump in rule 2; after it merges, rule 2 applies to every shared-file change.
2. Shared files change only via a PR labeled `contract-change` that updates `CONTRACT.md` (and bumps `contract_version`) in the same diff, with the reviewer agent passing and your approval.
3. A track needing a contract change opens that PR first; it does not work around the contract.
4. Source control is Git + GitHub on the existing `main`. Nobody runs `git init`. All changes land by PR, and `main` is protected by the CI gate.
5. Tracks interact only through the public interfaces above and `contract_examples/` fixtures.
