# CONTRACT.md — Self-Healing RAG

**Contract version:** 0.1.0 (draft, pending approval)
**Status:** PROPOSED — nothing in this file is binding until approved.

This file is the single source of truth for every interface shared between
tracks. If code and this file disagree, the code is wrong. If this file is
wrong, change it through the process in §9 — never by editing code first.

Corpus: three papers.

| `paper` id      | Title                                                                 | Source                |
|-----------------|-----------------------------------------------------------------------|-----------------------|
| `deepseek-r1`   | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL         | arXiv 2501.12948      |
| `deepseekmath`  | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open LMs | arXiv 2402.03300      |
| `dr-grpo`       | Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) | arXiv 2503.20783      |

The exact arXiv version (`vN`) and SHA-256 of each PDF are pinned in
`config/corpus.yaml` (shared file).

---

## 1. Global constraints (apply to every track)

1. **All LLM calls go through `src/llm.py`.** No other module may import
   `langchain_google_genai`, `google.generativeai`, `google.genai`, or call a
   Gemini HTTP endpoint. Gemini is accessed via `langchain-google-genai`.
2. **API key:** read from `GEMINI_API_KEY`, loaded from `.env` via
   `python-dotenv` in `src/llm.py` only. `.env` is git-ignored and never
   committed. `.env.example` (committed) lists variable names with empty values.
   In CI the key comes from the GitHub Actions secret `GEMINI_API_KEY`.
3. **`src/llm.py` must provide:**
   - a process-wide **concurrency cap** (`asyncio.Semaphore`, size from
     `LLM_MAX_CONCURRENCY`, default `2`) plus a requests-per-minute limiter
     (`LLM_MAX_RPM`, default `10`);
   - **retry with exponential backoff + jitter** on rate-limit / transient
     errors (HTTP 429, `ResourceExhausted`, 500, 503, timeouts): base 2 s,
     factor 2, cap 60 s, max 6 attempts; honour a server `retry-after` hint
     when present; never retry on 400/401/403;
   - per-call **usage accounting**: input tokens, output tokens, model,
     `latency_ms` (time inside the API call) and `throttle_ms` (time spent
     waiting on the semaphore, RPM limiter, and backoff sleeps).
4. **Model roles.** Every LLM call declares one role:
   `planner | generator | critic | judge`. Role → model + temperature is
   configured in `config/models.yaml`. **The `judge` role MUST use a different
   model than `generator` and `critic`, AND its own prompt files** (which live
   under `eval/`, never under `src/agent/`). `src/llm.py` raises at startup if
   `judge.model` equals `generator.model` or `critic.model`.
5. **Embeddings are local:** `BAAI/bge-small-en-v1.5` via
   `sentence-transformers`, 384-dim, L2-normalised. Queries are prefixed with
   `"Represent this sentence for searching relevant passages: "`; passages are
   not prefixed. Implemented once in `src/embeddings.py` (shared) and used by
   both indexing and querying.
6. **No secrets in code, configs, notebooks, fixtures, logs or eval outputs.**

### `src/llm.py` public interface (frozen by this contract)

```python
Role = Literal["planner", "generator", "critic", "judge"]

class LLMUsage(BaseModel):
    role: Role
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float      # time inside API calls (summed across retries)
    throttle_ms: float     # time waiting on caps/backoff
    attempts: int

async def ainvoke(role: Role, messages: list[BaseMessage],
                  *, schema: type[BaseModel] | None = None
                  ) -> tuple[AIMessage | BaseModel, LLMUsage]: ...

def usage_log() -> UsageLog   # context-local accumulator, reset per question
```

---

## 2. Chunk schema

Pydantic model `Chunk` in `src/contracts.py` (shared). Serialized one-per-line
in `data/index/chunks.jsonl`.

| Field             | Type   | Rules |
|-------------------|--------|-------|
| `id`              | `str`  | Deterministic: `"{paper}:p{page:03d}:c{n:03d}"`, where `n` is the 0-based chunk index within that page. Stable across re-runs with the same corpus version. |
| `paper`           | `str`  | One of `deepseek-r1`, `deepseekmath`, `dr-grpo`. |
| `section`         | `str`  | Nearest preceding section heading, normalised (e.g. `"2.3 Group Relative Policy Optimization"`). `"Abstract"`, `"References"`, `"Appendix A ..."` allowed. `"unknown"` if no heading detected. Never empty. |
| `page`            | `int`  | 1-based physical PDF page on which the chunk **starts**. |
| `text`            | `str`  | Extracted text, whitespace-normalised, non-empty, ≤ 512 bge tokens. |
| `embedding_model` | `str`  | Exactly `"BAAI/bge-small-en-v1.5"`. |

Example:

```json
{"id": "dr-grpo:p004:c002", "paper": "dr-grpo", "section": "3.1 GRPO Leads to Biased Optimization",
 "page": 4, "text": "…", "embedding_model": "BAAI/bge-small-en-v1.5"}
```

Index artifact (written by ingestion, read by agent), under `data/index/`
(git-ignored, rebuilt in CI):

- `chunks.jsonl` — one `Chunk` per line, ordered by `id`.
- `embeddings.npy` — `float32[N, 384]`, row *i* ↔ line *i* of `chunks.jsonl`.
- `manifest.json` — `{corpus_version, embedding_model, chunker_params, pdf_sha256: {paper: sha}, n_chunks, built_at}`.

`corpus_version` = first 12 hex chars of SHA-256 over
(sorted PDF SHA-256s + `chunker_params` JSON + `embedding_model`).

Retrieval interface (implemented by ingestion, consumed by agent):

```python
# src/ingestion/retriever.py
class ScoredChunk(BaseModel):     # defined in src/contracts.py
    chunk: Chunk
    score: float                  # cosine similarity, [-1, 1]

def load_retriever(index_dir: Path = Path("data/index")) -> Retriever
class Retriever:
    corpus_version: str
    def search(self, query: str, k: int = 6,
               papers: list[str] | None = None) -> list[ScoredChunk]: ...
```

---

## 3. LangGraph state

`GraphState` is a `TypedDict` in `src/contracts.py` (shared). All nodes read
and write only these keys. Default reducer (overwrite) unless stated.

| Key              | Type                         | Written by | Meaning |
|------------------|------------------------------|------------|---------|
| `question`       | `str`                        | caller     | User question, unchanged for the whole run. |
| `sub_queries`    | `list[str]`                  | planner / rewriter | Retrieval queries for the current attempt (1–4). Replaced on retry. |
| `chunks`         | `list[ScoredChunk]`          | retriever  | Deduplicated by `chunk.id`, sorted by score desc, ≤ 12. |
| `answer`         | `str \| None`                | generator / finaliser | Answer text with inline citations `[chunk_id]`. On refusal: the standard refusal text (§3.2). |
| `claims`         | `list[Claim]`                | claim extractor | Atomic claims extracted from `answer` (§5). |
| `critic_verdict` | `CriticVerdict \| None`      | critic     | See below. |
| `retry_count`    | `int`                        | rewriter   | Starts at `0`; +1 each time the critic sends the run back. |
| `final_status`   | `Literal["answered","refused"] \| None` | finaliser | `None` until the graph terminates; always set at END. |

```python
class Claim(BaseModel):
    id: str                       # "c1", "c2", … within one answer
    text: str
    cited_chunk_ids: list[str]    # may be empty (that alone makes it unsupported)
    supported: bool | None = None # set by critic; None = not yet checked

class CriticVerdict(BaseModel):
    decision: Literal["accept", "retry", "refuse"]
    unsupported_claim_ids: list[str]
    reason: str                   # short, human-readable
```

### 3.1 Control flow (proposed)

```
plan → retrieve → generate → extract_claims → critic
critic.accept                                   → finalise(answered)
critic.retry  and retry_count <  MAX_RETRIES    → rewrite → retrieve
critic.retry  and retry_count >= MAX_RETRIES    → finalise(refused)
critic.refuse                                   → finalise(refused)
```

- `MAX_RETRIES = 2` (i.e. at most 3 generation attempts).
- The critic accepts only if **every** claim is `supported=True`.
- Generator may itself output the refusal sentinel if retrieved chunks are
  insufficient; the critic then returns `refuse` (no retry needed).

### 3.2 Refusal

When `final_status == "refused"`, `answer` is exactly
`"I can't answer that from the DeepSeek-R1, DeepSeekMath, or Dr. GRPO papers."`
and `claims == []`.

### 3.3 Agent entry point

```python
# src/agent/__init__.py
async def answer(question: str) -> AgentResult

class AgentResult(BaseModel):     # src/contracts.py
    state: GraphState             # final state
    usage: list[LLMUsage]         # every LLM call made for this question
    prompt_version: str
    corpus_version: str
```

`prompt_version` = `"{semver}+{sha8}"`, where `semver` is declared in
`src/agent/prompts/VERSION` and `sha8` is the first 8 hex chars of SHA-256 over
all files in `src/agent/prompts/` (sorted by path). Bump `semver` on any
intentional prompt change.

---

## 4. Eval result JSON

Written by the eval runner to `eval/results/{run_id}.json` (git-ignored; CI
uploads it as an artifact). Pydantic model `EvalRun` in `src/contracts.py`.
`schema_version` changes whenever this section changes.

```jsonc
{
  "schema_version": "1.0",
  "run_id": "2026-11-19T10-00-00Z_ab12cd3",
  "timestamp": "2026-11-19T10:00:00Z",       // ISO-8601 UTC
  "git_sha": "ab12cd3…",
  "contract_version": "0.1.0",
  "prompt_version": "1.0.0+1a2b3c4d",          // from AgentResult
  "judge_prompt_version": "1.0.0+9f8e7d6c",    // same scheme, over eval/prompts/
  "corpus_version": "3f9a0c1d2e4b",
  "dataset": { "path": "eval/dataset/questions.jsonl", "sha8": "…", "split": "full" },
  "models": { "planner": "…", "generator": "…", "critic": "…", "judge": "…" },

  "results": [
    {
      "qid": "q017",
      "question": "…",
      "label": "answerable",                    // answerable | unanswerable
      "final_status": "answered",               // answered | refused | error
      "answer": "…",
      "retry_count": 1,
      "retrieved_chunk_ids": ["deepseekmath:p012:c001", "…"],
      "judged_claims": [
        { "text": "…", "verdict": "supported", "evidence_chunk_ids": ["…"] }
        // verdict: supported | unsupported | contradicted
      ],
      "n_claims": 4,
      "n_unsupported": 0,                       // unsupported + contradicted
      "hallucinated": false,                    // n_unsupported > 0
      "faithfulness": 1.0,                      // supported / n_claims; null if refused or n_claims == 0
      "outcome": "answered_ok",                 // see §5.5
      "latency_ms": 8421.0,                     // end-to-end wall clock for answer()
      "service_latency_ms": 6120.0,             // latency_ms − throttle_ms
      "throttle_ms": 2301.0,
      "llm_calls": 5,
      "input_tokens": 9123,
      "output_tokens": 812,
      "cost_usd": 0.0031,                       // agent calls only (not judge)
      "error": null                             // string if final_status == "error"
    }
  ],

  "aggregates": {
    "n_questions": 40, "n_answerable": 30, "n_unanswerable": 10, "n_errors": 0,
    "hallucination_rate": 0.033,     // §5.2
    "faithfulness": 0.97,            // §5.3
    "correct_refusals": 9,           // count
    "correct_refusal_rate": 0.90,    // correct_refusals / n_unanswerable
    "false_refusals": 2,             // count
    "false_refusal_rate": 0.067,     // false_refusals / n_answerable
    "latency_p50_ms": 7300.0,
    "latency_p95_ms": 14100.0,
    "service_latency_p50_ms": 5200.0,
    "service_latency_p95_ms": 9800.0,
    "cost_estimate_usd": 0.12,       // Σ cost_usd (agent) at paid-tier list price
    "judge_cost_estimate_usd": 0.05
  },

  "gate": {                          // filled by ci/check_gate.py, null from runner
    "passed": true,
    "baseline_run_id": "…",
    "checks": [ { "name": "hallucination_rate", "value": 0.033, "limit": 0.05, "passed": true } ]
  }
}
```

Percentiles use linear interpolation (`numpy.percentile`, default method) over
questions with `final_status != "error"`. Cost = tokens × per-1M-token prices
in `config/pricing.yaml` (shared); the free tier is billed $0, so this is an
*estimate at paid list price*.

---

## 5. Definitions

### 5.1 Claim
An **atomic, self-contained factual assertion** in an answer that can be
checked against the corpus — one subject, one predicate, with numbers, units,
model names, and conditions included (e.g. *"DeepSeekMath 7B scores 51.7% on
MATH without tool use"*). Hedges, restatements of the question, transitions
and the refusal text are not claims. Compound sentences are split.

For **evaluation**, claims are re-extracted from `answer` by the judge using
`eval/prompts/` — the agent's own `claims` list is *not* trusted by the eval.

### 5.2 Hallucination
A claim is **hallucinated** if the judge labels it `unsupported` (no retrieved
chunk entails it) or `contradicted` (a retrieved chunk contradicts it), judged
**only against the chunks retrieved for that question** (`retrieved_chunk_ids`),
not world knowledge. A question is hallucinated if it has ≥ 1 hallucinated claim.

`hallucination_rate` = (# questions with `final_status = answered` and
`hallucinated = true`) / (# questions with `final_status = answered`).
Answering an *unanswerable* question with any claim counts as hallucinated.

### 5.3 Faithfulness
`faithfulness` = Σ supported claims / Σ claims, pooled over all answered
questions (micro-average). Complements the question-level hallucination rate.

### 5.4 Correct refusal / false refusal
Each eval question carries a human label:
- `answerable` — the three papers contain enough information to answer it;
- `unanswerable` — they do not (out-of-corpus, false premise, or requires
  information the papers don't give).

- **Correct refusal:** label `unanswerable` and `final_status = refused`.
- **False refusal:** label `answerable` and `final_status = refused`.
- **Missed refusal:** label `unanswerable` and `final_status = answered`
  (also counted as a hallucination per §5.2).

### 5.5 `outcome` values
`answered_ok` (answerable, answered, not hallucinated) · `answered_hallucinated`
· `correct_refusal` · `false_refusal` · `missed_refusal` · `error`.

---

## 6. Gate rules (PROPOSED — adjust before approval)

Thresholds live in `config/gate.yaml` (shared). CI fails the PR if **any**
check fails.

| Check | Rule | Proposed value |
|-------|------|----------------|
| Hallucination | `hallucination_rate ≤ max` | **0.05** |
| False refusals | `false_refusal_rate ≤ max` | **0.15** |
| Correct refusals *(extra, optional)* | `correct_refusal_rate ≥ min` | **0.70** |
| Latency regression | `service_latency_p95_ms ≤ baseline × (1 + max_regression)` | **+25 %** |
| Latency absolute | `service_latency_p95_ms ≤ max` | **30 000 ms** |
| Errors | `n_errors / n_questions ≤ max` | **0.05** |

- Latency is gated on **service latency** (excluding throttle/backoff time) so
  free-tier rate limiting doesn't cause false regressions. Raw latency is
  still reported.
- **Baseline** = `eval/baselines/main.json`, the most recent accepted run on
  `main`. It is updated only by an explicit PR. If no baseline exists, the
  regression check is skipped (reported as `skipped`, not `passed`).
- If `prompt_version`, `corpus_version` or the dataset `sha8` differs from the
  baseline, the latency check still runs; quality checks are absolute and do
  not depend on the baseline.

```yaml
# config/gate.yaml
hallucination_rate_max: 0.05
false_refusal_rate_max: 0.15
correct_refusal_rate_min: 0.70
service_latency_p95_regression_max: 0.25
service_latency_p95_abs_max_ms: 30000
error_rate_max: 0.05
```

---

## 7. Ownership

| Owner        | Owns (may edit)                                                    |
|--------------|--------------------------------------------------------------------|
| `ingestion`  | `src/ingestion/`, `tests/ingestion/`, `scripts/ingest/`            |
| `agent`      | `src/agent/` (incl. `src/agent/prompts/`), `tests/agent/`          |
| `eval`       | `eval/` (dataset, runner, judge, `eval/prompts/`, baselines), `tests/eval/` |
| `ci`         | `.github/`, `ci/`, `tests/ci/`                                     |
| **contract** (shared) | `CONTRACT.md`, `PLAN.md`, `src/contracts.py`, `src/llm.py`, `src/embeddings.py`, `src/__init__.py`, `config/`, `pyproject.toml`, `.env.example`, `.gitignore`, `.claude/`, `README.md` |

**Rule:** a track branch edits **only** its own folders. Shared files change
**only** through a contract-change PR (§9). Generated/ignored paths
(`data/`, `eval/results/`, `.env`) are never committed.

Branch naming: `ingestion/<topic>`, `agent/<topic>`, `eval/<topic>`,
`ci/<topic>`, `contract/<topic>`. The reviewer subagent infers the track from
the prefix.

---

## 8. Shared configuration files

- `config/models.yaml` — role → `{model, temperature, max_output_tokens}`.
  Proposed: planner/generator/critic on one Gemini Flash-class model at low
  temperature; judge on a different Gemini model at temperature 0.
- `config/pricing.yaml` — per-model USD per 1M input/output tokens.
- `config/corpus.yaml` — paper id → arXiv id+version, URL, SHA-256.
- `config/gate.yaml` — §6.
- `.env.example` — `GEMINI_API_KEY=`, `LLM_MAX_CONCURRENCY=2`, `LLM_MAX_RPM=10`.

---

## 9. Changing the contract

1. Open a branch `contract/<topic>` touching only shared files.
2. Update this file first: the relevant section, the version number
   (patch = clarification, minor = additive field, major = breaking), and the
   changelog below.
3. Update `src/contracts.py` (and other shared code) to match in the same PR.
4. List affected tracks in the PR description; each affected track adapts in
   its own follow-up PR.
5. The reviewer subagent must report no violations before merge.

## Changelog

- **0.1.0** — initial draft.
