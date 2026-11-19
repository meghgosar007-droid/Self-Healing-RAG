# PLAN.md

Four tracks, each owning its folders per CONTRACT.md section 7. Shared files are created once in Step 0, then change only through the contract.

## Step 0 - Foundation (shared, merged first)

Not a fifth track; it unblocks the others and is done once.
- `src/schemas.py` (Chunk, Claim, State, EvalResult models matching CONTRACT.md)
- `src/llm.py` (key from `.env`, concurrency cap, backoff retry, token/wait accounting) with a mocked-API unit test
- `contract_examples/` (sample `chunk.json`, `state.json`, `eval_result.json`)
- `pyproject.toml`, `.env.example` (all model env vars), `.gitignore` (includes `.env`, `data/index/`, `eval/results/`; must be populated before the first commit). The repo already exists on `main`; no `git init`.
- `.claude/agents/reviewer.md`

## Track 1 - Ingestion

- **Owns:** `src/ingestion/`, `data/`, `tests/ingestion/`
- **Depends on:** Step 0 (schemas only)
- **Inputs:** the owner places the three PDFs in `data/papers/` (`deepseek-r1.pdf`, `deepseek-math.pdf`, `dr-grpo.pdf`). Ingestion reads local files only; no downloading.
- **Delivers:** PDF parse -> section-aware chunks -> local bge-small-en-v1.5 embeddings -> Chroma index (`data/index/`); `data/chunks.jsonl`, `data/manifest.json` (with `corpus_version`); `retrieve(query, k) -> list[Chunk]`
- **Done when:** chunks validate against the schema, ids are stable across re-runs, and a handful of known passages are retrievable in top-5.

## Track 2 - Agent graph

- **Owns:** `src/agent/` (graph, nodes, `prompts/` with `PROMPT_VERSION`), `tests/agent/`
- **Depends on:** Step 0; `retrieve()` interface (develops against a stub returning `contract_examples/` chunks until Track 1 lands)
- **Delivers:** `run(question) -> State` implementing planner / parallel retrievers / merge / generator / critic (with claim extraction) / rewriter / refuse, plus `run_plain(question) -> State` (retrieve -> generate baseline)
- **Done when:** state transitions are unit-tested with a mocked `llm.py`, retry cap and refusal paths are covered, and a live smoke run works on 3 questions with both `run` and `run_plain`.

## Track 3 - Eval suite

- **Owns:** `eval/` (dataset, judge + prompts, runner, `pricing.json`, `results/`), `tests/eval/`
- **Depends on:** Step 0; `run()` interface (develops against a fake agent fixture); real runs need Tracks 1 and 2
- **Delivers:** labeled dataset (answerable and unanswerable, with a 15-20 question `smoke` subset), independent judge, runner with `--mode smoke|full` that runs every question through both plain and self-healing systems and produces the CONTRACT.md eval JSON with per-system aggregates, metric computation, baseline candidates for both modes
- **Done when:** metric math is unit-tested on hand-built results, judge differs from generator in model and prompt, and one full real run validates against the schema.

## Track 4 - CI

- **Owns:** `ci/` (`gate.py`, `gate_config.json`, `baseline_smoke.json`, `baseline_full.json`), `.github/workflows/`, `tests/ci/`
- **Depends on:** Step 0; the eval result JSON format (develops against `contract_examples/eval_result.json`); needs Track 3's first real run to seed the baseline
- **Delivers:** workflow (lint, unit tests, reviewer-style checks, eval run, gate): smoke eval on PRs, full eval on `main` and nightly; `gate.py` applying CONTRACT.md section 5 against the per-mode baseline; `GEMINI_API_KEY` read from repo secrets; PR comment summarizing metrics for both systems (plain vs self-healing)
- **Done when:** gate passes on the baseline (self-healing aggregates only), and deliberately degraded fixtures fail each gate rule.

## Merge order

1. Step 0 foundation
2. Track 1 Ingestion
3. Track 2 Agent (can be built in parallel from step 1, merges after ingestion)
4. Track 3 Eval (parallel dev; merges after agent so real runs work)
5. Track 4 CI (parallel dev; merges last, then the first full run calibrates the gate thresholds and seeds both baselines in a follow-up `contract-change` PR)

Tracks 1-4 can be developed in parallel against `contract_examples/`; the order above is only the merge order. Every PR must pass the reviewer agent before merge.
