# contract_examples

Hand-written fixtures that validate against `src/schemas.py`. Tracks build and test against these instead of each other.

- The paper text and page numbers in the chunks are **fixtures for testing**, not extracted from the real PDFs.
- `eval_result.json` is internally consistent with CONTRACT.md section 4: the aggregates are the metrics of its rows. Latency percentiles use `latency_ms - llm_wait_ms` (nearest-rank). Costs are made up.
- `chunks.jsonl` is three chunks, one per paper, for stubbing `retrieve()`.
- `tests/shared/` validates every file here, so a schema change that breaks a fixture fails CI.
