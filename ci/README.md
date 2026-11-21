# ci

- `gate.py` - applies CONTRACT.md section 5 to an eval result (`python -m ci.gate --result <json> --expected-mode smoke|full`). Exit 0 pass, 1 fail, 2 unusable input.
- `gate_config.json` - thresholds; `checks.py` fails CI if they drift from CONTRACT.md section 5.
- `checks.py` - reviewer-style static checks (secrets, tracked `.env`, Gemini imports outside `src/llm.py`, track ownership).
- `baseline_smoke.json`, `baseline_full.json` - **PLACEHOLDERS** seeded from `contract_examples/eval_result.json` (corpus `a1b2c3d4e5f6`, results empty). The first real run will fail the corpus_version rule until both are replaced with real baseline runs via a `contract-change`/baseline PR.

Red -> green demo (offline): `python -m ci.gate --result contract_examples/eval_result.json` exits 1 (false-refusal rate 0.5 > 0.10); set `aggregates.self_healing.false_refusals` to `{"count": 0, "rate": 0.0}` in a copy and it exits 0.
