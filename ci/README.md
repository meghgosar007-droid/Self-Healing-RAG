# ci

- `gate.py` - applies CONTRACT.md section 5 to an eval result (`python -m ci.gate --result <json> --expected-mode smoke|full`). Exit 0 pass, 1 fail, 2 unusable input.
- `gate_config.json` - thresholds; `checks.py` fails CI if they drift from CONTRACT.md section 5.
- `checks.py` - reviewer-style static checks (secrets, tracked `.env`, Gemini imports outside `src/llm.py`, track ownership).
- `baseline_smoke.json` - real smoke run (corpus `cbf2cc00e93d`, git `eda32df`, contract v3), seeded from the eval track's local run.
- `baseline_full.json` - **PLACEHOLDER** (contract_examples copy, latencies 3.8 s / 5 s). Any real full run fails the corpus_version rule and both latency rules (fail closed) until it is replaced from a real `--mode full` run (30-question set) via PR.

Red -> green demo (offline): `python -m ci.gate --result contract_examples/eval_result.json` exits 1 (false-refusal rate 0.5 > 0.10); set `aggregates.self_healing.false_refusals` to `{"count": 0, "rate": 0.0}` in a copy and it exits 0.

Mode policy (temporary): the workflow runs `--mode smoke` (18 questions) on PRs, pushes to `main` and nightly. `--mode full` runs only on manual dispatch until a full baseline exists. This deviates from CONTRACT.md section 5; reconcile with a `contract-change` PR.
