"""Gate tests: contract_examples/eval_result.json plus one failing case per rule."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ci import checks, gate

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = json.loads((ROOT / "contract_examples" / "eval_result.json").read_text("utf-8"))
CONFIG = gate.load_config()


def passing() -> dict:
    """The contract example with the false refusal removed, so every rule passes."""
    r = copy.deepcopy(EXAMPLE)
    r["aggregates"]["self_healing"]["false_refusals"] = {"count": 0, "rate": 0.0}
    return r


def baseline(mode: str = "smoke") -> dict:
    b = passing()
    b["meta"]["eval_mode"] = mode
    return b


def run(result: dict, expected_mode: str | None = None, baselines=None) -> gate.GateReport:
    if baselines is None:
        baselines = {"smoke": baseline("smoke"), "full": baseline("full")}
    return gate.evaluate(result, baselines, CONFIG, expected_mode)


def failed_rules(report: gate.GateReport) -> set[str]:
    return {c.rule for c in report.checks if not c.passed}


def test_passing_fixture_passes():
    report = run(passing(), "smoke")
    assert report.passed, [c for c in report.checks if not c.passed]


def test_contract_example_itself_fails_on_false_refusals():
    # The shipped example has a 0.5 false-refusal rate: it is the "red" fixture for that rule.
    assert failed_rules(run(EXAMPLE)) == {"false_refusals.rate"}


def test_seeded_baselines_pass_against_themselves():
    for mode in ("smoke", "full"):
        b = gate.load_json(ROOT / "ci" / f"baseline_{mode}.json")
        assert gate.evaluate(
            b, {m: gate.load_json(ROOT / "ci" / f"baseline_{m}.json") for m in gate.MODES}, CONFIG
        ).passed


def test_hallucination_rate_fails():
    r = passing()
    r["aggregates"]["self_healing"]["hallucination_rate"] = 0.06
    assert failed_rules(run(r)) == {"hallucination_rate"}


def test_hallucination_boundary_passes():
    r = passing()
    r["aggregates"]["self_healing"]["hallucination_rate"] = 0.05
    assert run(r).passed


def test_false_refusal_rate_fails():
    r = passing()
    r["aggregates"]["self_healing"]["false_refusals"]["rate"] = 0.11
    assert failed_rules(run(r)) == {"false_refusals.rate"}


@pytest.mark.parametrize(
    "p,rule", [("p95", "p95 latency regression"), ("p50", "p50 latency regression")]
)
def test_latency_regression_fails_and_boundary_passes(p, rule):
    base = baseline()["aggregates"]["self_healing"]["latency_ms"][p]
    r = passing()
    r["aggregates"]["self_healing"]["latency_ms"][p] = base * 1.25
    assert run(r).passed
    r["aggregates"]["self_healing"]["latency_ms"][p] = base * 1.25 + 1
    assert failed_rules(run(r)) == {rule}


def test_latency_improvement_passes():
    r = passing()
    r["aggregates"]["self_healing"]["latency_ms"] = {"p50": 1, "p95": 1}
    assert run(r).passed


def test_corpus_version_mismatch_fails():
    r = passing()
    r["meta"]["corpus_version"] = "ffffffffffff"
    assert failed_rules(run(r)) == {"corpus_version matches baseline"}


def test_contract_version_mismatch_fails():
    r = passing()
    r["meta"]["contract_version"] = 3
    assert failed_rules(run(r)) == {"contract_version matches baseline"}


def test_baseline_updated_in_same_pr_passes():
    r = passing()
    r["meta"]["corpus_version"] = "ffffffffffff"
    b = baseline()
    b["meta"]["corpus_version"] = "ffffffffffff"
    assert run(r, baselines={"smoke": b}).passed


def test_missing_system_block_fails_either_way():
    for system in gate.SYSTEMS:
        r = passing()
        del r["aggregates"][system]
        report = run(r)
        assert not report.passed
        assert any(f"aggregates.{system}" in c.detail for c in report.checks)


def test_plain_is_never_gated():
    r = passing()
    plain = r["aggregates"]["plain"]
    plain["hallucination_rate"] = 1.0
    plain["false_refusals"]["rate"] = 1.0
    plain["latency_ms"] = {"p50": 10**9, "p95": 10**9}
    assert run(r).passed


def test_mode_mismatch_with_run_fails():
    assert failed_rules(run(passing(), expected_mode="full")) == {"mode matches run"}


def test_gate_picks_baseline_by_mode_not_the_other():
    r = passing()
    r["meta"]["eval_mode"] = "full"
    full_base = baseline("full")
    full_base["aggregates"]["self_healing"]["latency_ms"] = {"p50": 10, "p95": 10}  # tight
    smoke_base = baseline("smoke")  # lenient: would pass
    assert failed_rules(run(r, baselines={"smoke": smoke_base, "full": full_base})) == {
        "p95 latency regression",
        "p50 latency regression",
    }


def test_missing_or_wrong_mode_baseline_fails():
    assert failed_rules(run(passing(), baselines={"full": baseline("full")})) == {
        "baseline present"
    }
    wrong = baseline("full")  # file for smoke slot but says full
    assert not run(passing(), baselines={"smoke": wrong}).passed


def test_invalid_eval_mode_fails():
    r = passing()
    r["meta"]["eval_mode"] = "nightly"
    assert not run(r).passed


def test_zero_baseline_latency_fails_on_any_positive():
    b = baseline()
    b["aggregates"]["self_healing"]["latency_ms"] = {"p50": 0, "p95": 0}
    assert failed_rules(run(passing(), baselines={"smoke": b})) == {
        "p95 latency regression",
        "p50 latency regression",
    }


def test_markdown_reports_both_systems():
    md = gate.render_markdown(run(passing(), "smoke"))
    assert "PASS" in md and "plain" in md and "self_healing" in md and "<!-- eval-gate -->" in md
    assert "FAIL" in gate.render_markdown(run(EXAMPLE))


def test_cli_exit_codes(tmp_path):
    ok, bad = tmp_path / "ok.json", tmp_path / "bad.json"
    ok.write_text(json.dumps(passing()))
    r = passing()
    r["aggregates"]["self_healing"]["hallucination_rate"] = 0.5
    bad.write_text(json.dumps(r))
    for name in ("smoke", "full"):
        (tmp_path / f"baseline_{name}.json").write_text(json.dumps(baseline(name)))
    args = ["--baseline-dir", str(tmp_path)]
    md = tmp_path / "out.md"
    assert gate.main(["--result", str(ok), *args, "--markdown", str(md)]) == 0
    assert md.exists()
    assert gate.main(["--result", str(bad), *args]) == 1
    assert gate.main(["--result", str(tmp_path / "missing.json"), *args]) == 2


# --- config / checks ----------------------------------------------------------------------


def test_gate_config_matches_contract():
    contract = (ROOT / "CONTRACT.md").read_text("utf-8")
    assert checks.check_gate_config(contract, CONFIG) == []


def test_gate_config_drift_detected():
    contract = (ROOT / "CONTRACT.md").read_text("utf-8")
    assert checks.check_gate_config(contract, {**CONFIG, "hallucination_rate_max": 0.2})


def test_ownership():
    assert (
        checks.ownership_violations("track/ci/gate", ["ci/gate.py", ".github/workflows/a.yml"])
        == []
    )
    assert checks.ownership_violations("track/ci/gate", ["CONTRACT.md", "eval/x.py"])
    assert checks.ownership_violations("feature/x", [])


def test_secret_and_import_scanners():
    fake_key = "AIza" + "x" * 35
    assert checks.scan_secrets({"a.py": f'k = "{fake_key}"'})
    assert checks.scan_secrets({"a.py": "GEMINI_API_KEY="}) == []
    assert checks.scan_imports({"src/agent/x.py": "from langchain_google_genai import X"})
    assert checks.scan_imports({"src/llm.py": "from langchain_google_genai import X"}) == []
    assert checks.env_tracked([".env", ".env.example"]) == [".env: env file is tracked by git"]


def test_repo_passes_static_checks():
    assert checks.main([]) == 0
