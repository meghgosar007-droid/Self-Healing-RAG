"""CI gate: apply CONTRACT.md section 5 to an eval result.

    python -m ci.gate --result eval/results/<run>.json [--expected-mode smoke] [--markdown out.md]

Exit 0 = pass, 1 = fail, 2 = unusable input (missing/invalid file). Reads only
``aggregates.self_healing`` for gating; ``aggregates.plain`` is reported, never gated.
The baseline is picked by ``meta.eval_mode``. All thresholds come from ``gate_config.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

CI_DIR = Path(__file__).parent
CONFIG_PATH = CI_DIR / "gate_config.json"
MODES = ("smoke", "full")
SYSTEMS = ("plain", "self_healing")
CONFIG_KEYS = (
    "hallucination_rate_max",
    "false_refusal_rate_max",
    "p95_latency_regression_max",
    "p50_latency_regression_max",
)
SH = ("aggregates", "self_healing")


class GateInputError(Exception):
    """Result, baseline or config is missing or malformed (exit code 2)."""


@dataclass
class Check:
    rule: str
    passed: bool
    detail: str


@dataclass
class GateReport:
    mode: str
    checks: list[Check]
    result: dict
    baseline: dict | None

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


def load_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise GateInputError(f"cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise GateInputError(f"{path}: top level must be a JSON object")
    return data


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = load_json(path)
    missing = [k for k in CONFIG_KEYS if not isinstance(cfg.get(k), int | float)]
    if missing:
        raise GateInputError(f"{path}: missing/non-numeric keys {missing}")
    return cfg


def _num(obj: object, *path: str) -> float | None:
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    if isinstance(obj, bool) or not isinstance(obj, int | float):
        return None
    return float(obj)


def structural_problems(doc: dict, label: str) -> list[str]:
    """Everything the gate needs must be present; otherwise fail closed."""
    aggs = doc.get("aggregates")
    if not isinstance(aggs, dict):
        return [f"{label}: no aggregates block"]
    problems = [
        f"{label}: missing aggregates.{s} block"
        for s in SYSTEMS
        if not isinstance(aggs.get(s), dict)
    ]
    if isinstance(aggs.get("self_healing"), dict):
        for path in (
            ("hallucination_rate",),
            ("false_refusals", "rate"),
            ("latency_ms", "p50"),
            ("latency_ms", "p95"),
        ):
            if _num(doc, *SH, *path) is None:
                problems.append(
                    f"{label}: missing numeric aggregates.self_healing.{'.'.join(path)}"
                )
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        problems.append(f"{label}: no meta block")
    else:
        problems += [
            f"{label}: meta.{k} missing"
            for k in ("eval_mode", "corpus_version", "contract_version")
            if k not in meta
        ]
    return problems


def _limit_check(rule: str, value: float, limit: float) -> Check:
    ok = value <= limit
    return Check(rule, ok, f"{value:.4f} {'<=' if ok else '>'} {limit:.4f}")


def _regression_check(rule: str, current: float, base: float, max_pct: float) -> Check:
    allowed = base * (1 + max_pct)
    ok = current <= allowed
    change = f"{(current - base) / base:+.1%}" if base else "n/a, baseline is 0"
    return Check(
        rule,
        ok,
        f"{current:.0f} ms vs baseline {base:.0f} ms ({change}); "
        f"limit +{max_pct:.0%} = {allowed:.0f} ms",
    )


def evaluate(
    result: dict, baselines: dict[str, dict], config: dict, expected_mode: str | None = None
) -> GateReport:
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    mode = meta.get("eval_mode")

    problems = structural_problems(result, "result")
    if mode not in MODES:
        problems.append(f"result: meta.eval_mode must be one of {MODES}, got {mode!r}")
    if problems:
        return GateReport(
            str(mode), [Check("input valid", False, p) for p in problems], result, None
        )

    checks: list[Check] = []
    if expected_mode and expected_mode != mode:
        checks.append(
            Check("mode matches run", False, f"ran {expected_mode!r}, result is {mode!r}")
        )
    baseline = baselines.get(mode)
    if baseline is None:
        checks.append(Check("baseline present", False, f"no baseline file for mode {mode!r}"))
        return GateReport(mode, checks, result, None)
    bproblems = structural_problems(baseline, f"baseline_{mode}")
    if (baseline.get("meta") or {}).get("eval_mode") != mode:
        bproblems.append(f"baseline_{mode}: meta.eval_mode is not {mode!r} (mode mismatch)")
    if bproblems:
        checks += [Check("baseline valid", False, p) for p in bproblems]
        return GateReport(mode, checks, result, None)

    checks.append(
        _limit_check(
            "hallucination_rate",
            _num(result, *SH, "hallucination_rate"),
            config["hallucination_rate_max"],
        )
    )
    checks.append(
        _limit_check(
            "false_refusals.rate",
            _num(result, *SH, "false_refusals", "rate"),
            config["false_refusal_rate_max"],
        )
    )
    for p, key in (("p95", "p95_latency_regression_max"), ("p50", "p50_latency_regression_max")):
        checks.append(
            _regression_check(
                f"{p} latency regression",
                _num(result, *SH, "latency_ms", p),
                _num(baseline, *SH, "latency_ms", p),
                config[key],
            )
        )
    for key in ("corpus_version", "contract_version"):
        cur, base = result["meta"][key], baseline["meta"][key]
        detail = (
            f"{cur!r} == {base!r}"
            if cur == base
            else f"{cur!r} != baseline {base!r}; update the baseline in the same PR"
        )
        checks.append(Check(f"{key} matches baseline", cur == base, detail))
    return GateReport(mode, checks, result, baseline)


def _fmt(v: float | None, spec: str) -> str:
    return format(v, spec) if v is not None else "n/a"


def render_markdown(report: GateReport) -> str:
    lines = [
        "<!-- eval-gate -->",
        f"## Eval gate: {'PASS' if report.passed else 'FAIL'} (`{report.mode}` mode)",
        "",
        "| rule (self_healing) | result | detail |",
        "|---|---|---|",
    ]
    lines += [
        f"| {c.rule} | {'pass' if c.passed else '**FAIL**'} | {c.detail} |" for c in report.checks
    ]
    aggs = report.result.get("aggregates")
    aggs = aggs if isinstance(aggs, dict) else {}
    rows = [
        ("hallucination_rate", ("hallucination_rate",), ".3f"),
        ("faithfulness", ("faithfulness",), ".3f"),
        ("false_refusal_rate", ("false_refusals", "rate"), ".3f"),
        ("correct_refusal_rate", ("correct_refusals", "rate"), ".3f"),
        ("latency p50 (ms)", ("latency_ms", "p50"), ".0f"),
        ("latency p95 (ms)", ("latency_ms", "p95"), ".0f"),
        ("cost_usd_estimate", ("cost_usd_estimate",), ".4f"),
    ]
    lines += ["", "Plain vs self-healing (plain is reported, never gated):", ""]
    lines += ["| metric | plain | self_healing |", "|---|---|---|"]
    for label, path, spec in rows:
        vals = [_fmt(_num(aggs, s, *path), spec) for s in SYSTEMS]
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    meta = report.result.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    lines += ["", f"corpus `{meta.get('corpus_version')}` | sha `{str(meta.get('git_sha'))[:7]}`"]
    return "\n".join(lines) + "\n"


def run_gate(
    result_path: Path,
    baseline_dir: Path = CI_DIR,
    config_path: Path = CONFIG_PATH,
    expected_mode: str | None = None,
) -> GateReport:
    result = load_json(result_path)
    config = load_config(config_path)
    baselines = {}
    for mode in MODES:
        p = Path(baseline_dir) / f"baseline_{mode}.json"
        if p.exists():
            baselines[mode] = load_json(p)
    return evaluate(result, baselines, config, expected_mode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, default=CI_DIR)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--expected-mode", choices=MODES)
    ap.add_argument("--markdown", type=Path, help="also write the PR-comment markdown here")
    args = ap.parse_args(argv)
    try:
        report = run_gate(args.result, args.baseline_dir, args.config, args.expected_mode)
    except GateInputError as e:
        print(f"GATE ERROR: {e}", file=sys.stderr)
        return 2
    md = render_markdown(report)
    print(md)
    if args.markdown:
        args.markdown.write_text(md, encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
