"""Metric math on hand-built rows, plus the contract example as an oracle."""

import json
from pathlib import Path

import pytest

from eval import metrics
from src.schemas import EvalResult, JudgedClaim, ResultRow

EXAMPLE = Path(__file__).resolve().parents[2] / "contract_examples" / "eval_result.json"


def claim(verdict):
    return JudgedClaim(text="c", chunk_ids=[], verdict=verdict)


def row(
    status="answered",
    answerable=True,
    verdicts=(),
    latency=1000,
    wait=0,
    tin=0,
    tout=0,
    system="plain",
):
    claims = [claim(v) for v in verdicts]
    return ResultRow(
        id="q",
        system=system,
        question="?",
        answerable=answerable,
        final_status=status,
        answer="a" if status == "answered" else None,
        retry_count=0,
        retrieved_chunk_ids=[],
        claims=claims,
        outcome=metrics.classify_outcome(status, answerable, claims),
        latency_ms=latency,
        llm_wait_ms=wait,
        tokens_in=tin,
        tokens_out=tout,
    )


# --- classify_outcome ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,answerable,verdicts,expected",
    [
        ("refused", True, [], "false_refusal"),
        ("refused", False, [], "correct_refusal"),
        ("answered", True, ["supported", "supported"], "answered_ok"),
        ("answered", True, ["supported", "unsupported"], "hallucinated"),
        ("answered", True, ["contradicted"], "hallucinated"),
        ("answered", True, [], "answered_ok"),  # no claims: nothing unsupported
        ("answered", False, ["supported"], "hallucinated"),  # unanswerable + answered: always
        ("answered", False, [], "hallucinated"),
    ],
)
def test_classify_outcome(status, answerable, verdicts, expected):
    assert metrics.classify_outcome(status, answerable, [claim(v) for v in verdicts]) == expected


# --- percentiles --------------------------------------------------------------------------


def test_nearest_rank():
    assert metrics.nearest_rank([10, 20, 30, 40], 0.50) == 20  # ceil(2.0) = 2nd
    assert metrics.nearest_rank([10, 20, 30, 40], 0.95) == 40  # ceil(3.8) = 4th
    assert metrics.nearest_rank([5], 0.95) == 5
    assert metrics.nearest_rank([3, 1, 2], 0.50) == 2  # unsorted input
    assert metrics.nearest_rank([], 0.5) == 0


# --- aggregate block ----------------------------------------------------------------------


def test_aggregate_hand_built():
    rows = [
        row(verdicts=["supported", "supported"], latency=1000, wait=0, tin=10, tout=1),  # ok
        row(verdicts=["supported", "unsupported"], latency=2000, wait=500, tin=20, tout=2),  # bad
        row(status="refused", answerable=True, latency=500, wait=0, tin=5, tout=1),  # false ref
        row(status="refused", answerable=False, latency=400, wait=100, tin=5, tout=1),  # correct
        row(answerable=False, verdicts=["unsupported"], latency=3000, wait=0, tin=1, tout=1),
    ]
    a = metrics.aggregate_block(rows, cost_usd=0.1234567)
    assert (a.n_questions, a.n_answerable, a.n_unanswerable) == (5, 3, 2)
    assert (a.n_answered, a.n_refused) == (3, 2)
    # answered = 3; hallucinated = the unsupported answerable one + the answered unanswerable
    assert a.hallucination_rate == pytest.approx(2 / 3, abs=1e-4)
    # faithfulness over answered: (1.0 + 0.5 + 0.0) / 3
    assert a.faithfulness == pytest.approx(0.5)
    assert a.correct_refusals.count == 1 and a.correct_refusals.rate == pytest.approx(0.5)
    assert a.false_refusals.count == 1 and a.false_refusals.rate == pytest.approx(1 / 3, abs=1e-4)
    # adjusted latencies: 1000, 1500, 500, 300, 3000 -> sorted 300 500 1000 1500 3000
    assert a.latency_ms.p50 == 1000 and a.latency_ms.p95 == 3000
    assert (a.tokens_in, a.tokens_out) == (41, 6)
    assert a.cost_usd_estimate == 0.123457


def test_faithfulness_skips_answers_without_claims():
    rows = [row(verdicts=["supported"]), row(verdicts=[])]
    assert metrics.aggregate_block(rows).faithfulness == 1.0


def test_empty_and_all_refused_do_not_divide_by_zero():
    a = metrics.aggregate_block([])
    assert a.n_questions == 0 and a.hallucination_rate == 0 and a.latency_ms.p95 == 0
    b = metrics.aggregate_block([row(status="refused", answerable=False)])
    assert b.hallucination_rate == 0 and b.faithfulness == 0
    assert b.correct_refusals.rate == 1.0 and b.false_refusals.rate == 0


def test_build_aggregates_splits_by_system_and_costs():
    rows = [row(system="plain"), row(system="self_healing", status="refused", answerable=False)]
    agg = metrics.build_aggregates(rows, {"plain": 1.0, "self_healing": 2.0})
    assert agg.plain.n_answered == 1 and agg.self_healing.n_refused == 1
    assert agg.plain.cost_usd_estimate == 1.0 and agg.self_healing.cost_usd_estimate == 2.0


def test_contract_example_aggregates_reproduced():
    """Our math must reproduce the hand-checked aggregates in contract_examples (cost aside)."""
    result = EvalResult.model_validate(json.loads(EXAMPLE.read_text(encoding="utf-8")))
    ours = metrics.build_aggregates(result.results)
    for system in ("plain", "self_healing"):
        want = getattr(result.aggregates, system).model_dump()
        got = getattr(ours, system).model_dump()
        want.pop("cost_usd_estimate")
        got.pop("cost_usd_estimate")
        assert got == want  # same rounding (4 places) as the example


# --- cost ---------------------------------------------------------------------------------


def test_call_cost_uses_model_price_and_default():
    pricing = {
        "models": {"m": {"input": 1.0, "output": 10.0}},
        "default": {"input": 2.0, "output": 20.0},
    }
    assert metrics.call_cost_usd(pricing, "m", 1_000_000, 100_000) == pytest.approx(2.0)
    assert metrics.call_cost_usd(pricing, "unknown", 1_000_000, 0) == pytest.approx(2.0)


def test_pricing_file_covers_default_models():
    pricing = metrics.load_pricing()
    for model in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"):
        assert pricing["models"][model]["input"] > 0
