"""Outcome classification, aggregates and cost, as defined in CONTRACT.md sections 3-4.

Pure functions only (no I/O besides loading the pricing file), so they are unit-tested on
hand-built rows.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.schemas import (
    AggregateBlock,
    Aggregates,
    FinalStatus,
    JudgedClaim,
    Latency,
    Outcome,
    Rate,
    ResultRow,
)

log = logging.getLogger(__name__)

PRICING_PATH = Path(__file__).with_name("pricing.json")
_warned: set[str] = set()
RATE_DIGITS = 4
COST_DIGITS = 6


def classify_outcome(
    final_status: FinalStatus, answerable: bool, claims: Sequence[JudgedClaim]
) -> Outcome:
    """Section 4: refusal -> correct/false; answered -> hallucinated if any bad claim or the
    question is unanswerable, else answered_ok."""
    if final_status == "refused":
        return "false_refusal" if answerable else "correct_refusal"
    if not answerable:
        return "hallucinated"
    bad = any(c.verdict in ("unsupported", "contradicted") for c in claims)
    return "hallucinated" if bad else "answered_ok"


def nearest_rank(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile; 0 for no data."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p * len(ordered))) - 1]


def _rate(numerator: int, denominator: int) -> float:
    """Ratio rounded to 4 places; 0.0 when the denominator is 0 (undefined)."""
    return round(numerator / denominator, RATE_DIGITS) if denominator else 0.0


def row_faithfulness(row: ResultRow) -> float | None:
    """supported / total claims, or None when the answer has no claims."""
    if not row.claims:
        return None
    return sum(c.verdict == "supported" for c in row.claims) / len(row.claims)


def aggregate_block(rows: Sequence[ResultRow], cost_usd: float = 0.0) -> AggregateBlock:
    """Aggregates over ONE system's rows."""
    answered = [r for r in rows if r.final_status == "answered"]
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    hallucinated = [r for r in answered if r.outcome == "hallucinated"]
    correct = [r for r in rows if r.outcome == "correct_refusal"]
    false = [r for r in rows if r.outcome == "false_refusal"]

    scores = [s for s in map(row_faithfulness, answered) if s is not None]
    faithfulness = round(sum(scores) / len(scores), RATE_DIGITS) if scores else 0.0

    adjusted = [r.latency_ms - r.llm_wait_ms for r in rows]
    return AggregateBlock(
        n_questions=len(rows),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        n_answered=len(answered),
        n_refused=len(rows) - len(answered),
        hallucination_rate=_rate(len(hallucinated), len(answered)),
        faithfulness=faithfulness,
        correct_refusals=Rate(count=len(correct), rate=_rate(len(correct), len(unanswerable))),
        false_refusals=Rate(count=len(false), rate=_rate(len(false), len(answerable))),
        latency_ms=Latency(p50=nearest_rank(adjusted, 0.50), p95=nearest_rank(adjusted, 0.95)),
        cost_usd_estimate=round(cost_usd, COST_DIGITS),
        tokens_in=sum(r.tokens_in for r in rows),
        tokens_out=sum(r.tokens_out for r in rows),
    )


def build_aggregates(
    rows: Iterable[ResultRow], cost_usd_by_system: dict[str, float] | None = None
) -> Aggregates:
    rows = list(rows)
    costs = cost_usd_by_system or {}
    return Aggregates(
        plain=aggregate_block([r for r in rows if r.system == "plain"], costs.get("plain", 0.0)),
        self_healing=aggregate_block(
            [r for r in rows if r.system == "self_healing"], costs.get("self_healing", 0.0)
        ),
    )


# --- cost ---------------------------------------------------------------------------------


def load_pricing(path: Path = PRICING_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def call_cost_usd(pricing: dict[str, Any], model: str, tokens_in: int, tokens_out: int) -> float:
    """Notional USD for one call at list price (per 1M tokens)."""
    prices = pricing["models"].get(model)
    if prices is None:
        if model not in _warned:
            _warned.add(model)
            log.warning("no price for model %r in pricing.json; using the default price", model)
        prices = pricing["default"]
    return (tokens_in * prices["input"] + tokens_out * prices["output"]) / 1_000_000
