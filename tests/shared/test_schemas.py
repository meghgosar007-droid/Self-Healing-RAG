import json
import math
import re
from pathlib import Path
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from src import schemas
from src.schemas import Chunk, CriticVerdict, EvalResult, State, new_state

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "contract_examples"


def load(name: str):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


# --- fixtures validate --------------------------------------------------------------------


def test_chunk_example_validates():
    Chunk.model_validate(load("chunk.json"))


def test_chunks_jsonl_validates_and_covers_all_papers():
    lines = (EXAMPLES / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
    assert {c.paper for c in chunks} == {"deepseek-r1", "deepseek-math", "dr-grpo"}
    assert len({c.id for c in chunks}) == len(chunks)


def test_state_example_validates():
    data = load("state.json")
    assert set(data) == set(get_type_hints(State))
    [Chunk.model_validate(c) for c in data["chunks"]]
    CriticVerdict.model_validate(data["critic_verdict"])


def test_eval_result_example_validates():
    EvalResult.model_validate(load("eval_result.json"))


def test_contract_version_matches_contract_md():
    first_line = (ROOT / "CONTRACT.md").read_text(encoding="utf-8").splitlines()[0]
    m = re.search(r"contract_version:\s*(\d+)", first_line)
    assert m and int(m.group(1)) == schemas.CONTRACT_VERSION


def test_constants():
    assert schemas.MAX_RETRIES == 2
    assert schemas.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"


def test_state_keys_exact():
    assert set(get_type_hints(State)) == {
        "question",
        "sub_queries",
        "chunks",
        "answer",
        "claims",
        "critic_verdict",
        "retry_count",
        "final_status",
    }


def test_new_state_defaults():
    s = new_state("q?")
    assert s["question"] == "q?" and s["retry_count"] == 0 and s["final_status"] is None
    assert s["answer"] is None and s["chunks"] == [] and s["claims"] == []


# --- chunk rules --------------------------------------------------------------------------


def chunk(**over):
    return {**load("chunk.json"), **over}


def test_chunk_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Chunk.model_validate(chunk(extra="x"))


@pytest.mark.parametrize(
    "over",
    [
        {"paper": "other-paper"},
        {"embedding_model": "text-embedding-004"},
        {"page": 0},
        {"text": ""},
        {"text": "double  space"},
        {"text": " leading"},
        {"section": ""},
        {"id": "deepseek-math:4-1-grpo:p12"},  # missing n
        {"id": "deepseek-math:4_1 GRPO:p12:0"},  # bad slug
        {"id": "dr-grpo:4-1-grpo:p12:0"},  # paper mismatch
        {"id": "deepseek-math:4-1-grpo:p99:0"},  # page mismatch
    ],
)
def test_chunk_rejects_bad_values(over):
    with pytest.raises(ValidationError):
        Chunk.model_validate(chunk(**over))


def test_chunk_accepts_unknown_section():
    Chunk.model_validate(chunk(section="unknown", id="deepseek-math:unknown:p12:3"))


# --- eval result rules --------------------------------------------------------------------


def test_eval_result_rejects_wrong_contract_version():
    data = load("eval_result.json")
    data["meta"]["contract_version"] = 1
    with pytest.raises(ValidationError):
        EvalResult.model_validate(data)


def test_eval_result_requires_both_systems():
    data = load("eval_result.json")
    del data["aggregates"]["plain"]
    with pytest.raises(ValidationError):
        EvalResult.model_validate(data)


def test_eval_result_rejects_bad_claim_verdict():
    data = load("eval_result.json")
    data["results"][0]["claims"][0]["verdict"] = "maybe"
    with pytest.raises(ValidationError):
        EvalResult.model_validate(data)


@pytest.mark.parametrize(
    "idx,outcome",
    [
        (0, "correct_refusal"),  # answered answerable question
        (1, "correct_refusal"),  # refused answerable question must be false_refusal
        (2, "false_refusal"),  # refused unanswerable question must be correct_refusal
        (5, "answered_ok"),  # answering an unanswerable question is hallucination
    ],
)
def test_outcome_must_match_status_and_answerable(idx, outcome):
    data = load("eval_result.json")
    data["results"][idx]["outcome"] = outcome
    with pytest.raises(ValidationError):
        EvalResult.model_validate(data)


def test_refusals_cannot_have_claims():
    data = load("eval_result.json")
    data["results"][1]["claims"] = [
        {"text": "x", "chunk_ids": [], "verdict": "supported"},
    ]
    with pytest.raises(ValidationError):
        EvalResult.model_validate(data)


def test_wait_cannot_exceed_latency():
    data = load("eval_result.json")
    data["results"][0]["llm_wait_ms"] = data["results"][0]["latency_ms"] + 1
    with pytest.raises(ValidationError):
        EvalResult.model_validate(data)


def test_retry_count_capped_at_max_retries():
    data = load("eval_result.json")
    data["results"][0]["retry_count"] = schemas.MAX_RETRIES + 1
    with pytest.raises(ValidationError):
        EvalResult.model_validate(data)


# --- the example's aggregates equal the section 4 formulas ---------------------------------


def nearest_rank(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p * len(ordered))) - 1]


@pytest.mark.parametrize("system", ["plain", "self_healing"])
def test_example_aggregates_match_contract_formulas(system):
    result = EvalResult.model_validate(load("eval_result.json"))
    rows = [r for r in result.results if r.system == system]
    agg = getattr(result.aggregates, system)

    answered = [r for r in rows if r.final_status == "answered"]
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    halluc = [r for r in answered if r.outcome == "hallucinated"]
    correct_ref = [r for r in rows if r.outcome == "correct_refusal"]
    false_ref = [r for r in rows if r.outcome == "false_refusal"]

    assert agg.n_questions == len(rows)
    assert agg.n_answerable == len(answerable)
    assert agg.n_unanswerable == len(unanswerable)
    assert agg.n_answered == len(answered)
    assert agg.n_refused == len(rows) - len(answered)
    assert agg.hallucination_rate == pytest.approx(len(halluc) / len(answered), abs=1e-4)

    def faith(r):
        supported = sum(c.verdict == "supported" for c in r.claims)
        return supported / len(r.claims)

    assert agg.faithfulness == pytest.approx(sum(map(faith, answered)) / len(answered), abs=1e-4)
    assert agg.correct_refusals.count == len(correct_ref)
    assert agg.correct_refusals.rate == pytest.approx(len(correct_ref) / len(unanswerable))
    assert agg.false_refusals.count == len(false_ref)
    assert agg.false_refusals.rate == pytest.approx(len(false_ref) / len(answerable))

    adjusted = [r.latency_ms - r.llm_wait_ms for r in rows]
    assert agg.latency_ms.p50 == nearest_rank(adjusted, 0.50)
    assert agg.latency_ms.p95 == nearest_rank(adjusted, 0.95)
    assert agg.tokens_in == sum(r.tokens_in for r in rows)
    assert agg.tokens_out == sum(r.tokens_out for r in rows)
