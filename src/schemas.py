"""Pydantic models and the LangGraph state, mirroring CONTRACT.md.

Change this file only through a contract-change PR (CONTRACT.md section 7).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = 3
MAX_RETRIES = 2
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

Paper = Literal["deepseek-r1", "deepseek-math", "dr-grpo"]
FinalStatus = Literal["answered", "refused"]
System = Literal["plain", "self_healing"]
EvalMode = Literal["smoke", "full"]
ClaimVerdict = Literal["supported", "unsupported", "contradicted"]
Outcome = Literal["answered_ok", "hallucinated", "correct_refusal", "false_refusal"]

_ID_RE = re.compile(
    r"^(?P<paper>deepseek-r1|deepseek-math|dr-grpo)"
    r":(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)"
    r":p(?P<page>[1-9]\d*):(?P<n>\d+)$"
)


class Strict(BaseModel):
    """Base model: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")


# --- Section 1: chunk ---------------------------------------------------------------------


class Chunk(Strict):
    id: str
    paper: Paper
    section: str = Field(min_length=1)
    page: int = Field(ge=1)
    text: str
    embedding_model: Literal["BAAI/bge-small-en-v1.5"]

    @field_validator("text")
    @classmethod
    def _text_normalized(cls, v: str) -> str:
        if not v or v != " ".join(v.split()):
            raise ValueError("text must be non-empty and whitespace-normalized")
        return v

    @model_validator(mode="after")
    def _id_matches_fields(self) -> Chunk:
        m = _ID_RE.match(self.id)
        if m is None:
            raise ValueError("id must look like {paper}:{section_slug}:p{page}:{n}")
        if m["paper"] != self.paper or int(m["page"]) != self.page:
            raise ValueError("id paper/page must match the paper and page fields")
        return self


# --- Section 2: agent state ---------------------------------------------------------------


class Claim(Strict):
    text: str = Field(min_length=1)
    chunk_ids: list[str]


class CriticVerdict(Strict):
    verdict: Literal["pass", "fail"]
    unsupported_claims: list[str]
    reason: str


class State(TypedDict):
    question: str
    sub_queries: list[str]
    chunks: list[Chunk]
    answer: str | None
    claims: list[Claim]
    critic_verdict: CriticVerdict | None
    retry_count: int
    final_status: FinalStatus | None


def new_state(question: str) -> State:
    """Initial state for a question."""
    return State(
        question=question,
        sub_queries=[],
        chunks=[],
        answer=None,
        claims=[],
        critic_verdict=None,
        retry_count=0,
        final_status=None,
    )


# --- Section 3: eval result ---------------------------------------------------------------


class Meta(Strict):
    contract_version: int
    eval_mode: EvalMode
    prompt_version: str
    judge_prompt_version: str
    corpus_version: str = Field(pattern=r"^[0-9a-f]{12}$")
    generator_model: str
    critic_model: str
    judge_model: str
    git_sha: str
    timestamp: str

    @field_validator("contract_version")
    @classmethod
    def _version_current(cls, v: int) -> int:
        if v != CONTRACT_VERSION:
            raise ValueError(f"contract_version must be {CONTRACT_VERSION}, got {v}")
        return v

    @field_validator("timestamp")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class JudgedClaim(Strict):
    text: str = Field(min_length=1)
    chunk_ids: list[str]
    verdict: ClaimVerdict


class ResultRow(Strict):
    id: str
    system: System
    question: str
    answerable: bool
    final_status: FinalStatus
    answer: str | None
    retry_count: int = Field(ge=0, le=MAX_RETRIES)
    retrieved_chunk_ids: list[str]
    claims: list[JudgedClaim]
    outcome: Outcome
    latency_ms: int = Field(ge=0)
    llm_wait_ms: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)

    @model_validator(mode="after")
    def _outcome_consistent(self) -> ResultRow:
        if self.llm_wait_ms > self.latency_ms:
            raise ValueError("llm_wait_ms cannot exceed latency_ms")
        if self.final_status == "refused":
            if self.claims:
                raise ValueError("refusals contain no claims")
            expected = {"false_refusal"} if self.answerable else {"correct_refusal"}
        elif not self.answerable:
            expected = {"hallucinated"}
        else:
            expected = {"answered_ok", "hallucinated"}
        if self.outcome not in expected:
            raise ValueError(f"outcome {self.outcome!r} inconsistent with status/answerable")
        return self


class Rate(Strict):
    count: int = Field(ge=0)
    rate: float = Field(ge=0, le=1)


class Latency(Strict):
    p50: float = Field(ge=0)
    p95: float = Field(ge=0)


class AggregateBlock(Strict):
    n_questions: int = Field(ge=0)
    n_answerable: int = Field(ge=0)
    n_unanswerable: int = Field(ge=0)
    n_answered: int = Field(ge=0)
    n_refused: int = Field(ge=0)
    hallucination_rate: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)
    correct_refusals: Rate
    false_refusals: Rate
    latency_ms: Latency
    cost_usd_estimate: float = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)


class Aggregates(Strict):
    plain: AggregateBlock
    self_healing: AggregateBlock


class EvalResult(Strict):
    meta: Meta
    results: list[ResultRow]
    aggregates: Aggregates
