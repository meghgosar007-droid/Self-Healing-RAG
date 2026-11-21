"""Offline stand-ins for the agent and the judge LLM.

Used by ``python -m eval.runner --fake`` and by tests/eval. They follow the public interfaces
only: an agent is ``Callable[[str], State]``; a judge LLM is ``(system, user) -> LLMReply``.

The fake agents answer answerable questions from the contract fixtures; on unanswerable ones the
fake ``self_healing`` refuses while the fake ``plain`` makes something up, so a fake run exercises
every outcome.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from eval.judge import LLMReply
from src.schemas import Chunk, State, new_state

FIXTURES = Path(__file__).resolve().parents[1] / "contract_examples" / "chunks.jsonl"
FAKE_MODELS = {
    "generator": "fake-generator",
    "critic": "fake-critic",
    "judge": "fake-judge",
}
REFUSAL = "I can't answer this reliably from the papers."


def fixture_chunks() -> list[Chunk]:
    lines = FIXTURES.read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(line) for line in lines if line.strip()]


def fake_agents(answerable: Mapping[str, bool]) -> dict[str, Callable[[str], State]]:
    chunks = fixture_chunks()

    def pick(question: str) -> Chunk:
        return chunks[sum(map(ord, question)) % len(chunks)]

    def refuse(question: str) -> State:
        s = new_state(question)
        s.update(answer=REFUSAL, final_status="refused")
        return s

    def answer_from(question: str, retries: int) -> State:
        c = pick(question)
        s = new_state(question)
        s.update(
            chunks=[c],
            answer=f"{c.text} [{c.id}]",
            final_status="answered",
            retry_count=retries,
        )
        return s

    def plain(question: str) -> State:
        if answerable.get(question, True):
            return answer_from(question, 0)
        s = new_state(question)
        c = pick(question)
        s.update(
            chunks=[c],
            answer=f"The reported figure was 42 percent [{c.id}].",
            final_status="answered",
        )
        return s

    def self_healing(question: str) -> State:
        if not answerable.get(question, True):
            return refuse(question)
        retries = len(question) % 3  # 0..2 = MAX_RETRIES
        if retries == 2 and len(question) % 5 == 0:
            return refuse(question)  # an occasional false refusal
        return answer_from(question, retries)

    return {"plain": plain, "self_healing": self_healing}


# --- fake judge LLM -----------------------------------------------------------------------

_CITE = re.compile(r"\[([^\[\]]+)\]")
_WORD = re.compile(r"[a-z0-9]{4,}")


class FakeJudgeLLM:
    """Rule-based stand-in for the judge model: splits sentences, checks word overlap."""

    def __init__(self, model: str = FAKE_MODELS["judge"]) -> None:
        self.model = model
        self.calls = 0

    def __call__(self, system: str, user: str) -> LLMReply:
        self.calls += 1
        if "TASK: JUDGE_EXTRACT" in system:
            data = self._extract(user)
        elif "TASK: JUDGE_VERIFY" in system:
            data = self._verify(user)
        else:
            raise ValueError("unknown judge task")
        return LLMReply(json.dumps(data), self.model, len(user) // 4, 40)

    @staticmethod
    def _extract(user: str) -> dict:
        answer = user.split("Answer:\n", 1)[1]
        claims = []
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
            ids = _CITE.findall(sentence)
            text = _CITE.sub("", sentence).strip()
            if text:
                claims.append({"text": text, "chunk_ids": ids})
        return {"claims": claims}

    @staticmethod
    def _verify(user: str) -> dict:
        passages, claims = user.split("\n\nClaims:\n", 1)
        words = set(_WORD.findall(passages.lower()))
        verdicts = []
        for line in claims.splitlines():
            idx, _, text = line.partition(". ")
            claim_words = set(_WORD.findall(text.lower()))
            overlap = len(claim_words & words) / max(1, len(claim_words))
            verdicts.append(
                {"index": int(idx), "verdict": "supported" if overlap >= 0.8 else "unsupported"}
            )
        return {"verdicts": verdicts}
