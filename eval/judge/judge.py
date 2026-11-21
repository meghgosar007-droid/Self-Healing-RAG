"""Independent judge: re-extracts claims from an answer and labels each against the chunks.

* Uses the ``judge`` role of ``src/llm.py`` (a different model from the generator, enforced by the
  runner) and its own prompts in ``prompts.py`` (never the agent's).
* Never looks at the agent's own claims or critic verdict, so it cannot inherit the critic's
  mistakes.
* Every LLM answer is cached on disk, keyed by prompt, model and input.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from eval import transient
from eval.judge import prompts
from eval.judge.cache import DiskCache, NullCache, cache_key
from src import llm
from src.schemas import Chunk, ClaimVerdict, JudgedClaim

VERDICTS: tuple[str, ...] = ("supported", "unsupported", "contradicted")
_REMINDER = "\n\nReply with a single valid JSON object only."


class JudgeError(RuntimeError):
    """The judge could not produce a usable answer (e.g. non-JSON output twice)."""


@dataclass(frozen=True)
class LLMReply:
    text: str
    model: str
    tokens_in: int
    tokens_out: int


LLMCall = Callable[[str, str], LLMReply]


def default_llm_call(system: str, user: str) -> LLMReply:
    """One judge call through ``src.llm`` (concurrency cap, retry, fallback, token usage)."""
    with llm.track() as usage:
        response = llm.get_llm("judge").invoke([SystemMessage(system), HumanMessage(user)])
    content = response.content
    if isinstance(content, list):  # Gemini may return content parts
        content = "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    model = usage.calls[-1].model if usage.calls else llm.active_model("judge")
    return LLMReply(str(content).strip(), model, usage.tokens_in, usage.tokens_out)


@dataclass(frozen=True)
class JudgeCall:
    """One judge LLM call (live or served from cache), for cost accounting."""

    model: str
    tokens_in: int
    tokens_out: int
    cached: bool


@dataclass(frozen=True)
class JudgeResult:
    claims: list[JudgedClaim]
    calls: list[JudgeCall]


def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in judge output: {text[:200]!r}") from None
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge output JSON is not an object")
    return value


def format_context(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(
        f"[{c.id}] ({c.paper}, section {c.section}, page {c.page})\n{c.text}" for c in chunks
    )


class Judge:
    def __init__(
        self,
        llm_call: LLMCall = default_llm_call,
        cache: DiskCache | NullCache | None = None,
        model_name: Callable[[], str] = lambda: llm.active_model("judge"),
    ) -> None:
        self._call = llm_call
        self.cache = cache if cache is not None else NullCache()
        self._model_name = model_name
        self._calls: list[JudgeCall] = []

    # --- public ---------------------------------------------------------------------------

    def judge(self, question: str, answer: str, chunks: Sequence[Chunk]) -> JudgeResult:
        """Label every claim in ``answer`` against ``chunks`` (the retrieved set)."""
        self._calls = []
        known = {c.id for c in chunks}
        extracted = self._extract(question, answer, known)
        if not extracted:
            return JudgeResult([], self._calls)
        labels = self._verify(extracted, chunks)
        claims = [
            JudgedClaim(text=text, chunk_ids=ids, verdict=labels.get(i, "unsupported"))
            for i, (text, ids) in enumerate(extracted)
        ]
        return JudgeResult(claims, self._calls)

    # --- stages ---------------------------------------------------------------------------

    def _extract(self, question: str, answer: str, known: set[str]) -> list[tuple[str, list[str]]]:
        user = f"Question: {question}\n\nAnswer:\n{answer}"
        data = self._ask("extract", prompts.EXTRACT, user)
        out: list[tuple[str, list[str]]] = []
        for item in data.get("claims") or []:
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get("text", "")).split())
            if not text:
                continue
            ids = [str(i).strip("[] ") for i in item.get("chunk_ids") or []]
            out.append((text, [i for i in dict.fromkeys(ids) if i in known]))
        return out

    def _verify(
        self, extracted: list[tuple[str, list[str]]], chunks: Sequence[Chunk]
    ) -> dict[int, ClaimVerdict]:
        listing = "\n".join(f"{i}. {text}" for i, (text, _) in enumerate(extracted))
        user = f"Passages:\n\n{format_context(chunks)}\n\nClaims:\n{listing}"
        data = self._ask("verify", prompts.VERIFY, user)
        labels: dict[int, ClaimVerdict] = {}
        for v in data.get("verdicts") or []:
            if not isinstance(v, dict) or not isinstance(v.get("index"), int):
                continue
            label = str(v.get("verdict", "")).strip().lower()
            if label in VERDICTS and 0 <= v["index"] < len(extracted):
                labels.setdefault(v["index"], label)  # type: ignore[arg-type]
        return labels  # a claim without a valid label is "unsupported" (strict default)

    # --- cached call ----------------------------------------------------------------------

    def _key(self, stage: str, system: str, user: str, model: str) -> str:
        return cache_key(
            stage=stage,
            judge_prompt_version=prompts.JUDGE_PROMPT_VERSION,
            system=system,
            user=user,
            model=model,
        )

    def _ask(self, stage: str, system: str, user: str) -> dict[str, Any]:
        requested = self._model_name()
        key = self._key(stage, system, user, requested)
        entry = self.cache.get(key)
        if entry is not None:
            self._calls.append(
                JudgeCall(entry["model"], entry["tokens_in"], entry["tokens_out"], cached=True)
            )
            return entry["data"]

        tokens_in = tokens_out = 0
        data: dict[str, Any] | None = None
        model = requested
        for attempt in (user, user + _REMINDER):
            reply = transient.with_retry(
                lambda text=attempt: self._call(system, text), what=f"judge {stage}"
            )
            model = reply.model
            tokens_in += reply.tokens_in
            tokens_out += reply.tokens_out
            try:
                data = parse_json(reply.text)
                break
            except ValueError:
                continue
        self._calls.append(JudgeCall(model, tokens_in, tokens_out, cached=False))
        if data is None:
            raise JudgeError(f"judge returned invalid JSON twice for stage {stage!r}")
        if model != requested:  # a fallback answered: file it under the model that produced it
            key = self._key(stage, system, user, model)
        self.cache.put(
            key, {"data": data, "model": model, "tokens_in": tokens_in, "tokens_out": tokens_out}
        )
        return data
