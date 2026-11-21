"""Eval runner (CONTRACT.md section 3).

    python -m eval.runner --mode smoke     # smoke:true rows only
    python -m eval.runner --mode full      # every row
    python -m eval.runner --mode smoke --fake   # fully offline, fake agent + fake judge

Every selected question runs through both systems (``plain`` = ``run_plain``, ``self_healing`` =
``run``); the judge scores both with the same prompts. The result is validated against
``src.schemas.EvalResult`` before anything is written to ``eval/results/<run_id>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from eval import dataset, metrics, transient
from eval.judge import JUDGE_PROMPT_VERSION, DiskCache, Judge, NullCache
from src import llm
from src.schemas import CONTRACT_VERSION, EvalResult, Meta, ResultRow, State, System

log = logging.getLogger("eval")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_CACHE_DIR = RESULTS_DIR / ".judge_cache"  # eval/results/ is gitignored
MANIFEST_PATH = ROOT / "data" / "manifest.json"
SYSTEMS: tuple[System, ...] = ("plain", "self_healing")

AgentFn = Callable[[str], State]
Progress = Callable[[str], None]


class EvalError(RuntimeError):
    """The run cannot produce a trustworthy result; nothing is written."""


class ModelCollisionError(EvalError):
    """The judge resolved to the same model as the generator or the critic."""


# --- model separation (CONTRACT.md section 6) ---------------------------------------------


@dataclass
class ModelsSeen:
    """Models actually used per role, after any fallback."""

    by_role: dict[str, set[str]] = field(
        default_factory=lambda: {"generator": set(), "critic": set(), "judge": set()}
    )
    last: dict[str, str] = field(default_factory=dict)

    def add(self, role: str, model: str) -> None:
        self.by_role[role].add(model)
        self.last[role] = model

    def assert_judge_independent(self) -> None:
        """Fail the run when the judge shares a model with the generator or the critic.

        The contract requires judge != generator; the critic is checked too because a judge that
        shares the critic's model would grade the critic's own work.
        """
        judge = self.by_role["judge"]
        for role in ("generator", "critic"):
            clash = judge & self.by_role[role]
            if clash:
                raise ModelCollisionError(
                    f"judge model {sorted(clash)} is also used as the {role} model; "
                    "set JUDGE_MODEL / JUDGE_MODEL_FALLBACK to a different model"
                )

    def final(self, role: str, configured: Mapping[str, str]) -> str:
        return self.last.get(role) or configured[role]


# --- one question, one system -------------------------------------------------------------


@dataclass
class Scored:
    row: ResultRow
    cost_usd: float


def run_one(
    q: dataset.Question,
    system: System,
    agent: AgentFn,
    judge: Judge,
    pricing: dict[str, Any],
    seen: ModelsSeen,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> Scored:
    def attempt() -> tuple[State, int, Any]:
        with llm.track() as usage:  # fresh scope per attempt: a failed attempt is not billed
            start = clock()
            state = agent(q.question)
            return state, max(0, round((clock() - start) * 1000)), usage

    try:
        state, latency_ms, usage = transient.with_retry(
            attempt, sleep=sleep, what=f"{system} agent on {q.id}"
        )
    except Exception as exc:
        raise EvalError(f"{system} agent failed on {q.id}: {exc!r}") from exc

    status = state["final_status"]
    if status not in ("answered", "refused"):
        raise EvalError(f"{system} agent returned a non-terminal state on {q.id}: {status!r}")
    for role, model in usage.models.items():
        seen.add(role, model)
    cost = sum(
        metrics.call_cost_usd(pricing, c.model, c.tokens_in, c.tokens_out) for c in usage.calls
    )

    claims = []
    answer = state["answer"]
    if status == "answered":
        result = judge.judge(q.question, answer or "", state["chunks"])
        claims = result.claims
        for call in result.calls:
            seen.add("judge", call.model)
            cost += metrics.call_cost_usd(pricing, call.model, call.tokens_in, call.tokens_out)
    seen.assert_judge_independent()

    row = ResultRow(
        id=q.id,
        system=system,
        question=q.question,
        answerable=q.answerable,
        final_status=status,
        answer=answer if status == "answered" else None,
        retry_count=state["retry_count"],
        retrieved_chunk_ids=[c.id for c in state["chunks"]],
        claims=claims,
        outcome=metrics.classify_outcome(status, q.answerable, claims),
        latency_ms=latency_ms,
        llm_wait_ms=min(usage.wait_ms, latency_ms),
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
    )
    return Scored(row, cost)


# --- the whole run ------------------------------------------------------------------------


def run_eval(
    questions: list[dataset.Question],
    mode: Literal["smoke", "full"],
    *,
    agents: Mapping[str, AgentFn],
    judge: Judge,
    prompt_version: str,
    corpus_version: str,
    configured_models: Mapping[str, str],
    git_sha: str,
    pricing: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    progress: Progress = lambda _msg: None,
) -> EvalResult:
    missing = [s for s in SYSTEMS if s not in agents]
    if missing:
        raise EvalError(f"missing agent(s) for system(s): {missing}")
    pricing = pricing or metrics.load_pricing()
    seen = ModelsSeen()
    # Preflight on the configured models, so a bad config fails before any tokens are spent.
    for role in ("generator", "critic", "judge"):
        seen.add(role, configured_models[role])
    seen.assert_judge_independent()
    seen.by_role = {k: set() for k in seen.by_role}  # configured != used; track real usage below

    by_system: dict[str, list[ResultRow]] = {s: [] for s in SYSTEMS}
    costs = dict.fromkeys(SYSTEMS, 0.0)
    for i, q in enumerate(questions, 1):
        for system in SYSTEMS:
            scored = run_one(q, system, agents[system], judge, pricing, seen, clock)
            by_system[system].append(scored.row)
            costs[system] += scored.cost_usd
            progress(f"[{i}/{len(questions)}] {q.id} {system}: {scored.row.outcome}")

    rows = [r for s in SYSTEMS for r in by_system[s]]
    meta = Meta(
        contract_version=CONTRACT_VERSION,
        eval_mode=mode,
        prompt_version=prompt_version,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        corpus_version=corpus_version,
        generator_model=seen.final("generator", configured_models),
        critic_model=seen.final("critic", configured_models),
        judge_model=seen.final("judge", configured_models),
        git_sha=git_sha,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return EvalResult(meta=meta, results=rows, aggregates=metrics.build_aggregates(rows, costs))


def write_result(result: EvalResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# --- environment helpers ------------------------------------------------------------------


def read_corpus_version(path: Path = MANIFEST_PATH) -> str:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["corpus_version"]
    except (OSError, KeyError, ValueError) as exc:
        raise EvalError(f"cannot read corpus_version from {path}: {exc}") from exc


def read_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.getenv("GITHUB_SHA", "unknown")


def real_agents() -> dict[str, AgentFn]:
    from src.agent import graph  # imported lazily: pulls in langgraph and the retriever

    return {"plain": graph.run_plain, "self_healing": graph.run}


def real_prompt_version() -> str:
    from src.agent.prompts import PROMPT_VERSION

    return PROMPT_VERSION


def configured_models() -> dict[str, str]:
    return {role: llm.active_model(role) for role in ("generator", "critic", "judge")}


def summarize(result: EvalResult) -> str:
    lines = [
        f"{'system':<13}{'halluc':>8}{'faith':>8}{'false_ref':>11}{'corr_ref':>10}{'p50':>8}{'p95':>8}{'usd':>9}"
    ]
    for name in SYSTEMS:
        a = getattr(result.aggregates, name)
        lines.append(
            f"{name:<13}{a.hallucination_rate:>8.3f}{a.faithfulness:>8.3f}"
            f"{a.false_refusals.rate:>11.3f}{a.correct_refusals.rate:>10.3f}"
            f"{a.latency_ms.p50:>8.0f}{a.latency_ms.p95:>8.0f}{a.cost_usd_estimate:>9.4f}"
        )
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval.runner", description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--fake", action="store_true", help="offline: fake agent and fake judge")
    p.add_argument("--dataset", type=Path, default=dataset.DATASET_PATH)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.getenv("EVAL_JUDGE_CACHE_DIR", DEFAULT_CACHE_DIR)),
        help="judge output cache (default eval/results/.judge_cache, or $EVAL_JUDGE_CACHE_DIR)",
    )
    p.add_argument("--no-cache", action="store_true", help="ignore and do not write the cache")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cache = NullCache() if args.no_cache else DiskCache(args.cache_dir)

    questions = dataset.load_questions(args.dataset)
    dataset.validate_smoke_subset(questions)
    selected = dataset.select(questions, args.mode)

    if args.fake:
        from eval import fakes

        answerable = {q.question: q.answerable for q in questions}
        agents: dict[str, AgentFn] = fakes.fake_agents(answerable)
        judge = Judge(fakes.FakeJudgeLLM(), cache, model_name=lambda: fakes.FAKE_MODELS["judge"])
        models = dict(fakes.FAKE_MODELS)
        prompt_version = real_prompt_version()
    else:
        for role in ("generator", "critic", "judge"):
            llm.get_llm(role)  # fail fast if GEMINI_API_KEY is missing
        agents = real_agents()
        judge = Judge(cache=cache)
        models = configured_models()
        prompt_version = real_prompt_version()

    sha = read_git_sha()
    result = run_eval(
        selected,
        args.mode,
        agents=agents,
        judge=judge,
        prompt_version=prompt_version,
        corpus_version=read_corpus_version(),
        configured_models=models,
        git_sha=sha,
        progress=lambda m: print(m, file=sys.stderr),
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{'fake-' if args.fake else ''}{args.mode}-{stamp}-{sha[:7]}"
    out = args.out_dir / f"{run_id}.json"
    write_result(result, out)
    print(summarize(result))
    print(f"judge cache: {cache.hits} hits, {cache.misses} misses")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
