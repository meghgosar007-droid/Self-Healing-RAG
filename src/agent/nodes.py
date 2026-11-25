"""Graph nodes. Each takes the graph state and returns a partial update."""

from __future__ import annotations

from typing import Any

from src.agent import llm_utils, retrieval
from src.agent.prompts import (
    CLAIM_EXTRACTOR,
    CLAIM_VERIFIER,
    GENERATOR,
    PLANNER,
    REFUSAL_TEXT,
    REWRITER,
)
from src.schemas import MAX_RETRIES, Chunk, Claim, CriticVerdict

K = retrieval.K_DEFAULT
RETRY_K_STEP = 3  # each heal loop retrieves this many more chunks per query
_STOP = frozenset("a an and are does do how in is of the to what which why".split())


def _clean_queries(raw: Any, question: str) -> list[str]:
    """Normalize to 2-3 distinct non-empty queries; pad from the question if the model gave <2."""
    items = raw if isinstance(raw, list) else []
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        q = " ".join(str(item).split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    out = out[:3]
    if len(out) < 2:  # degenerate model output: add the question and a keyword form of it
        keywords = " ".join(w for w in question.rstrip("?").split() if w.lower() not in _STOP)
        for cand in (question, keywords):
            if cand and cand.lower() not in seen and len(out) < 2:
                seen.add(cand.lower())
                out.append(cand)
    return out


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{c.id}] ({c.paper}, section {c.section}, page {c.page})\n{c.text}" for c in chunks
    )


def planner(state: dict) -> dict:
    """Scope check and query plan in one call. An out-of-scope request gets no sub-queries,
    which routes straight to refuse. Anything unclear fails open (in scope)."""
    data = llm_utils.call_json("generator", PLANNER, f"Question: {state['question']}")
    data = data or {}  # unparseable plan: fail open, and queries are built from the question
    if data.get("in_scope") is False:
        return {"sub_queries": []}
    return {"sub_queries": _clean_queries(data.get("sub_queries"), state["question"])}


def retriever(payload: dict) -> dict:
    """One per sub-query (fan-out via Send). `idx` keeps merge order deterministic."""
    k = payload.get("k", K)
    return {"retrieved": [(payload["idx"], retrieval.retrieve(payload["query"], k))]}


def merge(state: dict) -> dict:
    """Fan-in: dedupe by chunk id (first occurrence wins), replace chunks, reset the buffer."""
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for _, found in sorted(state["retrieved"], key=lambda t: t[0]):
        for c in found:
            if c.id not in seen:
                seen.add(c.id)
                chunks.append(c)
    return {"chunks": chunks, "retrieved": None}


def generate_answer(question: str, chunks: list[Chunk]) -> str:
    """Shared by the self-healing graph and the plain baseline (fair comparison)."""
    user = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    return llm_utils.call("generator", GENERATOR, user)


def generator(state: dict) -> dict:
    return {"answer": generate_answer(state["question"], state["chunks"])}


def extract_claims(answer: str, chunks: list[Chunk]) -> list[Claim] | None:
    """None means the model output could not be parsed (distinct from no claims)."""
    known = {c.id for c in chunks}
    data = llm_utils.call_json("critic", CLAIM_EXTRACTOR, f"Answer:\n{answer}")
    if data is None:
        return None
    raw = data.get("claims")
    claims: list[Claim] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        raw_ids = item.get("chunk_ids")
        ids = [str(i).strip("[] ") for i in (raw_ids if isinstance(raw_ids, list) else [])]
        claims.append(Claim(text=text, chunk_ids=[i for i in dict.fromkeys(ids) if i in known]))
    return claims


def verify_claims(claims: list[Claim], chunks: list[Chunk]) -> CriticVerdict:
    if not claims:
        return CriticVerdict(
            verdict="fail", unsupported_claims=[], reason="The answer contains no checkable claims."
        )
    listing = "\n".join(f"{i}. {c.text}" for i, c in enumerate(claims))
    user = f"Context passages:\n\n{format_context(chunks)}\n\nClaims:\n{listing}"
    data = llm_utils.call_json("critic", CLAIM_VERIFIER, user)
    if data is None:  # unparseable verification: nothing is confirmed, so fail closed
        return CriticVerdict(
            verdict="fail",
            unsupported_claims=[c.text for c in claims],
            reason="The critic's verification output could not be parsed.",
        )
    labels: dict[int, str] = {}
    raw = data.get("verdicts")
    for v in raw if isinstance(raw, list) else []:
        if isinstance(v, dict) and isinstance(v.get("index"), int):
            labels[v["index"]] = str(v.get("verdict", "")).lower()
    # A claim the critic did not explicitly mark supported counts against the answer.
    bad = [c.text for i, c in enumerate(claims) if labels.get(i) != "supported"]
    reason = str(data.get("reason") or "").strip()
    if bad:
        return CriticVerdict(
            verdict="fail",
            unsupported_claims=bad,
            reason=reason or f"{len(bad)} claim(s) unsupported.",
        )
    return CriticVerdict(
        verdict="pass", unsupported_claims=[], reason=reason or "All claims are supported."
    )


def critic(state: dict) -> dict:
    claims = extract_claims(state["answer"], state["chunks"])
    if claims is None:  # unparseable extraction: cannot verify, so fail closed
        verdict = CriticVerdict(
            verdict="fail",
            unsupported_claims=[],
            reason="The critic's claim extraction output could not be parsed.",
        )
        return {"claims": [], "critic_verdict": verdict}
    verdict = verify_claims(claims, state["chunks"])
    update: dict[str, Any] = {"claims": claims, "critic_verdict": verdict}
    if verdict.verdict == "pass":
        update["final_status"] = "answered"
    return update


def rewriter(state: dict) -> dict:
    v = state["critic_verdict"]
    user = (
        f"Question: {state['question']}\n"
        f"Previous queries: {state['sub_queries']}\n"
        f"Previous answer: {state['answer']}\n"
        f"Unsupported claims: {v.unsupported_claims if v else []}\n"
        f"Critic reason: {v.reason if v else ''}"
    )
    data = llm_utils.call_json("generator", REWRITER, user) or {}  # fallback: from the question
    return {
        "sub_queries": _clean_queries(data.get("sub_queries"), state["question"]),
        "retry_count": state["retry_count"] + 1,
    }


def refuse(state: dict) -> dict:
    return {"answer": REFUSAL_TEXT, "claims": [], "final_status": "refused"}


# --- routing ------------------------------------------------------------------------------


def route_after_merge(state: dict) -> str:
    return "generator" if state["chunks"] else "refuse"


def route_after_critic(state: dict) -> str:
    if state["critic_verdict"].verdict == "pass":
        return "__end__"
    return "rewriter" if state["retry_count"] < MAX_RETRIES else "refuse"
