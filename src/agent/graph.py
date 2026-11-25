"""Self-healing graph and plain baseline (CONTRACT.md section 2).

planner -> (out of scope: refuse) retriever x N (fan-out) -> merge (fan-in) -> generator -> critic
critic: pass -> answered | fail and retry_count < MAX_RETRIES -> rewriter -> retrievers
        | otherwise -> refuse
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agent import nodes, retrieval
from src.agent.prompts import REFUSAL_TEXT
from src.schemas import Chunk, Claim, CriticVerdict, State, new_state


def _collect(old: list | None, new: list | None) -> list:
    """Reducer for parallel retriever output; `None` (sent by merge) clears the buffer."""
    return [] if new is None else (old or []) + new


class GraphState(TypedDict):
    """Contract State plus one internal fan-in buffer, stripped before returning."""

    question: str
    sub_queries: list[str]
    chunks: list[Chunk]
    answer: str | None
    claims: list[Claim]
    critic_verdict: CriticVerdict | None
    retry_count: int
    final_status: str | None
    retrieved: Annotated[list[tuple[int, list[Chunk]]], _collect]


def _fan_out(state: dict) -> list[Send] | str:
    """One retriever per sub-query; no sub-queries (out of scope) means refuse.

    Each retry widens retrieval: k grows by RETRY_K_STEP per heal loop.
    """
    if not state["sub_queries"]:
        return "refuse"
    k = nodes.K + nodes.RETRY_K_STEP * state["retry_count"]
    return [
        Send("retriever", {"idx": i, "query": q, "k": k})
        for i, q in enumerate(state["sub_queries"])
    ]


@lru_cache(maxsize=1)
def build_graph() -> Any:
    g = StateGraph(GraphState)
    g.add_node("planner", nodes.planner)
    g.add_node("retriever", nodes.retriever)
    g.add_node("merge", nodes.merge)
    g.add_node("generator", nodes.generator)
    g.add_node("critic", nodes.critic)
    g.add_node("rewriter", nodes.rewriter)
    g.add_node("refuse", nodes.refuse)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", _fan_out, ["retriever", "refuse"])
    g.add_edge("retriever", "merge")
    g.add_conditional_edges("merge", nodes.route_after_merge, ["generator", "refuse"])
    g.add_edge("generator", "critic")
    g.add_conditional_edges(
        "critic",
        nodes.route_after_critic,
        {"__end__": END, "rewriter": "rewriter", "refuse": "refuse"},
    )
    g.add_conditional_edges("rewriter", _fan_out, ["retriever"])
    g.add_edge("refuse", END)
    return g.compile()


def _to_state(raw: dict) -> State:
    return State(**{k: raw[k] for k in new_state("")})


def run(question: str) -> State:
    """Self-healing flow; returns the terminal state."""
    initial = {**new_state(question), "retrieved": []}
    return _to_state(build_graph().invoke(initial))


def run_plain(question: str) -> State:
    """Baseline: retrieve(question, k) -> generate. Same retriever, prompt and model as `run`.

    No planner, critic or retry; refuses only when retrieval returns nothing.
    `claims` stay empty (the judge extracts them) and `retry_count` stays 0.
    """
    state = new_state(question)
    state["chunks"] = retrieval.retrieve(question, nodes.K)
    if not state["chunks"]:
        state.update(answer=REFUSAL_TEXT, final_status="refused")
        return state
    state["answer"] = nodes.generate_answer(question, state["chunks"])
    state["final_status"] = "answered"
    return state
