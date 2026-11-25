"""Agent tests: fake LLM (keyed on the prompt's TASK line) and fake retriever; fully offline."""

import json
import re
from pathlib import Path

import pytest

from src import llm
from src.agent import graph, nodes, retrieval
from src.agent.prompts import PROMPT_VERSION, REFUSAL_TEXT
from src.schemas import MAX_RETRIES, Chunk

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = [
    Chunk.model_validate_json(line)
    for line in (ROOT / "contract_examples" / "chunks.jsonl").read_text().splitlines()
]
GRPO, R1, DRGRPO = FIXTURES
Q = "How does GRPO avoid needing a value model?"


class Resp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Scripts replies per task; the last reply for a task repeats. Records every call."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, str, str]] = []  # (role, task, user)

    def get_llm(self, role):
        outer = self

        class Client:
            def invoke(self, messages):
                task = re.search(r"TASK: (\w+)", messages[0].content).group(1)
                outer.calls.append((role, task, messages[1].content))
                replies = outer.script[task]
                reply = replies.pop(0) if len(replies) > 1 else replies[0]
                return Resp(reply if isinstance(reply, str) else json.dumps(reply))

        return Client()

    def tasks(self):
        return [t for _, t, _ in self.calls]


def planner(*qs):
    return {"sub_queries": list(qs)}


def claims(*items):
    return {"claims": [{"text": t, "chunk_ids": ids} for t, ids in items]}


def verdicts(*labels):
    return {"verdicts": [{"index": i, "verdict": v} for i, v in enumerate(labels)], "reason": "r"}


GOOD_ANSWER = f"GRPO uses group reward statistics as its baseline [{GRPO.id}]."
GOOD_CLAIMS = claims(("GRPO uses group reward statistics as the baseline.", [GRPO.id]))


@pytest.fixture
def fake(monkeypatch):
    def install(script, retrieve=None):
        f = FakeLLM(script)
        monkeypatch.setattr(llm, "get_llm", f.get_llm)
        monkeypatch.setattr(retrieval, "retrieve", retrieve or (lambda q, k=5: [GRPO]))
        return f

    return install


def happy_script(**over):
    s = {
        "PLANNER": [planner("GRPO baseline group reward", "GRPO value model PPO")],
        "GENERATOR": [GOOD_ANSWER],
        "CLAIM_EXTRACTOR": [GOOD_CLAIMS],
        "CLAIM_VERIFIER": [verdicts("supported")],
        "REWRITER": [planner("GRPO advantage estimation", "group relative baseline")],
    }
    s.update(over)
    return s


# --- self-healing flow --------------------------------------------------------------------


def test_happy_path_answers_without_retry(fake):
    f = fake(happy_script())
    s = graph.run(Q)
    assert s["final_status"] == "answered"
    assert s["retry_count"] == 0
    assert s["answer"] == GOOD_ANSWER
    assert [c.id for c in s["chunks"]] == [GRPO.id]
    assert s["critic_verdict"].verdict == "pass"
    assert s["claims"][0].chunk_ids == [GRPO.id]
    assert s["question"] == Q
    assert f.tasks() == ["PLANNER", "GENERATOR", "CLAIM_EXTRACTOR", "CLAIM_VERIFIER"]
    assert [r for r, t, _ in f.calls if t.startswith("CLAIM")] == ["critic", "critic"]


def test_state_has_exactly_contract_keys(fake):
    fake(happy_script())
    assert set(graph.run(Q)) == {
        "question", "sub_queries", "chunks", "answer", "claims",
        "critic_verdict", "retry_count", "final_status",
    }  # fmt: skip


def test_one_retriever_per_sub_query_and_merge_dedupes(fake):
    seen = []

    def retrieve(query, k=5):
        seen.append((query, k))
        return [GRPO, R1] if "a" in query.split() else [GRPO]

    fake(happy_script(PLANNER=[planner("a b", "c d", "e f")]), retrieve)
    s = graph.run(Q)
    assert sorted(q for q, _ in seen) == ["a b", "c d", "e f"]
    assert s["sub_queries"] == ["a b", "c d", "e f"]
    assert [c.id for c in s["chunks"]] == [GRPO.id, R1.id]  # deduped


def test_merge_order_follows_sub_query_order(fake):
    out = {"first": [R1], "second": [GRPO, R1], "third": [DRGRPO]}
    fake(happy_script(PLANNER=[planner("first", "second", "third")]), lambda q, k=5: out[q])
    assert [c.id for c in graph.run(Q)["chunks"]] == [R1.id, GRPO.id, DRGRPO.id]


def test_retry_then_pass_replaces_chunks_and_rewrites_queries(fake):
    def retrieve(query, k=5):
        return [R1] if "bad" in query else [GRPO]

    f = fake(
        happy_script(
            PLANNER=[planner("bad one", "bad two")],
            REWRITER=[planner("good one", "good two")],
            CLAIM_EXTRACTOR=[GOOD_CLAIMS, GOOD_CLAIMS],
            CLAIM_VERIFIER=[verdicts("unsupported"), verdicts("supported")],
        ),
        retrieve,
    )
    s = graph.run(Q)
    assert s["final_status"] == "answered"
    assert s["retry_count"] == 1
    assert s["sub_queries"] == ["good one", "good two"]
    assert [c.id for c in s["chunks"]] == [GRPO.id]  # replaced, not appended
    assert f.tasks().count("REWRITER") == 1
    assert f.tasks().count("GENERATOR") == 2


def test_retry_cap_then_refuse(fake):
    f = fake(happy_script(CLAIM_VERIFIER=[verdicts("contradicted")]))
    s = graph.run(Q)
    assert s["final_status"] == "refused"
    assert s["retry_count"] == MAX_RETRIES == 2
    assert s["answer"] == REFUSAL_TEXT
    assert s["claims"] == []
    assert s["critic_verdict"].verdict == "fail"
    assert f.tasks().count("GENERATOR") == MAX_RETRIES + 1
    assert f.tasks().count("REWRITER") == MAX_RETRIES


def test_empty_retrieval_refuses_straight_from_merge(fake):
    f = fake(happy_script(), lambda q, k=5: [])
    s = graph.run(Q)
    assert s["final_status"] == "refused"
    assert s["answer"] == REFUSAL_TEXT
    assert s["retry_count"] == 0
    assert s["claims"] == [] and s["critic_verdict"] is None
    assert f.tasks() == ["PLANNER"]


def test_empty_retrieval_on_retry_refuses(fake):
    calls = {"n": 0}

    def retrieve(q, k=5):
        calls["n"] += 1
        return [GRPO] if calls["n"] <= 2 else []

    fake(happy_script(CLAIM_VERIFIER=[verdicts("unsupported")]), retrieve)
    s = graph.run(Q)
    assert s["final_status"] == "refused"
    assert s["retry_count"] == 1
    assert s["chunks"] == []


def test_answer_with_no_claims_fails_critic(fake):
    fake(happy_script(CLAIM_EXTRACTOR=[claims()], GENERATOR=["The context lacks this."]))
    s = graph.run(Q)
    assert s["final_status"] == "refused"
    assert s["critic_verdict"].reason == "The answer contains no checkable claims."


def test_unlabelled_claim_is_not_supported(fake):
    two = claims(("claim one", [GRPO.id]), ("claim two", [GRPO.id]))
    fake(happy_script(CLAIM_EXTRACTOR=[two], CLAIM_VERIFIER=[verdicts("supported")]))
    s = graph.run(Q)  # verdict list covers only claim 0 -> claim 1 counts as unsupported
    assert s["final_status"] == "refused"
    assert s["critic_verdict"].unsupported_claims == ["claim two"]


def test_unknown_cited_ids_are_dropped_from_claims(fake):
    c = claims(("x", [GRPO.id, "deepseek-r1:nope:p1:0"]))
    fake(happy_script(CLAIM_EXTRACTOR=[c]))
    assert graph.run(Q)["claims"][0].chunk_ids == [GRPO.id]


def test_sub_queries_clamped_to_three_and_padded_to_two(fake):
    fake(happy_script(PLANNER=[planner("a", "b", "c", "d", "A")]))
    assert graph.run(Q)["sub_queries"] == ["a", "b", "c"]
    fake(happy_script(PLANNER=[planner("only one")]))
    sq = graph.run(Q)["sub_queries"]
    assert 2 <= len(sq) <= 3 and sq[0] == "only one"


def test_json_in_code_fence_and_one_retry_on_garbage(fake):
    fenced = "```json\n" + json.dumps(planner("a b", "c d")) + "\n```"
    f = fake(happy_script(PLANNER=["not json at all", fenced]))
    assert graph.run(Q)["sub_queries"] == ["a b", "c d"]
    assert f.tasks().count("PLANNER") == 2


# --- plain baseline -----------------------------------------------------------------------


def test_plain_flow(fake):
    got = []

    def retrieve(q, k=5):
        got.append((q, k))
        return [GRPO]

    f = fake(happy_script(), retrieve)
    s = graph.run_plain(Q)
    assert got == [(Q, 5)]
    assert f.tasks() == ["GENERATOR"]
    assert s["final_status"] == "answered"
    assert s["answer"] == GOOD_ANSWER
    assert s["claims"] == [] and s["retry_count"] == 0
    assert s["critic_verdict"] is None and s["sub_queries"] == []


def test_plain_refuses_only_on_empty_retrieval(fake):
    f = fake(happy_script(), lambda q, k=5: [])
    s = graph.run_plain(Q)
    assert s["final_status"] == "refused" and s["answer"] == REFUSAL_TEXT
    assert f.calls == []


def test_plain_does_not_refuse_even_if_answer_says_unknown(fake):
    fake(happy_script(GENERATOR=["The context does not contain it."]), lambda q, k=5: [GRPO])
    assert graph.run_plain(Q)["final_status"] == "answered"


def test_plain_and_graph_use_same_generator_prompt(fake):
    f = fake(happy_script())
    graph.run(Q)
    graph.run_plain(Q)
    gen_users = [u for _, t, u in f.calls if t == "GENERATOR"]
    assert len(gen_users) == 2 and gen_users[0] == gen_users[1]
    assert [r for r, t, _ in f.calls if t == "GENERATOR"] == ["generator", "generator"]


# --- retrieval seam, prompts, boundaries --------------------------------------------------


def test_stub_retriever_ranks_and_can_return_nothing():
    assert retrieval.stub_retrieve("GRPO value model baseline")[0].id == GRPO.id
    assert retrieval.stub_retrieve("zebra quantum soup") == []
    assert len(retrieval.stub_retrieve("GRPO reinforcement learning", k=1)) == 1


def test_retrieve_falls_back_to_stub_when_ingestion_missing():
    import importlib.util

    if importlib.util.find_spec("src.ingestion") is not None:
        pytest.skip("real retriever present")
    assert retrieval.retrieve("GRPO value model")[0].id == GRPO.id


def test_prompt_version_format():
    assert re.fullmatch(r"agent-\d+\.\d+\.\d+", PROMPT_VERSION)


def test_nodes_context_lists_ids():
    assert f"[{GRPO.id}]" in nodes.format_context([GRPO])


def test_agent_never_imports_gemini_sdk_or_judge_prompts():
    for path in (ROOT / "src" / "agent").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "langchain_google_genai" not in text, path
        assert "google.generativeai" not in text, path
        assert "eval.judge" not in text and "from eval" not in text, path


# --- scope refusal (off-topic / creative requests) ----------------------------------------


def test_out_of_scope_request_is_refused_before_retrieval(fake):
    calls = []

    def retrieve(q, k=5):
        calls.append(q)
        return [GRPO]

    f = fake(happy_script(PLANNER=[{"in_scope": False, "sub_queries": []}]), retrieve)
    s = graph.run("Write a poem about autumn.")
    assert s["final_status"] == "refused"
    assert s["answer"] == REFUSAL_TEXT
    assert s["claims"] == [] and s["chunks"] == []
    assert s["retry_count"] == 0 and s["sub_queries"] == []
    assert calls == [] and f.tasks() == ["PLANNER"]


def test_out_of_scope_ignores_leftover_sub_queries(fake):
    f = fake(happy_script(PLANNER=[{"in_scope": False, "sub_queries": ["a b", "c d"]}]))
    assert graph.run(Q)["final_status"] == "refused"
    assert f.tasks() == ["PLANNER"]


@pytest.mark.parametrize(
    "reply", [{"in_scope": True}, {}, {"in_scope": "false"}, {"in_scope": None}]
)
def test_scope_check_fails_open(fake, reply):
    fake(happy_script(PLANNER=[{**reply, "sub_queries": ["a b", "c d"]}]))
    assert graph.run(Q)["final_status"] == "answered"


def test_planner_prompt_asks_for_scope_and_generator_prompt_declines_non_facts():
    from src.agent import prompts

    assert "in_scope" in prompts.PLANNER
    assert "creative writing" in prompts.GENERATOR


# --- recall on retry (false refusals) ------------------------------------------------------


def test_retry_widens_k_each_heal_loop(fake):
    ks = []

    def retrieve(q, k=5):
        ks.append(k)
        return [GRPO]

    fake(happy_script(CLAIM_VERIFIER=[verdicts("unsupported")]), retrieve)
    graph.run(Q)
    # 2 sub-queries per attempt, 3 attempts (retry_count 0, 1, 2)
    assert ks == [5, 5, 8, 8, 11, 11]


def test_retry_with_better_queries_recovers_instead_of_refusing(fake):
    """First queries miss the evidence, so the generator reports no information; the rewriter
    gets that answer and its new queries find the right passage."""

    def retrieve(query, k=5):
        return [GRPO] if "precise" in query else [R1]

    miss = "The context does not contain it."
    f = fake(
        happy_script(
            PLANNER=[planner("broad topic", "general area")],
            REWRITER=[planner("precise fact", "precise wording")],
            GENERATOR=[miss, GOOD_ANSWER],
            CLAIM_EXTRACTOR=[claims(), GOOD_CLAIMS],
            CLAIM_VERIFIER=[verdicts("supported")],
        ),
        retrieve,
    )
    s = graph.run(Q)
    assert s["final_status"] == "answered" and s["retry_count"] == 1
    assert [c.id for c in s["chunks"]] == [GRPO.id]
    rewriter_user = next(u for _, t, u in f.calls if t == "REWRITER")
    assert miss in rewriter_user  # the rewriter sees what went wrong


def test_prompt_version_bumped_for_this_change():
    assert PROMPT_VERSION != "agent-1.0.0"


# --- malformed JSON never crashes the graph -------------------------------------------------

# The real failure seen live: JSONDecodeError "Expecting ',' delimiter".
BAD = '{"sub_queries": ["a b" "c d"]}'


def test_parse_json_raises_value_error_on_malformed():
    from src.agent import llm_utils

    for text in (BAD, "nope", "[1, 2]", '{"a": 1,', ""):
        with pytest.raises(ValueError):
            llm_utils.parse_json(text)


def test_call_json_retries_once_then_returns_none(fake):
    from src.agent import llm_utils

    f = fake(happy_script(PLANNER=[BAD]))
    assert llm_utils.call_json("generator", "TASK: PLANNER", "q") is None
    assert f.tasks() == ["PLANNER", "PLANNER"]


def test_planner_malformed_twice_fails_open(fake):
    f = fake(happy_script(PLANNER=[BAD]))
    s = graph.run(Q)
    assert s["final_status"] == "answered"  # in scope, queries built from the question
    assert 2 <= len(s["sub_queries"]) <= 3 and s["sub_queries"][0] == Q
    assert f.tasks().count("PLANNER") == 2


def test_planner_malformed_once_uses_the_retry(fake):
    f = fake(happy_script(PLANNER=[BAD, planner("a b", "c d")]))
    assert graph.run(Q)["sub_queries"] == ["a b", "c d"]
    assert f.tasks().count("PLANNER") == 2


def test_extractor_malformed_twice_fails_closed_and_refuses_at_cap(fake):
    f = fake(happy_script(CLAIM_EXTRACTOR=[BAD]))
    s = graph.run(Q)
    assert s["final_status"] == "refused" and s["claims"] == []
    assert s["retry_count"] == MAX_RETRIES
    assert "could not be parsed" in s["critic_verdict"].reason
    assert "CLAIM_VERIFIER" not in f.tasks()  # nothing to verify


def test_extractor_malformed_once_recovers_via_retry_call(fake):
    f = fake(happy_script(CLAIM_EXTRACTOR=[BAD, GOOD_CLAIMS]))
    s = graph.run(Q)
    assert s["final_status"] == "answered" and s["retry_count"] == 0
    assert f.tasks().count("CLAIM_EXTRACTOR") == 2


def test_verifier_malformed_twice_marks_every_claim_unsupported(fake):
    f = fake(happy_script(CLAIM_VERIFIER=[BAD]))
    s = graph.run(Q)
    assert s["final_status"] == "refused" and s["claims"] == []
    assert s["critic_verdict"].verdict == "fail"
    assert s["critic_verdict"].unsupported_claims == [GOOD_CLAIMS["claims"][0]["text"]]
    assert "could not be parsed" in s["critic_verdict"].reason
    assert f.tasks().count("CLAIM_VERIFIER") == 2 * (MAX_RETRIES + 1)


def test_verifier_malformed_once_recovers(fake):
    fake(happy_script(CLAIM_VERIFIER=[BAD, verdicts("supported")]))
    s = graph.run(Q)
    assert s["final_status"] == "answered" and s["retry_count"] == 0


def test_rewriter_malformed_twice_falls_back_to_question_queries(fake):
    f = fake(
        happy_script(
            REWRITER=[BAD],
            CLAIM_VERIFIER=[verdicts("unsupported"), verdicts("supported")],
        )
    )
    s = graph.run(Q)
    assert s["final_status"] == "answered" and s["retry_count"] == 1
    assert s["sub_queries"][0] == Q and 2 <= len(s["sub_queries"]) <= 3
    assert f.tasks().count("REWRITER") == 2


@pytest.mark.parametrize(
    "task, reply",
    [
        ("PLANNER", {"in_scope": True, "sub_queries": "not a list"}),
        ("CLAIM_EXTRACTOR", {"claims": "not a list"}),
        ("CLAIM_EXTRACTOR", {"claims": [{"text": "x", "chunk_ids": "not a list"}]}),
        ("CLAIM_VERIFIER", {"verdicts": {"0": "supported"}}),
        ("CLAIM_VERIFIER", {"verdicts": "supported"}),
    ],
)
def test_valid_json_with_wrong_shape_never_raises(fake, task, reply):
    fake(happy_script(**{task: [reply]}))
    assert graph.run(Q)["final_status"] in ("answered", "refused")
