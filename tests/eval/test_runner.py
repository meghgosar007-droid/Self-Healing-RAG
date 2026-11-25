"""Runner end to end with a fake agent and a fake judge LLM (no network, no real models)."""

import json
from itertools import count

import pytest

from eval import dataset, fakes, runner
from eval.judge import Judge, NullCache
from src import llm
from src.schemas import EvalResult, new_state

MODELS = {"generator": "gen-m", "critic": "critic-m", "judge": "judge-m"}


def qs():
    return [
        dataset.Question(
            id="q1", question="Answerable one?", answerable=True, smoke=True, reference_notes=""
        ),
        dataset.Question(
            id="q2", question="Unanswerable two?", answerable=False, smoke=True, reference_notes=""
        ),
        dataset.Question(
            id="q3", question="Only in full?", answerable=True, smoke=False, reference_notes=""
        ),
    ]


def ticking_clock(step=0.25):
    c = count()
    return lambda: next(c) * step


def fake_judge(model="judge-m"):
    return Judge(fakes.FakeJudgeLLM(model), NullCache(), model_name=lambda: model)


def go(questions=None, mode="full", agents=None, judge=None, models=MODELS, **kw):
    questions = questions or qs()
    answerable = {q.question: q.answerable for q in questions}
    return runner.run_eval(
        dataset.select(questions, mode),
        mode,
        agents=agents or fakes.fake_agents(answerable),
        judge=judge or fake_judge(),
        prompt_version="agent-test",
        corpus_version="a1b2c3d4e5f6",
        configured_models=models,
        git_sha="deadbeef",
        clock=ticking_clock(),
        **kw,
    )


def test_full_run_validates_against_schema_and_scores_both_systems():
    result = go()
    EvalResult.model_validate(result.model_dump(mode="json"))  # round-trips
    assert len(result.results) == 6
    assert {r.system for r in result.results} == {"plain", "self_healing"}
    by = {(r.id, r.system): r for r in result.results}
    assert by["q1", "plain"].outcome == "answered_ok"
    assert by["q2", "plain"].outcome == "hallucinated"  # makes something up
    assert by["q2", "self_healing"].outcome == "correct_refusal"
    assert by["q2", "self_healing"].answer is None and by["q2", "self_healing"].claims == []
    assert result.aggregates.plain.n_questions == 3
    assert result.aggregates.self_healing.correct_refusals.count == 1
    # plain: 3 answered, hallucinated on q2 only
    assert result.aggregates.plain.hallucination_rate == pytest.approx(1 / 3, abs=1e-4)


def test_meta_records_modes_versions_and_models():
    result = go(mode="smoke")
    m = result.meta
    assert (m.eval_mode, m.contract_version, m.prompt_version) == ("smoke", 3, "agent-test")
    assert m.judge_prompt_version.startswith("judge-")
    assert (m.generator_model, m.critic_model, m.judge_model) == ("gen-m", "critic-m", "judge-m")
    assert (m.corpus_version, m.git_sha) == ("a1b2c3d4e5f6", "deadbeef")
    assert len(result.results) == 4  # q1, q2 only


def test_latency_is_wall_clock_per_question():
    result = go()
    assert {r.latency_ms for r in result.results} == {250}  # one 0.25 s tick per agent call


def test_agent_usage_flows_into_tokens_wait_models_and_cost(monkeypatch):
    monkeypatch.setenv("GEN_MODEL", "gemini-2.5-flash")  # hermetic: ignore the real .env
    """Real src.llm accounting: tokens, queue/backoff wait and the model actually used."""

    class Chat:
        def invoke(self, messages, **kw):
            class R:
                content = "x"
                usage_metadata = {"input_tokens": 1000, "output_tokens": 100}

            return R()

    managed = llm.ManagedLLM("generator", "k", builder=lambda m, k: Chat())
    base = fakes.fake_agents({})

    def metered(question):
        managed.invoke("hi")
        return base["self_healing"](question)

    llm.reset_state()
    result = go(
        questions=qs()[:1],
        agents={"plain": base["plain"], "self_healing": metered},
        pricing={
            "models": {"gemini-2.5-flash": {"input": 1000.0, "output": 0.0}},
            "default": {"input": 0, "output": 0},
        },
    )
    row = next(r for r in result.results if r.system == "self_healing")
    assert (row.tokens_in, row.tokens_out) == (1000, 100)
    assert row.llm_wait_ms <= row.latency_ms
    assert result.meta.generator_model == "gemini-2.5-flash"  # observed, not configured
    # cost: 1000 tokens x $1000/M = $1.00 from the agent call; judge calls are priced at default 0
    assert result.aggregates.self_healing.cost_usd_estimate == pytest.approx(1.0)
    assert result.aggregates.plain.cost_usd_estimate == 0


def test_judge_calls_are_included_in_cost_but_not_in_row_tokens():
    result = go(
        questions=qs()[:1],
        pricing={"models": {}, "default": {"input": 1_000_000.0, "output": 0.0}},
    )
    plain = next(r for r in result.results if r.system == "plain")
    assert plain.tokens_in == 0  # agent made no LLM calls
    assert result.aggregates.plain.cost_usd_estimate > 0  # judge tokens priced
    assert result.aggregates.plain.tokens_in == 0


def test_refusals_are_not_sent_to_the_judge():
    judge_llm = fakes.FakeJudgeLLM("judge-m")
    only_refusal = [qs()[1]]
    agents = {
        s: fakes.fake_agents({qs()[1].question: False})["self_healing"] for s in runner.SYSTEMS
    }
    result = go(
        questions=only_refusal,
        agents=agents,
        judge=Judge(judge_llm, NullCache(), model_name=lambda: "judge-m"),
    )
    assert judge_llm.calls == 0
    assert result.aggregates.plain.correct_refusals.rate == 1.0


# --- model separation ---------------------------------------------------------------------


def test_preflight_rejects_judge_equal_to_generator():
    with pytest.raises(runner.ModelCollisionError, match="generator"):
        go(models={**MODELS, "judge": "gen-m"})


def test_preflight_rejects_judge_equal_to_critic():
    with pytest.raises(runner.ModelCollisionError, match="critic"):
        go(models={**MODELS, "judge": "critic-m"})


def test_fallback_collision_detected_at_runtime():
    """Configured models differ, but the judge falls back onto the generator's model."""
    result_models = {**MODELS, "judge": "judge-primary"}
    judge = Judge(fakes.FakeJudgeLLM("gen-m"), NullCache(), model_name=lambda: "judge-primary")
    agents = fakes.fake_agents({})

    class Chat:
        def invoke(self, messages, **kw):
            class R:
                content = "x"
                usage_metadata = {"input_tokens": 1, "output_tokens": 1}

            return R()

    gen = llm.ManagedLLM("generator", "k", builder=lambda m, k: Chat())

    def with_gen(question):
        gen.invoke("hi")
        return agents["plain"](question)

    llm.reset_state()
    import os

    os.environ["GEN_MODEL"] = "gen-m"
    try:
        with pytest.raises(runner.ModelCollisionError):
            go(
                agents={"plain": with_gen, "self_healing": agents["self_healing"]},
                judge=judge,
                models=result_models,
            )
    finally:
        del os.environ["GEN_MODEL"]
        llm.reset_state()


# --- failure handling ---------------------------------------------------------------------


def test_agent_failure_aborts_with_question_id():
    base = fakes.fake_agents({})

    def boom(question):
        raise RuntimeError("quota")

    with pytest.raises(runner.EvalError, match="q1"):
        go(agents={"plain": base["plain"], "self_healing": boom})


def test_non_terminal_state_is_an_error():
    base = fakes.fake_agents({})
    with pytest.raises(runner.EvalError, match="non-terminal"):
        go(agents={"plain": lambda q: new_state(q), "self_healing": base["self_healing"]})


def test_missing_system_is_an_error():
    with pytest.raises(runner.EvalError, match="missing"):
        go(agents={"plain": fakes.fake_agents({})["plain"]})


# --- CLI ----------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_cli_fake_run_writes_a_valid_result(tmp_path, mode, capsys):
    rc = runner.main(
        [
            "--mode",
            mode,
            "--fake",
            "--out-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "c"),
            "--run-id",
            "t",
        ]
    )
    assert rc == 0
    result = EvalResult.model_validate(
        json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    )
    n = len([q for q in dataset.load_questions() if mode == "full" or q.smoke])
    assert result.aggregates.plain.n_questions == n == result.aggregates.self_healing.n_questions
    assert result.meta.eval_mode == mode
    assert "wrote" in capsys.readouterr().out


def test_cli_second_run_is_served_from_judge_cache(tmp_path, capsys):
    args = [
        "--mode",
        "smoke",
        "--fake",
        "--out-dir",
        str(tmp_path),
        "--cache-dir",
        str(tmp_path / "c"),
    ]
    runner.main([*args, "--run-id", "a"])
    capsys.readouterr()
    runner.main([*args, "--run-id", "b"])
    assert " 0 misses" in capsys.readouterr().out
    a = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert a["results"] == b["results"]


def test_corpus_version_read_from_manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text('{"corpus_version": "abcdef012345"}', encoding="utf-8")
    assert runner.read_corpus_version(p) == "abcdef012345"
    with pytest.raises(runner.EvalError):
        runner.read_corpus_version(tmp_path / "missing.json")


# --- transient network errors -------------------------------------------------------------


class RemoteProtocolError(Exception):  # same class name httpx raises
    pass


@pytest.fixture
def no_backoff(monkeypatch):
    from eval import transient

    monkeypatch.setattr(transient, "BACKOFF_S", 0)


def flaky(real, failures, exc):
    state = {"left": failures}

    def agent(question):
        if state["left"] > 0:
            state["left"] -= 1
            raise exc
        return real(question)

    return agent, state


def test_agent_retries_dropped_connections_then_succeeds(no_backoff):
    base = fakes.fake_agents({})
    agent, state = flaky(base["self_healing"], 2, RemoteProtocolError("Server disconnected"))
    result = go(questions=qs()[:1], agents={"plain": base["plain"], "self_healing": agent})
    assert state["left"] == 0 and len(result.results) == 2


def test_agent_gives_up_after_three_transient_failures(no_backoff):
    base = fakes.fake_agents({})
    agent, _ = flaky(base["self_healing"], 5, RemoteProtocolError("Server disconnected"))
    with pytest.raises(runner.EvalError, match="q1"):
        go(questions=qs()[:1], agents={"plain": base["plain"], "self_healing": agent})


def test_non_transient_errors_are_not_retried(no_backoff):
    base = fakes.fake_agents({})
    agent, state = flaky(base["self_healing"], 1, ValueError("bug"))
    with pytest.raises(runner.EvalError):
        go(questions=qs()[:1], agents={"plain": base["plain"], "self_healing": agent})
    assert state["left"] == 0  # called once, not retried


def test_transient_detected_through_exception_chain():
    from eval import transient

    try:
        try:
            raise RemoteProtocolError("x")
        except RemoteProtocolError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert transient.is_transient(outer)
    assert not transient.is_transient(ValueError("nope"))


def test_judge_retries_dropped_connections(no_backoff):
    calls = {"n": 0}
    real = fakes.FakeJudgeLLM("judge-m")

    def flaky_llm(system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RemoteProtocolError("Server disconnected")
        return real(system, user)

    judge = Judge(flaky_llm, NullCache(), model_name=lambda: "judge-m")
    result = go(questions=qs()[:1], judge=judge)
    assert len(result.results) == 2
