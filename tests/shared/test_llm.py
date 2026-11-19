"""Unit tests for src/llm.py with a fake chat model; no network and no real key."""

import logging
import threading
import time

import pytest

from src import llm

SECRET = "sk-test-SECRET-1234567890"


class ResourceExhausted(Exception):
    """Same class name as google.api_core.exceptions.ResourceExhausted."""


class NotFound(Exception):
    """Same class name as google.api_core.exceptions.NotFound."""


class FakeResponse:
    def __init__(self, text="ok", tokens_in=10, tokens_out=5):
        self.content = text
        self.usage_metadata = {"input_tokens": tokens_in, "output_tokens": tokens_out}


class FakeChat:
    """Plays a script of outcomes: an exception to raise, or a response to return."""

    def __init__(self, model, script, log):
        self.model = model
        self.script = script
        self.log = log

    def invoke(self, input, **kwargs):  # noqa: A002
        self.log.append(self.model)
        step = self.script.get(self.model, [FakeResponse()])
        item = step.pop(0) if len(step) > 1 else step[0]
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    llm.reset_state()
    for var in (
        "GEN_MODEL",
        "GEN_MODEL_FALLBACK",
        "CRITIC_MODEL",
        "CRITIC_MODEL_FALLBACK",
        "JUDGE_MODEL",
        "JUDGE_MODEL_FALLBACK",
        "LLM_MAX_CONCURRENCY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: False)  # never read the real .env
    yield
    llm.reset_state()


def make(role="generator", script=None):
    calls: list[str] = []
    sleeps: list[float] = []
    script = script or {}
    client = llm.get_llm(
        role,
        _builder=lambda model, key: FakeChat(model, script, calls),
        _sleep=sleeps.append,
        _rng=lambda: 1.0,  # jitter at its ceiling: deterministic waits
    )
    return client, calls, sleeps


# --- configuration ------------------------------------------------------------------------


def test_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.get_llm("generator")


def test_blank_key_fails_fast(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(RuntimeError):
        llm.get_llm("judge")


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        llm.get_llm("planner")  # type: ignore[arg-type]


def test_default_models():
    assert llm.resolve_models("generator") == ("gemini-2.5-flash", "gemini-2.5-flash-lite")
    assert llm.resolve_models("critic") == ("gemini-2.5-flash", "gemini-2.5-flash-lite")
    assert llm.resolve_models("judge") == ("gemini-2.5-pro", "gemini-2.5-flash-lite")


def test_models_come_from_env(monkeypatch):
    monkeypatch.setenv("GEN_MODEL", "my-gen")
    monkeypatch.setenv("GEN_MODEL_FALLBACK", "my-gen-lite")
    monkeypatch.setenv("JUDGE_MODEL", "  ")  # blank -> default
    assert llm.resolve_models("generator") == ("my-gen", "my-gen-lite")
    assert llm.resolve_models("judge")[0] == "gemini-2.5-pro"


def test_default_judge_differs_from_default_generator():
    assert llm.resolve_models("judge")[0] != llm.resolve_models("generator")[0]


# --- retry --------------------------------------------------------------------------------


def test_success_records_tokens_and_model():
    client, calls, sleeps = make(script={"gemini-2.5-flash": [FakeResponse("hi", 12, 7)]})
    with llm.track() as usage:
        assert client.invoke("hello").content == "hi"
    assert calls == ["gemini-2.5-flash"] and sleeps == []
    assert (usage.tokens_in, usage.tokens_out) == (12, 7)
    assert usage.models == {"generator": "gemini-2.5-flash"}


def test_retries_429_with_exponential_backoff_then_succeeds():
    err = ResourceExhausted("429 quota")
    client, calls, sleeps = make(script={"gemini-2.5-flash": [err, err, err, FakeResponse()]})
    with llm.track() as usage:
        client.invoke("x")
    assert len(calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]  # rng=1.0 -> ceiling of each attempt
    assert usage.wait_ms >= 7000  # backoff time counts as llm_wait


def test_retries_503():
    client, calls, _ = make(
        script={"gemini-2.5-flash": [Exception("503 Service Unavailable"), FakeResponse()]}
    )
    client.invoke("x")
    assert len(calls) == 2


def test_honors_server_retry_hint():
    err = Exception("429 RESOURCE_EXHAUSTED. Please retry in 7.5s")
    client, _, sleeps = make(script={"gemini-2.5-flash": [err, FakeResponse()]})
    client.invoke("x")
    assert sleeps == [7.5]  # hint beats the 1.0s first-attempt ceiling


def test_wait_capped_at_60s():
    err = Exception("429 retry in 600s")
    client, _, sleeps = make(script={"gemini-2.5-flash": [err, FakeResponse()]})
    client.invoke("x")
    assert sleeps == [60.0]


def test_parse_retry_hint_variants():
    assert llm.parse_retry_hint(Exception("retry_delay { seconds: 33 }")) == 33
    assert llm.parse_retry_hint(Exception("'retryDelay': '12s'")) == 12
    assert llm.parse_retry_hint(Exception("nothing here")) is None


def test_fatal_error_not_retried():
    client, calls, sleeps = make(script={"gemini-2.5-flash": [ValueError("bad prompt")]})
    with pytest.raises(ValueError):
        client.invoke("x")
    assert calls == ["gemini-2.5-flash"] and sleeps == []


@pytest.mark.parametrize(
    "exc,kind",
    [
        (ResourceExhausted("x"), "retryable"),
        (Exception("HTTP 429"), "retryable"),
        (Exception("503"), "retryable"),
        (NotFound("x"), "unavailable"),
        (Exception("404 models/x is not found"), "unavailable"),
        (Exception("403 PERMISSION_DENIED"), "unavailable"),
        (ValueError("tokens: 15034"), "fatal"),  # digits inside a number are not a status code
        (RuntimeError("boom"), "fatal"),
    ],
)
def test_classify_error(exc, kind):
    assert llm.classify_error(exc) == kind


# --- fallback -----------------------------------------------------------------------------


def test_falls_back_after_max_attempts_and_stays_on_fallback():
    err = ResourceExhausted("429")
    client, calls, sleeps = make(
        script={"gemini-2.5-flash": [err], "gemini-2.5-flash-lite": [FakeResponse("lite")]}
    )
    with llm.track() as usage:
        assert client.invoke("x").content == "lite"
    assert calls.count("gemini-2.5-flash") == llm.MAX_ATTEMPTS
    assert calls[-1] == "gemini-2.5-flash-lite"
    assert len(sleeps) == llm.MAX_ATTEMPTS - 1
    assert usage.models["generator"] == "gemini-2.5-flash-lite"
    assert client.model_name == "gemini-2.5-flash-lite"

    calls.clear()
    client.invoke("again")  # sticky: no more attempts on the dead primary
    assert calls == ["gemini-2.5-flash-lite"]


def test_model_not_found_falls_back_immediately():
    client, calls, sleeps = make(
        script={"gemini-2.5-flash": [NotFound("model missing")]},
    )
    client.invoke("x")
    assert calls == ["gemini-2.5-flash", "gemini-2.5-flash-lite"] and sleeps == []


def test_error_propagates_when_fallback_also_fails():
    client, calls, _ = make(
        script={
            "gemini-2.5-flash": [NotFound("a")],
            "gemini-2.5-flash-lite": [NotFound("b")],
        }
    )
    with pytest.raises(NotFound, match="b"):
        client.invoke("x")


def test_no_fallback_loop_when_fallback_equals_primary(monkeypatch):
    monkeypatch.setenv("GEN_MODEL_FALLBACK", "gemini-2.5-flash")
    client, calls, _ = make(script={"gemini-2.5-flash": [NotFound("gone")]})
    with pytest.raises(NotFound):
        client.invoke("x")
    assert calls == ["gemini-2.5-flash"]


def test_fallback_is_per_role():
    err = NotFound("gone")
    gen, _, _ = make("generator", {"gemini-2.5-flash": [err]})
    gen.invoke("x")
    assert llm.active_model("generator") == "gemini-2.5-flash-lite"
    assert llm.active_model("judge") == "gemini-2.5-pro"


# --- concurrency --------------------------------------------------------------------------


def test_concurrency_cap_respected(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "2")
    llm.reset_state()
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class SlowChat:
        def invoke(self, input, **kwargs):  # noqa: A002
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return FakeResponse()

    client = llm.get_llm("generator", _builder=lambda m, k: SlowChat())
    waits = []

    def worker():
        with llm.track() as usage:
            client.invoke("x")
        waits.append(usage.wait_ms)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["peak"] == 2
    assert max(waits) > 0  # queueing shows up as llm_wait_ms


# --- accounting and secrets ---------------------------------------------------------------


def test_track_scopes_are_isolated():
    client, _, _ = make()
    with llm.track() as outer:
        client.invoke("a")
        with llm.track() as inner:
            client.invoke("b")
        client.invoke("c")
    assert len(inner.calls) == 1 and len(outer.calls) == 2


def test_calls_outside_track_are_fine():
    client, _, _ = make()
    client.invoke("x")


def test_failed_retries_still_count_wait():
    err = ResourceExhausted("429")
    client, _, _ = make(script={"gemini-2.5-flash": [err, ValueError("stop")]})
    with llm.track() as usage, pytest.raises(ValueError):
        client.invoke("x")
    assert usage.wait_ms >= 1000


def test_api_key_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    err = ResourceExhausted("429")
    client, _, _ = make(
        script={"gemini-2.5-flash": [err], "gemini-2.5-flash-lite": [FakeResponse()]}
    )
    client.invoke("x")
    assert caplog.records, "expected retry/fallback log lines"
    assert SECRET not in caplog.text
