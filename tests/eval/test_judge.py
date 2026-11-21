"""Judge behaviour with a scripted LLM: parsing, strictness, retry, disk cache, independence."""

import json
import re
from pathlib import Path

import pytest

from eval.judge import DiskCache, Judge, JudgeError, LLMReply, NullCache, prompts
from eval.judge import judge as judge_mod
from src import llm
from src.schemas import Chunk

ROOT = Path(__file__).resolve().parents[2]
C1 = Chunk.model_validate_json(
    (ROOT / "contract_examples" / "chunk.json").read_text(encoding="utf-8")
)


class Script:
    """Scripted judge LLM: replies keyed by task, last reply repeats; records every call."""

    def __init__(self, extract, verify, model="judge-m"):
        self.replies = {"JUDGE_EXTRACT": list(extract), "JUDGE_VERIFY": list(verify)}
        self.model = model
        self.calls = []

    def __call__(self, system, user):
        task = re.search(r"TASK: (\w+)", system).group(1)
        self.calls.append((task, user))
        queue = self.replies[task]
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMReply(text, self.model, 100, 10)


def claims(*items):
    return {"claims": [{"text": t, "chunk_ids": ids} for t, ids in items]}


def verdicts(*labels):
    return {"verdicts": [{"index": i, "verdict": v} for i, v in enumerate(labels)]}


def make(script, cache=None, model="judge-m"):
    return Judge(script, cache, model_name=lambda: model)


def test_labels_each_claim_and_keeps_only_known_cited_ids():
    s = Script(
        [claims(("A is 1.", [C1.id, "ghost:id:p1:0"]), ("B is 2.", []))],
        [verdicts("supported", "contradicted")],
    )
    out = make(s).judge("q?", "answer text", [C1])
    assert [(c.text, c.chunk_ids, c.verdict) for c in out.claims] == [
        ("A is 1.", [C1.id], "supported"),
        ("B is 2.", [], "contradicted"),
    ]
    assert [c.cached for c in out.calls] == [False, False]
    assert sum(c.tokens_in for c in out.calls) == 200


def test_judge_never_sees_the_agents_claims_only_question_and_answer():
    s = Script([claims(("A.", []))], [verdicts("supported")])
    make(s).judge("What is A?", "A is true [x].", [C1])
    (_, extract_user), (_, verify_user) = s.calls
    assert extract_user == "Question: What is A?\n\nAnswer:\nA is true [x]."
    assert C1.text in verify_user and "1. " not in verify_user.split("Claims:")[0]


def test_unlabeled_or_invalid_verdicts_default_to_unsupported():
    s = Script(
        [claims(("a", []), ("b", []), ("c", []))],
        [{"verdicts": [{"index": 0, "verdict": "SUPPORTED"}, {"index": 1, "verdict": "maybe"}]}],
    )
    out = make(s).judge("q", "a", [C1])
    assert [c.verdict for c in out.claims] == ["supported", "unsupported", "unsupported"]


def test_no_claims_skips_verification_call():
    s = Script([claims()], [verdicts()])
    out = make(s).judge("q", "I do not know.", [C1])
    assert out.claims == [] and [t for t, _ in s.calls] == ["JUDGE_EXTRACT"]


def test_invalid_json_is_retried_once_then_fails():
    s = Script(["not json at all", claims(("a", []))], [verdicts("supported")])
    out = make(s).judge("q", "a", [C1])
    assert out.claims[0].verdict == "supported"
    assert s.calls[1][1].endswith("Reply with a single valid JSON object only.")
    assert out.calls[0].tokens_in == 200  # both attempts are billed

    bad = Script(["nope"], [verdicts()])
    with pytest.raises(JudgeError):
        make(bad).judge("q", "a", [C1])


def test_fenced_json_is_accepted():
    s = Script(["```json\n" + json.dumps(claims(("a", []))) + "\n```"], [verdicts("supported")])
    assert make(s).judge("q", "a", [C1]).claims[0].text == "a"


# --- disk cache ---------------------------------------------------------------------------


def run_twice(tmp_path, model="judge-m", second_model=None, edit_answer=False):
    cache_dir = tmp_path / "cache"
    s1 = Script([claims(("a", []))], [verdicts("supported")], model=model)
    make(s1, DiskCache(cache_dir), model).judge("q", "answer", [C1])
    s2 = Script([claims(("a", []))], [verdicts("unsupported")], model=second_model or model)
    cache2 = DiskCache(cache_dir)
    out = make(s2, cache2, second_model or model).judge(
        "q", "answer 2" if edit_answer else "answer", [C1]
    )
    return s1, s2, cache2, out


def test_cache_hit_skips_llm_and_returns_stored_labels(tmp_path):
    s1, s2, cache, out = run_twice(tmp_path)
    assert len(s1.calls) == 2 and s2.calls == []
    assert cache.hits == 2 and cache.misses == 0
    assert out.claims[0].verdict == "supported"  # the first run's answer, not s2's
    assert all(c.cached for c in out.calls)
    assert sum(c.tokens_in for c in out.calls) == 200  # notional cost survives caching


def test_cache_key_changes_with_model_input_and_prompt(tmp_path, monkeypatch):
    _, s2, _, _ = run_twice(tmp_path, second_model="other-model")
    assert len(s2.calls) == 2  # new model -> miss
    _, s2, _, _ = run_twice(tmp_path / "b", edit_answer=True)
    # new answer -> extraction misses; verification (same claims, same chunks) rightly hits
    assert [t for t, _ in s2.calls] == ["JUDGE_EXTRACT"]
    monkeypatch.setattr(prompts, "JUDGE_PROMPT_VERSION", "judge-9.9.9")
    monkeypatch.setattr(judge_mod.prompts, "JUDGE_PROMPT_VERSION", "judge-9.9.9")
    cache_dir = tmp_path / "cache"
    s3 = Script([claims(("a", []))], [verdicts("supported")])
    make(s3, DiskCache(cache_dir)).judge("q", "answer", [C1])
    assert len(s3.calls) == 2  # new prompt version -> miss


def test_fallback_answer_is_filed_under_the_model_that_produced_it(tmp_path):
    cache_dir = tmp_path / "c"
    # asked for primary, answered by the fallback
    s = Script([claims(("a", []))], [verdicts("supported")], model="fallback-m")
    make(s, DiskCache(cache_dir), model="primary-m").judge("q", "answer", [C1])
    again = Script([claims(("a", []))], [verdicts("supported")], model="primary-m")
    make(again, DiskCache(cache_dir), model="primary-m").judge("q", "answer", [C1])
    assert len(again.calls) == 2  # primary was not served the fallback's answer


def test_corrupt_cache_entry_is_a_miss(tmp_path):
    cache = DiskCache(tmp_path)
    key = "ab" + "0" * 62
    path = tmp_path / "ab" / f"{key}.json"
    path.parent.mkdir()
    path.write_text("{ broken", encoding="utf-8")
    assert cache.get(key) is None and cache.misses == 1
    cache.put(key, {"x": 1})
    assert cache.get(key) == {"x": 1}


def test_null_cache_never_stores():
    c = NullCache()
    c.put("k", {"a": 1})
    assert c.get("k") is None


# --- default LLM call goes through src.llm with the judge role ----------------------------


def test_default_call_uses_judge_role_and_reports_actual_model(monkeypatch):
    class Chat:
        def invoke(self, messages, **kw):
            class R:
                content = '{"claims": []}'
                usage_metadata = {"input_tokens": 7, "output_tokens": 3}

            return R()

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("JUDGE_MODEL", "judge-primary")
    seen_roles = []
    real = llm.get_llm

    def get(role, **kw):
        seen_roles.append(role)
        return real(role, _builder=lambda m, k: Chat())

    monkeypatch.setattr(llm, "get_llm", get)
    llm.reset_state()
    reply = judge_mod.default_llm_call("sys", "user")
    assert seen_roles == ["judge"]
    assert (reply.text, reply.model, reply.tokens_in, reply.tokens_out) == (
        '{"claims": []}',
        "judge-primary",
        7,
        3,
    )


# --- independence from the agent (CONTRACT.md section 6) ----------------------------------


def test_judge_prompts_differ_from_agent_prompts():
    from src.agent import prompts as agent_prompts

    agent_texts = {v for k, v in vars(agent_prompts).items() if k.isupper() and isinstance(v, str)}
    judge_texts = {prompts.EXTRACT, prompts.VERIFY}
    assert not judge_texts & agent_texts
    for text in judge_texts:
        assert not any(text.splitlines()[0] == a.splitlines()[0] for a in agent_texts)


def test_judge_code_does_not_import_agent_or_the_gemini_sdk():
    for path in (ROOT / "eval").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "langchain_google_genai" not in src and "google.generativeai" not in src
        if "judge" in path.parts:
            assert not re.search(r"^\s*(from|import)\s+src\.agent", src, re.M), path
