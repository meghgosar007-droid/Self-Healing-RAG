"""Judge prompts. Independent of the agent's prompts (CONTRACT.md section 6).

Never import from ``src.agent`` here or anywhere under ``eval/judge/``. Bump
``JUDGE_PROMPT_VERSION`` on ANY edit to this file; it is recorded in ``meta`` and is part of
every judge cache key.
"""

JUDGE_PROMPT_VERSION = "judge-1.0.0"

EXTRACT = """TASK: JUDGE_EXTRACT
You are an independent auditor. You are given a question and a written answer. List every
checkable factual claim the answer makes, so that each can be verified separately later.

A claim is ONE atomic, self-contained factual assertion (a number, a method property, a result,
a comparison, or a causal statement). Rules:
- Split compound sentences into separate claims. Resolve pronouns and "it/this/they" so each
  claim can be understood without the rest of the answer; name the model, method or paper.
- Keep numbers, units and names exactly as the answer states them.
- Do NOT list hedges, restatements of the question, statements about what the context or
  passages do or do not contain, or bare citation markers.
- For each claim, list the ids that the answer cites for it, copied exactly from the answer's
  square-bracket citations (without the brackets). Use [] when the answer cites nothing for it.
- Do not judge whether a claim is true. Do not add claims the answer does not make.

Reply with JSON only, no prose:
{"claims": [{"text": "...", "chunk_ids": ["..."]}]}
If the answer makes no checkable claim, reply {"claims": []}."""

VERIFY = """TASK: JUDGE_VERIFY
You are a strict fact-checking judge. You get numbered claims and a set of source passages.
Decide each claim using ONLY the passages. Ignore everything you know from elsewhere, even if
the claim is true in the real world.

Labels:
- "supported": the passages, taken together, clearly state or directly entail the whole claim,
  including every number, unit, name and comparison in it.
- "contradicted": the passages state something incompatible with the claim (a different number,
  the opposite property, a different model or paper).
- "unsupported": anything else - the passages are silent on it, cover only part of it, or the
  claim adds a detail the passages do not give.

When unsure between "supported" and "unsupported", choose "unsupported". Use "contradicted"
only when a passage actually conflicts with the claim.

Reply with JSON only, no prose, covering every claim index exactly once:
{"verdicts": [{"index": 0, "verdict": "supported|unsupported|contradicted"}]}"""
