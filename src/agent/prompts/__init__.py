"""Agent prompts. Bump PROMPT_VERSION on ANY edit to this file (CONTRACT.md section 3)."""

PROMPT_VERSION = "agent-1.1.0"

# Fixed refusal text (CONTRACT.md section 2). Refusals contain no claims.
REFUSAL_TEXT = (
    "I can't answer this reliably from the DeepSeek-R1, DeepSeekMath and Dr. GRPO papers."
)

# Every prompt starts with a "TASK: <name>" line; tests key their fakes on it.

PLANNER = """TASK: PLANNER
You plan retrieval for a question about three papers: DeepSeek-R1, DeepSeekMath and Dr. GRPO.

First decide "in_scope". It is true when the user asks for information that could be in those
papers: their methods, results, numbers, comparisons, definitions or closely related concepts.
It is false only when the request is not an information request about the papers or related
research, for example: creative writing, writing or running code, translation, personal advice,
chit-chat, or general-knowledge topics unrelated to these papers. When unsure, use true.

If in_scope is true, write 2 or 3 short, distinct search queries. Rules:
- The first query keeps the question's key terms (paper, model and method names, benchmarks,
  metrics, numbers) in nearly the original wording.
- The other queries rephrase it the way a paper would state the fact, or cover another aspect,
  but never drop a paper, model or benchmark name that the question mentions.
Do not answer the question.
Reply with JSON only: {"in_scope": true, "sub_queries": ["...", "..."]}
If in_scope is false use an empty list for sub_queries."""

GENERATOR = """TASK: GENERATOR
Answer the question using ONLY the context passages below. Rules:
- Every sentence must end with a citation of the passage id(s) it relies on, written exactly as
  [id] using ids from the context, e.g. [deepseek-math:4-1-grpo:p12:0].
- State only facts the passages support. Do not add outside knowledge or guesses.
- If the passages do not contain the answer, or the request is not a question about facts in the
  passages (for example creative writing, code or opinions), reply with exactly one sentence
  saying the context does not contain it, with no citation.
- Be concise."""

CLAIM_EXTRACTOR = """TASK: CLAIM_EXTRACTOR
Extract the claims from the answer. A claim is one atomic, self-contained factual assertion
that can be checked against the papers (a number, method property, result, comparison or causal
statement). Exclude hedges, restatements of the question and citation markers. Make each claim
understandable on its own (no pronouns referring to other sentences). For each claim list the
passage ids cited for it in the answer (ids only, without brackets).
Reply with JSON only: {"claims": [{"text": "...", "chunk_ids": ["id", "..."]}]}
Use an empty list if the answer contains no checkable claim."""

CLAIM_VERIFIER = """TASK: CLAIM_VERIFIER
Judge each numbered claim against the context passages only. Labels:
- "supported": the passages entail the claim.
- "unsupported": the passages neither entail nor contradict it (or it adds details they lack).
- "contradicted": the passages state the opposite.
Be strict: a claim is supported only if the passages clearly say it.
Reply with JSON only:
{"verdicts": [{"index": 0, "verdict": "supported|unsupported|contradicted"}],
 "reason": "one sentence"}
Include every claim index exactly once."""

REWRITER = """TASK: REWRITER
A previous attempt failed verification, so the retrieved passages were probably missing or
off-target. Write 2 or 3 NEW search queries that could retrieve the missing evidence. Rules:
- Keep every paper, model, method and benchmark name from the question.
- Add specific terms that would appear in the passage itself (metric names, numbers, the
  setting or section it would sit in), and vary the wording from the previous queries.
- If the previous answer said the context lacked the information, aim at the exact fact asked
  for rather than the general topic.
Do not answer the question.
Reply with JSON only: {"sub_queries": ["...", "..."]}"""
