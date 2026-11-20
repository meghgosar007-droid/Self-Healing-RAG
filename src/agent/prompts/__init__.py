"""Agent prompts. Bump PROMPT_VERSION on ANY edit to this file (CONTRACT.md section 3)."""

PROMPT_VERSION = "agent-1.0.0"

# Fixed refusal text (CONTRACT.md section 2). Refusals contain no claims.
REFUSAL_TEXT = (
    "I can't answer this reliably from the DeepSeek-R1, DeepSeekMath and Dr. GRPO papers."
)

# Every prompt starts with a "TASK: <name>" line; tests key their fakes on it.

PLANNER = """TASK: PLANNER
You plan retrieval for a question about three papers: DeepSeek-R1, DeepSeekMath and Dr. GRPO.
Write 2 or 3 short, distinct search queries that together cover what is needed to answer the
question (different aspects, wordings or papers). Do not answer the question.
Reply with JSON only: {"sub_queries": ["...", "..."]}"""

GENERATOR = """TASK: GENERATOR
Answer the question using ONLY the context passages below. Rules:
- Every sentence must end with a citation of the passage id(s) it relies on, written exactly as
  [id] using ids from the context, e.g. [deepseek-math:4-1-grpo:p12:0].
- State only facts the passages support. Do not add outside knowledge or guesses.
- If the passages do not contain the answer, reply with exactly one sentence saying the context
  does not contain it, with no citation.
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
A previous answer failed verification. Write 2 or 3 NEW search queries that could retrieve the
missing or better evidence. Use different wordings, more specific terms or other papers than
the previous queries. Do not answer the question.
Reply with JSON only: {"sub_queries": ["...", "..."]}"""
