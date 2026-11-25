"""Helpers around `src.llm`: every call goes through `llm.get_llm(role)`."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src import llm

log = logging.getLogger(__name__)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # Gemini may return content parts
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return str(content)


def call(role: str, system: str, user: str) -> str:
    response = llm.get_llm(role).invoke([SystemMessage(system), HumanMessage(user)])
    return _text(response.content).strip()


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in model output: {text[:200]!r}") from None
        value = json.loads(text[start : end + 1])  # JSONDecodeError is a ValueError
    if not isinstance(value, dict):
        raise ValueError("model output JSON is not an object")
    return value


def call_json(role: str, system: str, user: str) -> dict[str, Any] | None:
    """Call and parse JSON. On malformed output, retry once with a reminder; if that also
    fails return None so the caller can use its safe default. Never raises on bad JSON."""
    reminder = "\n\n" + "Reply with a single valid JSON object only."
    for attempt, suffix in enumerate(("", reminder)):
        try:
            return parse_json(call(role, system, user + suffix))
        except ValueError:
            log.warning("llm role=%s: malformed JSON from model (attempt %d/2)", role, attempt + 1)
    return None
