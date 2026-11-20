"""Helpers around `src.llm`: every call goes through `llm.get_llm(role)`."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src import llm


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
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output JSON is not an object")
    return value


def call_json(role: str, system: str, user: str) -> dict[str, Any]:
    """Call and parse JSON; one retry with a reminder if the output is not valid JSON."""
    try:
        return parse_json(call(role, system, user))
    except ValueError:
        return parse_json(
            call(role, system, user + "\n\nReply with a single valid JSON object only.")
        )
