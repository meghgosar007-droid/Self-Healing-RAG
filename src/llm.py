"""The only gateway to Gemini (CONTRACT.md section 6).

Every LLM call in the repo goes through ``get_llm(role).invoke(...)``. This module adds:

* a global concurrency cap (``LLM_MAX_CONCURRENCY``, default 2),
* retry with exponential backoff and jitter on 429 / ResourceExhausted / 503, honoring
  server retry-delay hints (max 6 attempts, 60 s cap per wait),
* a per-role fallback model, used when the primary is unavailable or its quota is exhausted,
* usage accounting (wait time and tokens) for the active ``track()`` scope.

No other file may import ``langchain_google_genai`` or ``google.generativeai``.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv

log = logging.getLogger(__name__)

Role = Literal["generator", "critic", "judge"]
ROLES: tuple[str, ...] = ("generator", "critic", "judge")

# role -> (primary env var, fallback env var, primary default, fallback default)
_MODEL_CONFIG: dict[str, tuple[str, str, str, str]] = {
    "generator": ("GEN_MODEL", "GEN_MODEL_FALLBACK", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
    "critic": (
        "CRITIC_MODEL",
        "CRITIC_MODEL_FALLBACK",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ),
    "judge": ("JUDGE_MODEL", "JUDGE_MODEL_FALLBACK", "gemini-2.5-pro", "gemini-2.5-flash-lite"),
}

MAX_ATTEMPTS = 6
MAX_WAIT_S = 60.0
BASE_WAIT_S = 1.0
DEFAULT_CONCURRENCY = 2


# --- usage accounting ---------------------------------------------------------------------


@dataclass(frozen=True)
class CallRecord:
    role: str
    model: str
    tokens_in: int
    tokens_out: int


class Usage:
    """Thread-safe accumulator for one ``track()`` scope."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[CallRecord] = []
        self._wait_s = 0.0

    def add_wait(self, seconds: float) -> None:
        with self._lock:
            self._wait_s += seconds

    def add_call(self, record: CallRecord) -> None:
        with self._lock:
            self.calls.append(record)

    @property
    def wait_ms(self) -> int:
        return round(self._wait_s * 1000)

    @property
    def tokens_in(self) -> int:
        return sum(c.tokens_in for c in self.calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.tokens_out for c in self.calls)

    @property
    def models(self) -> dict[str, str]:
        """Model actually used per role (the last one that answered)."""
        return {c.role: c.model for c in self.calls}


_current_usage: contextvars.ContextVar[Usage | None] = contextvars.ContextVar(
    "llm_usage", default=None
)


@contextlib.contextmanager
def track() -> Iterator[Usage]:
    """Collect wait time and token usage for every LLM call made inside the block.

    The scope follows the context, so threads you start yourself must run inside
    ``contextvars.copy_context().run(...)``; LangChain/LangGraph executors already do this.
    """
    usage = Usage()
    token = _current_usage.set(usage)
    try:
        yield usage
    finally:
        _current_usage.reset(token)


# --- configuration ------------------------------------------------------------------------


def resolve_models(role: str) -> tuple[str, str]:
    """Return ``(primary, fallback)`` model names for a role from env, else defaults."""
    if role not in _MODEL_CONFIG:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    primary_var, fallback_var, primary_default, fallback_default = _MODEL_CONFIG[role]
    primary = os.getenv(primary_var, "").strip() or primary_default
    fallback = os.getenv(fallback_var, "").strip() or fallback_default
    return primary, fallback


_state_lock = threading.Lock()
_semaphore: threading.Semaphore | None = None
_fallback_active: dict[str, bool] = {}


def _get_semaphore() -> threading.Semaphore:
    global _semaphore
    with _state_lock:
        if _semaphore is None:
            raw = os.getenv("LLM_MAX_CONCURRENCY", "").strip()
            size = int(raw) if raw else DEFAULT_CONCURRENCY
            if size < 1:
                raise ValueError("LLM_MAX_CONCURRENCY must be >= 1")
            _semaphore = threading.Semaphore(size)
        return _semaphore


def reset_state() -> None:
    """Forget the semaphore and any sticky fallback. For tests and long-lived processes."""
    global _semaphore
    with _state_lock:
        _semaphore = None
        _fallback_active.clear()


def active_model(role: str) -> str:
    """Model the next call for this role will use (primary, or fallback once switched)."""
    primary, fallback = resolve_models(role)
    return fallback if _fallback_active.get(role) else primary


# --- error handling -----------------------------------------------------------------------

Kind = Literal["retryable", "unavailable", "fatal"]

_RETRYABLE_NAMES = {"ResourceExhausted", "ServiceUnavailable", "TooManyRequests"}
_UNAVAILABLE_NAMES = {"NotFound", "PermissionDenied"}


def _status_code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        value = getattr(value, "value", value)  # enums such as grpc.StatusCode
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _has_code(text: str, code: int) -> bool:
    return re.search(rf"(?<!\d){code}(?!\d)", text) is not None


def classify_error(exc: BaseException) -> Kind:
    """retryable: 429 / ResourceExhausted / 503. unavailable: model missing or no access."""
    names = {c.__name__ for c in type(exc).__mro__}
    status = _status_code(exc)
    text = str(exc)
    upper = text.upper()

    if status in (404, 403) or names & _UNAVAILABLE_NAMES:
        return "unavailable"
    if "NOT_FOUND" in upper or "PERMISSION_DENIED" in upper:
        return "unavailable"
    if _has_code(text, 404) or _has_code(text, 403):
        return "unavailable"
    if status in (429, 503) or names & _RETRYABLE_NAMES:
        return "retryable"
    if _has_code(text, 429) or _has_code(text, 503):
        return "retryable"
    if "RESOURCE_EXHAUSTED" in upper or "UNAVAILABLE" in upper:
        return "retryable"
    return "fatal"


_HINT_PATTERNS = (
    re.compile(r"retry in\s+(\d+(?:\.\d+)?)\s*s", re.IGNORECASE),
    re.compile(r"retry[_ ]?delay\W+(?:seconds\W+)?(\d+(?:\.\d+)?)", re.IGNORECASE),
)


def parse_retry_hint(exc: BaseException) -> float | None:
    """Seconds the server asked us to wait, if the error message says so."""
    text = str(exc)
    for pattern in _HINT_PATTERNS:
        m = pattern.search(text)
        if m:
            return float(m.group(1))
    return None


def compute_wait(attempt: int, hint: float | None, rng: Callable[[], float]) -> float:
    """Backoff before attempt ``attempt + 1``: full jitter, never below the server hint."""
    ceiling = min(MAX_WAIT_S, BASE_WAIT_S * 2 ** (attempt - 1))
    wait = rng() * ceiling
    if hint is not None:
        wait = max(wait, hint)
    return min(wait, MAX_WAIT_S)


class _SwitchModel(Exception):
    """Internal: this model cannot serve the call; try the fallback."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


# --- the client ---------------------------------------------------------------------------


def _default_builder(model: str, api_key: str) -> Any:
    # Imported here so this is the only module that ever loads the Gemini SDK.
    from langchain_google_genai import ChatGoogleGenerativeAI

    # retries=0: the SDK's own retry is disabled so the policy above is the only one.
    return ChatGoogleGenerativeAI(model=model, api_key=api_key, temperature=0, retries=0)


def _token_counts(response: Any) -> tuple[int, int]:
    meta = getattr(response, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0) or 0), int(meta.get("output_tokens", 0) or 0)


class ManagedLLM:
    """Wraps a chat model with the concurrency cap, retry, fallback and accounting."""

    def __init__(
        self,
        role: str,
        api_key: str,
        *,
        builder: Callable[[str, str], Any] = _default_builder,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.role = role
        self._api_key = api_key
        self._builder = builder
        self._sleep = sleep
        self._rng = rng
        self._chats: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return active_model(self.role)

    def invoke(self, input: Any, **kwargs: Any) -> Any:  # noqa: A002 - mirrors LangChain
        primary, fallback = resolve_models(self.role)
        if _fallback_active.get(self.role):
            candidates = [fallback]
        else:
            candidates = [primary] if primary == fallback else [primary, fallback]

        for i, model in enumerate(candidates):
            try:
                return self._call_model(model, input, kwargs)
            except _SwitchModel as switch:
                if i + 1 >= len(candidates):
                    raise switch.original from None
                log.warning(
                    "llm role=%s: model %s unavailable, falling back to %s",
                    self.role,
                    model,
                    candidates[i + 1],
                )
                _fallback_active[self.role] = True
        raise AssertionError("unreachable")

    def _chat(self, model: str) -> Any:
        if model not in self._chats:
            self._chats[model] = self._builder(model, self._api_key)
        return self._chats[model]

    def _call_model(self, model: str, input: Any, kwargs: dict[str, Any]) -> Any:  # noqa: A002
        usage = _current_usage.get()
        semaphore = _get_semaphore()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            queued_at = time.monotonic()
            with semaphore:
                if usage:
                    usage.add_wait(time.monotonic() - queued_at)
                try:
                    response = self._chat(model).invoke(input, **kwargs)
                except Exception as exc:  # noqa: BLE001 - classified below
                    error = exc
                else:
                    tokens_in, tokens_out = _token_counts(response)
                    if usage:
                        usage.add_call(CallRecord(self.role, model, tokens_in, tokens_out))
                    return response

            kind = classify_error(error)
            if kind == "fatal":
                raise error
            if kind == "unavailable" or attempt == MAX_ATTEMPTS:
                raise _SwitchModel(error) from error
            wait = compute_wait(attempt, parse_retry_hint(error), self._rng)
            log.info(
                "llm role=%s model=%s attempt=%d/%d: retrying in %.1fs",
                self.role,
                model,
                attempt,
                MAX_ATTEMPTS,
                wait,
            )
            self._sleep(wait)
            if usage:
                usage.add_wait(wait)
        raise AssertionError("unreachable")


def get_llm(
    role: Role,
    *,
    _builder: Callable[[str, str], Any] = _default_builder,
    _sleep: Callable[[float], None] = time.sleep,
    _rng: Callable[[], float] = random.random,
) -> ManagedLLM:
    """Return the managed client for a role. Fails fast if ``GEMINI_API_KEY`` is missing.

    The underscore arguments exist for tests only.
    """
    if role not in _MODEL_CONFIG:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ManagedLLM(role, api_key, builder=_builder, sleep=_sleep, rng=_rng)
