"""Retry for transient network failures during long eval runs.

``src/llm.py`` (shared file) retries only 429 / ResourceExhausted / 503. A dropped connection
("Server disconnected without sending a response") is classified fatal there and would abort a
run that has already spent quota. The eval track retries those at its own level instead.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger("eval")
T = TypeVar("T")

ATTEMPTS = 3
BACKOFF_S = 5.0
# Provider overload that src/llm.py has already retried and given up on (503 "high demand",
# 429 quota). Spikes clear in minutes, so wait longer and try more times than for a dropped socket.
OVERLOAD_ATTEMPTS = 5
OVERLOAD_BACKOFF_S = 60.0
_OVERLOAD_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
_TRANSIENT_NAMES = {
    "RemoteProtocolError",
    "ReadError",
    "WriteError",
    "ReadTimeout",
    "ConnectError",
    "ConnectTimeout",
    "PoolTimeout",
    "ServerDisconnectedError",
    "ConnectionError",
    "ConnectionResetError",
    "TimeoutError",
}


def is_transient(exc: BaseException) -> bool:
    """True if the exception, or anything it was raised from, is a connection-level failure."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(c.__name__ in _TRANSIENT_NAMES for c in type(exc).__mro__):
            return True
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]
    return False


def is_overload(exc: BaseException) -> bool:
    """True if the exception, or anything it was raised from, reports provider overload."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = str(exc)
        if any(m in text for m in _OVERLOAD_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]
    return False


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = ATTEMPTS,
    backoff_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    what: str = "call",
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:
            overload = is_overload(exc)
            limit = max(attempts, OVERLOAD_ATTEMPTS) if overload else attempts
            if attempt >= limit or not (overload or is_transient(exc)):
                raise
            base = OVERLOAD_BACKOFF_S if overload else BACKOFF_S
            wait = (base if backoff_s is None else backoff_s) * attempt
            log.warning(
                "%s: %r (attempt %d/%d), retrying in %.0fs", what, exc, attempt, limit, wait
            )
            sleep(wait)
