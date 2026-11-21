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


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = ATTEMPTS,
    backoff_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    what: str = "call",
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            log.warning("%s: transient %r (attempt %d/%d), retrying", what, exc, attempt, attempts)
            sleep((BACKOFF_S if backoff_s is None else backoff_s) * attempt)
    raise AssertionError("unreachable")
