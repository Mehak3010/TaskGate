"""A simple token-aware pacer for tight tokens-per-minute budgets.

Groq's free tier binds on tokens per minute (8K TPM), not on requests, so a
fast loop can trip it even at concurrency 1. :class:`TokenPacer` keeps a rolling
window of the tokens each request actually consumed and blocks the *next*
request until the window has drained enough room for it.

It is deliberately simple: a plain sliding window over the last ``window_s``
seconds, checked before each request and fed the real ``usage`` afterwards. At
concurrency 1 (the mode the budget assumes) that is exact enough to never trip
the limit, and it never invents a token count it did not see.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Callable


class TokenPacer:
    def __init__(
        self,
        tokens_per_minute: int,
        *,
        window_s: float = 60.0,
        safety: float = 0.9,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], "asyncio.Future"] = asyncio.sleep,
    ) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        self.tokens_per_minute = tokens_per_minute
        self.window_s = window_s
        self.budget = max(1, int(tokens_per_minute * safety))
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _used(self) -> int:
        return sum(tokens for _, tokens in self._events)

    def record(self, tokens: int) -> None:
        """Record the tokens a completed request consumed."""
        if tokens > 0:
            self._events.append((self._clock(), int(tokens)))

    async def acquire(self, estimated_tokens: int) -> None:
        """Block until ``estimated_tokens`` fit inside the rolling window."""
        estimate = max(0, int(estimated_tokens))
        while True:
            now = self._clock()
            self._prune(now)
            if self._used() + estimate <= self.budget or not self._events:
                return
            oldest = self._events[0][0]
            wait = self.window_s - (now - oldest)
            await self._sleep(max(wait, 0.01))

    def time_until_room(self, estimated_tokens: int) -> float:
        """Seconds until ``estimated_tokens`` fit in the window (0 if they fit now).

        Used after a provider *still* returned a tokens-per-minute 429 despite
        the reservation: the caller sleeps exactly until the window has drained
        enough, rather than guessing a backoff. Never returns a negative value.
        """
        now = self._clock()
        self._prune(now)
        estimate = max(0, int(estimated_tokens))
        if self._used() + estimate <= self.budget or not self._events:
            return 0.0
        oldest = self._events[0][0]
        return max(self.window_s - (now - oldest), 0.0)
