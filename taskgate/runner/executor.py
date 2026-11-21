"""Async rollout executor: bounded concurrency, retries, spend guard.

Three properties matter more than raw speed here:

* a transient provider error is retried with backoff and then **excluded from the
  denominator** rather than counted as a task failure;
* the budget guard stops the run early instead of silently overspending;
* cleanup/timeouts always produce a rollout record, so one bad rollout can never
  abort the whole run.

Provider failures are handled *by class*:

* an authentication/authorisation error (401/402/403) is not retryable -- the run
  aborts within seconds with one message and nothing is scored;
* a rate limit (429) backs off (honouring ``Retry-After``) and retries, and when
  the *daily* quota is gone the run stops cleanly and is marked
  "rate-limited, resume tomorrow" rather than failing the tasks.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..models import Rollout


# --------------------------------------------------------------------------
# provider error classes
# --------------------------------------------------------------------------
class ProviderError(RuntimeError):
    """A provider-side failure. Carries the status code and provider name only.

    The API key is never part of a ``ProviderError`` message: these strings can
    end up in ``Rollout.error`` and therefore in ``report.json``.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class NonRetryableProviderError(ProviderError):
    """401/402/403 (and other hard 4xx): abort the run; never score the tasks."""


class RateLimitError(ProviderError):
    """429: retryable, honouring ``Retry-After`` unless the daily quota is gone."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int = 429,
        retry_after_s: float | None = None,
        daily_quota: bool = False,
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code)
        self.retry_after_s = retry_after_s
        self.daily_quota = daily_quota

    def delay(self, base_backoff: float, attempt: int) -> float:
        """The backoff to apply before the next attempt."""
        if self.retry_after_s is not None:
            return max(0.0, self.retry_after_s)
        return base_backoff * (2 ** (attempt - 1))


@dataclass
class Job:
    task_id: str
    index: int
    run: Callable[[], Awaitable[Rollout]]


@dataclass
class ExecutorReport:
    rollouts: dict[str, list[Rollout]] = field(default_factory=dict)
    aborted: bool = False
    #: "budget" | "auth" | "quota" | None -- why the run stopped early.
    abort_kind: str | None = None
    abort_reason: str | None = None
    #: True when a daily quota reset (not a task failure) stopped the run.
    rate_limited: bool = False
    aborted_tasks: list[str] = field(default_factory=list)
    skipped_jobs: int = 0
    retries: int = 0
    total_cost_usd: float = 0.0
    duration_s: float = 0.0

    @property
    def completed(self) -> int:
        return sum(len(v) for v in self.rollouts.values())


class Executor:
    def __init__(
        self,
        *,
        concurrency: int = 4,
        max_retries: int = 2,
        budget_usd: float | None = None,
        backoff_s: float = 0.4,
        rollout_timeout_s: float | None = None,
        on_rollout: Callable[[Job, Rollout], None] | None = None,
    ) -> None:
        self.concurrency = max(1, concurrency)
        self.max_retries = max(0, max_retries)
        self.budget_usd = budget_usd
        self.backoff_s = backoff_s
        self.rollout_timeout_s = rollout_timeout_s
        self.on_rollout = on_rollout
        self.aborted = False
        self.abort_kind: str | None = None
        self.abort_reason: str | None = None
        self.rate_limited = False
        self._cost = 0.0
        self._retries = 0

    # -- internals ---------------------------------------------------------
    def _charge(self, rollout: Rollout) -> None:
        self._cost += rollout.cost_usd or 0.0
        # A free model spends nothing, so a $0 budget must never trip on it:
        # the guard only fires once real spend reaches the cap.
        if self.budget_usd is not None and self._cost > 0 and self._cost >= self.budget_usd:
            self._abort("budget", f"budget guard tripped at ${self.budget_usd}")

    def _abort(self, kind: str, reason: str, *, rate_limited: bool = False) -> None:
        if self.aborted:
            return
        self.aborted = True
        self.abort_kind = kind
        self.abort_reason = reason
        self.rate_limited = rate_limited

    async def _run_with_timeout(self, job: Job) -> Rollout:
        if self.rollout_timeout_s is not None:
            return await asyncio.wait_for(job.run(), timeout=self.rollout_timeout_s)
        return await job.run()

    async def _attempt(self, job: Job) -> Rollout:
        try:
            return await self._run_with_timeout(job)
        except NonRetryableProviderError as exc:
            # 401/402/403: the key is wrong or the account is not allowed. Retrying
            # cannot help, so stop the whole run now and score nothing.
            self._abort("auth", str(exc))
            return Rollout(index=job.index, status="error", error=str(exc))
        except RateLimitError as exc:
            if exc.daily_quota:
                self._abort("quota", str(exc), rate_limited=True)
                return Rollout(index=job.index, status="rate_limited", error=str(exc))
            raise  # a short-lived 429: let _run_job honour Retry-After and retry
        except asyncio.TimeoutError:
            return Rollout(
                index=job.index,
                status="timeout",
                error=f"wall-clock timeout after {self.rollout_timeout_s:g}s",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad rollout must not kill the run
            return Rollout(
                index=job.index,
                status="infra_error",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _run_job(self, job: Job, semaphore: asyncio.Semaphore) -> Rollout | None:
        if self.aborted:
            return None
        async with semaphore:
            if self.aborted:
                return None
            attempt = 0
            while True:
                try:
                    rollout = await self._attempt(job)
                except RateLimitError as exc:
                    if attempt < self.max_retries:
                        attempt += 1
                        self._retries += 1
                        await asyncio.sleep(exc.delay(self.backoff_s, attempt))
                        if self.aborted:
                            return None
                        continue
                    rollout = Rollout(
                        index=job.index, status="infra_error", error=str(exc)
                    )
                if rollout.status == "infra_error" and attempt < self.max_retries:
                    attempt += 1
                    self._retries += 1
                    await asyncio.sleep(self.backoff_s * (2 ** (attempt - 1)))
                    if self.aborted:
                        return None
                    continue
                self._charge(rollout)
                if self.on_rollout is not None:
                    self.on_rollout(job, rollout)
                return rollout

    # -- public ------------------------------------------------------------
    async def run(self, jobs: list[Job]) -> ExecutorReport:
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(self.concurrency)
        report = ExecutorReport()
        results = await asyncio.gather(*(self._run_job(job, semaphore) for job in jobs))
        for job, rollout in zip(jobs, results):
            if rollout is None:
                report.skipped_jobs += 1
                report.aborted_tasks.append(job.task_id)
                continue
            report.rollouts.setdefault(job.task_id, []).append(rollout)
        for values in report.rollouts.values():
            values.sort(key=lambda r: r.index)
        report.aborted = self.aborted
        report.abort_kind = self.abort_kind
        report.abort_reason = self.abort_reason
        report.rate_limited = self.rate_limited
        report.aborted_tasks = sorted(set(report.aborted_tasks))
        report.retries = self._retries
        report.total_cost_usd = round(self._cost, 6)
        report.duration_s = round(time.perf_counter() - started, 3)
        return report
