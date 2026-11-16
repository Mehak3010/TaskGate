"""Async rollout executor: bounded concurrency, retries, spend guard.

Three properties matter more than raw speed here:

* a transient provider error is retried with backoff and then **excluded from the
  denominator** rather than counted as a task failure;
* the budget guard stops the run early instead of silently overspending;
* cleanup/timeouts always produce a rollout record, so one bad rollout can never
  abort the whole run.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..models import Rollout


@dataclass
class Job:
    task_id: str
    index: int
    run: Callable[[], Awaitable[Rollout]]


@dataclass
class ExecutorReport:
    rollouts: dict[str, list[Rollout]] = field(default_factory=dict)
    aborted: bool = False
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
        self._cost = 0.0
        self._retries = 0

    # -- internals ---------------------------------------------------------
    def _charge(self, rollout: Rollout) -> None:
        self._cost += rollout.cost_usd or 0.0
        if self.budget_usd is not None and self._cost >= self.budget_usd:
            self.aborted = True

    async def _attempt(self, job: Job) -> Rollout:
        try:
            if self.rollout_timeout_s is not None:
                return await asyncio.wait_for(job.run(), timeout=self.rollout_timeout_s)
            return await job.run()
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
                rollout = await self._attempt(job)
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
        report.aborted_tasks = sorted(set(report.aborted_tasks))
        report.retries = self._retries
        report.total_cost_usd = round(self._cost, 6)
        report.duration_s = round(time.perf_counter() - started, 3)
        return report
