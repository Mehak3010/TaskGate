"""Executor behaviour under adverse conditions.

Three promises are tested here because they are what makes a calibration run
trustworthy: transient errors are retried and then excluded from the denominator,
the budget guard stops the run, and one bad rollout never kills the run.
"""

from __future__ import annotations

import asyncio

from taskgate.models import Rollout
from taskgate.runner.executor import Executor, Job


def _job(task_id: str, index: int, fn) -> Job:
    return Job(task_id=task_id, index=index, run=fn)


async def test_non_retryable_error_is_recorded_and_not_raised():
    async def boom():
        raise ValueError("no such image")

    executor = Executor(concurrency=1, max_retries=2, backoff_s=0.0)
    report = await executor.run([_job("t", 0, boom)])

    rollouts = report.rollouts["t"]
    assert len(rollouts) == 1
    assert rollouts[0].status == "infra_error"
    assert rollouts[0].counted is False  # excluded from the denominator
    assert "ValueError" in (rollouts[0].error or "")


async def test_transient_error_is_retried_before_giving_up():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        raise RuntimeError("429 rate limited")

    executor = Executor(concurrency=1, max_retries=1, backoff_s=0.0)
    report = await executor.run([_job("t", 0, flaky)])

    assert calls["n"] == 2  # one attempt + one retry
    assert report.retries == 1
    assert report.rollouts["t"][0].status == "infra_error"


async def test_retry_can_recover_a_transient_failure():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 rate limited")
        return Rollout(index=0, status="ok", passed=True)

    executor = Executor(concurrency=1, max_retries=2, backoff_s=0.0)
    report = await executor.run([_job("t", 0, flaky)])

    assert calls["n"] == 2
    assert report.rollouts["t"][0].status == "ok"
    assert report.rollouts["t"][0].passed is True


async def test_budget_guard_aborts_the_remaining_rollouts():
    async def paid():
        return Rollout(index=0, status="ok", passed=False, cost_usd=0.30)

    executor = Executor(concurrency=1, budget_usd=0.5, backoff_s=0.0)
    jobs = [_job("t", index, paid) for index in range(4)]
    report = await executor.run(jobs)

    assert report.aborted is True
    assert report.completed == 2
    assert report.skipped_jobs == 2
    assert report.total_cost_usd == 0.6


async def test_no_budget_guard_runs_everything():
    async def paid():
        return Rollout(index=0, status="ok", passed=True, cost_usd=1.0)

    executor = Executor(concurrency=2, budget_usd=None, backoff_s=0.0)
    report = await executor.run([_job("t", index, paid) for index in range(6)])

    assert report.aborted is False
    assert report.completed == 6
    assert report.skipped_jobs == 0


async def test_rollout_timeout_produces_a_record_not_an_exception():
    async def slow():
        await asyncio.sleep(5)
        return Rollout(index=0, status="ok")

    executor = Executor(concurrency=1, rollout_timeout_s=0.05, backoff_s=0.0)
    report = await executor.run([_job("t", 0, slow)])

    assert report.rollouts["t"][0].status == "timeout"


async def test_rollouts_are_grouped_by_task_and_sorted_by_index():
    def make(index: int):
        async def runner():
            # rollouts finish out of order; results must still come back sorted
            await asyncio.sleep(0.001 * (3 - index))
            return Rollout(index=index, status="ok", passed=True)

        return runner

    executor = Executor(concurrency=4, backoff_s=0.0)
    jobs = [
        _job("a", 0, make(0)),
        _job("b", 0, make(0)),
        _job("a", 1, make(1)),
        _job("a", 2, make(2)),
    ]
    report = await executor.run(jobs)

    assert sorted(report.rollouts) == ["a", "b"]
    assert [r.index for r in report.rollouts["a"]] == [0, 1, 2]
