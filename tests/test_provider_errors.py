"""Provider failures handled by class, and key hygiene.

All HTTP is mocked with ``httpx.MockTransport``: these tests never touch the
network, so they are safe to run without an API key.

The promises under test:

* 401/402/403 abort the whole run in seconds and score nothing;
* a 429 backs off and resumes;
* a *daily* 429 stops the run cleanly and is not a task failure;
* a provider key never appears in an error message or in the serialised report;
* a failed preflight starts no containers;
* infra errors never enter the solve-rate denominator.
"""

from __future__ import annotations

import httpx
import pytest

from taskgate.models import Report, Rollout, RunConfig
from taskgate.pipeline import PipelineOptions, run_pipeline_sync
from taskgate.runner.agent import AgentRunner, preflight
from taskgate.runner.executor import (
    Executor,
    Job,
    NonRetryableProviderError,
    RateLimitError,
)
from taskgate.stats.calibration import build_calibration_result

FAKE_KEY = "gsk_TESTONLY_do_not_log_1234567890"


# ---------------------------------------------------------------- helpers
def _job(task_id: str, index: int, fn) -> Job:
    return Job(task_id=task_id, index=index, run=fn)


def _install_transport(monkeypatch, handler):
    """Route every AsyncClient through a MockTransport running ``handler``."""
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def _runner(provider: str = "groq") -> AgentRunner:
    # _complete only reads model/provider/endpoint/api_key; task and box are unused.
    return AgentRunner(
        task=None,  # type: ignore[arg-type]
        box=None,  # type: ignore[arg-type]
        model="openai/gpt-oss-120b",
        api_key=FAKE_KEY,
        provider=provider,
    )


# ---------------------------------------------------------------- executor
@pytest.mark.parametrize("status", [401, 402, 403])
async def test_auth_errors_abort_the_run_and_score_nothing(status):
    async def boom():
        raise NonRetryableProviderError(
            f"Groq: HTTP {status}. Check the API key.",
            provider="groq",
            status_code=status,
        )

    executor = Executor(concurrency=1, max_retries=3, backoff_s=0.0)
    jobs = [_job("t", index, boom) for index in range(3)]
    report = await executor.run(jobs)

    assert report.aborted is True
    assert report.abort_kind == "auth"
    assert report.completed == 1  # only the first job was attempted
    assert report.skipped_jobs == 2
    scored = report.rollouts["t"]
    assert len(scored) == 1
    assert scored[0].counted is False  # excluded from the denominator


async def test_rate_limit_is_retried_then_resumes():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("groq: rate limited", provider="groq", retry_after_s=0.0)
        return Rollout(index=0, status="ok", passed=True)

    executor = Executor(concurrency=1, max_retries=2, backoff_s=0.0)
    report = await executor.run([_job("t", 0, flaky)])

    assert calls["n"] == 2
    assert report.retries == 1
    assert report.rollouts["t"][0].status == "ok"
    assert report.aborted is False


async def test_daily_quota_exhaustion_stops_the_run_cleanly():
    async def exhausted():
        raise RateLimitError(
            "groq: daily quota exhausted; rate-limited, resume tomorrow.",
            provider="groq",
            daily_quota=True,
        )

    executor = Executor(concurrency=1, max_retries=5, backoff_s=0.0)
    jobs = [_job("t", index, exhausted) for index in range(3)]
    report = await executor.run(jobs)

    assert report.aborted is True
    assert report.abort_kind == "quota"
    assert report.rate_limited is True
    rollout = report.rollouts["t"][0]
    assert rollout.status == "rate_limited"
    assert rollout.counted is False  # not a task failure, not in the denominator
    assert report.skipped_jobs == 2


# ---------------------------------------------------------------- agent HTTP
async def test_hard_4xx_raises_non_retryable_without_the_key(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    _install_transport(monkeypatch, handler)
    runner = _runner()
    async with httpx.AsyncClient() as client:
        with pytest.raises(NonRetryableProviderError) as caught:
            await runner._complete(client, [{"role": "user", "content": "hi"}])

    assert FAKE_KEY not in str(caught.value)
    assert "Groq" in str(caught.value)
    assert "401" in str(caught.value) or "HTTP 401" in str(caught.value)


async def test_daily_429_raises_daily_quota_rate_limit(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "Rate limit reached ... tokens per day (TPD)"}},
        )

    _install_transport(monkeypatch, handler)
    runner = _runner()
    async with httpx.AsyncClient() as client:
        with pytest.raises(RateLimitError) as caught:
            await runner._complete(client, [{"role": "user", "content": "hi"}])

    assert caught.value.daily_quota is True
    assert FAKE_KEY not in str(caught.value)


async def test_short_429_is_retryable_and_honours_retry_after(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"}, json={})

    _install_transport(monkeypatch, handler)
    runner = _runner()
    async with httpx.AsyncClient() as client:
        with pytest.raises(RateLimitError) as caught:
            await runner._complete(client, [{"role": "user", "content": "hi"}])

    assert caught.value.daily_quota is False
    assert caught.value.retry_after_s == 7.0
    assert caught.value.delay(base_backoff=0.4, attempt=1) == 7.0


# ---------------------------------------------------------------- preflight
async def test_preflight_reports_auth_failure_without_the_key(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    _install_transport(monkeypatch, handler)
    ok, message = await preflight(
        provider="groq", model="openai/gpt-oss-120b", api_key=FAKE_KEY
    )

    assert ok is False
    assert FAKE_KEY not in message
    assert "401" in message


async def test_preflight_succeeds_on_200(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _install_transport(monkeypatch, handler)
    ok, message = await preflight(
        provider="groq", model="openai/gpt-oss-120b", api_key=FAKE_KEY
    )
    assert ok is True
    assert FAKE_KEY not in message


# ---------------------------------------------------------------- pipeline
def test_preflight_failure_starts_no_containers(monkeypatch, make_task):
    task = make_task(task_yaml="id: t_simple\n", instruction="do a thing\n")

    monkeypatch.setattr("taskgate.pipeline.docker_available", lambda *a, **k: True)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)

    async def fake_preflight(*, provider, model, api_key, base_url=None, timeout_s=30.0):
        return False, "Groq: preflight returned HTTP 401."

    monkeypatch.setattr("taskgate.runner.agent.preflight", fake_preflight)

    # Any attempt to build an image would construct a Sandbox; make that fatal.
    built = {"n": 0}

    class _NoSandbox:
        def __init__(self, *a, **k):
            built["n"] += 1
            raise AssertionError("no container should be built after a failed preflight")

    monkeypatch.setattr("taskgate.pipeline.Sandbox", _NoSandbox)

    config = RunConfig(
        model="openai/gpt-oss-120b",
        provider="groq",
        mock=False,
        rollouts=3,
        concurrency=1,
        out_dir="",
    )
    options = PipelineOptions(audit=True, dynamic=False, calibrate=True, diagnose=False)
    report = run_pipeline_sync([task], config, options=options)

    assert built["n"] == 0
    assert any("preflight failed" in note for note in report.notes)
    for task_report in report.tasks:
        assert task_report.calibration is None or task_report.calibration.n_valid == 0


def test_mock_run_never_calls_preflight(monkeypatch, make_task):
    task = make_task(task_yaml="id: t_simple\n", instruction="do a thing\n")
    monkeypatch.setattr("taskgate.pipeline.docker_available", lambda *a, **k: False)

    called = {"n": 0}

    async def fake_preflight(*a, **k):
        called["n"] += 1
        return True, "should not run"

    monkeypatch.setattr("taskgate.runner.agent.preflight", fake_preflight)

    config = RunConfig(model="mock", mock=True, rollouts=2, concurrency=1, out_dir="")
    options = PipelineOptions(audit=False, dynamic=False, calibrate=True, diagnose=False)
    report = run_pipeline_sync([task], config, options=options)

    assert called["n"] == 0
    assert report.simulated is True


# ---------------------------------------------------------------- denominator
def test_infra_and_rate_limited_rollouts_leave_the_denominator():
    rollouts = [
        Rollout(index=0, status="ok", passed=True),
        Rollout(index=1, status="ok", passed=False),
        Rollout(index=2, status="infra_error", error="boom"),
        Rollout(index=3, status="timeout", error="too slow"),
        Rollout(index=4, status="rate_limited", error="daily quota"),
    ]
    result = build_calibration_result(
        task_id="t", rollouts=rollouts, n_planned=5, model="openai/gpt-oss-120b"
    )

    assert result.n_valid == 2  # only the two "ok" rollouts
    assert result.n_passed == 1
    assert result.solve_rate == pytest.approx(0.5)
    assert result.n_infra_errors == 1
    assert result.n_timeouts == 1
    assert result.n_rate_limited == 1
    assert result.n_planned == 5  # the planned count is unchanged


# ---------------------------------------------------------------- key hygiene
async def test_key_never_appears_in_a_serialised_report(monkeypatch):
    async def boom():
        raise NonRetryableProviderError(
            "Groq: HTTP 401. Check the API key.", provider="groq", status_code=401
        )

    executor = Executor(concurrency=1, max_retries=0, backoff_s=0.0)
    exec_report = await executor.run([_job("t", 0, boom)])

    report = Report(config=RunConfig(model="openai/gpt-oss-120b", provider="groq"))
    for rollout in exec_report.rollouts["t"]:
        assert FAKE_KEY not in (rollout.error or "")
        report.tasks = []

    serialised = report.model_dump_json()
    assert FAKE_KEY not in serialised
    assert "gsk_" not in serialised
