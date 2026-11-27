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

import io
import json

import httpx
import pytest
from rich.console import Console

from taskgate.models import Report, Rollout, Route, RunConfig, TaskReport
from taskgate.pipeline import PipelineOptions, run_pipeline_sync
from taskgate.report.terminal import print_report_summary
from taskgate.runner.agent import (
    FAILED_GENERATION_LIMIT,
    TRUNCATION_MARKER,
    AgentRunner,
    ToolCallError,
    billable_tokens,
    echo_assistant_message,
    preflight,
    provider_failed_generation,
    provider_message,
    request_schema_rejected,
    scrub_secrets,
    truncate_tool_output,
)
from taskgate.runner.executor import (
    Executor,
    Job,
    NonRetryableProviderError,
    RateLimitError,
)
from taskgate.runner.sandbox import RunResult
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
    task = make_task(
        {"Dockerfile": "FROM python:3.11-slim\n"},
        task_yaml="id: t_simple\n",
        instruction="do a thing\n",
    )

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


def test_tool_call_failures_are_reported_apart_from_infra_errors():
    rollouts = [
        Rollout(index=0, status="ok", passed=True),
        Rollout(index=1, status="ok", passed=False, tool_call_failed=True),
        Rollout(index=2, status="infra_error", error="boom"),
    ]
    result = build_calibration_result(
        task_id="t", rollouts=rollouts, n_planned=3, model="openai/gpt-oss-120b"
    )
    # A tool-call failure is a counted task failure, not an infra error.
    assert result.n_valid == 2
    assert result.n_tool_call_failures == 1
    assert result.n_infra_errors == 1


def test_summary_aggregates_tool_call_failures():
    from taskgate.models import CalibrationResult
    from taskgate.pipeline import _summarise

    calibration = CalibrationResult(
        task_id="t", n_planned=3, n_valid=2, n_tool_call_failures=1
    )
    task_report = TaskReport(
        task_id="t", path="t", route=Route.SHIP, calibration=calibration
    )
    summary = _summarise([task_report], RunConfig(model="m"), wall_time_s=1.0)
    assert summary.total_tool_call_failures == 1


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


# ---------------------------------------------------------------- request shape
async def test_groq_request_sends_no_field_groq_rejects(monkeypatch):
    """The regression: only schema-legal fields go to Groq's endpoint."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    _install_transport(monkeypatch, handler)
    runner = AgentRunner(
        task=None,  # type: ignore[arg-type]
        box=None,  # type: ignore[arg-type]
        model="openai/gpt-oss-120b",
        api_key=FAKE_KEY,
        provider="groq",
        max_completion_tokens=2000,
    )
    async with httpx.AsyncClient() as client:
        await runner._complete(client, [{"role": "user", "content": "hi"}])

    body = captured["body"]
    # Generous completion headroom: reasoning tokens live inside this cap.
    assert body["max_completion_tokens"] == 2000
    # Fields Groq rejects (or that TaskGate simply must not send) are absent.
    for offender in ("parallel_tool_calls", "stream_options", "reasoning", "reasoning_content"):
        assert offender not in body
    assert "strict" not in body["tools"][0]["function"]


async def test_openrouter_request_is_unchanged(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    _install_transport(monkeypatch, handler)
    runner = AgentRunner(
        task=None,  # type: ignore[arg-type]
        box=None,  # type: ignore[arg-type]
        model="openai/gpt-4o-mini",
        api_key="sk-or-TESTONLY",
        provider="openrouter",
    )
    async with httpx.AsyncClient() as client:
        await runner._complete(client, [{"role": "user", "content": "hi"}])

    body = captured["body"]
    assert "max_completion_tokens" not in body  # openrouter path is untouched
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["name"] == "bash"


# ---------------------------------------------------------------- 400 surfacing
async def test_400_body_is_surfaced_without_the_key(monkeypatch):
    groq_message = (
        "invalid JSON schema for tool bash, tools[0].function.parameters: "
        "`additionalProperties:false` must be set on every object"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": f"{groq_message} ({FAKE_KEY})"}})

    _install_transport(monkeypatch, handler)
    runner = _runner()
    async with httpx.AsyncClient() as client:
        with pytest.raises(NonRetryableProviderError) as caught:
            await runner._complete(client, [{"role": "user", "content": "hi"}])

    text = str(caught.value)
    assert "provider rejected the request" in text
    assert "invalid JSON schema" in text  # the provider's own words
    assert "check the key" not in text.lower()  # only 401/403 may say this
    assert FAKE_KEY not in text


async def test_only_auth_failures_tell_you_to_check_the_key(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    _install_transport(monkeypatch, handler)
    runner = _runner()
    async with httpx.AsyncClient() as client:
        with pytest.raises(NonRetryableProviderError) as caught:
            await runner._complete(client, [{"role": "user", "content": "hi"}])

    assert "Check the API key" in str(caught.value)


def test_provider_message_is_scrubbed_and_truncated():
    response = httpx.Response(
        400, json={"error": {"message": f"boom {FAKE_KEY} " + "x" * 400}}
    )
    message = provider_message(response, key=FAKE_KEY)
    assert FAKE_KEY not in message
    assert len(message) <= 300


def test_provider_message_carries_the_error_code_and_type():
    """Groq's machine-readable code is what separates a transient tool_use_failed
    from a schema complaint, so it must survive into Rollout.error."""
    response = httpx.Response(
        400,
        json={
            "error": {
                "message": "Failed to parse tool call arguments as JSON",
                "type": "invalid_request_error",
                "code": "tool_use_failed",
            }
        },
    )
    message = provider_message(response, key=FAKE_KEY)
    assert "Failed to parse tool call arguments as JSON" in message
    assert "code=tool_use_failed" in message
    assert "type=invalid_request_error" in message
    assert len(message) <= 300


def test_schema_rejection_is_told_apart_from_a_bad_generation():
    assert request_schema_rejected(
        "'messages.2' : property 'reasoning_content' is unsupported"
    )
    assert request_schema_rejected(
        "'messages.2.tool_calls.0.function.arguments' : value must be a string"
    )
    assert request_schema_rejected("additionalProperties:false must be set")
    # The model's own bad tool call names no request field.
    assert not request_schema_rejected("Failed to parse tool call arguments as JSON")


def test_scrub_removes_tokens_and_key_urls():
    text = "key gsk_ABCDEFGH12345678 at https://openrouter.ai/workspaces/abc/keys"
    scrubbed = scrub_secrets(text)
    assert "gsk_" not in scrubbed
    assert "openrouter.ai/workspaces" not in scrubbed


def test_scrub_removes_account_ids_from_quota_messages():
    """Groq's 429 body names the billing org; that id must not be persisted."""
    text = (
        "Rate limit reached for model `openai/gpt-oss-120b` in organization "
        "`org_01fake000000000000000000` service tier `on_demand` on tokens per "
        "day (TPD): Limit 200000, Used 197737, Requested 3904."
    )
    scrubbed = scrub_secrets(text, key=FAKE_KEY)
    assert "org_01fake000000000000000000" not in scrubbed
    assert "org_0" not in scrubbed
    assert "organization `[redacted-org]`" in scrubbed
    # The parts a human needs to act on survive.
    assert "tokens per day" in scrubbed and "Requested 3904" in scrubbed


def test_provider_message_scrubs_the_account_id_too():
    response = httpx.Response(
        429,
        json={
            "error": {
                "message": "in organization `org_01fake000000000000000000`",
                "code": "rate_limit_exceeded",
                "type": "tokens",
            }
        },
    )
    detail = provider_message(response, key=FAKE_KEY)
    assert "org_0" not in detail
    assert "rate_limit_exceeded" in detail


# ---------------------------------------------------------------- echo hygiene
def test_echo_assistant_message_drops_reasoning_fields():
    message = {
        "role": "assistant",
        "content": None,
        "reasoning": "hidden chain of thought",
        "reasoning_content": "the property Groq rejects",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ],
    }
    cleaned = echo_assistant_message(message)
    assert "reasoning" not in cleaned
    assert "reasoning_content" not in cleaned
    assert cleaned["role"] == "assistant"
    assert cleaned["tool_calls"] == message["tool_calls"]


class _StubBox:
    """A Sandbox stand-in: no container, deterministic output."""

    def start(self) -> None:  # noqa: D102
        pass

    def exec(self, command: str, timeout_s: float = 0.0) -> RunResult:  # noqa: D102
        return RunResult(exit_code=0, output="ok", duration_s=0.0, timed_out=False)

    def copy_in(self, src, dest) -> None:  # noqa: D102
        pass

    def remove(self, keep_image: bool = False) -> None:  # noqa: D102
        pass


async def test_second_request_never_echoes_reasoning(monkeypatch, make_task):
    """A real turn-2 body carries no provider reasoning back to the endpoint."""
    bodies: list[dict] = []
    turn = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        turn["n"] += 1
        if turn["n"] == 1:
            message = {
                "role": "assistant",
                "reasoning": "chain of thought",
                "reasoning_content": "cot part two",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})},
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "done", "reasoning": "final"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    _install_transport(monkeypatch, handler)
    task = make_task(task_yaml="id: t_reason\n", instruction="do the thing\n")
    runner = AgentRunner(
        task, _StubBox(), model="openai/gpt-oss-120b", api_key=FAKE_KEY, provider="groq"
    )
    rollout = await runner.run_rollout(0)

    assert rollout.status == "ok"
    assert len(bodies) == 2
    echoed = [m for m in bodies[1]["messages"] if m.get("role") == "assistant"]
    assert echoed, "the assistant turn must be echoed back"
    for message in echoed:
        assert "reasoning" not in message
        assert "reasoning_content" not in message


# ---------------------------------------------------------------- pacer + usage
def test_billable_tokens_count_reasoning_once():
    # Groq folds reasoning into completion_tokens: count it once, not twice.
    folded = billable_tokens(
        {
            "prompt_tokens": 408,
            "completion_tokens": 147,
            "completion_tokens_details": {"reasoning_tokens": 116},
        }
    )
    assert folded == (408, 147)
    # A provider that reports reasoning separately must not under-reserve.
    separate = billable_tokens(
        {
            "prompt_tokens": 100,
            "completion_tokens": 31,
            "completion_tokens_details": {"reasoning_tokens": 116},
        }
    )
    assert separate == (100, 147)
    # No reasoning block at all: plain prompt + completion.
    assert billable_tokens({"prompt_tokens": 10, "completion_tokens": 5}) == (10, 5)


# ------------------------------------------------- TPM 429 recovery
async def test_a_short_429_is_waited_out_and_the_same_turn_succeeds(
    monkeypatch, make_task
):
    """A tokens-per-minute burst must be retried on the *same* turn.

    Groq charges the *requested* tokens (prompt + max_completion_tokens) against
    its 8K/min budget, so a 429 here is a pacing problem, not a task failure: the
    turn is waited out and retried rather than recorded as an infra error.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"message": "rate limit reached", "type": "tokens"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    _install_transport(monkeypatch, handler)
    task = make_task(task_yaml="id: t_tpm\n", instruction="do the thing\n")
    runner = AgentRunner(
        task, _StubBox(), model="openai/gpt-oss-120b", api_key=FAKE_KEY, provider="groq"
    )
    executor = Executor(concurrency=1, max_retries=0, backoff_s=0.0)
    report = await executor.run([_job(task.task_id, 0, lambda: runner.run_rollout(0))])

    assert calls["n"] == 3  # two 429s, then the retried turn succeeds
    assert report.aborted is False
    rollout = report.rollouts[task.task_id][0]
    assert rollout.status == "ok"  # never an infra error
    assert rollout.counted is True


async def test_a_persistent_429_only_becomes_an_infra_error_after_five_retries(
    monkeypatch, make_task
):
    """Only after the same turn is retried 5 times may a 429 be an infra error."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"retry-after": "0"}, json={})

    _install_transport(monkeypatch, handler)
    task = make_task(task_yaml="id: t_tpm2\n", instruction="do the thing\n")
    runner = AgentRunner(
        task,
        _StubBox(),
        model="openai/gpt-oss-120b",
        api_key=FAKE_KEY,
        provider="groq",
        rate_limit_retries=5,
    )
    executor = Executor(concurrency=1, max_retries=0, backoff_s=0.0)
    report = await executor.run([_job(task.task_id, 0, lambda: runner.run_rollout(0))])

    assert calls["n"] == 6  # the initial turn plus exactly five retries
    assert report.aborted is False
    rollout = report.rollouts[task.task_id][0]
    assert rollout.status == "infra_error"
    assert rollout.counted is False


# ------------------------------------------------- tool-output truncation
def test_tool_output_is_truncated_on_a_line_boundary():
    text = "alpha\nbeta\ngamma\ndelta\n"
    out = truncate_tool_output(text, 12)
    # Cut at the last newline at or before the limit, marker on its own line.
    assert out == "alpha\nbeta" + TRUNCATION_MARKER
    assert out.endswith(TRUNCATION_MARKER)
    assert "gamma" not in out  # never a partial line / mid-token cut


def test_truncation_is_a_noop_when_short_or_disabled():
    assert truncate_tool_output("short", 100) == "short"
    assert truncate_tool_output("x" * 50, 0) == "x" * 50


def test_truncation_of_a_single_long_line_still_marks_the_cut():
    out = truncate_tool_output("x" * 50, 10)
    assert out == "x" * 10 + TRUNCATION_MARKER


# ------------------------------------------------- transient tool-call 400
#: A realistic Groq ``failed_generation``: a tool call whose arguments embed a raw
#: newline inside a heredoc (the JSON never parses), followed by an account id that
#: must be scrubbed before it is stored.
FAILED_GENERATION = (
    '{"name": "bash", "arguments": "{\\"command\\": '
    '\\"cat > t.py <<\'PY\'\\nprint(1)\\nPY\\"}"} org_0deadbeefcafe'
)


def _tool_use_failed_handler(calls: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Failed to parse tool call arguments as JSON",
                    "type": "invalid_request_error",
                    "code": "tool_use_failed",
                    "failed_generation": FAILED_GENERATION,
                }
            },
        )

    return handler


async def test_groq_tool_use_failed_400_is_a_transient_tool_call_error(monkeypatch):
    _install_transport(monkeypatch, _tool_use_failed_handler({"n": 0}))
    runner = _runner("groq")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ToolCallError):
            await runner._complete(client, [{"role": "user", "content": "hi"}])


async def test_groq_tool_use_failed_is_retried_then_counted_as_a_failed_rollout(
    monkeypatch, make_task
):
    """Retry the turn twice, then keep going: no abort, no infra error."""
    calls = {"n": 0}
    _install_transport(monkeypatch, _tool_use_failed_handler(calls))
    task = make_task(task_yaml="id: t_tool\n", instruction="do the thing\n")
    runner = AgentRunner(
        task,
        _StubBox(),
        model="openai/gpt-oss-120b",
        api_key=FAKE_KEY,
        provider="groq",
        tool_call_retries=2,
        tool_call_backoff_s=0.0,
    )
    executor = Executor(concurrency=1, max_retries=0, backoff_s=0.0)
    report = await executor.run([_job(task.task_id, 0, lambda: runner.run_rollout(0))])

    assert calls["n"] == 3  # the initial call plus exactly two retries
    assert report.aborted is False
    rollout = report.rollouts[task.task_id][0]
    assert rollout.status == "ok"  # not "infra_error"
    assert rollout.passed is False  # a failed rollout
    assert rollout.counted is True  # still in the solve-rate denominator
    assert rollout.tool_call_failed is True  # disclosed separately from infra errors
    assert "Failed to parse tool call arguments as JSON" in (rollout.error or "")
    assert "tool_use_failed" in (rollout.error or "")  # the code is surfaced
    assert FAKE_KEY not in (rollout.error or "")
    # The model's own rejected output is kept for offline diagnosis, scrubbed.
    assert rollout.failed_generation is not None
    assert "cat > t.py" in rollout.failed_generation  # the malformed shape survives
    assert "org_0" not in rollout.failed_generation  # account id removed
    assert FAKE_KEY not in rollout.failed_generation


def test_provider_failed_generation_is_scrubbed_and_capped():
    response = httpx.Response(
        400,
        json={"error": {"code": "tool_use_failed", "failed_generation": FAILED_GENERATION}},
    )
    gen = provider_failed_generation(response, key=FAKE_KEY)
    assert gen is not None
    assert "org_0" not in gen
    assert "cat > t.py" in gen

    # A too-long generation is capped at the limit, not stored whole.
    long_body = "x" * (FAILED_GENERATION_LIMIT + 250)
    capped = provider_failed_generation(
        httpx.Response(400, json={"error": {"failed_generation": long_body}})
    )
    assert capped is not None and len(capped) == FAILED_GENERATION_LIMIT

    # No generation in the body -> nothing stored.
    assert provider_failed_generation(httpx.Response(400, json={"error": {}})) is None


async def test_a_schema_400_still_aborts_the_run(monkeypatch, make_task):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "'messages.1' : property 'reasoning_content' is unsupported",
                    "type": "invalid_request_error",
                }
            },
        )

    _install_transport(monkeypatch, handler)
    task = make_task(task_yaml="id: t_schema\n", instruction="do the thing\n")
    runner = AgentRunner(
        task,
        _StubBox(),
        model="openai/gpt-oss-120b",
        api_key=FAKE_KEY,
        provider="groq",
        tool_call_backoff_s=0.0,
    )
    executor = Executor(concurrency=1, max_retries=0, backoff_s=0.0)
    jobs = [
        _job(task.task_id, index, lambda i=index: runner.run_rollout(i))
        for index in range(2)
    ]
    report = await executor.run(jobs)

    assert report.aborted is True
    assert report.abort_kind == "auth"
    assert report.skipped_jobs == 1  # nothing else was run
    scored = report.rollouts[task.task_id][0]
    assert scored.counted is False  # nothing is scored after a schema rejection
    assert FAKE_KEY not in (scored.error or "")


async def test_openrouter_400_behaviour_is_unchanged(monkeypatch):
    """Only Groq gets the transient-tool-call treatment."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": "Failed to parse tool call arguments as JSON"}}
        )

    _install_transport(monkeypatch, handler)
    runner = AgentRunner(
        task=None,  # type: ignore[arg-type]
        box=None,  # type: ignore[arg-type]
        model="openai/gpt-4o-mini",
        api_key="sk-or-TESTONLY",
        provider="openrouter",
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(NonRetryableProviderError):
            await runner._complete(client, [{"role": "user", "content": "hi"}])


# ------------------------------------------------- terminal surfacing
def test_a_rollout_error_is_printed_to_the_terminal():
    rollout = Rollout(
        index=0,
        status="ok",
        passed=False,
        error=(
            "Groq: provider rejected the request: Failed to parse tool call "
            "arguments as JSON [code=tool_use_failed type=invalid_request_error]"
        ),
    )
    calibration = build_calibration_result(
        task_id="t01_log_parser",
        rollouts=[rollout],
        n_planned=1,
        model="openai/gpt-oss-120b",
    )
    report = Report(
        config=RunConfig(model="openai/gpt-oss-120b", provider="groq"),
        tasks=[
            TaskReport(
                task_id="t01_log_parser",
                path="tasks/t01_log_parser",
                route=Route.REWRITE,
                calibration=calibration,
            )
        ],
    )
    stream = io.StringIO()
    print_report_summary(report, Console(file=stream, width=220, no_color=True))
    printed = stream.getvalue()

    assert "Failed to parse tool call arguments as JSON" in printed
    assert "code=tool_use_failed" in printed
