"""Minimal OpenAI-style agent loop with a single ``bash`` tool.

This is deliberately *not* a production agent scaffold: it exists so
``taskgate calibrate`` can produce a real, if approximate, solve rate. The
measured rate depends on this loop, which is why the report says so, and why the
extension point (a pluggable ``AgentAdapter``) is called out in the README.

The endpoint is provider configurable (OpenRouter by default, Groq, or any
OpenAI-compatible ``--base-url``). Only the ``Authorization`` header ever sees
the key; it is never logged, never put in an error message and never stored.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx

from ..models import Rollout, Task, TranscriptStep
from .executor import NonRetryableProviderError, RateLimitError
from .pacing import TokenPacer
from .pricing import estimate_cost_usd
from .providers import api_key_for, provider_display, resolve_endpoint
from .sandbox import Sandbox

SYSTEM_PROMPT = (
    "You are an autonomous engineer working inside a Linux container. You are given a "
    "task in the user message. Use the `bash` tool to inspect the filesystem and to "
    "create the required artefacts. Work until the task is done, then stop calling "
    "tools and reply with a one-line summary. Do not ask questions: nobody is watching. "
    "The container has no network access."
)

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command inside the container and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
}

#: Fallback truncation for tool output when a provider default is not supplied.
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 4000

#: Appended to any tool output that had to be cut, on its own line so the text
#: never ends mid-token.
TRUNCATION_MARKER = "\n[output truncated]"


def truncate_tool_output(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters on a line boundary.

    Cutting at an arbitrary index can split a token or a multi-byte character and
    makes the truncated text hard to read. The cut is instead made at the last
    newline at or before ``limit`` and :data:`TRUNCATION_MARKER` is appended on
    its own line, so the result always ends on a clean line.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    if cut <= 0:
        # A single very long line: fall back to a hard cut, still clearly marked.
        return text[:limit] + TRUNCATION_MARKER
    return text[:cut] + TRUNCATION_MARKER


#: 4xx codes that mean "do not retry": the key, account or request is wrong.
_HARD_4XX = frozenset({400, 401, 402, 403, 404, 405, 406, 422})


class AgentError(RuntimeError):
    """Transient provider failure; the executor retries and excludes it."""


class ToolCallError(AgentError):
    """The model's own tool call was unparseable, not our request.

    Groq answers HTTP 400 (``error.code`` ``tool_use_failed`` / "Failed to parse
    tool call arguments as JSON") when the *generation* produced arguments it
    could not parse. That is a transient failure of one turn, not a bad request:
    re-asking usually succeeds, so the turn is retried, and if it keeps failing
    the rollout is counted as a *failed* rollout - never an infra error, never a
    whole-run abort.

    ``failed_generation`` carries the model's raw (already scrubbed and
    truncated) output from the rejected turn, when the provider includes it, so
    the failure can be diagnosed offline without re-spending quota.
    """

    def __init__(self, message: str, *, failed_generation: str | None = None) -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


#: A 400 that quotes the request schema is the caller's fault and must abort the
#: run. Groq's message-validation failures always name the offending field, e.g.
#: "'messages.2' : property 'reasoning_content' is unsupported".
_SCHEMA_400_RE = re.compile(
    r"("
    r"messages\.\d"
    r"|is unsupported|unsupported (parameter|property|field)"
    r"|unknown (parameter|property|field)"
    r"|not supported"
    r"|must be satisfied|value must be"
    r"|json schema|additionalproperties"
    r")",
    re.IGNORECASE,
)


def request_schema_rejected(detail: str) -> bool:
    """True when a 400 body blames the request shape, not the model's output."""
    return bool(_SCHEMA_400_RE.search(detail or ""))


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap upper-ish bound on prompt tokens, for the pacer only.

    Roughly one token per four characters of the serialised messages plus a small
    constant for the tool schema. Exactness is unnecessary: the pacer is fed the
    real ``usage`` after each call and only needs a sane reservation.
    """
    chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    return chars // 4 + 64


#: Key-shaped tokens scrubbed from any text that could be persisted (a 400 body,
#: a terminal message). The live key is scrubbed by value as well.
_SECRET_RE = re.compile(r"(gsk_[A-Za-z0-9]{8,}|sk-or-[A-Za-z0-9\-]{8,}|sk-[A-Za-z0-9]{16,})")
#: Account/organisation ids that providers echo in quota messages (Groq's
#: ``org_...``). They name the billing account rather than the key, but they are
#: still identifiers we should not persist, so they are redacted alongside it.
_ACCOUNT_ID_RE = re.compile(r"org_[0-9a-z]+")
#: Provider key-management URLs: never useful in an error, and they can carry a
#: secret id, so they are removed wholesale.
_KEY_URL_RE = re.compile(
    r"https?://\S*(?:/keys?\b|/workspaces?\b|api_?key)\S*", re.IGNORECASE
)

#: Cap on the stored failed generation: enough to see the shape of the malformed
#: tool call, small enough not to bloat report.json.
FAILED_GENERATION_LIMIT = 500


def scrub_secrets(text: str, *, key: str | None = None) -> str:
    """Strip the live API key, account ids and any key-shaped token/URL.

    Anything derived from a provider response may end up in ``Rollout.error``
    (and therefore ``report.json``), so it passes through here first.
    """
    if not text:
        return ""
    out = text.replace(key, "[redacted]") if key else text
    out = _SECRET_RE.sub("[redacted]", out)
    out = _ACCOUNT_ID_RE.sub("[redacted-org]", out)
    return _KEY_URL_RE.sub("[redacted-url]", out)


def provider_message(
    response: httpx.Response, *, key: str | None = None, limit: int = 300
) -> str:
    """The provider's own error, scrubbed and truncated: message, code and type.

    A non-2xx is only diagnosable from the provider's body, so this is what makes
    a 400 actionable; it never contains the key after :func:`scrub_secrets`.
    Groq's machine-readable ``error.code`` ("tool_use_failed" vs a schema
    complaint) is what tells a transient generation failure apart from a bad
    request, so it travels alongside the human message.
    """
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - a non-JSON body is still worth surfacing
        payload = None
    message = ""
    code = etype = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
            code = error.get("code")
            etype = error.get("type")
        elif isinstance(error, str):
            message = error
        if not message:
            message = str(payload.get("message") or "")
        if code is None:
            code = payload.get("code")
    if not message:
        message = (response.text or "").strip()
    text = scrub_secrets(message, key=key)
    extras = [
        f"{name}={scrub_secrets(str(value), key=key)}"
        for name, value in (("code", code), ("type", etype))
        if value
    ]
    if extras:
        suffix = f" [{' '.join(extras)}]"
        # Keep the machine-readable code/type intact: trim the human message,
        # never the suffix, so a long body cannot hide ``code``.
        text = f"{text[: max(limit - len(suffix), 0)]}{suffix}" if text else suffix.strip()
    return text[:limit]


def provider_failed_generation(
    response: httpx.Response, *, key: str | None = None, limit: int = FAILED_GENERATION_LIMIT
) -> str | None:
    """The model's raw output from a rejected tool call, scrubbed and truncated.

    Groq's ``tool_use_failed`` body usually carries ``error.failed_generation``:
    the text the model actually produced before its arguments failed to parse.
    That is the only way to diagnose the malformed calls offline, so it is kept
    (scrubbed of secrets, capped at ``limit``) on the rollout record. Returns
    ``None`` when the provider does not include it.
    """
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - a non-JSON body simply has no generation
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    raw = None
    if isinstance(error, dict):
        raw = error.get("failed_generation")
    if raw is None:
        raw = payload.get("failed_generation")
    if not raw:
        return None
    return scrub_secrets(str(raw), key=key)[:limit]


#: Keys allowed on an assistant message echoed back to the provider. Anything
#: else the provider returned (``reasoning``, ``reasoning_content``, vendor
#: extras) is dropped: some endpoints reject unknown properties on re-send
#: (Groq 400: "property 'reasoning_content' is unsupported"), and hidden
#: reasoning must never be echoed back anyway.
_ECHO_ASSISTANT_KEYS = ("role", "content", "tool_calls", "name", "function_call", "refusal")


def echo_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """A schema-clean copy of an assistant message, safe to send back."""
    return {k: message[k] for k in _ECHO_ASSISTANT_KEYS if k in message}


def billable_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    """``(prompt, output)`` tokens the pacer should charge for this turn.

    OpenAI-style usage folds reasoning into ``completion_tokens`` (Groq's
    ``t01`` turn reports ``completion_tokens_details.reasoning_tokens`` *inside*
    it). Some OpenAI-compatible endpoints report reasoning separately instead, so
    add it only when it cannot already be inside ``completion_tokens``: that
    never double counts, and a reasoning model never under-reserves the window.
    """
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0)
    output = completion + reasoning if reasoning > completion else completion
    return prompt, output


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _is_daily_quota(response: httpx.Response, retry_after_s: float | None) -> bool:
    """A 429 is a *daily* cap when the body says so or the reset is far away."""
    text = (response.text or "").lower()
    if any(token in text for token in ("per day", "daily", "tokens per day", "tpd")):
        return True
    if "tomorrow" in text:
        return True
    # A reset an hour or more out is not a short burst: treat it as the daily cap.
    return retry_after_s is not None and retry_after_s > 3600


class AgentRunner:
    def __init__(
        self,
        task: Task,
        box: Sandbox,
        *,
        model: str,
        api_key: str,
        provider: str = "openrouter",
        base_url: str | None = None,
        max_turns: int = 30,
        max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
        max_completion_tokens: int | None = None,
        temperature: float = 0.7,
        timeout_s: float = 900.0,
        seed: int | None = None,
        pacer: TokenPacer | None = None,
        tool_call_retries: int = 2,
        tool_call_backoff_s: float = 1.0,
        rate_limit_retries: int = 5,
    ) -> None:
        self.task = task
        self.box = box
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.endpoint = resolve_endpoint(provider, base_url)
        self.max_turns = max_turns
        self.max_tool_output_chars = max_tool_output_chars
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.seed = seed
        self.pacer = pacer
        self.tool_call_retries = max(0, tool_call_retries)
        self.tool_call_backoff_s = tool_call_backoff_s
        #: How many times a short (tokens-per-minute) 429 is retried *for the
        #: same turn* before the rollout is allowed to become an infra error.
        self.rate_limit_retries = max(0, rate_limit_retries)

    def _reserved_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Tokens a request reserves against the window: prompt + completion cap.

        Groq counts the *requested* tokens (the prompt plus
        ``max_completion_tokens``) against its tokens-per-minute budget, not the
        tokens the response turns out to use, so the pacer must reserve the cap
        up front or a fast loop trips the limit even at concurrency 1.
        """
        return estimate_tokens(messages) + (self.max_completion_tokens or 0)

    async def run_rollout(self, index: int) -> Rollout:
        started = time.perf_counter()
        transcript: list[TranscriptStep] = []
        tokens_in = tokens_out = 0
        turns = 0
        self.box.start()
        try:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._user_prompt()},
            ]
            turn_error: str | None = None
            turn_failed_generation: str | None = None
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                while turns < self.max_turns:
                    turns += 1
                    if self.pacer is not None:
                        await self.pacer.acquire(self._reserved_tokens(messages))
                    try:
                        payload = await self._complete_turn(client, messages)
                    except ToolCallError as exc:
                        # The model never produced a parseable tool call, even
                        # after the retries. That is a failed rollout, not an
                        # infra error and not a reason to abort the whole run.
                        turn_error = str(exc)
                        turn_failed_generation = exc.failed_generation
                        break
                    choice = self._choice(payload)
                    usage = payload.get("usage") or {}
                    turn_in, turn_pacer_out = billable_tokens(usage)
                    tokens_in += turn_in
                    tokens_out += int(usage.get("completion_tokens") or 0)
                    if self.pacer is not None:
                        # Charge prompt + output with reasoning counted in, so a
                        # reasoning model's hidden tokens never slip the window.
                        self.pacer.record(turn_in + turn_pacer_out)

                    calls = choice["message"].get("tool_calls") or []
                    text = choice["message"].get("content") or ""
                    if text:
                        transcript.append(TranscriptStep(turn=turns, role="agent", text=text[:1000]))
                    if not calls:
                        break
                    # Echo only schema-legal keys: never send the provider's
                    # hidden ``reasoning``/``reasoning_content`` back to it.
                    messages.append(echo_assistant_message(choice["message"]))
                    for call in calls:
                        command = self._tool_command(call)
                        transcript.append(
                            TranscriptStep(turn=turns, role="agent", text=f"$ {command[:400]}")
                        )
                        result = self.box.exec(command, timeout_s=self.timeout_s)
                        output = truncate_tool_output(result.output, self.max_tool_output_chars)
                        transcript.append(
                            TranscriptStep(
                                turn=turns,
                                role="tool",
                                text=f"(exit {result.exit_code})\n{output}",
                            )
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id", ""),
                                "content": output or "(no output)",
                            }
                        )

            if turn_error is not None:
                # Counted as a *failed* rollout (status "ok", passed False) so it
                # stays in the solve-rate denominator instead of being excused as
                # infrastructure.
                return Rollout(
                    index=index,
                    status="ok",
                    passed=False,
                    turns=turns,
                    duration_s=round(time.perf_counter() - started, 3),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=estimate_cost_usd(
                        self.model, tokens_in, tokens_out, provider=self.provider
                    ),
                    error=turn_error,
                    tool_call_failed=True,
                    failed_generation=turn_failed_generation,
                    transcript=transcript,
                )

            # The verifier is mounted only now, once the agent has stopped. The
            # reference solution is never mounted into an agent rollout.
            if self.task.tests_dir is not None:
                self.box.copy_in(self.task.tests_dir, "/tests")
            verifier = self.task.spec.verifier_cmd or "python -m pytest -q /tests"
            verdict = self.box.exec(verifier, timeout_s=max(120.0, self.timeout_s / 4))
            return Rollout(
                index=index,
                status="ok",
                passed=verdict.exit_code == 0,
                turns=turns,
                duration_s=round(time.perf_counter() - started, 3),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=estimate_cost_usd(
                    self.model, tokens_in, tokens_out, provider=self.provider
                ),
                verifier_exit_code=verdict.exit_code,
                verifier_output=verdict.output[:4000],
                transcript=transcript,
            )
        finally:
            self.box.remove(keep_image=True)

    # -- helpers -----------------------------------------------------------
    def _400_is_transient(self, detail: str) -> bool:
        """Is a 400 the model's malformed generation rather than our request?

        Only Groq is treated this way: it 400s on a tool call its own model
        emitted with unparseable arguments. Everything else keeps the previous
        behaviour, so the OpenRouter path is byte-for-byte unchanged. A 400 that
        names the request schema still aborts the run.
        """
        if self.provider != "groq":
            return False
        return not request_schema_rejected(detail)

    async def _complete_turn(
        self, client: httpx.AsyncClient, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """``_complete`` with the two per-turn retries this loop needs.

        * a malformed tool call (Groq's ``tool_use_failed`` 400) is the model's
          own generation, so the turn is re-asked ``tool_call_retries`` times;
        * a short tokens-per-minute 429 is waited out and the *same turn* is
          retried up to ``rate_limit_retries`` times, so a burst against the free
          tier never turns into an infra error on the first hit.
        """
        attempts = max(1, self.tool_call_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                return await self._complete_with_rate_limit_retry(client, messages)
            except ToolCallError:
                if attempt >= attempts:
                    raise
                await asyncio.sleep(self.tool_call_backoff_s * (2 ** (attempt - 1)))
        raise AssertionError("unreachable")  # pragma: no cover

    async def _complete_with_rate_limit_retry(
        self, client: httpx.AsyncClient, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Retry a short (tokens-per-minute) 429 for the *same turn*.

        The daily-quota 429 is not retried here: it stops the whole run cleanly.
        Only after ``rate_limit_retries`` failed attempts is the error allowed to
        propagate, where the executor may treat it as an infra error.
        """
        reserve = self._reserved_tokens(messages)
        attempts = max(1, self.rate_limit_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                return await self._complete(client, messages)
            except RateLimitError as exc:
                if exc.daily_quota or attempt >= attempts:
                    raise
                await self._wait_out_rate_limit(exc, reserve, attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _wait_out_rate_limit(
        self, exc: RateLimitError, reserve: int, attempt: int
    ) -> None:
        """Sleep until a tokens-per-minute window has cleared before retrying.

        ``Retry-After`` is honoured when the provider sends it; otherwise the
        pacer says exactly how long the window needs to drain. With no pacer
        (e.g. the limit is request-based) a bounded exponential backoff applies.
        """
        if exc.retry_after_s is not None:
            await asyncio.sleep(max(0.0, exc.retry_after_s))
            return
        if self.pacer is not None:
            wait = self.pacer.time_until_room(reserve)
            if wait > 0:
                await asyncio.sleep(wait)
                return
        await asyncio.sleep(min(30.0, 2.0 ** attempt))

    def _user_prompt(self) -> str:
        instruction = self.task.instruction_md.strip() or self.task.spec.instruction
        return f"# Task: {self.task.task_id}\n\n{instruction}\n"

    async def _complete(self, client: httpx.AsyncClient, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [BASH_TOOL],
            "tool_choice": "auto",
            "temperature": self.temperature,
        }
        if self.max_completion_tokens is not None:
            body["max_completion_tokens"] = self.max_completion_tokens
        if self.seed is not None:
            body["seed"] = self.seed
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = await client.post(self.endpoint, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise AgentError(f"{provider_display(self.provider)}: transport error") from exc

        status = response.status_code
        display = provider_display(self.provider)
        detail = provider_message(response, key=self.api_key)
        tail = f" {detail}" if detail else ""
        if status == 429:
            retry_after = _retry_after_seconds(response)
            daily = _is_daily_quota(response, retry_after)
            if daily:
                raise RateLimitError(
                    f"{display}: daily quota exhausted (HTTP 429). "
                    f"rate-limited, resume tomorrow.{tail}",
                    provider=self.provider,
                    retry_after_s=retry_after,
                    daily_quota=True,
                )
            raise RateLimitError(
                f"{display}: rate limited (HTTP 429); will retry.{tail}",
                provider=self.provider,
                retry_after_s=retry_after,
            )
        if status in (401, 403):
            # Only an auth failure may tell the user to check the key.
            raise NonRetryableProviderError(
                f"{display}: authentication failed (HTTP {status}). Check the API key "
                f"and the account's access to model '{self.model}'.{tail}",
                provider=self.provider,
                status_code=status,
            )
        if status == 400 and self._400_is_transient(detail):
            # Groq 400s when the model's *own* tool call could not be parsed: a
            # transient failure of this turn, not a bad request, so it must not
            # abort the run. Keep the raw rejected generation for offline
            # diagnosis.
            raise ToolCallError(
                f"{display}: provider rejected the request: {detail or f'HTTP {status}'}",
                failed_generation=provider_failed_generation(response, key=self.api_key),
            )
        if status in _HARD_4XX:
            # A 400/402/404/... is a bad request, not necessarily a bad key: the
            # provider's own message is the only thing that makes it diagnosable.
            raise NonRetryableProviderError(
                f"{display}: provider rejected the request: {detail or f'HTTP {status}'}",
                provider=self.provider,
                status_code=status,
            )
        if status >= 500:
            raise AgentError(f"{display}: server error (HTTP {status}){tail}")
        if status >= 400:
            raise AgentError(f"{display}: provider rejected the request: {detail or f'HTTP {status}'}")
        return response.json()

    @staticmethod
    def _choice(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return payload["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError(f"malformed provider response: {str(payload)[:200]}") from exc

    @staticmethod
    def _tool_command(call: dict[str, Any]) -> str:
        try:
            arguments = call["function"]["arguments"]
            data = json.loads(arguments) if isinstance(arguments, str) else arguments
            return str(data.get("command", ""))
        except Exception:  # noqa: BLE001 - a malformed tool call is just an empty command
            return ""


async def preflight(
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    timeout_s: float = 30.0,
) -> tuple[bool, str]:
    """One cheap request before any image build or rollout.

    Returns ``(ok, message)``. ``ok=False`` means the key, base URL or model is
    bad and the run must not start. The message names the provider and the HTTP
    status but never the key.
    """
    display = provider_display(provider)
    try:
        endpoint = resolve_endpoint(provider, base_url)
    except ValueError as exc:
        return False, f"{display}: {exc}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(endpoint, json=body, headers=headers)
    except httpx.HTTPError:
        return False, (
            f"{display}: preflight request failed (network/transport). "
            "Check the base URL and connectivity."
        )
    status = response.status_code
    detail = provider_message(response, key=api_key)
    tail = f" {detail}" if detail else ""
    if status == 429:
        return True, f"{display}: preflight hit a rate limit (HTTP 429); proceeding."
    if status in (401, 403):
        return False, (
            f"{display}: preflight failed (HTTP {status}). Check the API key and that "
            f"model '{model}' is available.{tail}"
        )
    if status >= 400:
        return False, (
            f"{display}: preflight rejected the request: {detail or f'HTTP {status}'}"
        )
    return True, f"{display}: preflight ok."


def resolve_api_key(provider: str) -> str | None:
    """Provider-aware key lookup (kept here so callers never touch the env list)."""
    return api_key_for(provider)
