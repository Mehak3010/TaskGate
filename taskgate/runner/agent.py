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

import json
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

#: 4xx codes that mean "do not retry": the key, account or request is wrong.
_HARD_4XX = frozenset({400, 401, 402, 403, 404, 405, 406, 422})


class AgentError(RuntimeError):
    """Transient provider failure; the executor retries and excludes it."""


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap upper-ish bound on prompt tokens, for the pacer only.

    Roughly one token per four characters of the serialised messages plus a small
    constant for the tool schema. Exactness is unnecessary: the pacer is fed the
    real ``usage`` after each call and only needs a sane reservation.
    """
    chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    return chars // 4 + 64


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
        temperature: float = 0.7,
        timeout_s: float = 900.0,
        seed: int | None = None,
        pacer: TokenPacer | None = None,
    ) -> None:
        self.task = task
        self.box = box
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.endpoint = resolve_endpoint(provider, base_url)
        self.max_turns = max_turns
        self.max_tool_output_chars = max_tool_output_chars
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.seed = seed
        self.pacer = pacer

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
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                while turns < self.max_turns:
                    turns += 1
                    if self.pacer is not None:
                        await self.pacer.acquire(estimate_tokens(messages))
                    payload = await self._complete(client, messages)
                    choice = self._choice(payload)
                    usage = payload.get("usage") or {}
                    turn_in = int(usage.get("prompt_tokens") or 0)
                    turn_out = int(usage.get("completion_tokens") or 0)
                    tokens_in += turn_in
                    tokens_out += turn_out
                    if self.pacer is not None:
                        self.pacer.record(turn_in + turn_out)

                    calls = choice["message"].get("tool_calls") or []
                    text = choice["message"].get("content") or ""
                    if text:
                        transcript.append(TranscriptStep(turn=turns, role="agent", text=text[:1000]))
                    if not calls:
                        break
                    messages.append(choice["message"])
                    for call in calls:
                        command = self._tool_command(call)
                        transcript.append(
                            TranscriptStep(turn=turns, role="agent", text=f"$ {command[:400]}")
                        )
                        result = self.box.exec(command, timeout_s=self.timeout_s)
                        output = result.output[: self.max_tool_output_chars]
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
        if self.seed is not None:
            body["seed"] = self.seed
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = await client.post(self.endpoint, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise AgentError(f"{provider_display(self.provider)}: transport error") from exc

        status = response.status_code
        if status == 429:
            retry_after = _retry_after_seconds(response)
            daily = _is_daily_quota(response, retry_after)
            if daily:
                raise RateLimitError(
                    f"{provider_display(self.provider)}: daily quota exhausted (HTTP 429). "
                    "rate-limited, resume tomorrow.",
                    provider=self.provider,
                    retry_after_s=retry_after,
                    daily_quota=True,
                )
            raise RateLimitError(
                f"{provider_display(self.provider)}: rate limited (HTTP 429); will retry.",
                provider=self.provider,
                retry_after_s=retry_after,
            )
        if status in _HARD_4XX:
            raise NonRetryableProviderError(
                f"{provider_display(self.provider)}: HTTP {status}. Check the API key, the "
                f"account's access and that model '{self.model}' is available. No rollouts were scored.",
                provider=self.provider,
                status_code=status,
            )
        if status >= 500:
            raise AgentError(f"{provider_display(self.provider)}: server error (HTTP {status})")
        if status >= 400:
            raise AgentError(f"{provider_display(self.provider)}: HTTP {status}")
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
    if status == 429:
        return True, f"{display}: preflight hit a rate limit (HTTP 429); proceeding."
    if status >= 400:
        return False, (
            f"{display}: preflight returned HTTP {status}. Check the API key and that "
            f"model '{model}' is available."
        )
    return True, f"{display}: preflight ok."


def resolve_api_key(provider: str) -> str | None:
    """Provider-aware key lookup (kept here so callers never touch the env list)."""
    return api_key_for(provider)
