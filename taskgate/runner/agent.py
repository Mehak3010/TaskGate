"""Minimal OpenRouter agent loop with a single ``bash`` tool.

This is deliberately *not* a production agent scaffold: it exists so
``taskgate calibrate`` can produce a real, if approximate, solve rate. The
measured rate depends on this loop, which is why the report says so, and why the
extension point (a pluggable ``AgentAdapter``) is called out in the README.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ..models import Rollout, Task, TranscriptStep
from .pricing import estimate_cost_usd
from .sandbox import Sandbox

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

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

#: Output is truncated before it goes back to the model (and into the report).
MAX_TOOL_OUTPUT_CHARS = 4000


class AgentError(RuntimeError):
    """Transient provider failure; the executor retries and excludes it."""


class AgentRunner:
    def __init__(
        self,
        task: Task,
        box: Sandbox,
        *,
        model: str,
        api_key: str,
        max_turns: int = 30,
        temperature: float = 0.7,
        timeout_s: float = 900.0,
        seed: int | None = None,
    ) -> None:
        self.task = task
        self.box = box
        self.model = model
        self.api_key = api_key
        self.max_turns = max_turns
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.seed = seed

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
                    payload = await self._complete(client, messages)
                    choice = self._choice(payload)
                    usage = payload.get("usage") or {}
                    tokens_in += int(usage.get("prompt_tokens") or 0)
                    tokens_out += int(usage.get("completion_tokens") or 0)

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
                        output = result.output[:MAX_TOOL_OUTPUT_CHARS]
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
                cost_usd=estimate_cost_usd(self.model, tokens_in, tokens_out),
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
        for attempt in range(3):
            try:
                response = await client.post(OPENROUTER_URL, json=body, headers=headers)
            except httpx.HTTPError as exc:
                raise AgentError(f"transport error: {exc}") from exc
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == 2:
                    raise AgentError(f"provider returned {response.status_code}")
                time.sleep(0.5 * (2 ** attempt))
                continue
            if response.status_code >= 400:
                raise AgentError(f"provider returned {response.status_code}: {response.text[:200]}")
            return response.json()
        raise AgentError("provider unavailable")

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
