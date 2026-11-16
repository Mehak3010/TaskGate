"""Optional LLM-assisted diagnosis (OpenRouter).

Unused without an API key: the pipeline silently falls back to the heuristics.
The answer is schema-validated against the closed cause set and **stripped** if
it does not validate -- a plausible-sounding but unvalidated diagnosis is worse
than none, because a human is going to act on it.

The API key is read from the environment and never written to a log or a report.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from ..models import (
    LIKELY_CAUSES,
    CalibrationResult,
    Diagnosis,
    Finding,
    Rollout,
    Task,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = (
    "You audit benchmark tasks for an RL data company. You are given one task, its "
    "hidden verifier summary and digests of model rollouts. Decide why the solve rate "
    "is out of band.\n"
    "Answer with JSON only, no prose, no markdown fences:\n"
    '{"likely_cause": "<one of: '
    + ", ".join(LIKELY_CAUSES)
    + '>", "evidence": ["<short factual bullet>", "..."], "suggested_edit": "<one or two sentences>"}\n'
    "Rules: `evidence` must cite something present in the input. Do not invent test names "
    "or files. Never suggest deleting a task when a rewrite is plausible."
)


def _step_digest(rollout: Rollout, max_steps: int = 5) -> str:
    lines: list[str] = []
    for step in rollout.transcript[:max_steps]:
        text = " ".join(step.text.split())
        lines.append(f"  [{step.role}] {text[:180]}")
    if rollout.verifier_output:
        tail = " / ".join(
            line for line in rollout.verifier_output.splitlines() if line.strip()
        )[-220:]
        lines.append(f"  [verifier] {tail}")
    return "\n".join(lines)


def _prompt(task: Task, calibration: CalibrationResult, findings: list[Finding]) -> str:
    blockers = [f"{f.rule_id} {f.severity.value} {f.file}:{f.line or '-'} {f.message}" for f in findings[:8]]
    passing = [r for r in calibration.rollouts if r.counted and r.passed][:3]
    failing = [r for r in calibration.rollouts if r.counted and not r.passed][:3]
    parts = [
        f"task_id: {task.task_id}",
        f"instruction:\n{task.instruction_md.strip()[:1500]}",
        f"declared output format: {task.spec.expected_output_format or 'unspecified'}",
        (
            "measurement: "
            f"{calibration.n_passed}/{calibration.n_valid} rollouts passed "
            f"(rate {calibration.solve_rate}, Wilson {calibration.ci_confidence:.0%} CI "
            f"[{calibration.ci_low:.2f}, {calibration.ci_high:.2f}]), "
            f"mean turns {calibration.mean_turns:.1f}"
        ),
        "audit findings:\n" + ("\n".join(f"- {b}" for b in blockers) if blockers else "- none"),
        f"verifier: {task.spec.verifier_cmd or 'pytest on tests/'}",
        f"passing rollout digests ({len(passing)}):",
    ]
    for rollout in passing:
        parts.append(f"- rollout #{rollout.index}\n{_step_digest(rollout)}")
    parts.append(f"failing rollout digests ({len(failing)}):")
    for rollout in failing:
        parts.append(f"- rollout #{rollout.index}\n{_step_digest(rollout)}")
    return "\n".join(parts)


def parse_diagnosis_payload(payload: Any, task_id: str) -> Diagnosis | None:
    """Validate a decoded JSON payload against the closed cause set."""
    if not isinstance(payload, dict):
        return None
    cause = payload.get("likely_cause")
    if not isinstance(cause, str) or cause not in LIKELY_CAUSES:
        return None
    evidence = payload.get("evidence")
    if isinstance(evidence, list):
        evidence = [str(item) for item in evidence if isinstance(item, (str, int, float))][:6]
    else:
        evidence = []
    edit = payload.get("suggested_edit")
    if not isinstance(edit, str):
        edit = ""
    return Diagnosis(
        task_id=task_id,
        source="llm",
        likely_cause=cause,
        evidence=evidence,
        suggested_edit=edit.strip()[:600],
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    """LLMs like markdown fences and trailing prose; take the first JSON object."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def api_key_from_env() -> str | None:
    for name in ("OPENROUTER_API_KEY", "TASKGATE_OPENROUTER_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


async def diagnose_with_llm(
    task: Task,
    calibration: CalibrationResult,
    findings: list[Finding],
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout_s: float = 60.0,
) -> tuple[Diagnosis | None, bool]:
    """Returns ``(diagnosis, was_stripped)``.

    ``diagnosis`` is None when there was no usable answer (caller keeps the
    heuristics); ``was_stripped`` is True when the model answered but the answer
    failed schema validation.
    """
    key = api_key or api_key_from_env()
    if not key:
        return None, False
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt(task, calibration, findings)},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(OPENROUTER_URL, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception:  # noqa: BLE001 - diagnosis is best effort
        return None, False

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, False
    payload = extract_json_object(text if isinstance(text, str) else "")
    if payload is None:
        return None, True
    diagnosis = parse_diagnosis_payload(payload, task.task_id)
    if diagnosis is None:
        return None, True
    return diagnosis, False
