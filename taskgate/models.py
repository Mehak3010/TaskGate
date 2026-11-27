"""Pydantic v2 data models.

Everything that crosses a module boundary or lands in ``report.json`` is defined
here, so the JSON schema is the single source of truth for the pipeline.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

#: Closed set of causes a diagnosis may report (section 8.5 of the brief).
LIKELY_CAUSES: tuple[str, ...] = (
    "underspecified_instruction",
    "trivially_solvable",
    "test_too_strict",
    "environment_issue",
    "genuinely_hard",
    "shortcut_available",
)

#: Causes for which a rewrite is a plausible fix. Routing uses this to turn a
#: decisively-bad statistical verdict into REWRITE instead of DROP.
RECOVERABLE_CAUSES: frozenset[str] = frozenset(
    {
        "underspecified_instruction",
        "test_too_strict",
        "environment_issue",
        "shortcut_available",
    }
)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Severity(str, Enum):
    BLOCKER = "BLOCKER"
    WARN = "WARN"
    INFO = "INFO"


SEVERITY_ORDER: dict[str, int] = {"BLOCKER": 0, "WARN": 1, "INFO": 2}

#: Marker persisted wherever a dynamic check was skipped, so a report never
#: shows an empty value that a reader could mistake for a pass.
NOT_RUN = "not run"
NotRun = Literal["not run"]


class Route(str, Enum):
    SHIP = "SHIP"
    REWRITE = "REWRITE"
    DROP = "DROP"
    NEEDS_MORE_RUNS = "NEEDS-MORE-RUNS"
    AUDIT_ONLY = "AUDIT-ONLY"
    BLOCKED = "BLOCKED"


class Finding(BaseModel):
    """One audit finding. ``rule_id`` is stable so reports stay diffable."""

    rule_id: str
    severity: Severity
    file: str
    line: int | None = None
    message: str
    fix_hint: str = ""
    #: True when the finding came from the mock-mode simulator, never a real check.
    simulated: bool = False

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}" if self.line else self.file


class DemoMeta(BaseModel):
    """Demo-corpus-only metadata (``task.yaml`` -> ``demo:``).

    Real tasks never carry this block. It exists so ``taskgate demo`` can seed a
    known ground truth and simulate container findings without Docker.
    """

    true_pass_rate: float | None = None
    transcript_profile: str = "normal"
    simulate_dynamic: list[str] = Field(default_factory=list)
    expected_route: str | None = None
    expected_cause: str | None = None


class TaskSpec(BaseModel):
    """Parsed ``task.yaml``. The only thing that knows this schema is loader.py."""

    model_config = ConfigDict(extra="allow")

    id: str
    instruction: str = ""
    difficulty: str | None = None
    author: str | None = None
    timeout_s: int | None = None
    allow_network: bool = False
    expected_output_format: Literal["json", "yaml", "csv", "text"] | None = None
    solution_cmd: str | None = None
    verifier_cmd: str | None = None
    demo: DemoMeta | None = None


class Task(BaseModel):
    task_id: str
    path: Path
    spec: TaskSpec
    instruction_md: str = ""
    rel_files: list[str] = Field(default_factory=list)
    #: Set when the task directory could not be loaded; the task is then reported
    #: as BLOCKED (LOAD-ERROR) instead of aborting the whole folder.
    load_error: str | None = None

    @property
    def build_context_path(self) -> Path:
        """Docker build context: ``environment/`` for Terminal-Bench, else the task root."""
        rel = getattr(self.spec, "build_context", None)
        if isinstance(rel, str) and rel and (self.path / rel).is_dir():
            return self.path / rel
        return self.path

    @property
    def verifier_command(self) -> str:
        """Command that runs the hidden tests.

        An explicit ``verifier_cmd`` wins. A Terminal-Bench task is verified by its
        ``tests/test.sh`` entry point (mounted at /tests). Everything else defaults to pytest.
        """
        if self.spec.verifier_cmd:
            return self.spec.verifier_cmd
        entry = getattr(self.spec, "verifier_entry", None)
        if isinstance(entry, str) and entry:
            rel = entry[len("tests/"):] if entry.startswith("tests/") else entry
            return f"bash /tests/{rel}"
        return "python -m pytest -q /tests"

    def docker_build_args(self) -> tuple[Path, str] | None:
        """``(context_dir, dockerfile_relative_to_context)`` for ``images.build``, or None."""
        dockerfile = self.dockerfile_path
        if dockerfile is None:
            return None
        context = self.build_context_path
        try:
            return context, dockerfile.resolve().relative_to(context.resolve()).as_posix()
        except ValueError:  # Dockerfile outside the declared context: build from the task root
            return self.path, dockerfile.resolve().relative_to(self.path.resolve()).as_posix()

    def file(self, rel: str) -> Path:
        return self.path / rel

    def has(self, rel: str) -> bool:
        return (self.path / rel).exists()

    @property
    def instruction_path(self) -> Path:
        return self.path / "instruction.md"

    @property
    def dockerfile_path(self) -> Path | None:
        # Terminal-Bench/Harbor records the agent image's Dockerfile on the spec
        # as a task-relative path ("environment/Dockerfile"); the demo format has
        # it at the task root. Prefer the spec, then fall back to the root names.
        rel = getattr(self.spec, "dockerfile", None)
        if isinstance(rel, str) and rel and (self.path / rel).is_file():
            return self.path / rel
        for candidate in ("Dockerfile", "dockerfile", "Dockerfile.task"):
            if (self.path / candidate).is_file():
                return self.path / candidate
        return None

    @property
    def tests_dir(self) -> Path | None:
        d = self.path / "tests"
        return d if d.is_dir() else None

    @property
    def solution_dir(self) -> Path | None:
        d = self.path / "solution"
        return d if d.is_dir() else None

    @property
    def demo(self) -> DemoMeta | None:
        return self.spec.demo


class TranscriptStep(BaseModel):
    turn: int
    role: Literal["agent", "tool", "system"] = "agent"
    text: str


class Rollout(BaseModel):
    index: int
    #: Only "ok" rollouts count in the denominator. "infra_error", "timeout" and
    #: "error" are excluded; "rate_limited" is a provider daily-quota stop, not a
    #: task failure, so it is excluded too and the run is marked resumable.
    status: Literal["ok", "infra_error", "timeout", "error", "rate_limited"] = "ok"
    passed: bool = False
    turns: int = 0
    duration_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    verifier_exit_code: int | None = None
    verifier_output: str = ""
    error: str | None = None
    #: True when the rollout failed because the model never produced a parseable
    #: tool call (e.g. Groq's ``tool_use_failed``), even after the per-turn
    #: retries. Such a rollout stays counted (a task failure), unlike an infra
    #: error, but it is tracked separately so the report can disclose it.
    tool_call_failed: bool = False
    #: The model's own rejected output when a turn failed on an unparseable tool
    #: call, already scrubbed of secrets and capped at 500 chars. Kept for offline
    #: diagnosis; it is written to ``report.json`` but deliberately not rendered in
    #: ``report.html``.
    failed_generation: str | None = None
    simulated: bool = False
    transcript: list[TranscriptStep] = Field(default_factory=list)

    @property
    def counted(self) -> bool:
        return self.status == "ok"

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


class CalibrationResult(BaseModel):
    task_id: str
    status: Literal["ok", "skipped", "error"] = "ok"
    error: str | None = None
    n_planned: int = 0
    #: Rollouts that produced a verdict (excludes infra errors).
    n_valid: int = 0
    n_passed: int = 0
    n_infra_errors: int = 0
    n_timeouts: int = 0
    #: Rollouts stopped by a provider daily quota; excluded from the denominator.
    n_rate_limited: int = 0
    #: Rollouts that failed because the model never produced a parseable tool
    #: call (Groq's ``tool_use_failed``), even after the per-turn retries. These
    #: are counted as *task* failures, so they are reported separately from
    #: infrastructure errors rather than hidden inside ``n_infra_errors``.
    n_tool_call_failures: int = 0
    solve_rate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_confidence: float = 0.90
    pass_at_k: dict[str, float | None] = Field(default_factory=dict)
    route: Route | None = None
    route_reason: str = ""
    suggested_extra_runs: int | None = None
    cost_usd: float = 0.0
    duration_s: float = 0.0
    model: str | None = None
    simulated: bool = False
    rollouts: list[Rollout] = Field(default_factory=list)

    @property
    def infra_error_rate(self) -> float:
        if self.n_planned <= 0:
            return 0.0
        return self.n_infra_errors / self.n_planned

    @property
    def mean_turns(self) -> float:
        counted = [r.turns for r in self.rollouts if r.counted]
        return sum(counted) / len(counted) if counted else 0.0


class Diagnosis(BaseModel):
    task_id: str
    advisory: bool = True
    source: Literal["heuristic", "llm", "none"] = "heuristic"
    likely_cause: str | None = None
    evidence: list[str] = Field(default_factory=list)
    suggested_edit: str = ""
    #: True when the model's answer failed schema validation and was discarded.
    stripped: bool = False
    simulated: bool = False


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "mock"
    #: "openrouter" | "groq" | "openai-compatible".
    provider: str = "openrouter"
    #: Overrides the provider's default chat-completions endpoint.
    base_url: str | None = None
    #: One cheap request before any image build/rollout; skipped for mock runs.
    skip_preflight: bool = False
    rollouts: int = 20
    concurrency: int = 4
    ship_min: float = 0.10
    ship_max: float = 0.30
    #: DROP when the lower CI bound is >= this (degenerately easy task).
    drop_if_lower_bound_at_or_above: float = 0.80
    ci_confidence: float = 0.90
    mock: bool = False
    budget_usd: float | None = None
    timeout_s: int = 900
    max_turns: int = 30
    #: Provider-derived; None means "use the provider default" (see providers.py).
    max_tool_output_chars: int | None = None
    #: Tokens-per-minute budget for the pacer; None disables pacing.
    tpm: int | None = None
    #: Completion cap sent as ``max_completion_tokens``; None sends none. A
    #: reasoning model (gpt-oss) spends part of this on hidden reasoning tokens,
    #: so the cap must be generous or the visible content comes back empty.
    max_completion_tokens: int | None = None
    audit_reruns: int = 5
    out_dir: str = ""
    seed: int = 1234
    temperature: float = 0.7
    created_at: str = Field(default_factory=utcnow_iso)
    git_hash: str | None = None
    taskgate_version: str = "0.1.0"
    python_version: str = Field(default_factory=lambda: platform.python_version())
    #: Assumption knobs for the "engineer-hours saved" estimate (clearly labelled).
    manual_minutes_per_task: int = 90
    manual_minutes_per_finding: int = 10


class DynamicAuditEvidence(BaseModel):
    """Persisted evidence from the container-based dynamic audit, per task.

    Every check records an explicit value; anything the audit skipped is the
    literal ``"not run"`` rather than an empty field, so a reader can never
    mistake "we did not look" for "we looked and it was fine".
    """

    #: True when the container audit actually executed for this task.
    ran: bool = False
    #: Why it did not run (Docker unreachable, ``--no-dynamic``, ...); null when it ran.
    skipped_reason: str | None = NOT_RUN

    #: image build (D000) and whether two builds agreed (D005)
    build: Literal["ok", "failed"] | NotRun = NOT_RUN
    reproducible: Literal["yes", "no"] | NotRun = NOT_RUN

    #: reference solution runs (D001): K, how many passed, per-run durations
    reruns: int | NotRun = NOT_RUN
    rerun_passes: int | NotRun = NOT_RUN
    durations_s: list[float] | NotRun = NOT_RUN

    #: vacuity check (D002)
    vacuous: Literal["yes", "no"] | NotRun = NOT_RUN

    #: network-off result and the hidden-network-dependency verdict (D003)
    network_off_passed: Literal["yes", "no"] | NotRun = NOT_RUN
    network_dependent: Literal["yes", "no"] | NotRun = NOT_RUN

    #: mutation probe (D004): how many sabotages were tried, caught, which were
    #: tried, and which were skipped (with the reason), so the counts are explained.
    mutations_probed: int | NotRun = NOT_RUN
    mutations_caught: int | NotRun = NOT_RUN
    mutations_tried: list[str] | NotRun = NOT_RUN
    #: Empty list means "the probe ran and skipped nothing" -- distinct from
    #: "not run" (the probe did not execute at all).
    mutations_skipped: list[str] | NotRun = NOT_RUN

    #: Any exception the audit swallowed; None when it completed cleanly.
    error: str | None = None

    @classmethod
    def skipped(cls, reason: str) -> "DynamicAuditEvidence":
        """Evidence for a task whose dynamic audit did not run at all."""
        return cls(ran=False, skipped_reason=reason or NOT_RUN)

    @classmethod
    def from_info(cls, info: Any) -> "DynamicAuditEvidence":
        """Map a ``DynamicAuditInfo`` (duck-typed) into persisted evidence.

        Duck-typed with ``Any`` on purpose: ``audit.dynamic_checks`` imports this
        module, so a real import here would be circular. Any value the audit did
        not produce becomes ``"not run"`` rather than a misleading default.
        """
        reruns = int(getattr(info, "reruns", 0) or 0)
        rerun_passes = int(getattr(info, "rerun_passes", 0) or 0)
        durations = list(getattr(info, "durations_s", []) or [])
        tried = list(getattr(info, "mutations_tried", []) or [])
        skipped = list(getattr(info, "mutations_skipped", []) or [])
        reproducible = getattr(info, "reproducible", None)
        vacuous = getattr(info, "vacuous", None)
        network_dependent = getattr(info, "network_dependent", None)
        # D004 only runs once the reference solution is stable across all K runs.
        mutation_probe_ran = bool(reruns) and rerun_passes == reruns

        def _tri(value: bool | None) -> str:
            return NOT_RUN if value is None else ("yes" if value else "no")

        return cls(
            ran=True,
            skipped_reason=None,
            build="ok" if getattr(info, "build_ok", False) else "failed",
            reproducible=_tri(reproducible),
            reruns=reruns if reruns else NOT_RUN,
            rerun_passes=rerun_passes if reruns else NOT_RUN,
            durations_s=durations if durations else NOT_RUN,
            vacuous=_tri(vacuous),
            network_off_passed=(
                NOT_RUN if not reruns else ("yes" if rerun_passes == reruns else "no")
            ),
            network_dependent=_tri(network_dependent),
            mutations_probed=(
                int(getattr(info, "mutations_probed", 0) or 0)
                if mutation_probe_ran
                else NOT_RUN
            ),
            mutations_caught=(
                int(getattr(info, "mutations_caught", 0) or 0)
                if mutation_probe_ran
                else NOT_RUN
            ),
            mutations_tried=(
                (tried if tried else NOT_RUN) if mutation_probe_ran else NOT_RUN
            ),
            # When the probe ran, an empty list legitimately means "nothing was
            # skipped"; only a probe that did not run is "not run".
            mutations_skipped=(skipped if mutation_probe_ran else NOT_RUN),
            error=getattr(info, "error", None),
        )


class TaskReport(BaseModel):
    task_id: str
    path: str
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    route: Route
    route_reason: str = ""
    findings: list[Finding] = Field(default_factory=list)
    calibration: CalibrationResult | None = None
    diagnosis: Diagnosis | None = None
    dynamic_audit: DynamicAuditEvidence = Field(default_factory=DynamicAuditEvidence)
    simulated: bool = False

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.BLOCKER]

    @property
    def severity_counts(self) -> dict[str, int]:
        counts = {"BLOCKER": 0, "WARN": 0, "INFO": 0}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts

    @property
    def top_finding(self) -> Finding | None:
        ordered = sorted(self.findings, key=lambda f: (SEVERITY_ORDER[f.severity.value], f.rule_id))
        return ordered[0] if ordered else None


class Summary(BaseModel):
    n_tasks: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    n_in_window: int = 0
    total_cost_usd: float = 0.0
    total_rollouts: int = 0
    total_valid_rollouts: int = 0
    total_infra_errors: int = 0
    total_rate_limited: int = 0
    #: Rollouts that failed on an unparseable model tool call (counted as task
    #: failures, not infra errors); disclosed separately so the README can name
    #: them instead of letting them inflate the failure count silently.
    total_tool_call_failures: int = 0
    #: Total tokens the provider reported for all rollouts (input + output).
    total_tokens: int = 0
    infra_error_rate: float = 0.0
    wall_time_s: float = 0.0
    #: "completed" | "rate-limited, resume tomorrow" | "aborted: <kind>".
    run_status: str = "completed"
    engineer_minutes_saved: int = 0
    findings_by_severity: dict[str, int] = Field(default_factory=dict)


class Report(BaseModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: str = Field(default_factory=utcnow_iso)
    #: True when at least one number in this report came from the simulator.
    simulated: bool = False
    config: RunConfig
    summary: Summary = Field(default_factory=Summary)
    tasks: list[TaskReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    docker_available: bool | None = None
    dynamic_audit_ran: bool = False

    @property
    def n_audit_errors(self) -> int:
        """Tasks whose static audit crashed (an AUDIT-ERROR finding)."""
        return sum(
            1 for t in self.tasks if any(f.rule_id == "AUDIT-ERROR" for f in t.findings)
        )

    @property
    def audit_error_rate(self) -> float:
        return self.n_audit_errors / len(self.tasks) if self.tasks else 0.0
