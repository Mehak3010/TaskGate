"""End-to-end pipeline: audit -> dynamic audit -> calibrate -> diagnose -> report.

Order matters and mirrors the brief:

1. the static audit runs for every task;
2. any BLOCKER stops that task, and **no calibration spend happens for it**;
3. the dynamic audit runs in Docker when a daemon is reachable, and otherwise
   degrades to clearly-labelled SIMULATED findings;
4. calibration runs the remaining tasks concurrently through one executor;
5. only out-of-window tasks get a diagnosis, and a recoverable cause can turn a
   DROP / NEEDS-MORE-RUNS into a REWRITE.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .audit.dynamic_checks import (
    DynamicAuditInfo,
    run_dynamic_audit,
    simulated_dynamic_findings,
)
from .audit.static_checks import run_static_audit
from .diagnose.heuristics import diagnose_heuristically
from .diagnose.llm_diagnosis import api_key_from_env, diagnose_with_llm
from .models import (
    CalibrationResult,
    Diagnosis,
    DynamicAuditEvidence,
    Finding,
    Report,
    Route,
    RunConfig,
    Severity,
    Summary,
    Task,
    TaskReport,
    utcnow_iso,
)
from .runner.executor import Executor, ExecutorReport, Job
from .runner.mock_agent import MockAgent
from .runner.pacing import TokenPacer
from .runner.providers import (
    api_key_for,
    apply_defaults,
    key_env_names,
    provider_display,
)
from .runner.sandbox import Sandbox, docker_available
from .stats.calibration import (
    apply_diagnosis,
    build_calibration_result,
    route_calibration_result,
)

ProgressFn = Callable[..., None]


@dataclass
class PipelineOptions:
    audit: bool = True
    dynamic: bool = True
    calibrate: bool = True
    diagnose: bool = True
    llm_diagnosis: bool = True
    use_docker: bool = True  # False: never touch Docker; dynamic findings are simulated


@dataclass
class PipelineContext:
    """Mutable bookkeeping shared between the stages and the report."""

    docker_ok: bool = False
    dynamic_audit_ran: bool = False
    simulated: bool = False
    notes: list[str] = field(default_factory=list)
    dynamic_info: dict[str, DynamicAuditInfo] = field(default_factory=dict)


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def detect_git_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        value = out.stdout.strip()
        return value or None
    except Exception:  # noqa: BLE001 - not a repo, or no git
        return None


def _has_blocker(findings: list[Finding]) -> bool:
    return any(f.severity == Severity.BLOCKER for f in findings)


def _dynamic_evidence(
    task: Task, ctx: PipelineContext, options: PipelineOptions
) -> DynamicAuditEvidence:
    """Persist this task's dynamic-audit evidence, or an explicit "not run"."""
    info = ctx.dynamic_info.get(task.task_id)
    if info is not None:
        return DynamicAuditEvidence.from_info(info)
    if not options.dynamic:
        return DynamicAuditEvidence.skipped("dynamic audit disabled (--no-dynamic)")
    if not options.use_docker:
        return DynamicAuditEvidence.skipped(
            "demo mode: container checks not run; dynamic findings simulated"
        )
    if not ctx.docker_ok:
        return DynamicAuditEvidence.skipped(
            "docker daemon unreachable; dynamic findings simulated"
        )
    return DynamicAuditEvidence.skipped("dynamic audit did not run")


def _is_mock(config: RunConfig, docker_ok: bool) -> bool:
    if config.mock or config.model in ("mock", ""):
        return True
    # A real model without a container runtime cannot be measured; degrade.
    return not docker_ok


def _task_simulated(ctx: PipelineContext, options: PipelineOptions) -> bool:
    """True only when something in this task's row came from the simulator.

    That is: its calibration actually ran through the mock agent, or its dynamic
    checks were simulated because Docker was unavailable. A genuine audit-only
    run has neither, so it must not be labelled simulated.
    """
    return bool(ctx.simulated or (options.dynamic and not ctx.docker_ok))


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def _audit_task(task: Task, ctx: PipelineContext, options: PipelineOptions) -> list[Finding]:
    findings = run_static_audit(task) if options.audit else []
    if not options.dynamic:
        return findings
    if ctx.docker_ok:
        dyn, info = run_dynamic_audit(task, reruns=5)
        ctx.dynamic_info[task.task_id] = info
        ctx.dynamic_audit_ran = True
        findings = findings + dyn
    else:
        findings = findings + simulated_dynamic_findings(task)
    return findings


def _build_jobs(
    tasks: list[Task],
    config: RunConfig,
    ctx: PipelineContext,
    *,
    api_key: str | None = None,
    pacer: TokenPacer | None = None,
) -> list[Job]:
    jobs: list[Job] = []
    if ctx.simulated:
        for task in tasks:
            agent = MockAgent(
                task,
                seed=config.seed,
                n_rollouts=config.rollouts,
                max_turns=config.max_turns,
                model=config.model,
            )
            agent.plan()  # draw the whole schedule up front: concurrency cannot change it
            for index in range(config.rollouts):
                jobs.append(
                    Job(task_id=task.task_id, index=index, run=_bind(agent.run_rollout, index))
                )
        return jobs

    for task in tasks:
        box = Sandbox(task, rebuild=True)
        build = box.build()
        if not build.ok:
            ctx.notes.append(f"{task.task_id}: image build failed; calibration skipped")
            continue
        for index in range(config.rollouts):
            jobs.append(
                Job(
                    task_id=task.task_id,
                    index=index,
                    run=_real_job(task, index, config, api_key, pacer),
                )
            )
    return jobs


def _bind(coro_fn, index: int):
    async def runner():
        return await coro_fn(index)

    return runner


def _real_job(
    task: Task,
    index: int,
    config: RunConfig,
    api_key: str | None,
    pacer: TokenPacer | None,
):
    async def runner():
        from .runner.agent import AgentRunner  # imported lazily: needs docker + key

        key = api_key
        if not key:
            names = ", ".join(key_env_names(config.provider))
            raise RuntimeError(f"{names} is not set")
        box = Sandbox(task, rebuild=False)
        runner_obj = AgentRunner(
            task,
            box,
            model=config.model,
            api_key=key,
            provider=config.provider,
            base_url=config.base_url,
            max_turns=config.max_turns,
            max_tool_output_chars=config.max_tool_output_chars,
            temperature=config.temperature,
            timeout_s=config.timeout_s,
            seed=config.seed + index,
            pacer=pacer,
        )
        return await runner_obj.run_rollout(index)

    return runner


async def _preflight(config: RunConfig) -> tuple[bool, str]:
    """One cheap request before any image build or rollout.

    Returns ``(ok, message)``. The message names the provider and the HTTP status
    but never the key. A missing key is reported with the *variable name* only.
    """
    from .runner.agent import preflight

    display = provider_display(config.provider)
    key = api_key_for(config.provider)
    if not key:
        names = " or ".join(key_env_names(config.provider))
        return False, f"{display}: no API key found. Set {names} and retry."
    return await preflight(
        provider=config.provider,
        model=config.model,
        api_key=key,
        base_url=config.base_url,
    )


async def _diagnose(
    task: Task,
    calibration: CalibrationResult,
    findings: list[Finding],
    config: RunConfig,
    options: PipelineOptions,
) -> Diagnosis | None:
    heuristic = diagnose_heuristically(
        task.task_id, calibration, findings, max_turns=config.max_turns, ship_min=config.ship_min
    )
    if not (options.diagnose and options.llm_diagnosis):
        return heuristic
    if not api_key_from_env():
        return heuristic
    diagnosis, stripped = await diagnose_with_llm(task, calibration, findings)
    if diagnosis is None:
        if stripped:
            heuristic.stripped = True
        return heuristic
    return diagnosis


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
async def run_pipeline(
    tasks: list[Task],
    config: RunConfig,
    *,
    options: PipelineOptions | None = None,
    on_task: ProgressFn | None = None,
    on_rollout: ProgressFn | None = None,
) -> Report:
    options = options or PipelineOptions()
    started = time.perf_counter()
    apply_defaults(config)  # fill provider-derived tool-output / TPM defaults
    ctx = PipelineContext(docker_ok=bool(options.use_docker and docker_available()))
    # Calibration is simulated only when it actually runs through the mock agent
    # (mock config, or no container runtime to measure with). An audit-only run
    # has no calibration, so it stays a real run rather than a simulation.
    ctx.simulated = bool(options.calibrate and _is_mock(config, ctx.docker_ok))

    if config.git_hash is None:
        config.git_hash = detect_git_hash()

    if not options.use_docker:
        ctx.notes.append(
            "Demo mode: container checks were not run. Dynamic findings in this report "
            "are SIMULATED from demo metadata. Use `taskgate demo --docker` for real ones."
        )
    elif not ctx.docker_ok:
        ctx.notes.append(
            "Docker daemon not reachable: the dynamic audit and real rollouts are "
            "SKIPPED. Dynamic findings in this report are SIMULATED from demo metadata."
        )
    elif ctx.simulated:
        ctx.notes.append(
            "Mock mode: rollout results are SIMULATED from each task's declared true "
            "pass rate; no model was called and no container was run for calibration."
        )

    # -- stage 1 + 2: audit -------------------------------------------------
    findings_by_task: dict[str, list[Finding]] = {}
    for task in tasks:
        findings_by_task[task.task_id] = _audit_task(task, ctx, options)
        if on_task is not None:
            on_task(task, findings_by_task[task.task_id])

    # -- stage 3: calibration ----------------------------------------------
    blocked = [t for t in tasks if _has_blocker(findings_by_task[t.task_id])]
    calibratable = (
        [t for t in tasks if t not in blocked] if options.calibrate else []
    )
    report = ExecutorReport()
    if calibratable:
        # Preflight one cheap request before building any image or starting any
        # rollout, so a bad key/model fails in seconds. Mock runs never touch the
        # network and skip it entirely.
        if not ctx.simulated and not config.skip_preflight:
            ok, message = await _preflight(config)
            ctx.notes.append(f"preflight: {message}")
            if not ok:
                ctx.notes.append(
                    "preflight failed: no images were built and no rollouts were started."
                )
                calibratable = []
        elif not ctx.simulated:
            ctx.notes.append("preflight skipped (--skip-preflight).")

    if calibratable:
        api_key = None if ctx.simulated else api_key_for(config.provider)
        pacer = TokenPacer(config.tpm) if (config.tpm and not ctx.simulated) else None
        jobs = _build_jobs(calibratable, config, ctx, api_key=api_key, pacer=pacer)
        executor = Executor(
            concurrency=config.concurrency,
            budget_usd=config.budget_usd,
            rollout_timeout_s=None if ctx.simulated else config.timeout_s,
            on_rollout=on_rollout,
        )
        report = await executor.run(jobs)

    if report.abort_kind == "budget":
        ctx.notes.append(
            f"budget guard tripped at ${config.budget_usd}: "
            f"{report.skipped_jobs} rollout(s) were not run."
        )
    elif report.abort_kind == "auth":
        ctx.notes.append(
            f"run aborted (authentication/authorisation): {report.abort_reason} "
            f"{report.skipped_jobs} rollout(s) were not run and nothing was scored."
        )
    elif report.abort_kind == "quota":
        ctx.notes.append(
            "run stopped: rate-limited, resume tomorrow. The provider's daily quota "
            "is exhausted, so this is not a task failure and none of these rollouts "
            "count in the solve rate. Re-run tomorrow to finish the remaining rollouts."
        )

    # -- stage 4 + 5: route and diagnose -----------------------------------
    task_reports: list[TaskReport] = []
    for task in tasks:
        findings = findings_by_task[task.task_id]
        task_reports.append(
            await _finalise_task(task, findings, report.rollouts, config, options, ctx)
        )

    run_status = "completed"
    if report.abort_kind == "quota":
        run_status = "rate-limited, resume tomorrow"
    elif report.abort_kind == "auth":
        run_status = "aborted: auth"
    elif report.abort_kind == "budget":
        run_status = "aborted: budget"

    summary = _summarise(
        task_reports,
        config,
        wall_time_s=time.perf_counter() - started,
        run_status=run_status,
    )
    report_simulated = any(t.simulated for t in task_reports)
    if report_simulated:
        ctx.notes.append(
            "SIMULATED banner: every solve rate, cost and transcript in this report "
            "came from the mock agent and must not be read as a measurement."
        )
    ctx.notes.append(
        "Wilson score interval at 90% confidence; at N=20 a measured 20% has an "
        "interval of roughly 8-42%, so a point estimate alone is not a decision."
    )
    ctx.notes.append(
        "Audit rules are heuristics: findings are routed to a human, and TaskGate "
        "never auto-approves a task."
    )

    return Report(
        generated_at=utcnow_iso(),
        simulated=report_simulated,
        config=config,
        summary=summary,
        tasks=task_reports,
        notes=ctx.notes,
        docker_available=ctx.docker_ok if options.use_docker else None,
        dynamic_audit_ran=ctx.dynamic_audit_ran,
    )


async def _finalise_task(
    task: Task,
    findings: list[Finding],
    rollouts: dict[str, list],
    config: RunConfig,
    options: PipelineOptions,
    ctx: PipelineContext,
) -> TaskReport:
    has_blocker = _has_blocker(findings)
    dynamic_audit = _dynamic_evidence(task, ctx, options)
    task_simulated = _task_simulated(ctx, options)

    if has_blocker:
        return TaskReport(
            task_id=task.task_id,
            path=str(task.path),
            route=Route.BLOCKED,
            route_reason="blocker audit finding present; calibration was not run (no spend)",
            findings=findings,
            dynamic_audit=dynamic_audit,
            simulated=task_simulated,
        )

    if not options.calibrate:
        return TaskReport(
            task_id=task.task_id,
            path=str(task.path),
            route=Route.AUDIT_ONLY,
            route_reason="audit only: calibration was not run",
            findings=findings,
            dynamic_audit=dynamic_audit,
            simulated=task_simulated,
        )

    task_rollouts = rollouts.get(task.task_id, [])
    calibration = build_calibration_result(
        task_id=task.task_id,
        rollouts=task_rollouts,
        n_planned=config.rollouts,
        model=config.model,
        confidence=config.ci_confidence,
        simulated=ctx.simulated,
    )
    route_calibration_result(calibration, config, has_blocker=False)

    diagnosis: Diagnosis | None = None
    if calibration.route in (Route.DROP, Route.REWRITE, Route.NEEDS_MORE_RUNS):
        diagnosis = await _diagnose(task, calibration, findings, config, options)
        if diagnosis is not None:
            new_route, new_reason = apply_diagnosis(
                calibration.route,
                calibration.route_reason,
                diagnosis.likely_cause,
                diagnosis.source,
            )
            calibration.route = new_route
            calibration.route_reason = new_reason
            if new_route != Route.NEEDS_MORE_RUNS:
                calibration.suggested_extra_runs = None

    if diagnosis is not None:
        diagnosis.simulated = ctx.simulated and diagnosis.source == "heuristic"

    return TaskReport(
        task_id=task.task_id,
        path=str(task.path),
        route=calibration.route or Route.NEEDS_MORE_RUNS,
        route_reason=calibration.route_reason,
        findings=findings,
        calibration=calibration,
        diagnosis=diagnosis,
        dynamic_audit=dynamic_audit,
        simulated=task_simulated,
    )


def _summarise(
    task_reports: list[TaskReport],
    config: RunConfig,
    *,
    wall_time_s: float,
    run_status: str = "completed",
) -> Summary:
    counts: dict[str, int] = {r.value: 0 for r in Route}
    n_findings = 0
    findings_by_severity = {"BLOCKER": 0, "WARN": 0, "INFO": 0}
    n_in_window = 0
    total_cost = 0.0
    total_rollouts = 0
    total_valid = 0
    total_infra = 0
    total_rate_limited = 0
    total_tokens = 0

    for report in task_reports:
        counts[report.route.value] += 1
        n_findings += len(report.findings)
        for severity, count in report.severity_counts.items():
            findings_by_severity[severity] += count
        cal = report.calibration
        if cal is not None:
            total_cost += cal.cost_usd
            total_rollouts += cal.n_planned
            total_valid += cal.n_valid
            total_infra += cal.n_infra_errors
            total_rate_limited += cal.n_rate_limited
            total_tokens += sum(r.tokens_total for r in cal.rollouts)
            if cal.solve_rate is not None and config.ship_min <= cal.solve_rate <= config.ship_max:
                n_in_window += 1

    engineer_minutes = (
        len(task_reports) * config.manual_minutes_per_task
        + n_findings * config.manual_minutes_per_finding
    )
    return Summary(
        n_tasks=len(task_reports),
        counts=counts,
        n_in_window=n_in_window,
        total_cost_usd=round(total_cost, 6),
        total_rollouts=total_rollouts,
        total_valid_rollouts=total_valid,
        total_infra_errors=total_infra,
        total_rate_limited=total_rate_limited,
        total_tokens=total_tokens,
        infra_error_rate=round(total_infra / total_rollouts, 4) if total_rollouts else 0.0,
        wall_time_s=round(wall_time_s, 3),
        run_status=run_status,
        engineer_minutes_saved=engineer_minutes,
        findings_by_severity=findings_by_severity,
    )


def run_pipeline_sync(*args, **kwargs) -> Report:
    return asyncio.run(run_pipeline(*args, **kwargs))
