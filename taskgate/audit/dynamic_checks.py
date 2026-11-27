"""Dynamic audit: checks that need a container.

Implemented rules (all best-effort; a Docker failure degrades to a notice, never
a crash):

* D000  image fails to build
* D001  reference solution is flaky or broken across K fresh runs
* D002  tests pass on an unmodified environment (vacuous task)
* D003  hidden network dependency (passes with networking, fails without)
* D004  tests are too weak to detect a trivial sabotage of the solution
* D005  two builds of the same Dockerfile install different package sets

In mock mode the container work is replaced by
:func:`simulated_dynamic_findings`, which turns the demo metadata into clearly
labelled SIMULATED findings so the story still holds without Docker.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Finding, Task
from .rules import rule as get_rule
from .static_checks import make_finding
from ..runner.sandbox import Sandbox

#: A reference run slower than this multiple of the median is flagged as unstable.
RUNTIME_VARIANCE_FACTOR = 3.0
DEFAULT_VERIFIER = "python -m pytest -q /tests"
TEST_MOUNT = "/tests"
SOLUTION_MOUNT = "/solution"


@dataclass
class DynamicAuditInfo:
    """What the dynamic audit found, for the report's methodology section."""

    build_ok: bool = False
    reproducible: bool | None = None
    reruns: int = 0
    rerun_passes: int = 0
    vacuous: bool | None = None
    network_dependent: bool | None = None
    mutations_probed: int = 0
    mutations_caught: int = 0
    mutations_tried: list[str] = field(default_factory=list)
    mutations_skipped: list[str] = field(default_factory=list)
    durations_s: list[float] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _verifier_cmd(task: Task) -> str:
    return task.verifier_command


def _solution_cmd(task: Task) -> str | None:
    if task.spec.solution_cmd:
        return task.spec.solution_cmd
    solution = task.solution_dir
    if solution is None:
        return None
    solve = solution / "solve.sh"
    if solve.is_file():
        return f"bash {SOLUTION_MOUNT}/solve.sh"
    return None


def _prepare_verifier(box: Sandbox) -> None:
    if box.task.tests_dir is not None:
        box.copy_in(box.task.tests_dir, TEST_MOUNT)


def _run_reference_once(
    box: Sandbox, timeout_s: float, *, network_disabled: bool = True
) -> tuple[bool, float]:
    """Fresh container -> solution -> mount tests -> verifier.

    Returns ``(passed, verifier_duration_s)``.
    """
    box.spec.network_disabled = network_disabled
    box.start()
    try:
        if box.task.solution_dir is not None:
            box.copy_in(box.task.solution_dir, SOLUTION_MOUNT)
        solution = _solution_cmd(box.task)
        if solution:
            box.exec(solution, timeout_s=timeout_s)
        _prepare_verifier(box)
        result = box.exec(_verifier_cmd(box.task), timeout_s=timeout_s)
        return result.exit_code == 0, result.duration_s
    finally:
        box.remove(keep_image=True)


# --------------------------------------------------------------------------
# mutations (D004)
# --------------------------------------------------------------------------
def _candidate_scripts(solution_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(solution_dir.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".sh", ".bash"):
            out.append(path)
    return out


def _mutation_delete_line(text: str) -> str | None:
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith(("#", "import ", "from ", "set ")):
            return "\n".join(lines[:index] + lines[index + 1:])
    return None


def _mutation_bump_constant(text: str) -> str | None:
    match = re.search(r"(?<![\w.])(\d{2,})(?![\w.])", text)
    if not match:
        return None
    bumped = str(int(match.group(1)) + 1)
    start, end = match.span(1)
    return text[:start] + bumped + text[end:]


def _mutation_stub(text: str) -> str:
    # "truncate output": produce nothing at all and exit cleanly.
    return "#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n"


def _mutation_candidates(text: str) -> list[tuple[str, str, str]]:
    """Return ``(name, mutated_text, skip_reason)`` for each mutation we could try.

    When the mutation applies, ``mutated_text`` holds the sabotaged script and
    ``skip_reason`` is empty. When it does not apply, ``mutated_text`` is empty
    and ``skip_reason`` explains why, so the probe count is never unexplained.
    Pure and Docker-free, which makes the reporting contract unit-testable.
    """
    plan: list[tuple[str, str, str]] = []

    delete_line = _mutation_delete_line(text)
    if delete_line and delete_line != text:
        plan.append(("delete_line", delete_line, ""))
    else:
        plan.append(("delete_line", "", "no removable line found in solution"))

    change_constant = _mutation_bump_constant(text)
    if change_constant and change_constant != text:
        plan.append(("change_constant", change_constant, ""))
    else:
        plan.append(("change_constant", "", "no constant found in solution"))

    plan.append(("stub", _mutation_stub(text), ""))
    return plan


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------
def run_dynamic_audit(
    task: Task,
    *,
    reruns: int = 5,
    mutation_probes: int = 3,
    timeout_s: float = 600.0,
    rebuild: bool = True,
) -> tuple[list[Finding], DynamicAuditInfo]:
    """Run the container checks for one task. Returns ``(findings, info)``."""
    findings: list[Finding] = []
    info = DynamicAuditInfo()
    solution = _solution_cmd(task)
    if solution is None:
        info.error = "no reference solution declared (no solution_cmd and no solution/solve.sh)"
        return findings, info

    box = Sandbox(task, rebuild=rebuild)
    try:
        build = box.build()
        if not build.ok:
            info.error = build.output[:500]
            findings.append(
                make_finding("D000", str(task.dockerfile_path or "Dockerfile"), None,
                             f"image build failed: {build.output.strip()[:300]}")
            )
            return findings, info
        info.build_ok = True

        # D005 -- reproducibility: two builds must install the same package set.
        if rebuild:
            first = box.installed_packages()
            box2 = Sandbox(task, rebuild=True)
            second = box2.installed_packages()
            box2.remove(keep_image=True)
            info.reproducible = bool(first) and first == second
            if first and not info.reproducible:
                only_a = sorted(first - second)[:5]
                only_b = sorted(second - first)[:5]
                findings.append(
                    make_finding(
                        "D005",
                        str(task.dockerfile_path or "Dockerfile"),
                        None,
                        "two builds produced different package sets "
                        f"(only in build A: {', '.join(only_a) or '-'}; "
                        f"only in build B: {', '.join(only_b) or '-'}).",
                    )
                )

        # D002 -- vacuity: the verifier must fail before the solution runs.
        box2 = Sandbox(task, rebuild=False)
        box2.build()
        box2.start()
        try:
            _prepare_verifier(box2)
            pre = box2.exec(_verifier_cmd(task), timeout_s=timeout_s)
            info.vacuous = pre.exit_code == 0
        finally:
            box2.remove(keep_image=True)
        if info.vacuous:
            findings.append(
                make_finding("D002", TEST_MOUNT, None,
                             "the hidden verifier passed on an untouched environment: "
                             "the task can be solved without doing any work.")
            )

        # D001 -- reference solution stability across K fresh runs.
        runs: list[tuple[bool, float]] = []
        for _ in range(max(1, reruns)):
            runs.append(_run_reference_once(box, timeout_s))
        info.reruns = len(runs)
        info.rerun_passes = sum(1 for passed, _ in runs if passed)
        info.durations_s = [round(duration, 3) for _, duration in runs]
        if info.rerun_passes < info.reruns:
            # D003 -- does enabling networking rescue it? then it is a hidden net dep.
            network_ok, _ = _run_reference_once(box, timeout_s, network_disabled=False)
            if network_ok:
                info.network_dependent = True
                findings.append(
                    make_finding("D003", TEST_MOUNT, None,
                                 "the reference solution only passes with networking enabled: "
                                 "the task has a hidden network dependency.")
                )
            else:
                findings.append(
                    make_finding(
                        "D001",
                        SOLUTION_MOUNT,
                        None,
                        f"reference solution passed {info.rerun_passes}/{info.reruns} fresh runs: "
                        "flaky or broken.",
                    )
                )

        # D004 -- mutation probe: sabotage the solution, tests must notice.
        if info.rerun_passes == info.reruns and info.reruns:
            (
                info.mutations_probed,
                info.mutations_caught,
                info.mutations_tried,
                info.mutations_skipped,
            ) = _mutation_probe(box, task, mutation_probes, timeout_s)
            if info.mutations_caught < info.mutations_probed:
                findings.append(
                    make_finding(
                        "D004",
                        TEST_MOUNT,
                        None,
                        f"only {info.mutations_caught}/{info.mutations_probed} trivial sabotages of "
                        "the solution were detected by the tests.",
                    )
                )
    except Exception as exc:  # noqa: BLE001 - dynamic audit is best effort
        info.error = f"{type(exc).__name__}: {exc}"
    finally:
        box.remove(keep_image=True)
    return findings, info


def _mutation_probe(
    box: Sandbox, task: Task, probes: int, timeout_s: float
) -> tuple[int, int, list[str], list[str]]:
    """Sabotage the solution and check that the verifier notices.

    Returns ``(probed, caught, tried, skipped)``. ``skipped`` records every
    mutation that could not be applied together with the reason, so the probe
    count is always explained rather than silently short.
    """
    if task.solution_dir is None:
        return 0, 0, [], ["no solution directory to mutate"]
    scripts = _candidate_scripts(task.solution_dir)
    if not scripts:
        return 0, 0, [], ["no .py/.sh script in the solution to mutate"]
    primary = max(scripts, key=lambda p: p.stat().st_size)
    original = primary.read_text(encoding="utf-8", errors="replace")

    import tempfile

    probed = 0
    caught = 0
    tried: list[str] = []
    skipped: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, mutated, skip_reason in _mutation_candidates(original):
            if probed >= max(1, probes):
                break
            if not mutated or mutated == original:
                skipped.append(f"{name}: {skip_reason}")
                continue
            target = Path(tmp) / f"mutation_{probed}_{primary.name}"
            target.write_text(mutated, encoding="utf-8", newline="\n")
            target.chmod(0o755)
            tried.append(name)
            probed += 1
            if _mutation_is_caught(box, task, target, primary.name, timeout_s):
                caught += 1
    return probed, caught, tried, skipped


def _mutation_is_caught(
    box: Sandbox, task: Task, mutated_script: Path, script_name: str, timeout_s: float
) -> bool:
    """True when the verifier fails after the mutated solution runs."""
    box.start()
    try:
        if task.solution_dir is not None:
            box.copy_in(task.solution_dir, SOLUTION_MOUNT)
        box.copy_in(mutated_script, f"{SOLUTION_MOUNT}/{script_name}")
        cmd = _solution_cmd(task)
        if cmd:
            box.exec(cmd, timeout_s=timeout_s)
        _prepare_verifier(box)
        result = box.exec(_verifier_cmd(task), timeout_s=timeout_s)
        return result.exit_code != 0
    finally:
        box.remove(keep_image=True)


# --------------------------------------------------------------------------
# simulated findings (mock mode)
# --------------------------------------------------------------------------
def simulated_dynamic_findings(task: Task) -> list[Finding]:
    """Turn ``task.yaml`` demo metadata into labelled SIMULATED findings."""
    demo = task.demo
    if demo is None or not demo.simulate_dynamic:
        return []
    findings: list[Finding] = []
    for rule_id in demo.simulate_dynamic:
        try:
            r = get_rule(rule_id)
        except KeyError:
            continue
        findings.append(
            Finding(
                rule_id=rule_id,
                severity=r.severity,
                file="(simulated container run)",
                line=None,
                message=f"{r.title} — simulated in mock mode (no container was run).",
                fix_hint=r.fix_hint,
                simulated=True,
            )
        )
    return findings


def audit_available() -> bool:
    from ..runner.sandbox import docker_available

    return docker_available()
