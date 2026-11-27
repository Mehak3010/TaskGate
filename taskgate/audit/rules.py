"""Rule registry.

Rule IDs are stable forever, so two runs of TaskGate on the same task produce
diffable reports. Severities follow the brief: BLOCKER stops calibration,
WARN/INFO are reported for a human to judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Severity


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    severity: Severity
    fix_hint: str
    kind: str  # "static" or "dynamic"


_RULES: tuple[Rule, ...] = (
    Rule(
        id="S001",
        title="Hidden tests are reachable from the agent's filesystem",
        severity=Severity.BLOCKER,
        fix_hint=(
            "Do not COPY tests/ into the image. Add tests/ and solution/ to "
            ".dockerignore and copy only the files the agent needs."
        ),
        kind="static",
    ),
    Rule(
        id="S002",
        title="Solution or expected-output files baked into the image",
        severity=Severity.BLOCKER,
        fix_hint=(
            "Keep solution/ and any answer/expected/golden file out of the build "
            "context (add them to .dockerignore)."
        ),
        kind="static",
    ),
    Rule(
        id="S003",
        title="Test assertions compare against literals that also appear in instruction.md",
        severity=Severity.WARN,
        fix_hint=(
            "Answer leakage: derive the expected value in the test instead of "
            "hardcoding a literal that the instruction already reveals."
        ),
        kind="static",
    ),
    Rule(
        id="S004",
        title="Tests only check existence or non-emptiness, not behaviour",
        severity=Severity.WARN,
        fix_hint=(
            "Assert on the produced content (parsed structure, values, invariants), "
            "not just that a file exists or is non-empty."
        ),
        kind="static",
    ),
    Rule(
        id="S005",
        title="Unpinned dependencies",
        severity=Severity.WARN,
        fix_hint="Pin versions (x==1.2.3, package=1.2.3-1) and avoid :latest image tags.",
        kind="static",
    ),
    Rule(
        id="S006",
        title="Network use in tests or at run time",
        severity=Severity.BLOCKER,
        fix_hint=(
            "Fetch everything at image build time. Tests and solutions must run "
            "with network_mode=none."
        ),
        kind="static",
    ),
    Rule(
        id="S007",
        title="Non-determinism sources in tests",
        severity=Severity.WARN,
        fix_hint=(
            "Seed RNGs, inject a clock instead of reading time.time(), and replace "
            "sleep-based sync with a real readiness check."
        ),
        kind="static",
    ),
    Rule(
        id="S008",
        title="Declared output format is not strictly validated by the tests",
        severity=Severity.WARN,
        fix_hint=(
            "Parse the declared format strictly (json.loads / yaml.safe_load / "
            "csv reader) and assert on the parsed structure."
        ),
        kind="static",
    ),
    Rule(
        id="S009",
        title="Instruction may be ambiguous",
        severity=Severity.INFO,
        fix_hint=(
            "State the exact output path and format, and replace pronouns with the "
            "name of the artefact. Advisory only."
        ),
        kind="static",
    ),
    Rule(
        id="S010",
        title="The task can be solved by editing the tests",
        severity=Severity.BLOCKER,
        fix_hint=(
            "Never make tests/ writable by the agent user. Do not chmod it, and run "
            "tests from a read-only mount after the agent stops."
        ),
        kind="static",
    ),
    Rule(
        id="LOAD-ERROR",
        title="Task could not be loaded",
        severity=Severity.BLOCKER,
        fix_hint=(
            "Fix the task.yaml / task.toml named in the message (see 'Task format' in the "
            "README); the other tasks in the folder were still audited."
        ),
        kind="load",
    ),
    Rule(
        id="D000",
        title="Task image failed to build",
        severity=Severity.BLOCKER,
        fix_hint="Fix the Dockerfile: without a build there is nothing to calibrate.",
        kind="dynamic",
    ),
    Rule(
        id="D001",
        title="Reference solution is flaky or broken",
        severity=Severity.BLOCKER,
        fix_hint=(
            "The reference solution must pass 5/5 fresh runs. Fix the solution or "
            "remove the nondeterminism before calibrating."
        ),
        kind="dynamic",
    ),
    Rule(
        id="D002",
        title="Tests pass on an unmodified environment (vacuous task)",
        severity=Severity.BLOCKER,
        fix_hint=(
            "The verifier must fail before the solution runs. Add a precondition or "
            "assert the work was actually done."
        ),
        kind="dynamic",
    ),
    Rule(
        id="D003",
        title="Hidden network dependency",
        severity=Severity.BLOCKER,
        fix_hint="The task must still pass with network_mode=none.",
        kind="dynamic",
    ),
    Rule(
        id="D004",
        title="Tests are too weak to detect sabotage",
        severity=Severity.WARN,
        fix_hint="At least one assertion should fail for each trivial mutation of the solution.",
        kind="dynamic",
    ),
    Rule(
        id="D005",
        title="Image build is not reproducible",
        severity=Severity.BLOCKER,
        fix_hint="Pin dependencies so two builds of the same Dockerfile install the same package set.",
        kind="dynamic",
    ),
)

RULES: dict[str, Rule] = {r.id: r for r in _RULES}

STATIC_RULES: tuple[Rule, ...] = tuple(r for r in _RULES if r.kind == "static")
DYNAMIC_RULES: tuple[Rule, ...] = tuple(r for r in _RULES if r.kind == "dynamic")


def rule(rule_id: str) -> Rule:
    return RULES[rule_id]


def fix_hint(rule_id: str) -> str:
    return RULES[rule_id].fix_hint if rule_id in RULES else ""
