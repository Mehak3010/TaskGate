"""Heuristic diagnosis: always available, zero cost.

The output is advisory. A human decides. Every branch must be justifiable from
evidence that is actually in the report (findings, turn counts, transcript and
verifier text) -- no hidden signals from the demo metadata.
"""

from __future__ import annotations

import re
from collections import Counter

from ..models import CalibrationResult, Diagnosis, Finding

FAILED_TEST_RE = re.compile(r"FAILED\s+([\w./]+::[\w\[\]-]+)")
TURN_LIMIT_RATIO = 0.9
DOMINANT_FAILURE_RATIO = 0.8


def _failed_test_ids(calibration: CalibrationResult) -> Counter[str]:
    counter: Counter[str] = Counter()
    for rollout in calibration.rollouts:
        if rollout.counted and not rollout.passed:
            for test_id in set(FAILED_TEST_RE.findall(rollout.verifier_output or "")):
                counter[test_id] += 1
    return counter


def diagnose_heuristically(
    task_id: str,
    calibration: CalibrationResult,
    findings: list[Finding],
    *,
    max_turns: int = 30,
    ship_min: float = 0.10,
) -> Diagnosis:
    rule_ids = {f.rule_id for f in findings}
    rate = calibration.solve_rate if calibration.solve_rate is not None else 0.0
    failures = [r for r in calibration.rollouts if r.counted and not r.passed]
    passes = [r for r in calibration.rollouts if r.counted and r.passed]
    evidence: list[str] = []

    def diag(cause: str, edit: str, extra: list[str] | None = None) -> Diagnosis:
        return Diagnosis(
            task_id=task_id,
            source="heuristic",
            likely_cause=cause,
            evidence=evidence + (extra or []),
            suggested_edit=edit,
        )

    # 1. solved far too often
    if calibration.n_valid and rate >= 0.80:
        evidence.append(f"solve rate {rate:.2f} ({calibration.n_passed}/{calibration.n_valid} rollouts)")
        if passes:
            evidence.append(f"successful rollouts needed {min(r.turns for r in passes)}-{max(r.turns for r in passes)} turns")
        return diag(
            "trivially_solvable",
            "Raise the difficulty or delete the task: the model already solves it almost every time.",
        )

    # 2. ambiguity in the spec, almost nothing passes
    s009 = [f for f in findings if f.rule_id == "S009"]
    if s009 and rate <= 0.10:
        evidence.append(f"solve rate {rate:.2f} ({calibration.n_passed}/{calibration.n_valid} rollouts)")
        evidence.extend(f"{f.rule_id}: {f.message}" for f in s009[:3])
        return diag(
            "underspecified_instruction",
            "Name the exact output path and format in instruction.md; the hidden verifier "
            "assumes an artefact the agent was never told about.",
        )

    # 3. the run itself is broken
    env_rules = rule_ids & {"D002", "D003", "S006", "S001", "S010"}
    if env_rules or calibration.infra_error_rate > 0.10:
        if calibration.infra_error_rate > 0.10:
            evidence.append(
                f"infra error rate {calibration.infra_error_rate:.2f} over {calibration.n_planned} rollouts"
            )
        evidence.extend(sorted(env_rules))
        return diag(
            "environment_issue",
            "Fix the environment first: the failures say more about the setup than about the task.",
        )

    # 4. the model never finishes
    if failures:
        at_limit = [r for r in failures if r.turns >= max_turns * TURN_LIMIT_RATIO]
        if len(at_limit) >= 0.6 * len(failures):
            evidence.append(f"{len(at_limit)} of {len(failures)} failures used all {max_turns} turns")
            evidence.append(f"solve rate {rate:.2f} ({calibration.n_passed}/{calibration.n_valid} rollouts)")
            return diag(
                "genuinely_hard",
                "The task is not obviously broken; it is just at or beyond the model's reach. "
                "Consider it for a harder tier rather than rewriting the verifier.",
            )

    # 5. a dominant single failing assertion -> possibly too strict
    failed = _failed_test_ids(calibration)
    if failures and failed:
        test_id, count = failed.most_common(1)[0]
        if count >= DOMINANT_FAILURE_RATIO * len(failures):
            evidence.append(f"{count}/{len(failures)} failures stop at the same assertion: {test_id}")
            if rate < ship_min:
                evidence.append(f"solve rate {rate:.2f} is below ship_min {ship_min:.2f}")
            return diag(
                "test_too_strict",
                f"Inspect {test_id}: decide whether the assertion is wrong or the instruction "
                "sandbags the requirement.",
            )

    # 6. passing rollouts are suspiciously short
    if passes and sum(r.turns for r in passes) / len(passes) <= 3:
        evidence.append(
            f"passing rollouts averaged {sum(r.turns for r in passes) / len(passes):.1f} turns"
        )
        return diag(
            "trivially_solvable",
            "The solution path is only a couple of commands; add a requirement the model must reason about.",
        )

    evidence.append(f"solve rate {rate:.2f} ({calibration.n_passed}/{calibration.n_valid} rollouts)")
    evidence.append("no single dominant failure mode found in the transcripts")
    return diag(
        "genuinely_hard",
        "No actionable defect identified. Review the failing transcripts by hand before dropping.",
    )
