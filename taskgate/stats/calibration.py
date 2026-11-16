"""Solve-rate statistics and routing. Pure functions, no I/O.

This module is the credibility core of TaskGate: the whole point of the tool is
that a point estimate from ~20 rollouts is *not* a decision. Everything here is
deterministic and unit-tested against known values.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from ..models import (
    RECOVERABLE_CAUSES,
    CalibrationResult,
    Route,
    Rollout,
)

PASS_AT_K_VALUES = (1, 3, 5)


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------
def z_for_confidence(confidence: float) -> float:
    """Two-sided z multiplier, e.g. 0.90 -> 1.6449."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


def wilson_interval(passes: int, n: int, confidence: float = 0.90) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because it stays inside [0, 1] and
    behaves sensibly for small n and for 0/n or n/n -- exactly the regime this
    tool operates in.
    """
    if n <= 0:
        return (0.0, 1.0)
    if passes < 0 or passes > n:
        raise ValueError(f"passes must be in [0, {n}], got {passes}")
    z = z_for_confidence(confidence)
    phat = passes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of the probability that k samples contain >=1 success.

    1 - C(n-c, k) / C(n, k), as in the HumanEval report. Clamped to 0 when n < k.
    """
    if n <= 0 or k <= 0 or c <= 0 or n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_curve(n_valid: int, n_passed: int, ks=PASS_AT_K_VALUES) -> dict[str, float | None]:
    """pass@k for the k values that are defined at this sample size (None otherwise)."""
    return {
        f"pass@{k}": (pass_at_k(n_valid, n_passed, k) if n_valid >= k else None)
        for k in ks
    }


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
def straddles(lo: float, hi: float, boundary: float) -> bool:
    return lo < boundary < hi


def in_window(rate: float, ship_min: float, ship_max: float) -> bool:
    return ship_min <= rate <= ship_max


def suggest_extra_runs(
    solve_rate: float,
    n_valid: int,
    ship_min: float,
    ship_max: float,
    drop_easy: float = 0.80,
    confidence: float = 0.90,
    max_total: int = 400,
) -> int | None:
    """Smallest number of extra rollouts that would resolve the boundary straddle.

    Assumes the observed rate holds. A larger sample resolves the question as soon
    as the interval stops straddling a window boundary -- either by clearing it
    (a decisive SHIP/DROP/REWRITE) or by escaping to a clearly-too-hard /
    clearly-too-easy verdict. Returns None when no count within ``max_total``
    resolves it (e.g. a rate sitting exactly on a boundary).
    """
    if n_valid <= 0:
        return None
    for total in range(n_valid + 1, max_total + 1):
        c = round(solve_rate * total)
        lo, hi = wilson_interval(c, total, confidence)
        if hi < ship_min or lo >= drop_easy:
            return total - n_valid
        if not straddles(lo, hi, ship_min) and not straddles(lo, hi, ship_max):
            return total - n_valid
    return None


def route_task(
    *,
    solve_rate: float,
    ci_low: float,
    ci_high: float,
    n_valid: int,
    ship_min: float = 0.10,
    ship_max: float = 0.30,
    drop_easy: float = 0.80,
    has_blocker: bool = False,
    min_rollouts_for_decision: int = 5,
    confidence: float = 0.90,
) -> tuple[Route, str, int | None]:
    """Route one task from its statistics alone.

    Decision table (priority order, first match wins):

    1. BLOCKED          -- any BLOCKER audit finding; calibration is not spent.
    2. NEEDS-MORE-RUNS  -- fewer than ``min_rollouts_for_decision`` valid rollouts.
    3. DROP             -- CI entirely below ship_min (too hard).
    4. DROP             -- lower bound >= drop_easy (degenerately easy).
    5. SHIP             -- point estimate inside [ship_min, ship_max].
    6. NEEDS-MORE-RUNS  -- CI straddles a window boundary -> suggest extra N.
    7. REWRITE          -- CI sits entirely above the window but is not degenerate.
    8. REWRITE          -- anything else: out of window, interval unresolved.

    Returns ``(route, human_readable_reason, suggested_extra_runs)``.
    """
    if has_blocker:
        return Route.BLOCKED, "blocker audit finding present; no calibration spend", None

    if n_valid <= 0:
        return (
            Route.NEEDS_MORE_RUNS,
            "no valid rollouts (all of them errored or timed out)",
            None,
        )

    window = f"[{ship_min:.2f}, {ship_max:.2f}]"
    extra = suggest_extra_runs(solve_rate, n_valid, ship_min, ship_max, drop_easy, confidence)

    if n_valid < min_rollouts_for_decision:
        return (
            Route.NEEDS_MORE_RUNS,
            f"only {n_valid} valid rollout(s); too few to route reliably",
            max(min_rollouts_for_decision - n_valid, extra or 0),
        )

    if ci_high < ship_min:
        return (
            Route.DROP,
            f"upper CI bound {ci_high:.2f} < ship_min {ship_min:.2f}: too hard for the target model",
            None,
        )
    if ci_low >= drop_easy:
        return (
            Route.DROP,
            f"lower CI bound {ci_low:.2f} >= {drop_easy:.2f}: solved (almost) every time",
            None,
        )
    if in_window(solve_rate, ship_min, ship_max):
        return (
            Route.SHIP,
            f"point estimate {solve_rate:.2f} inside target window {window}",
            None,
        )
    if straddles(ci_low, ci_high, ship_min) or straddles(ci_low, ci_high, ship_max):
        return (
            Route.NEEDS_MORE_RUNS,
            (
                f"CI [{ci_low:.2f}, {ci_high:.2f}] straddles a window boundary; "
                f"point estimate {solve_rate:.2f} is not decisive"
            ),
            extra,
        )
    if ci_low >= ship_max:
        return (
            Route.REWRITE,
            f"CI sits above the window (lower bound {ci_low:.2f} > ship_max {ship_max:.2f}): too easy to be useful",
            None,
        )
    return (
        Route.REWRITE,
        f"point estimate {solve_rate:.2f} is out of window {window} and the interval does not resolve it",
        None,
    )


def apply_diagnosis(
    base_route: Route,
    base_reason: str,
    likely_cause: str | None,
    diagnosis_source: str,
) -> tuple[Route, str]:
    """Let an actionable diagnosis turn a DROP / NEEDS-MORE-RUNS into a REWRITE.

    A task whose failures are explained by an underspecified instruction or a
    too-strict test is not a lost task: it is a rewrite. A statistically decisive
    verdict from a cause nobody can fix (``genuinely_hard``) is left alone.
    """
    if base_route in (Route.BLOCKED, Route.SHIP):
        return base_route, base_reason
    if likely_cause in RECOVERABLE_CAUSES:
        return (
            Route.REWRITE,
            f"{base_reason}; overridden to REWRITE: recoverable per {diagnosis_source} diagnosis ({likely_cause})",
        )
    return base_route, base_reason


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def build_calibration_result(
    *,
    task_id: str,
    rollouts: list[Rollout],
    n_planned: int,
    model: str,
    confidence: float = 0.90,
    simulated: bool = False,
    duration_s: float = 0.0,
    status: str = "ok",
    error: str | None = None,
) -> CalibrationResult:
    """Aggregate rollouts into a calibration result (no routing applied yet)."""
    valid = [r for r in rollouts if r.counted]
    n_valid = len(valid)
    n_passed = sum(1 for r in valid if r.passed)
    n_infra = sum(1 for r in rollouts if r.status == "infra_error")
    n_timeout = sum(1 for r in rollouts if r.status == "timeout")

    solve_rate = (n_passed / n_valid) if n_valid else None
    ci_low, ci_high = (wilson_interval(n_passed, n_valid, confidence) if n_valid else (None, None))

    return CalibrationResult(
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        error=error,
        n_planned=n_planned,
        n_valid=n_valid,
        n_passed=n_passed,
        n_infra_errors=n_infra,
        n_timeouts=n_timeout,
        solve_rate=solve_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_confidence=confidence,
        pass_at_k=pass_at_k_curve(n_valid, n_passed),
        cost_usd=sum(r.cost_usd for r in rollouts),
        duration_s=duration_s,
        model=model,
        simulated=simulated,
        rollouts=rollouts,
    )


def route_calibration_result(result: CalibrationResult, config, has_blocker: bool) -> CalibrationResult:
    """Attach route + reason + suggested extra runs to a calibration result."""
    if result.status != "ok" or result.solve_rate is None:
        result.route = None
        result.route_reason = result.error or "calibration did not produce a usable estimate"
        return result
    route, reason, extra = route_task(
        solve_rate=result.solve_rate,
        ci_low=result.ci_low or 0.0,
        ci_high=result.ci_high if result.ci_high is not None else 1.0,
        n_valid=result.n_valid,
        ship_min=config.ship_min,
        ship_max=config.ship_max,
        drop_easy=config.drop_if_lower_bound_at_or_above,
        has_blocker=has_blocker,
        confidence=config.ci_confidence,
    )
    result.route = route
    result.route_reason = reason
    result.suggested_extra_runs = extra if route == Route.NEEDS_MORE_RUNS else None
    return result
