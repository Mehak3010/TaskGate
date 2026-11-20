"""Statistics and routing: the credibility core.

Every expected value here is either a published constant or hand-computed from
the Wilson formula, so a regression in the maths fails the build.
"""

from __future__ import annotations

import pytest

from taskgate.models import Route
from taskgate.stats.calibration import (
    apply_diagnosis,
    in_window,
    pass_at_k,
    pass_at_k_curve,
    route_task,
    straddles,
    suggest_extra_runs,
    wilson_interval,
    z_for_confidence,
)


# ---------------------------------------------------------------- intervals
def test_z_multiplier_for_90_percent():
    assert z_for_confidence(0.90) == pytest.approx(1.6448536, abs=1e-6)


def test_wilson_interval_known_value_5_of_20():
    low, high = wilson_interval(5, 20)
    assert low == pytest.approx(0.1274, abs=1e-3)
    assert high == pytest.approx(0.4322, abs=1e-3)


def test_wilson_interval_known_value_0_of_20():
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert high == pytest.approx(0.1192, abs=1e-3)


def test_wilson_interval_stays_inside_unit_interval():
    low, high = wilson_interval(20, 20)
    assert low == pytest.approx(0.8804, abs=1e-3)
    assert high == pytest.approx(1.0)


def test_wilson_interval_wide_at_small_n():
    # The whole point of the tool: a 20% point estimate at N=20 is not a decision.
    low, high = wilson_interval(4, 20)
    assert high - low > 0.25


def test_wilson_interval_without_samples():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(21, 20)


# ---------------------------------------------------------------- pass@k
def test_pass_at_k_known_values():
    assert pass_at_k(20, 1, 1) == pytest.approx(0.05)
    assert pass_at_k(20, 3, 1) == pytest.approx(0.15)
    assert pass_at_k(20, 3, 3) == pytest.approx(0.403509, abs=1e-6)
    assert pass_at_k(20, 5, 5) == pytest.approx(0.806308, abs=1e-6)


def test_pass_at_k_edge_cases():
    assert pass_at_k(20, 0, 1) == 0.0
    assert pass_at_k(2, 1, 5) == 0.0  # k > n is undefined -> 0
    assert pass_at_k(20, 19, 3) == 1.0  # fewer than k failures left


def test_pass_at_k_curve_blanks_undefined_k():
    curve = pass_at_k_curve(2, 1)
    assert curve["pass@1"] is not None
    assert curve["pass@3"] is None
    assert curve["pass@5"] is None


# ---------------------------------------------------------------- helpers
def test_straddles_and_in_window_boundaries():
    assert straddles(0.28, 0.63, 0.30) is True
    assert straddles(0.31, 0.63, 0.30) is False
    assert in_window(0.10, 0.10, 0.30) is True
    assert in_window(0.30, 0.10, 0.30) is True
    assert in_window(0.3000001, 0.10, 0.30) is False


# ---------------------------------------------------------------- routing
def _route(**overrides):
    kwargs = dict(solve_rate=0.20, ci_low=0.10, ci_high=0.35, n_valid=20)
    kwargs.update(overrides)
    return route_task(**kwargs)


def test_route_blocker_overrides_any_statistic():
    route, reason, extra = _route(
        solve_rate=0.20, ci_low=0.15, ci_high=0.25, has_blocker=True
    )
    assert route is Route.BLOCKED
    assert extra is None
    assert "blocker" in reason


def test_route_no_valid_rollouts_needs_more_runs():
    route, _, _ = _route(solve_rate=0.0, ci_low=0.0, ci_high=0.0, n_valid=0)
    assert route is Route.NEEDS_MORE_RUNS


def test_route_below_min_rollouts_suggests_at_least_the_shortfall():
    route, _, extra = _route(solve_rate=0.20, ci_low=0.02, ci_high=0.50, n_valid=3)
    assert route is Route.NEEDS_MORE_RUNS
    assert extra is not None and extra >= 2


def test_route_drop_when_interval_is_entirely_below_the_window():
    route, _, _ = _route(solve_rate=0.02, ci_low=0.0, ci_high=0.05)
    assert route is Route.DROP


def test_route_drop_when_lower_bound_is_degenerately_high():
    route, _, _ = _route(solve_rate=0.95, ci_low=0.85, ci_high=1.0)
    assert route is Route.DROP


def test_route_ship_uses_inclusive_window_bounds():
    assert _route(solve_rate=0.10, ci_low=0.03, ci_high=0.28)[0] is Route.SHIP
    assert _route(solve_rate=0.30, ci_low=0.16, ci_high=0.48)[0] is Route.SHIP


def test_route_needs_more_runs_when_interval_straddles_upper_bound():
    route, reason, extra = _route(solve_rate=0.35, ci_low=0.20, ci_high=0.53)
    assert route is Route.NEEDS_MORE_RUNS
    assert extra is not None and extra > 0
    assert "straddles" in reason


def test_route_needs_more_runs_when_interval_straddles_lower_bound():
    route, _, _ = _route(solve_rate=0.05, ci_low=0.02, ci_high=0.14)
    assert route is Route.NEEDS_MORE_RUNS


def test_route_rewrite_when_interval_sits_above_the_window():
    route, _, _ = _route(solve_rate=0.55, ci_low=0.40, ci_high=0.70)
    assert route is Route.REWRITE


# ---------------------------------------------------------------- diagnosis override
def test_recoverable_cause_turns_drop_into_rewrite():
    route, reason = apply_diagnosis(
        Route.DROP, "ci below window", "underspecified_instruction", "heuristic"
    )
    assert route is Route.REWRITE
    assert "overridden" in reason


def test_unfixable_cause_leaves_the_verdict_alone():
    route, _ = apply_diagnosis(Route.DROP, "ci below window", "genuinely_hard", "heuristic")
    assert route is Route.DROP


def test_diagnosis_never_touches_ship_or_blocked():
    assert apply_diagnosis(Route.SHIP, "ok", "test_too_strict", "heuristic")[0] is Route.SHIP
    assert (
        apply_diagnosis(Route.BLOCKED, "blocked", "test_too_strict", "heuristic")[0]
        is Route.BLOCKED
    )


def test_needs_more_runs_survives_a_recoverable_cause():
    """Too few valid rollouts is a sampling verdict, not a fixable editing one.

    README: "<5 valid rollouts -> NEEDS-MORE-RUNS". A recoverable diagnosis must
    not silently promote it to REWRITE (which would hide the missing sample).
    """
    route, reason = apply_diagnosis(
        Route.NEEDS_MORE_RUNS,
        "only 1 valid rollout(s); too few to route reliably",
        "environment_issue",
        "heuristic",
        underpowered=True,
    )
    assert route is Route.NEEDS_MORE_RUNS
    assert "overridden" not in reason


def test_needs_more_runs_from_a_straddle_can_still_be_rewritten():
    """A straddle on an *adequate* sample is a different situation.

    There the rollouts exist and the interval is merely inconclusive, so a
    recoverable diagnosis may legitimately resolve it to REWRITE.
    """
    route, reason = apply_diagnosis(
        Route.NEEDS_MORE_RUNS,
        "CI [0.30, 0.34] straddles a window boundary",
        "underspecified_instruction",
        "heuristic",
        underpowered=False,
    )
    assert route is Route.REWRITE
    assert "overridden" in reason


# ---------------------------------------------------------------- extra runs
def test_suggest_extra_runs_resolves_a_straddle():
    extra = suggest_extra_runs(0.35, 20, 0.10, 0.30)
    assert extra is not None and extra > 0


def test_suggest_extra_runs_is_small_for_in_window_rates():
    assert suggest_extra_runs(0.20, 20, 0.10, 0.30) is not None


def test_suggest_extra_runs_returns_none_on_an_exact_boundary():
    # A rate sitting exactly on ship_max can never be resolved by more samples:
    # the interval keeps containing the boundary. Saying so beats inventing a number.
    assert suggest_extra_runs(0.30, 20, 0.10, 0.30) is None

def test_build_calibration_result_with_no_valid_rollouts():
    from taskgate.stats.calibration import build_calibration_result

    r = build_calibration_result(task_id="x", rollouts=[], n_planned=10, model="m")
    assert r.solve_rate is None
    assert r.pass_at_k == {"pass@1": None, "pass@3": None, "pass@5": None}
