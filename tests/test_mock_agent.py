"""The mock agent must be deterministic and index-order independent.

``taskgate demo`` asserts eight specific routes with the default seed, so any
drift in the simulated schedule is a demo regression. Two properties matter:

* the whole schedule is pre-drawn, so concurrency cannot change the result;
* pass/fail is drawn from its own RNG stream, so the realised rate follows the
  declared ``true_pass_rate`` instead of being perturbed by turn-count draws.
"""

from __future__ import annotations

from pathlib import Path

from taskgate.models import DemoMeta, Task, TaskSpec
from taskgate.runner.mock_agent import MockAgent


def _demo_task(task_id: str, rate: float, profile: str = "normal") -> Task:
    return Task(
        task_id=task_id,
        path=Path("."),
        spec=TaskSpec(
            id=task_id,
            demo=DemoMeta(true_pass_rate=rate, transcript_profile=profile),
        ),
    )


def test_mock_plan_is_reproducible_for_the_same_seed():
    a = MockAgent(_demo_task("t01_log_parser", 0.20), seed=1234, n_rollouts=20)
    b = MockAgent(_demo_task("t01_log_parser", 0.20), seed=1234, n_rollouts=20)
    assert [o.passed for o in a.plan()] == [o.passed for o in b.plan()]


def test_mock_plan_does_not_depend_on_draw_order():
    agent = MockAgent(_demo_task("t01_log_parser", 0.20), seed=1234, n_rollouts=20)
    reference = [o.passed for o in agent.plan()]
    shuffled = MockAgent(_demo_task("t01_log_parser", 0.20), seed=1234, n_rollouts=20)
    for index in (7, 2, 19, 0, 11):
        shuffled.outcome(index)
    assert [o.passed for o in shuffled.plan()] == reference


def test_different_seeds_produce_different_schedules():
    a = MockAgent(_demo_task("t01_log_parser", 0.20), seed=1234, n_rollouts=20)
    b = MockAgent(_demo_task("t01_log_parser", 0.20), seed=9999, n_rollouts=20)
    assert [o.passed for o in a.plan()] != [o.passed for o in b.plan()]


def test_pass_count_tracks_the_declared_rate_at_n_20():
    # Pinned on purpose: these are the ground-truth demo outcomes at seed 1234.
    expected = {
        "t01_log_parser": 3,
        "t02_csv_dedupe": 3,
        "t03_trivial_rename": 19,
        "t08_borderline": 7,
    }
    rates = {
        "t01_log_parser": (0.20, "normal"),
        "t02_csv_dedupe": (0.15, "normal"),
        "t03_trivial_rename": (0.95, "too_easy"),
        "t08_borderline": (0.35, "turn_limit"),
    }
    for task_id, (rate, profile) in rates.items():
        agent = MockAgent(_demo_task(task_id, rate, profile), seed=1234, n_rollouts=20)
        passes = sum(o.passed for o in agent.plan())
        assert passes == expected[task_id], task_id


def test_mock_rollouts_are_flagged_simulated():
    agent = MockAgent(_demo_task("t01_log_parser", 0.20), seed=1234, n_rollouts=3)
    rollout = agent.rollout(0)
    assert rollout.simulated is True
    assert rollout.transcript  # a plausible fake transcript is produced


def test_infra_error_rate_removes_rollouts_from_the_denominator():
    agent = MockAgent(
        _demo_task("t01_log_parser", 0.20), seed=1234, n_rollouts=20, infra_error_rate=0.5
    )
    plan = agent.plan()
    infra = [o for o in plan if o.status == "infra_error"]
    assert infra  # the rate is applied
    assert all(o.error and o.verifier_exit_code is None for o in infra)
