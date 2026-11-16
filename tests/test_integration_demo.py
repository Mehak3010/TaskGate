"""Integration test for the demo corpus.

This is the end-to-end acceptance check from the brief: building the 8 seeded
tasks and running the full pipeline in mock mode (no Docker, no API key) must
route every task to its expected verdict, deterministically, at the default
seed.

The run is shared across the whole test session via the ``demo_report`` fixture
so the corpus is built and calibrated exactly once.
"""

from __future__ import annotations

import pytest

from taskgate.models import Route

EXPECTED_ROUTES = {
    "t01_log_parser": Route.SHIP,
    "t02_csv_dedupe": Route.SHIP,
    "t03_trivial_rename": Route.DROP,
    "t04_ambiguous_spec": Route.REWRITE,
    "t05_leaky_tests": Route.BLOCKED,
    "t06_flaky_timing": Route.BLOCKED,
    "t07_network_dep": Route.BLOCKED,
    "t08_borderline": Route.NEEDS_MORE_RUNS,
}

EXPECTED_COUNTS = {
    "SHIP": 2,
    "REWRITE": 1,
    "NEEDS-MORE-RUNS": 1,
    "AUDIT-ONLY": 0,
    "DROP": 1,
    "BLOCKED": 3,
}


@pytest.fixture(scope="module")
def routes(demo_report) -> dict[str, Route]:
    _tasks, report = demo_report
    return {t.task_id: t.route for t in report.tasks}


def test_every_demo_task_reaches_expected_route(routes):
    assert routes == EXPECTED_ROUTES, (
        "route mismatches vs expectations: "
        + ", ".join(
            f"{task_id}: got {routes.get(task_id)} want {want}"
            for task_id, want in EXPECTED_ROUTES.items()
            if routes.get(task_id) != want
        )
    )


def test_summary_counts_match_expected(demo_report):
    _tasks, report = demo_report
    assert report.summary.counts == EXPECTED_COUNTS
    assert report.summary.n_tasks == len(EXPECTED_ROUTES)
    assert report.summary.n_tasks == sum(EXPECTED_COUNTS.values())


def test_demo_run_is_flagged_simulated(demo_report):
    _tasks, report = demo_report
    assert report.simulated is True
    assert all(t.simulated for t in report.tasks)
    # Every calibration in a mock run must also be labelled simulated.
    calibrated = [t for t in report.tasks if t.calibration is not None]
    assert calibrated, "expected at least one calibrated task in the demo run"
    assert all(t.calibration.simulated for t in calibrated)


def test_blocked_tasks_have_no_calibration_spend(demo_report):
    """A BLOCKER must stop the task before any rollout spend happens."""
    _tasks, report = demo_report
    for task in report.tasks:
        if task.route is Route.BLOCKED:
            assert task.blockers, f"{task.task_id} is BLOCKED without a BLOCKER finding"
            assert task.calibration is None, (
                f"{task.task_id} is BLOCKED but was calibrated anyway"
            )


def test_blocked_tasks_expose_exact_findings(demo_report):
    """The three planted leak/network/flakiness tasks point at a real file:line."""
    _tasks, report = demo_report
    by_id = {t.task_id: t for t in report.tasks}

    leaky = by_id["t05_leaky_tests"]
    assert {f.rule_id for f in leaky.blockers} == {"S001"}
    assert all(f.file for f in leaky.findings)

    network = by_id["t07_network_dep"]
    assert "S006" in {f.rule_id for f in network.findings}

    flaky = by_id["t06_flaky_timing"]
    assert flaky.findings


def test_needs_more_runs_suggests_extra_runs(demo_report):
    _tasks, report = demo_report
    borderline = next(t for t in report.tasks if t.task_id == "t08_borderline")
    assert borderline.route is Route.NEEDS_MORE_RUNS
    assert borderline.calibration is not None
    assert borderline.calibration.suggested_extra_runs
    assert borderline.calibration.suggested_extra_runs > 0


def test_in_window_tasks_have_no_blockers(demo_report):
    _tasks, report = demo_report
    for task in report.tasks:
        if task.route is Route.SHIP:
            assert not task.blockers
            assert task.calibration is not None
            assert task.calibration.ci_low is not None
            assert report.config.ship_min <= task.calibration.solve_rate <= report.config.ship_max


def test_demo_does_not_touch_docker_by_default(tmp_path, monkeypatch):
    """`taskgate demo` is simulated: it must never probe or use Docker unless --docker."""
    from typer.testing import CliRunner

    from taskgate import pipeline
    from taskgate.cli import app

    def boom():
        raise AssertionError("demo touched Docker without --docker")

    from taskgate import cli as cli_mod

    monkeypatch.setattr(pipeline, "docker_available", boom)
    monkeypatch.setattr(cli_mod, "docker_available", boom)
    result = CliRunner().invoke(app, ["demo", "--work", str(tmp_path / "w"), "--no-open"])
    assert result.exit_code == 0, result.output
    out = result.output.replace("\n", " ")
    assert "all 8 demo tasks reached their expected route" in out
    assert "Docker unreachable" not in out
    report_html = (tmp_path / "w" / "results" / "report.html").read_text(encoding="utf-8")
    assert "docker daemon unreachable" not in report_html
    assert "demo mode: container checks not run" in report_html
