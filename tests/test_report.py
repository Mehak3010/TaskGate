"""Report rendering tests.

Two acceptance criteria live here:

* ``report.json`` round-trips through the Pydantic model unchanged (the report is
  the machine-readable artefact, so it must be complete and schema-valid);
* ``report.html`` renders as a single offline file — no CDN, no remote fetches —
  with the SIMULATED banner, the solve-rate chart, and the expandable rows the
  brief calls for.
"""

from __future__ import annotations

from pathlib import Path

from taskgate.audit.dynamic_checks import DynamicAuditInfo
from taskgate.models import (
    DynamicAuditEvidence,
    Report,
    Rollout,
    Route,
    RunConfig,
    TaskReport,
)
from taskgate.report.html_report import (
    _severity_badge,
    load_report_json,
    render_report_html,
    write_report_html,
    write_report_json,
)
from taskgate.report.terminal import _audit_cell
from taskgate.stats.calibration import build_calibration_result


def test_report_json_round_trips(demo_report, tmp_path: Path):
    _tasks, report = demo_report
    path = write_report_json(report, tmp_path)
    reloaded = load_report_json(path)
    assert reloaded == report
    assert reloaded.model_dump(mode="json") == report.model_dump(mode="json")


def test_report_json_records_reproducibility_metadata(demo_report, tmp_path: Path):
    _tasks, report = demo_report
    path = write_report_json(report, tmp_path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config"]["seed"] == 1234
    assert payload["config"]["model"] == "mock"
    assert payload["generated_at"]
    assert payload["schema_version"]
    assert "git_hash" in payload["config"]


def test_html_renders_the_simulated_banner(demo_report):
    _tasks, report = demo_report
    html = render_report_html(report)
    assert "SIMULATED DATA" in html


def test_html_has_inline_chart(demo_report):
    _tasks, report = demo_report
    html = render_report_html(report)
    assert '<svg class="chart"' in html
    assert "target window" in html


def test_html_is_self_contained_and_offline(demo_report):
    _tasks, report = demo_report
    html = render_report_html(report)
    # No remote scripts/styles/fonts, no CDN.
    assert 'src="http' not in html
    assert 'href="http' not in html
    assert "cdn" not in html.lower()
    assert "<link" not in html


def test_html_has_sortable_and_expandable_rows(demo_report):
    _tasks, report = demo_report
    html = render_report_html(report)
    assert 'data-sort' in html
    assert 'class="data-row"' in html
    assert "<script>" in html


def test_html_written_to_disk_is_single_file(demo_report, tmp_path: Path):
    _tasks, report = demo_report
    path = write_report_html(report, tmp_path)
    assert path.is_file()
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
    # A single file: nothing else was emitted alongside it.
    assert [p.name for p in tmp_path.iterdir()] == ["report.html"]


def test_failed_generation_is_stored_but_not_rendered_in_html():
    """The rejected generation lives in report.json for offline diagnosis only.

    The HTML is what a reviewer reads, so the raw model output stays out of it.
    """
    import json

    marker = "MALFORMED-GENERATION-MARKER-7f3a"
    rollout = Rollout(
        index=0,
        status="ok",
        passed=False,
        tool_call_failed=True,
        failed_generation=marker,
    )
    cal = build_calibration_result(
        task_id="t_gen", rollouts=[rollout], n_planned=1, model="m"
    )
    task = TaskReport(
        task_id="t_gen", path="/tmp/t_gen", route=Route.NEEDS_MORE_RUNS, calibration=cal
    )
    report = Report(config=RunConfig(model="m"), tasks=[task])

    assert marker in json.dumps(report.model_dump(mode="json"))
    assert marker not in render_report_html(report)


def test_severity_badge_reports_info_separately():
    """INFO is advisory: it must not be folded into a pass checkmark."""
    assert _severity_badge({"INFO": 1, "WARN": 2, "BLOCKER": 0}) == "I1/W2/B0"
    assert _severity_badge({"INFO": 0, "WARN": 0, "BLOCKER": 3}) == "I0/W0/B3"
    assert "✓" not in _severity_badge({"INFO": 4, "WARN": 0, "BLOCKER": 0})


def test_terminal_audit_cell_reports_info_separately(demo_report):
    _tasks, report = demo_report
    by_id = {t.task_id: t for t in report.tasks}

    # t04 carries an advisory S009 (INFO) finding, so it is the regression case.
    t04 = by_id["t04_ambiguous_spec"]
    counts = t04.severity_counts
    assert counts["INFO"] >= 1

    cell = _audit_cell(t04)
    assert cell == f"I{counts['INFO']} W{counts['WARN']} B{counts['BLOCKER']}"
    assert "✓" not in cell

    # A clean task reads I0 W0 B0 rather than a green checkmark.
    assert _audit_cell(by_id["t01_log_parser"]) == "I0 W0 B0"


# --------------------------------------------------------------------------
# dynamic-audit evidence is persisted per task
# --------------------------------------------------------------------------
def _report_from_info(info: DynamicAuditInfo) -> Report:
    task = TaskReport(
        task_id="t_dyn",
        path="/tmp/t_dyn",
        route=Route.NEEDS_MORE_RUNS,
        dynamic_audit=DynamicAuditEvidence.from_info(info),
    )
    return Report(config=RunConfig(model="mock", mock=True), tasks=[task])


def test_report_persists_dynamic_audit_evidence(tmp_path: Path):
    """A report built from a fake DynamicAuditInfo carries every evidence field."""
    import json

    info = DynamicAuditInfo(
        build_ok=True,
        reproducible=True,
        reruns=5,
        rerun_passes=5,
        vacuous=False,
        network_dependent=False,
        mutations_probed=3,
        mutations_caught=2,
        mutations_tried=["delete_line", "change_constant", "stub"],
        durations_s=[1.25, 1.5, 2.0, 1.75, 1.6],
    )
    report = _report_from_info(info)
    payload = json.loads(write_report_json(report, tmp_path).read_text(encoding="utf-8"))
    dyn = payload["tasks"][0]["dynamic_audit"]

    assert dyn["ran"] is True
    assert dyn["skipped_reason"] is None
    assert dyn["build"] == "ok"
    assert dyn["reproducible"] == "yes"
    assert dyn["reruns"] == 5
    assert dyn["rerun_passes"] == 5
    assert dyn["durations_s"] == [1.25, 1.5, 2.0, 1.75, 1.6]
    assert dyn["vacuous"] == "no"
    assert dyn["network_off_passed"] == "yes"
    assert dyn["network_dependent"] == "no"
    assert dyn["mutations_probed"] == 3
    assert dyn["mutations_caught"] == 2
    assert dyn["mutations_tried"] == ["delete_line", "change_constant", "stub"]
    # The probe ran and skipped nothing: an empty list, not the "not run" marker.
    assert dyn["mutations_skipped"] == []

    # It also round-trips through the Pydantic model unchanged...
    reloaded = load_report_json(tmp_path / "report.json")
    assert reloaded.tasks[0].dynamic_audit.model_dump(mode="json") == dyn

    # ...and renders in the HTML report.
    html = render_report_html(report)
    assert "Dynamic audit (container evidence)" in html
    assert "change_constant" in html  # which mutation was tried
    assert "2</b> / <b>3" in html  # mutations_caught / mutations_probed
    assert "mutations skipped: none" in html  # nothing was skipped


def test_dynamic_audit_evidence_shows_not_run_when_skipped(tmp_path: Path):
    """A skipped check is published as "not run", never as an empty value."""
    import json

    task = TaskReport(
        task_id="t_skip",
        path="/tmp/t_skip",
        route=Route.NEEDS_MORE_RUNS,
        dynamic_audit=DynamicAuditEvidence.skipped("docker daemon unreachable"),
    )
    report = Report(config=RunConfig(model="mock", mock=True), tasks=[task])
    payload = json.loads(write_report_json(report, tmp_path).read_text(encoding="utf-8"))
    dyn = payload["tasks"][0]["dynamic_audit"]

    assert dyn["ran"] is False
    assert dyn["skipped_reason"] == "docker daemon unreachable"
    for field in (
        "build",
        "reproducible",
        "reruns",
        "rerun_passes",
        "durations_s",
        "vacuous",
        "network_off_passed",
        "network_dependent",
        "mutations_probed",
        "mutations_caught",
        "mutations_tried",
        "mutations_skipped",
    ):
        assert dyn[field] == "not run", f"{field} was not published as 'not run'"

    html = render_report_html(report)
    assert "Dynamic audit (container evidence)" in html
    assert "not run" in html
    assert "docker daemon unreachable" in html


# --------------------------------------------------------------------------
# issue 1 -- audit-only runs are real runs, not simulations
# --------------------------------------------------------------------------
def _solo_task(make_task):
    """A minimal, blocker-free task: it only needs to audit cleanly."""
    return make_task(
        files={"Dockerfile": "FROM python:3.11-slim\n"},
        task_yaml="id: t_solo\n",
        instruction="Write the answer to /app/out.json as JSON.",
        task_id="t_solo",
    )


def test_audit_only_run_is_not_simulated(make_task, monkeypatch, tmp_path: Path):
    """A real audit (Docker up, no calibration) must not be labelled simulated."""
    from taskgate.pipeline import PipelineOptions, run_pipeline_sync

    monkeypatch.setattr("taskgate.pipeline.docker_available", lambda *a, **k: True)
    config = RunConfig(model="cohere/north-mini-code:free", rollouts=2, out_dir=str(tmp_path))
    report = run_pipeline_sync(
        [_solo_task(make_task)],
        config,
        options=PipelineOptions(audit=True, dynamic=False, calibrate=False, diagnose=False),
    )

    assert report.simulated is False
    assert report.dynamic_audit_ran is False
    task = report.tasks[0]
    assert task.route is Route.AUDIT_ONLY
    assert task.simulated is False
    # The dynamic audit was disabled, so it says so rather than claiming a pass.
    assert task.dynamic_audit.ran is False
    assert task.dynamic_audit.skipped_reason

    html = render_report_html(report)
    assert "SIMULATED" not in html
    assert "AUDIT-ONLY" in html
    assert not any("SIMULATED" in note for note in report.notes)


def test_audit_only_without_docker_is_simulated(make_task, monkeypatch, tmp_path: Path):
    """With Docker down the dynamic checks are simulated, so the task is marked.

    It still routes to AUDIT-ONLY: it is an audit-only run either way.
    """
    from taskgate.pipeline import PipelineOptions, run_pipeline_sync

    monkeypatch.setattr("taskgate.pipeline.docker_available", lambda *a, **k: False)
    config = RunConfig(model="cohere/north-mini-code:free", rollouts=2, out_dir=str(tmp_path))
    report = run_pipeline_sync(
        [_solo_task(make_task)],
        config,
        options=PipelineOptions(audit=True, dynamic=True, calibrate=False, diagnose=False),
    )

    assert report.simulated is True
    task = report.tasks[0]
    assert task.route is Route.AUDIT_ONLY
    assert task.simulated is True
    assert task.dynamic_audit.ran is False
    assert "docker daemon unreachable" in (task.dynamic_audit.skipped_reason or "")


# --------------------------------------------------------------------------
# issue 3 -- the mutation count is explained: what was tried, what was skipped
# --------------------------------------------------------------------------
def test_mutation_candidates_report_skip_reasons():
    """Every mutation carries an explanation, so the probe count is never silent."""
    from taskgate.audit.dynamic_checks import _mutation_candidates

    plan = {
        name: (mutated, reason) for name, mutated, reason in _mutation_candidates("print(hello)\n")
    }
    # No numeric constant in the script -> that mutation is skipped, with a reason.
    assert plan["change_constant"] == ("", "no constant found in solution")
    # The stub always applies, so at least one probe is always possible.
    stub, stub_reason = plan["stub"]
    assert stub and not stub_reason

    plan = {
        name: (mutated, reason)
        for name, mutated, reason in _mutation_candidates("total = compute(42)\nprint(total)\n")
    }
    mutated, reason = plan["change_constant"]
    assert mutated and "43" in mutated and not reason


def test_report_renders_skipped_mutations_with_reason(tmp_path: Path):
    """The HTML reports both the mutations tried and those skipped, with reasons."""
    info = DynamicAuditInfo(
        build_ok=True,
        reproducible=True,
        reruns=5,
        rerun_passes=5,
        vacuous=False,
        network_dependent=False,
        mutations_probed=2,
        mutations_caught=2,
        mutations_tried=["delete_line", "stub"],
        mutations_skipped=["change_constant: no constant found in solution"],
        durations_s=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    report = _report_from_info(info)

    import json

    payload = json.loads(write_report_json(report, tmp_path).read_text(encoding="utf-8"))
    dyn = payload["tasks"][0]["dynamic_audit"]
    assert dyn["mutations_tried"] == ["delete_line", "stub"]
    assert dyn["mutations_skipped"] == ["change_constant: no constant found in solution"]

    html = render_report_html(report)
    assert "mutations tried" in html
    assert "mutations skipped" in html
    assert "change_constant: no constant found in solution" in html



# ---------------------------------------------------------------- report polish
def test_html_has_no_engineer_time_headline_but_keeps_the_estimate_in_methodology(demo_report):
    html = render_report_html(demo_report[1])
    assert "<b>Engineer time saved" not in html  # no headline tile
    assert "modelled estimate" in html and "not a measurement" in html


def test_html_states_the_wilson_note_once(demo_report):
    html = render_report_html(demo_report[1])
    assert html.count("Wilson score interval") == 1


def test_html_extra_runs_label_is_explicit(demo_report):
    html = render_report_html(demo_report[1])
    assert "more runs to resolve" in html
    assert "runs?)" not in html


def test_summary_tiles_call_out_simulated_findings(demo_report):
    html = render_report_html(demo_report[1])
    assert "simulated)" in html  # e.g. "blocker findings (1 simulated)"
