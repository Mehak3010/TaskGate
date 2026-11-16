"""Single-file, offline HTML report.

No CDN, no external asset, no build step: the report opens from a USB stick with
Wi-Fi off. The chart is inline SVG generated here; the table is vanilla JS.

A SIMULATED banner is rendered whenever any number came from the mock agent, so
nobody mistakes a simulation for a measurement.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import NOT_RUN, Report

TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.html.j2"

ROUTE_COLORS: dict[str, str] = {
    "SHIP": "#1a7f37",
    "REWRITE": "#bf8700",
    "DROP": "#cf222e",
    "NEEDS-MORE-RUNS": "#0969da",
    "AUDIT-ONLY": "#8250df",
    "BLOCKED": "#57606a",
}

ROUTE_ORDER = ["SHIP", "REWRITE", "NEEDS-MORE-RUNS", "AUDIT-ONLY", "DROP", "BLOCKED"]


# --------------------------------------------------------------------------
# chart
# --------------------------------------------------------------------------
def _x(rate: float, left: float, width: float) -> float:
    return left + max(0.0, min(1.0, rate)) * width


def solve_rate_chart(tasks: list[Any], ship_min: float, ship_max: float) -> str:
    """One dot per task with a 90% CI whisker against the shaded target window."""
    rows = len(tasks)
    left, right, top, row_h = 160.0, 40.0, 34.0, 26.0
    width_total, plot_w = 900.0, 900.0 - 96.0 - 40.0
    height = top + row_h * max(rows, 1) + 54.0

    parts: list[str] = [
        f'<svg class="chart" viewBox="0 0 {width_total:.0f} {height:.0f}" '
        f'role="img" aria-label="Solve rate per task with confidence intervals" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]
    # target window
    wx0, wx1 = _x(ship_min, left, plot_w), _x(ship_max, left, plot_w)
    parts.append(
        f'<rect x="{wx0:.1f}" y="{top - 12:.1f}" width="{wx1 - wx0:.1f}" '
        f'height="{row_h * max(rows, 1) + 20:.1f}" fill="#1a7f37" opacity="0.09"/>'
    )
    parts.append(
        f'<text x="{(wx0 + wx1) / 2:.1f}" y="{top - 18:.1f}" text-anchor="middle" '
        f'class="chart-note">target window {ship_min:.0%}–{ship_max:.0%}</text>'
    )
    # gridlines + x ticks
    for tick in range(0, 101, 10):
        x = _x(tick / 100.0, left, plot_w)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 12:.1f}" x2="{x:.1f}" '
            f'y2="{top + row_h * max(rows, 1) + 8:.1f}" stroke="#d0d7de" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + row_h * max(rows, 1) + 26:.1f}" text-anchor="middle" '
            f'class="chart-tick">{tick}%</text>'
        )
    # rows
    for index, report in enumerate(tasks):
        y = top + row_h * index + row_h / 2 - 6
        color = ROUTE_COLORS.get(report.route.value, "#57606a")
        label = html.escape(report.task_id)
        parts.append(
            f'<text x="{left - 12:.1f}" y="{y + 4:.1f}" text-anchor="end" class="chart-label">{label}</text>'
        )
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_w:.1f}" y2="{y:.1f}" '
            f'stroke="#eaeef2" stroke-width="1"/>'
        )
        cal = report.calibration
        if cal is None or cal.solve_rate is None:
            parts.append(
                f'<text x="{left + 6:.1f}" y="{y + 4:.1f}" class="chart-none">no measurement</text>'
            )
            continue
        lo = _x(cal.ci_low or 0.0, left, plot_w)
        hi = _x(cal.ci_high if cal.ci_high is not None else 1.0, left, plot_w)
        cx = _x(cal.solve_rate, left, plot_w)
        parts.append(
            f'<line x1="{lo:.1f}" y1="{y:.1f}" x2="{hi:.1f}" y2="{y:.1f}" stroke="{color}" '
            f'stroke-width="2" opacity="0.55"/>'
        )
        for edge in (lo, hi):
            parts.append(
                f'<line x1="{edge:.1f}" y1="{y - 5:.1f}" x2="{edge:.1f}" y2="{y + 5:.1f}" '
                f'stroke="{color}" stroke-width="2" opacity="0.55"/>'
            )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="5.5" fill="{color}">'
            f'<title>{label}: {cal.solve_rate:.0%} (90% CI {cal.ci_low:.0%}–{cal.ci_high:.0%})</title>'
            f'</circle>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# view model
# --------------------------------------------------------------------------
def _severity_badge(counts: dict[str, int]) -> str:
    # INFO findings are advisory, so they must not be shown as a pass checkmark.
    return f"I{counts.get('INFO', 0)}/W{counts.get('WARN', 0)}/B{counts.get('BLOCKER', 0)}"


def _rate_text(report: Any) -> str:
    cal = report.calibration
    if cal is None or cal.solve_rate is None:
        return "—"
    return f"{cal.solve_rate:.0%} ({cal.n_passed}/{cal.n_valid})"


def _ci_text(report: Any) -> str:
    cal = report.calibration
    if cal is None or cal.solve_rate is None:
        return "—"
    return f"{cal.ci_low:.0%}–{cal.ci_high:.0%}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _durations_text(durations: list[float] | str) -> str:
    # A skipped run is published as the literal "not run", never a blank cell.
    if durations == NOT_RUN or not durations:
        return NOT_RUN
    return ", ".join(f"{d:.2f}s" for d in durations)


def _list_text(values: list[str] | str) -> str:
    if values == NOT_RUN or not values:
        return NOT_RUN
    return ", ".join(str(v) for v in values)


def _skipped_text(values: list[str] | str) -> str:
    # An empty list here means the probe ran and skipped nothing: "none", not
    # "not run" (which would wrongly imply the probe never executed).
    if values == NOT_RUN:
        return NOT_RUN
    return ", ".join(str(v) for v in values) if values else "none"


def build_view(report: Report) -> dict[str, Any]:
    data = report.model_dump(mode="json")
    counts = data["summary"]["counts"]
    variables = [
        {
            "name": name,
            "value": json.dumps(value, ensure_ascii=False),
            "bad": any(part in name.lower() for part in ("key", "token", "secret", "password")),
        }
        for name, value in sorted((report.config.model_extra or {}).items())
    ]
    rows: list[dict[str, Any]] = []
    for task in report.tasks:
        cal = task.calibration
        rows.append(
            {
                "task_id": task.task_id,
                "route": task.route.value,
                "color": ROUTE_COLORS.get(task.route.value, "#57606a"),
                "route_reason": task.route_reason,
                "path": task.path,
                "status": task.status,
                "error": task.error,
                "audit": _severity_badge(task.severity_counts),
                "blockers": len(task.blockers),
                "rate": _rate_text(task),
                "ci": _ci_text(task),
                "pass_at_1": _pct((cal.pass_at_k or {}).get("pass@1")) if cal else "—",
                "extra_runs": cal.suggested_extra_runs if cal else None,
                "cost": f"${cal.cost_usd:.4f}" if cal else "—",
                "model": (cal.model if cal else report.config.model),
                "top_finding": (
                    f"{task.top_finding.rule_id} {task.top_finding.message}" if task.top_finding else "—"
                ),
                "findings": [
                    {
                        "rule_id": f.rule_id,
                        "severity": f.severity.value,
                        "severity_class": f.severity.value.lower(),
                        "file": f.file,
                        "line": f.line,
                        "message": f.message,
                        "fix_hint": f.fix_hint,
                        "simulated": f.simulated,
                    }
                    for f in task.findings
                ],
                "diagnosis": (
                    {
                        "source": task.diagnosis.source,
                        "likely_cause": task.diagnosis.likely_cause,
                        "evidence": task.diagnosis.evidence,
                        "suggested_edit": task.diagnosis.suggested_edit,
                        "stripped": task.diagnosis.stripped,
                    }
                    if task.diagnosis
                    else None
                ),
                "rollouts": [
                    {
                        "index": r.index,
                        "status": r.status,
                        "passed": r.passed,
                        "turns": r.turns,
                        "duration_s": r.duration_s,
                        "cost_usd": r.cost_usd,
                        "verifier_exit_code": r.verifier_exit_code,
                        "transcript": [
                            {"turn": s.turn, "role": s.role, "text": s.text[:400]}
                            for s in r.transcript[:8]
                        ],
                        "verifier_output": (r.verifier_output or "")[:600],
                    }
                    for r in (cal.rollouts if cal else [])
                ],
                "dynamic_audit": {
                    "ran": task.dynamic_audit.ran,
                    "skipped_reason": task.dynamic_audit.skipped_reason,
                    "build": task.dynamic_audit.build,
                    "reproducible": task.dynamic_audit.reproducible,
                    "reruns": task.dynamic_audit.reruns,
                    "rerun_passes": task.dynamic_audit.rerun_passes,
                    "durations": _durations_text(task.dynamic_audit.durations_s),
                    "vacuous": task.dynamic_audit.vacuous,
                    "network_off_passed": task.dynamic_audit.network_off_passed,
                    "network_dependent": task.dynamic_audit.network_dependent,
                    "mutations_probed": task.dynamic_audit.mutations_probed,
                    "mutations_caught": task.dynamic_audit.mutations_caught,
                    "mutations_tried": _list_text(task.dynamic_audit.mutations_tried),
                    "mutations_skipped": _skipped_text(task.dynamic_audit.mutations_skipped),
                    "error": task.dynamic_audit.error,
                },
                "simulated": task.simulated,
            }
        )

    return {
        "report": data,
        "rows": rows,
        "counts": counts,
        "route_order": ROUTE_ORDER,
        "route_colors": ROUTE_COLORS,
        "chart_svg": solve_rate_chart(report.tasks, report.config.ship_min, report.config.ship_max),
        "extra_variables": variables,
        # Loud header warning when the static audit crashed on too many tasks.
        "audit_error_warn": report.audit_error_rate > 0.10,
        "audit_error_count": report.n_audit_errors,
        "audit_error_pct": f"{report.audit_error_rate:.0%}",
    }


def render_report_html(report: Report) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_NAME)
    return template.render(**build_view(report))


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def write_report_json(report: Report, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_report_html(report: Report, out_dir: Path, filename: str = "report.html") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(render_report_html(report), encoding="utf-8", newline="\n")
    return path


def load_report_json(path: Path) -> Report:
    return Report.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def find_report_json(results_dir: Path) -> Path:
    candidate = Path(results_dir) / "report.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"no report.json in {results_dir}")
