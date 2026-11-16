"""Rich terminal output: the summary table and the audit listing."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ..models import Report, Route, Severity, TaskReport

ROUTE_STYLE = {
    Route.SHIP: "bold green",
    Route.REWRITE: "bold yellow",
    Route.NEEDS_MORE_RUNS: "bold cyan",
    Route.AUDIT_ONLY: "bold magenta",
    Route.DROP: "bold red",
    Route.BLOCKED: "bold grey62",
}


def _rate_cell(report: TaskReport) -> str:
    cal = report.calibration
    if cal is None or cal.solve_rate is None:
        return "—"
    return f"{cal.solve_rate:.0%} [{cal.ci_low:.0%}–{cal.ci_high:.0%}] ({cal.n_passed}/{cal.n_valid})"


def _audit_cell(report: TaskReport) -> str:
    # INFO findings are advisory, so they must not be shown as a pass checkmark.
    counts = report.severity_counts
    return f"I{counts['INFO']} W{counts['WARN']} B{counts['BLOCKER']}"


def summary_table(report: Report) -> Table:
    table = Table(
        title=f"TaskGate — model {report.config.model}"
        + ("  [SIMULATED]" if report.simulated else ""),
        header_style="bold",
        expand=True,
    )
    table.add_column("task", style="cyan", no_wrap=True)
    table.add_column("audit", no_wrap=True)
    table.add_column("solve rate (90% CI)", no_wrap=True)
    table.add_column("pass@1", justify="right", no_wrap=True)
    table.add_column("route", no_wrap=True)
    table.add_column("cost", justify="right", no_wrap=True)
    table.add_column("top finding", overflow="fold")

    for task in report.tasks:
        cal = task.calibration
        pass_at_1 = (cal.pass_at_k or {}).get("pass@1") if cal else None
        top = task.top_finding
        top_text = f"{top.rule_id} {top.message}" if top else "—"
        route_style = ROUTE_STYLE.get(task.route, "")
        table.add_row(
            task.task_id,
            _audit_cell(task),
            _rate_cell(task),
            "—" if pass_at_1 is None else f"{pass_at_1:.0%}",
            f"[{route_style}]{task.route.value}[/]",
            f"${cal.cost_usd:.4f}" if cal else "—",
            top_text,
        )
    return table


def counts_line(report: Report) -> str:
    counts = report.summary.counts
    bits = [
        f"{name}: {counts.get(name, 0)}"
        for name in ("SHIP", "REWRITE", "NEEDS-MORE-RUNS", "AUDIT-ONLY", "DROP", "BLOCKED")
    ]
    return " · ".join(bits)


def print_report_summary(report: Report, console: Console | None = None) -> None:
    console = console or Console()
    console.print(summary_table(report))
    console.print(
        f"[bold]Routes:[/] {counts_line(report)}   "
        f"[bold]in window:[/] {report.summary.n_in_window}/{report.summary.n_tasks}   "
        f"[bold]cost:[/] ${report.summary.total_cost_usd:.4f}   "
        f"[bold]wall:[/] {report.summary.wall_time_s:.1f}s   "
        f"[bold]engineer time saved (est.):[/] {report.summary.engineer_minutes_saved} min"
    )
    if report.simulated:
        console.print("[bold yellow]SIMULATED[/]: mock-mode numbers, not measurements.", highlight=False)
    if report.docker_available is False:
        console.print(
            "[yellow]Docker unreachable:[/] static audit only; dynamic findings are simulated.",
            highlight=False,
        )
    if report.audit_error_rate > 0.10:
        console.print(
            f"[bold white on red] AUDIT-ERROR: {report.n_audit_errors}/{len(report.tasks)} "
            f"tasks ({report.audit_error_rate:.0%}) crashed during the static audit. [/]"
            "[bold red]Their findings are incomplete — do not read them as clean.[/]",
            highlight=False,
        )


def findings_table(task: TaskReport) -> Table:
    table = Table(title=f"{task.task_id} — {task.route.value}", header_style="bold", expand=True)
    table.add_column("rule", no_wrap=True)
    table.add_column("severity", no_wrap=True)
    table.add_column("location", no_wrap=True)
    table.add_column("message", overflow="fold")
    styles = {Severity.BLOCKER: "bold red", Severity.WARN: "yellow", Severity.INFO: "cyan"}
    for finding in task.findings:
        table.add_row(
            finding.rule_id,
            f"[{styles[finding.severity]}]{finding.severity.value}[/]",
            finding.location,
            finding.message,
        )
    return table


def print_note(message: str, console: Console | None = None, style: str = "yellow") -> None:
    (console or Console()).print(f"[{style}]{message}[/]", highlight=False)
