"""TaskGate command line interface."""

from __future__ import annotations

import asyncio
import time
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from . import __version__
from .demo.build_demo_tasks import build_demo_corpus
from .loader import TaskLoadError, load_tasks
from .models import Report, RunConfig, utcnow_iso
from .pipeline import PipelineOptions, detect_git_hash, run_pipeline
from .report.html_report import load_report_json, write_report_html, write_report_json
from .report.terminal import findings_table, print_note, print_report_summary
from .runner.providers import PROVIDERS, normalise_provider, provider_defaults
from .runner.sandbox import cleanup_all, docker_available

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Automated calibration and QA gate for human-authored RL/SFT tasks.",
)
console = Console()

PathArg = Annotated[Path, typer.Argument(help="A task directory, or a folder of task directories.")]

ModelOpt = Annotated[str, typer.Option("--model", "-m", help="Model id (for the chosen provider) or 'mock'.")]
ProviderOpt = Annotated[str, typer.Option("--provider", help=f"Provider: {'|'.join(PROVIDERS)}.")]
BaseUrlOpt = Annotated[Optional[str], typer.Option("--base-url", help="Override the chat-completions endpoint (required for openai-compatible).")]
SkipPreflightOpt = Annotated[bool, typer.Option("--skip-preflight", help="Skip the preflight request (mock runs never call the network).")]
MaxTurnsOpt = Annotated[Optional[int], typer.Option("--max-turns", help="Max agent turns per rollout (default: provider-specific).")]
MaxToolOutputOpt = Annotated[Optional[int], typer.Option("--max-tool-output-chars", help="Truncate tool output to this many characters (default: provider-specific).")]
TpmOpt = Annotated[Optional[int], typer.Option("--tpm", help="Tokens-per-minute budget for the pacer (default: provider-specific; 0 disables).")]
RolloutsOpt = Annotated[int, typer.Option("--rollouts", "-n", help="Rollouts per task.")]
ConcurrencyOpt = Annotated[int, typer.Option("--concurrency", "-c", help="Max concurrent rollouts.")]
ShipMinOpt = Annotated[float, typer.Option("--ship-min", help="Lower edge of the target solve-rate window.")]
ShipMaxOpt = Annotated[float, typer.Option("--ship-max", help="Upper edge of the target solve-rate window.")]
MockOpt = Annotated[bool, typer.Option("--mock", help="Simulate every model call and container run.")]
BudgetOpt = Annotated[Optional[float], typer.Option("--budget-usd", help="Hard stop once estimated spend reaches this.")]
TimeoutOpt = Annotated[int, typer.Option("--timeout-s", help="Per-rollout wall-clock timeout, seconds.")]
SeedOpt = Annotated[int, typer.Option("--seed", help="Seed for mock mode and sampling.")]
OutOpt = Annotated[Path, typer.Option("--out", "-o", help="Directory for report.json and report.html.")]


def _config(
    model: str,
    rollouts: int,
    concurrency: int,
    ship_min: float,
    ship_max: float,
    mock: bool,
    budget_usd: float | None,
    timeout_s: int,
    seed: int,
    out: Path,
    *,
    provider: str = "openrouter",
    base_url: str | None = None,
    skip_preflight: bool = False,
    max_turns: int | None = None,
    max_tool_output_chars: int | None = None,
    tpm: int | None = None,
) -> RunConfig:
    try:
        provider = normalise_provider(provider)
    except ValueError as exc:
        print_note(str(exc), console, style="bold red")
        raise typer.BadParameter(str(exc), param_hint="--provider") from exc
    defaults = provider_defaults(provider)
    return RunConfig(
        model=model,
        provider=provider,
        base_url=base_url,
        skip_preflight=skip_preflight,
        rollouts=max(1, rollouts),
        concurrency=max(1, concurrency),
        ship_min=ship_min,
        ship_max=ship_max,
        mock=mock,
        budget_usd=budget_usd,
        timeout_s=timeout_s,
        max_turns=max_turns if max_turns is not None else defaults.max_turns,
        max_tool_output_chars=(
            max_tool_output_chars
            if max_tool_output_chars is not None
            else defaults.max_tool_output_chars
        ),
        tpm=tpm if tpm is not None else defaults.tpm,
        max_completion_tokens=defaults.completion_tokens,
        seed=seed,
        out_dir=str(out),
        git_hash=detect_git_hash(),
    )


def _load(path: Path):
    try:
        return load_tasks(path)
    except TaskLoadError as exc:
        print_note(f"could not load tasks: {exc}", console, style="bold red")
        raise typer.Exit(code=2) from exc


def _emit(report: Report, out: Path, *, html: bool = True) -> Path:
    json_path = write_report_json(report, out)
    if html:
        write_report_html(report, out)
    return json_path


def _wait_for_docker(attempts: int = 3, delay_s: float = 2.0) -> bool:
    """Probe for a container runtime, re-probing a few times before giving up.

    A daemon that is starting (or was just restarted) can be reachable a few
    seconds later, so a single failed probe is not enough to declare Docker down.
    """
    if docker_available():
        return True
    for _ in range(max(0, attempts)):
        time.sleep(delay_s)
        if docker_available():
            return True
    return False


def _run(
    tasks,
    config: RunConfig,
    options: PipelineOptions,
    *,
    label: str,
) -> Report:
    console.print(f"[bold]{label}[/]: {len(tasks)} task(s), model {config.model}")
    explicit_mock = bool(config.mock or config.model in ("mock", ""))
    if options.calibrate and explicit_mock:
        print_note("mock mode: no API key needed, no containers, everything labelled SIMULATED.", console)
    if options.use_docker and not explicit_mock and not _wait_for_docker():
        # A real calibration needs a container runtime. Degrading to the mock
        # agent here would publish simulations as if they were measurements, so
        # refuse instead of falling back silently.
        print_note(
            "Docker daemon not reachable after re-probing: a real calibration needs a "
            "container runtime, and TaskGate will not fall back to the mock agent. "
            "Start Docker and retry, or pass --mock for a clearly-labelled simulation.",
            console,
            style="bold red",
        )
        raise typer.Exit(code=3)
    if options.use_docker and explicit_mock and not docker_available():
        print_note("Docker daemon not reachable: static audit only; dynamic findings are simulated.", console)

    def on_task(task, findings):
        blockers = sum(1 for f in findings if f.severity.value == "BLOCKER")
        console.print(f"  audited [cyan]{task.task_id}[/] — {len(findings)} finding(s), {blockers} blocker(s)")

    try:
        return asyncio.run(run_pipeline(tasks, config, options=options, on_task=on_task))
    except KeyboardInterrupt:
        cleanup_all()
        print_note("interrupted: containers and images torn down.", console)
        raise typer.Exit(code=130) from None
    finally:
        cleanup_all()


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
@app.command()
def audit(
    path: PathArg,
    out: OutOpt = Path("taskgate_out"),
    dynamic: Annotated[bool, typer.Option("--dynamic/--no-dynamic", help="Run container checks when Docker is available.")] = True,
) -> None:
    """Static and dynamic lint. No model calls, no calibration spend."""
    tasks = _load(path)
    config = _config("mock", 1, 1, 0.10, 0.30, True, None, 900, 1234, out)
    options = PipelineOptions(audit=True, dynamic=dynamic, calibrate=False, diagnose=False)
    report = _run(tasks, config, options, label="audit")
    for task in report.tasks:
        if task.findings:
            console.print(findings_table(task))
    print_report_summary(report, console)
    json_path = _emit(report, out)
    print_note(f"wrote {json_path} and {out / 'report.html'}", console, style="green")


@app.command()
def calibrate(
    path: PathArg,
    model: ModelOpt = "mock",
    provider: ProviderOpt = "openrouter",
    base_url: BaseUrlOpt = None,
    skip_preflight: SkipPreflightOpt = False,
    max_turns: MaxTurnsOpt = None,
    max_tool_output_chars: MaxToolOutputOpt = None,
    tpm: TpmOpt = None,
    rollouts: RolloutsOpt = 20,
    concurrency: ConcurrencyOpt = 4,
    ship_min: ShipMinOpt = 0.10,
    ship_max: ShipMaxOpt = 0.30,
    mock: MockOpt = False,
    budget_usd: BudgetOpt = None,
    timeout_s: TimeoutOpt = 900,
    seed: SeedOpt = 1234,
    out: OutOpt = Path("taskgate_out"),
    audit_first: Annotated[bool, typer.Option("--audit/--no-audit", help="Run the static audit first (recommended).")] = True,
) -> None:
    """Run N rollouts per task, compute the solve rate and route each task."""
    tasks = _load(path)
    config = _config(
        model, rollouts, concurrency, ship_min, ship_max, mock, budget_usd, timeout_s,
        seed, out, provider=provider, base_url=base_url, skip_preflight=skip_preflight,
        max_turns=max_turns, max_tool_output_chars=max_tool_output_chars, tpm=tpm,
    )
    options = PipelineOptions(audit=audit_first, dynamic=audit_first, calibrate=True, diagnose=True)
    report = _run(tasks, config, options, label="calibrate")
    print_report_summary(report, console)
    _emit(report, out)
    print_note(f"wrote {out / 'report.json'} and {out / 'report.html'}", console, style="green")


@app.command()
def run(
    path: PathArg,
    model: ModelOpt = "mock",
    provider: ProviderOpt = "openrouter",
    base_url: BaseUrlOpt = None,
    skip_preflight: SkipPreflightOpt = False,
    max_turns: MaxTurnsOpt = None,
    max_tool_output_chars: MaxToolOutputOpt = None,
    tpm: TpmOpt = None,
    rollouts: RolloutsOpt = 20,
    concurrency: ConcurrencyOpt = 4,
    ship_min: ShipMinOpt = 0.10,
    ship_max: ShipMaxOpt = 0.30,
    mock: MockOpt = False,
    budget_usd: BudgetOpt = None,
    timeout_s: TimeoutOpt = 900,
    seed: SeedOpt = 1234,
    out: OutOpt = Path("taskgate_out"),
) -> None:
    """Full pipeline: audit -> calibrate -> report."""
    tasks = _load(path)
    config = _config(
        model, rollouts, concurrency, ship_min, ship_max, mock, budget_usd, timeout_s,
        seed, out, provider=provider, base_url=base_url, skip_preflight=skip_preflight,
        max_turns=max_turns, max_tool_output_chars=max_tool_output_chars, tpm=tpm,
    )
    options = PipelineOptions(audit=True, dynamic=True, calibrate=True, diagnose=True)
    report = _run(tasks, config, options, label="run")
    print_report_summary(report, console)
    _emit(report, out)
    print_note(f"wrote {out / 'report.json'} and {out / 'report.html'}", console, style="green")


@app.command()
def report(
    results_dir: Annotated[Path, typer.Argument(help="Directory containing report.json.")],
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Where to write report.html (default: in place).")] = None,
) -> None:
    """Regenerate report.html from a stored report.json."""
    json_path = results_dir / "report.json"
    if not json_path.is_file():
        print_note(f"no report.json in {results_dir}", console, style="bold red")
        raise typer.Exit(code=2)
    loaded = load_report_json(json_path)
    target = write_report_html(loaded, out or results_dir)
    print_note(f"wrote {target}", console, style="green")


@app.command()
def demo(
    work: Annotated[Path, typer.Option("--work", help="Working directory for the generated corpus and results.")] = Path("taskgate_demo"),
    rollouts: RolloutsOpt = 20,
    seed: SeedOpt = 1234,
    concurrency: ConcurrencyOpt = 4,
    mock: MockOpt = True,
    model: ModelOpt = "mock",
    open_report: Annotated[bool, typer.Option("--open/--no-open", help="Open report.html in a browser.")] = True,
    docker: Annotated[
        bool,
        typer.Option("--docker/--no-docker", help="Run the real container audit (slow, several minutes per task)."),
    ] = False,
) -> None:
    """Build the 8-task demo corpus, run the full pipeline in mock mode, open the report."""
    corpus = Path(work) / "tasks"
    results = Path(work) / "results"
    build_demo_corpus(corpus, force=True)
    print_note(f"built demo corpus in {corpus}", console, style="green")

    config = _config(model, rollouts, concurrency, 0.10, 0.30, mock, None, 900, seed, results)
    options = PipelineOptions(audit=True, dynamic=True, calibrate=True, diagnose=True, use_docker=docker)
    report = _run(_load(corpus), config, options, label="demo")

    print_report_summary(report, console)
    _emit(report, results)

    routes = {t.task_id: t.route for t in report.tasks}
    expected = _expected_routes(corpus)
    mismatches = [
        f"{task_id}: got {routes.get(task_id).value if routes.get(task_id) else '?'} want {want}"
        for task_id, want in expected.items()
        if routes.get(task_id) and routes[task_id].value != want
    ]
    if expected and docker:
        print_note(
            "route expectations assume simulated dynamic findings and were not checked "
            "because the real container audit ran (e.g. t06 only gets S007 warnings).",
            console,
        )
    elif expected:
        if mismatches:
            print_note("route mismatches vs expectations: " + "; ".join(mismatches), console, style="bold red")
        else:
            print_note(f"all {len(expected)} demo tasks reached their expected route.", console, style="green")

    html_path = results / "report.html"
    print_note(f"report: {html_path}", console, style="green")
    if open_report:
        try:
            webbrowser.open(html_path.resolve().as_uri())
        except Exception:  # noqa: BLE001 - headless environment
            print_note("could not open a browser; open the file above manually.", console)


def _expected_routes(corpus: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for task in load_tasks(corpus):
        demo = task.demo
        if demo and demo.expected_route:
            expected[task.task_id] = demo.expected_route
    return expected


@app.command()
def version() -> None:
    """Print the TaskGate version."""
    console.print(f"taskgate {__version__} · {utcnow_iso()}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
