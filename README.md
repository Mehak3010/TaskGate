# TaskGate

**An automated calibration and QA gate for human-authored RL/SFT tasks.**

TaskGate takes a folder of Docker-based terminal/coding tasks and does the
repetitive QA work around them:

1. **Lint** each task for determinism, environment hygiene, and
   reward-hacking/leak risks (static AST/filesystem rules + container probes).
2. **Calibrate**: run N model rollouts per task in sandboxed containers and
   measure the solve rate *with a confidence interval*, not just a point estimate.
3. **Route** each task to `SHIP` / `REWRITE` / `DROP` / `NEEDS-MORE-RUNS` (plus
   `BLOCKED`) with a human-readable diagnosis.
4. **Report**: a terminal summary plus `report.json` and a single-file,
   offline `report.html`.

The creative task writing stays human. TaskGate removes the manual loop of
"hand-tune a task, run the model 20–50 times, compute the pass rate, argue about
fairness in a PR".

## Status: what is verified and what is not

| | Status |
|---|---|
| `taskgate demo` | **Simulated.** Mock agent, no containers. Every number is labelled SIMULATED. |
| `taskgate audit` (static + dynamic) | **Real.** Verified on Docker Desktop (WSL2) for the demo tasks: image build, 5 reference runs, network-off run and mutation probe are recorded in `report.json`. |
| `taskgate calibrate` with a real model | Implemented; only a small plumbing test has been run on free-tier models. Not a measurement. |
| Public-benchmark test | Static audit of 66 Terminal-Bench tasks. See [docs/real_findings.md](docs/real_findings.md): **no confirmed defects found**, rules tightened from what the review showed. |
| Not built | Dynamic audit for separate-verifier tasks (Terminal-Bench format), visual verifier stack, trace recorder, dataset ETL. |

## The problem

Calibrating one task by hand looks like this: write it, write the hidden tests,
write a reference solution, run it in a container a few times, run the model
20–50 times, compute a pass rate in a notebook, discover the task leaked its
tests into the image three weeks later. The expensive part is not judgement, it
is repetition — and repetition is exactly what should be automated.

TaskGate makes the machine do the repetition and a human do the deciding.

## 60-second quickstart

```bash
uv sync
uv run taskgate demo
```

No API key, no network beyond image pulls (and none at all if you skip the
dynamic audit). It builds an 8-task demo corpus, runs the whole pipeline in mock
mode, prints a routed summary table and opens `report.html`.

Against your own tasks:

```bash
uv run taskgate audit     tasks/my_task          # lint only, no spend
uv run taskgate calibrate tasks/ --model mock    # rollouts + stats + routing
uv run taskgate run       tasks/ --model openai/gpt-4o-mini --rollouts 30
uv run taskgate report    taskgate_out           # regenerate HTML from report.json
```

Common flags: `--model`, `--rollouts N` (default 20), `--concurrency C`
(default 4), `--ship-min`, `--ship-max`, `--mock`, `--budget-usd`,
`--timeout-s`, `--out DIR`, `--seed`.

## Architecture

```
taskgate/
  cli.py                 Typer app: audit / calibrate / run / report / demo
  pipeline.py            orchestration: audit -> dynamic -> calibrate -> diagnose -> report
  models.py              Pydantic v2 models; the JSON schema is the source of truth
  loader.py              TaskLoader -- the ONLY place that knows the task format
  audit/
    rules.py             stable rule registry (S001-S010, D000-D005)
    static_checks.py     AST + filesystem rules, one function per rule
    dynamic_checks.py    container probes D000-D005 (degrade when Docker is down)
  runner/
    sandbox.py           Docker SDK: build/run/exec/copy/teardown + Ctrl-C cleanup
    agent.py             minimal OpenRouter bash-tool agent loop
    mock_agent.py        deterministic fake agent (seeded true pass rate)
    executor.py          asyncio semaphore pool, retries, budget guard
    pricing.py           token pricing for the budget guard and cost column
  stats/
    calibration.py       Wilson CI, pass@k, routing -- pure functions
  diagnose/
    heuristics.py        zero-cost diagnosis (always available)
    llm_diagnosis.py     optional OpenRouter diagnosis, schema-validated
  report/
    html_report.py       single-file offline HTML report (inline SVG chart)
    terminal.py          Rich summary tables
  demo/
    build_demo_tasks.py  the 8 seeded demo tasks
tests/                   pytest suite for TaskGate itself
```

**Two adapters, that is the whole integration surface:**

* a different task format → edit [`loader.py`](taskgate/loader.py);
* a different model or agent scaffold → implement the same
  `async def run_rollout(index) -> Rollout` as
  [`mock_agent.py`](taskgate/runner/mock_agent.py) / [`agent.py`](taskgate/runner/agent.py).

### Assumed task format

Terminal-Bench / Harbor style. TaskGate assumes this layout; if the real format
differs, only `loader.py` changes.

```
tasks/<task_id>/
  task.yaml            # id, instruction, difficulty, timeouts, author, allow_network
  Dockerfile           # environment
  docker-compose.yaml  # optional
  instruction.md       # what the agent sees
  tests/               # hidden verifier (pytest or shell) -- NOT agent visible
  solution/            # reference solution (solve.sh or a patch)
```

## Rule reference

Findings are `{rule_id, severity, file, line, message, fix_hint}`. Rule IDs are
stable forever so two runs produce diffable reports.

| ID | Check | Severity |
|----|-------|----------|
| S001 | Hidden tests reachable from the agent's filesystem (Dockerfile COPY/mount) | BLOCKER |
| S002 | Solution or expected-output files baked into the image | BLOCKER |
| S003 | Test assertions compare against literals that also appear in `instruction.md` | WARN |
| S004 | Tests only check existence / non-emptiness, not behaviour | WARN |
| S005 | Unpinned dependencies (`pip install x`, `:latest` tags) | WARN |
| S006 | Network use in tests or at run time (`curl`, `requests.get`, `git clone`, …) | BLOCKER |
| S007 | Non-determinism in tests (unseeded RNG, `time.time()`, `datetime.now()`, sleep-based sync) | WARN |
| S008 | Declared output format is not strictly parsed/asserted by the tests | WARN |
| S009 | Instruction may be ambiguous (no output path, no format, unresolved pronoun) | INFO |
| S010 | The task can be solved by editing the tests (tests writable by the agent user) | BLOCKER |
| D000 | Task image failed to build | BLOCKER |
| D001 | Reference solution is flaky or broken across K fresh runs | BLOCKER |
| D002 | Tests pass on an unmodified environment (vacuous task) | BLOCKER |
| D003 | Hidden network dependency (passes with networking, fails without) | BLOCKER |
| D004 | Tests are too weak to detect a trivial sabotage of the solution | WARN |
| D005 | Image build is not reproducible (two builds install different package sets) | BLOCKER |

**Any BLOCKER stops the task and no calibration spend happens for it.**

## Statistical methodology

This is the credibility core, so it is deliberately conservative.

* **Solve rate** `p̂ = passes / valid_rollouts`. Rollouts that fail with an
  infrastructure error are excluded from the denominator, not counted as task
  failures; the infrastructure-error rate is tracked and warned about above 10%.
* **Wilson score interval at 90% confidence.** Chosen over the normal
  approximation because it stays inside `[0, 1]` and behaves sensibly for `0/n`
  and `n/n`. **At N = 20, a measured 20% has an interval of roughly 8–42%.** A
  point estimate from 20 rollouts is not a decision, and the report says so.
* **pass@k** via the unbiased estimator `1 − C(n−c, k) / C(n, k)` for
  `k ∈ {1, 3, 5}` when `N ≥ k`.
* **Routing** (priority order, first match wins):

  | Verdict | Condition |
  |---------|-----------|
  | `BLOCKED` | any BLOCKER audit finding (calibration skipped) |
  | `NEEDS-MORE-RUNS` | fewer than 5 valid rollouts, or the CI straddles a window boundary |
  | `DROP` | upper CI bound `< ship_min` (too hard), or lower bound `≥ 0.80` (degenerately easy) |
  | `SHIP` | point estimate inside `[ship_min, ship_max]` |
  | `REWRITE` | CI entirely above the window but not degenerate, or out of window and unresolved |

  Messages for recoverable causes (underspecified instruction, too-strict test,
  environment issue, shortcut available) turn a `DROP`/`NEEDS-MORE-RUNS` into a
  `REWRITE`. `NEEDS-MORE-RUNS` also reports a suggested extra N computed to
  resolve the boundary.

Default window is 10–30% (`--ship-min 0.10 --ship-max 0.30`).

## Diagnosis

For out-of-window tasks TaskGate produces an **advisory** diagnosis:

* with `OPENROUTER_API_KEY` set, it asks a model for a JSON answer validated
  against a closed cause set (`underspecified_instruction`, `trivially_solvable`,
  `test_too_strict`, `environment_issue`, `genuinely_hard`, `shortcut_available`).
  Answers that fail schema validation are **stripped**, not shown;
* without a key, heuristics are used (all failures hitting the turn limit →
  likely too hard; passes in under 3 turns → trivially solvable; a dominant
  single failing assertion → test too strict).

A human always decides. TaskGate never auto-approves a task.

## Testing against a real model (OpenRouter)

The agent cannot verify this without a key, so here is how to do it yourself:

```bash
export OPENROUTER_API_KEY=sk-or-...            # PowerShell: $env:OPENROUTER_API_KEY="sk-or-..."
docker info                                     # a running daemon is required
uv run taskgate calibrate taskgate_demo/tasks/t01_log_parser \
    --model openai/gpt-4o-mini --rollouts 10 --out taskgate_out
```

What to expect: real containers are built and torn down, the agent loop calls
OpenRouter with a single `bash` tool, the hidden tests are mounted only after the
agent stops, and the report is no longer labelled SIMULATED. Start with
`--rollouts 5 --budget-usd 1` to keep the first run cheap.

## Reports

* **Terminal**: a Rich table — task, audit counts, solve rate with CI, pass@1,
  route, cost, top finding.
* **`report.json`**: the complete, schema-validated result (`Report.model_dump`),
  including config, git hash, model, seed and timestamps.
* **`report.html`**: a single file with no CDN dependency, readable offline. It
  contains a visible `SIMULATED` banner when applicable, summary tiles, a
  solve-rate chart (one dot per task with a 90% CI whisker against the shaded
  10–30% window), a sortable table with expandable rows (findings with file and
  line, per-rollout results, diagnosis, transcript snippets), and a methodology
  footer.

## Demo corpus

`taskgate demo` generates 8 small, fully working tasks; each triggers exactly one
story. Routes are deterministic with the default seed.

| Task | Planted issue | Expected route |
|------|---------------|----------------|
| `t01_log_parser` | healthy control, true pass rate ≈ 20% | SHIP |
| `t02_csv_dedupe` | healthy control, ≈ 15% | SHIP |
| `t03_trivial_rename` | degenerately easy, ≈ 95% | DROP (`trivially_solvable`) |
| `t04_ambiguous_spec` | instruction never names the output path | REWRITE (`underspecified_instruction`) |
| `t05_leaky_tests` | `tests/` copied into the image | BLOCKED (S001) |
| `t06_flaky_timing` | sleep-based sync, unseeded RNG, wall-clock reads | BLOCKED (D001 / S007) |
| `t07_network_dep` | `requests.get` in the test, unpinned `pip install` | BLOCKED (S006 / D003) |
| `t08_borderline` | true rate ≈ 28%, N = 20 | NEEDS-MORE-RUNS |

Mock mode reproduces these routes deterministically. Real-model runs on the demo
tasks are a bonus, not a requirement.

## Limitations (read this before quoting any number)

* Static rules are heuristics: they produce false positives and false negatives.
  Findings are routed to humans and TaskGate never auto-approves.
* Solve-rate estimates at N ≈ 20 are wide. The tool shows the interval instead of
  hiding it.
* The minimal agent loop is **not** the client's real agent scaffold, so a
  measured solve rate may differ from the client's own. A pluggable
  `AgentAdapter` is the extension point.
* The task directory layout is an assumption (see above).
* **Mock-mode numbers are simulations and must never be presented as
  measurements.** Every mock report carries a visible SIMULATED banner.
* Docker is required for the dynamic audit and for real rollouts. Without a
  daemon, TaskGate falls back to the static audit plus mock calibration and says
  so in the report and in the terminal.

## Non-functional behaviour

* Agent containers run as a non-root user, with CPU/memory/pid limits, with
  networking off unless the task allows it, and with a read-only root where
  feasible. The host Docker socket is never mounted into a container.
* Containers and images are registered for teardown and cleaned up on the normal
  path, on an exception, and on Ctrl-C.
* Secrets come from environment variables only; they are never written to a log
  or a report. The API key is never echoed.
* A failure in one task never aborts the run: each task carries its own status
  and its error is captured in the report.

## Roadmap (documented, deliberately not built)

* A local specialist verifier stack for visual/UI tasks (OpenCV / CLIP /
  imagehash) instead of frontier VLM judges.
* A terminal trajectory recorder and sanitiser for SFT/PRM JSONL.
* Dataset ETL, schema conversion, and MinHash/embedding originality checks.
* A task scaffolding generator.

## Development

```bash
uv sync
uv run pytest
```

The test suite covers every static rule (positive and negative case), Wilson CI
and pass@k against known values, the routing decision table including boundary
cases, the budget guard, infrastructure-error exclusion, mock determinism, and an
integration test asserting all 8 demo routes.
