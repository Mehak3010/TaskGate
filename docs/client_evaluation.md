# Independent client evaluation

An AI agent that had never seen the code was told to use TaskGate like a customer: learn it from the
README and `--help` only, write its own tasks, plant known defects, and report what the audit caught.
It wrote six tasks (five in the TaskGate layout, one in the Terminal-Bench layout) and recorded the
ground truth **before** auditing. The static results were then reproduced independently against
commit `06f58eb`. The numbers below are from those runs.

## Static audit, before and after

| Task | Planted defect | Before | After |
|---|---|---|---|
| `c01_nginx_log_report` | none (healthy control) | `S003` INFO | `S003` INFO (unchanged: a human check by design) |
| `c02_leaky_normaliser` | hidden tests copied into the image | `S001` BLOCKER, `Dockerfile:5` | same |
| `c03_flaky_metrics` | `sleep`, clock and unseeded `random` in tests | `S007` x3 (5 present; `random.uniform` missed) | `S007` x5 |
| `c04_ambiguous_clean` | instruction never names the output | **missed** (0 findings) | `S009` x3 |
| `c05_vacuous_env` | tests only check the file exists / is non-empty | **missed**, plus an `S009` false alarm | `S004` WARN, false alarm gone |
| `tb01_port_check` | none (Terminal-Bench control) | `S009` false alarm | clean |

Root causes: `S007` only listed six `random` functions; `S004` did not treat
`stat().st_size > 0` as a bare emptiness check; `S009` counted an input file name (`raw_sales.csv`)
as both an output path and a format statement.

## Other findings

| Finding | Status |
|---|---|
| One invalid `task.yaml` aborted the whole folder (exit 2, nothing audited) | Fixed: the task becomes a `LOAD-ERROR` BLOCKER with the field and a valid example; the rest are audited |
| Duplicate task ids produced two identical rows | Fixed: the later one is reported as a `LOAD-ERROR` |
| A missing Dockerfile was skipped silently | Fixed: reported as `D000` |
| `calibrate --mock` still ran the container audit (about 7 to 11 minutes) | Fixed: offline by default with `--mock`; `--dynamic/--no-dynamic` added |
| Exit code was 0 even with BLOCKED tasks | Added `--fail-on-blocker` |
| Terminal-Bench container audit failed with `D000` | Fixed in two steps. The image is now built from `environment/`, and the verifier is now `tests/test.sh` instead of pytest, which had produced a false `D001`. Confirmed on Docker with the evaluation's `tb01` task: build ok, reference solution 5/5 in fresh containers, verifier fails on an untouched environment, network-off run passes, sabotage probe 2/2 |
| `task.yaml` fields were undocumented | README "Task format" added |
| README said networking is off "unless the task allows it" | Corrected: `allow_network` and `timeout_s` are parsed but not yet used |

## Still open

* The mutation probe only deletes the output, so tests that accept an empty or placeholder file are
  not caught by it (the static rule `S004` now covers the common patterns).
* Real Terminal-Bench tasks (as opposed to the one written for this evaluation) have not been run
  through the container audit, and a separate verifier image (`tests/Dockerfile`) is not supported.
* Shell verifiers (`tests/test.sh`) are not analysed statically. The evaluation's `tb01` task passes the
  audit although its `test.sh` never checks what the script prints; the sabotage probe caught 2/2
  mutations only because breaking the solution removes the script entirely.
* `S003` is noisy on a healthy task when the instruction contains an example value.
* There is no `taskgate init` scaffold yet.

## Limits of this evaluation

Six tasks, one evaluator, and defects planted by the same agent that wrote the tasks. It shows the
audit's blind spots on those patterns; it is not a measured recall figure.
