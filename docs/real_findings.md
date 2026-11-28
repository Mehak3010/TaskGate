# TaskGate on a public benchmark: Terminal-Bench (static audit)

**Short version:** TaskGate's static audit ran on 66 public Terminal-Bench tasks. I hand-read
20 of its findings against the task files. **None was a confirmed task defect.** Every one was
deliberate design in a carefully hardened benchmark, which means the original rules were too
noisy. I tightened six of them from what I read, added regression tests, and re-ran. This
document reports both the numbers and that result honestly.

## What was audited

| | |
|---|---|
| Dataset | Terminal-Bench, https://github.com/harbor-framework/terminal-bench |
| License | Apache-2.0 (LICENSE read in the repo) |
| Tasks audited | 66 (every directory under `tasks/` with `task.toml`) |
| Mode | Static audit only (`--no-dynamic`). No containers, no model calls, $0. |
| Dynamic audit | Not applied. All tasks use a separate verifier image (own `tests/Dockerfile`, RewardKit, artifacts copied between containers). Supporting that is roadmap, not built. |

The only adaptation for this format is in `loader.py` (reads `task.toml` and
`environment/Dockerfile`), plus build-context awareness in S001/S002/S010.

## Counts

First run (original rules) versus the run after precision fixes:

| Rule | Before: findings (tasks) | After: findings (tasks), severity | Note |
|---|---|---|---|
| S001 | crashed on 59 tasks, 1 finding | 0 | crash fixed; the 1 finding was a false positive |
| S002 | 1 (1) BLOCKER | 0 | false positive |
| S003 | 37 (15) WARN | 34 (13) INFO | specified output literals, now advisory |
| S004 | 11 (11) WARN | 2 (2) WARN | harness asserts excluded; the 2 left were not hand-reviewed |
| S005 | 54 (52) | 54 (52) INFO | apt unpinned is near-universal; not a defect signal |
| S006 | 35,733 (66) BLOCKER | 136 (23) WARN | lock files, docs, localhost excluded |
| S007 | 180 (42) WARN | 180 (42) WARN | not reviewed in depth |
| S009 | 80 (33) INFO | 80 (33) INFO | advisory |
| S008, S010 | 0 | 0 | not applicable to this layout |

Routes after the fixes: 66 AUDIT-ONLY, 0 BLOCKED. Before: 66 of 66 BLOCKED (meaningless).

## Hand review: 20 findings, 0 confirmed defects

| Rule | Task and location | What the code actually does | Verdict |
|---|---|---|---|
| S001 | medical-claims-processing `Dockerfile:17` | `mkdir -p ... /tests ...` creates an empty directory; tests live in a separate verifier image | False positive |
| S002 | medical-claims-processing `Dockerfile:22` | Copies `shared_seed/reference/` (T-* dev cases). The instruction sends the agent to `/shared/reference/` on purpose; hidden checks use separate R-* cases (no T-* ids appear in the hidden fixtures) | False positive (by design) |
| S003 | bun-sourcemap-leak `tests/checks.py:350` | Asserts output `Hello, Ada!`, which the instruction specifies as the required output | False positive (output contract) |
| S003 | wdm-design `tests/checks.py:134` | Asserts dtype `float64`, specified in the instruction | False positive |
| S003 | shadow-relay `tests/checks.py:177` | Asserts a 64-char key; the instruction states the format | False positive |
| S003 | production-planning `tests/checks.py:586` | Asserts status `WIP_CONT`, a field value named in the instruction | False positive |
| S003 | fp8-rmsnorm-gemm, lake-temp-glm `tests/uv.lock` | A lock file scanned as test code | Tool bug, fixed |
| S004 | freecad-impeller `tests/report.py:30`; intrastat-meldung `tests/checks.py:293`; ctr-optimization `tests/checks.py:143` | `assert rows, "RewardKit produced no criterion results"` inside reporting code | False positive (harness sanity check) |
| S004 | vba-userform-port `tests/test_trace_runner.py:125`, `tests/test_scoring.py:74` | Asserts inside a worker class and after scoring (reward file written) | False positive |
| S004 | fp8-rmsnorm-gemm `tests/conftest.py:254` | `assert not bad` on banned imports in a fixture helper | False positive |
| S006 | live-database-cutover `tests/test.sh:130` | Verifier installs the agent-modified `requirements.txt`; a pre-flight control turns an unresolvable pristine baseline into an *infra error*, not a score of 0 | By design, handled by the authors |
| S006 | payments-pipeline-fix `tests/test.sh:53` | `pip install` of agent-supplied worker dependencies, commented as impossible to bake at build time | By design; failure handling not checked |
| S006 | vba-userform-port `tests/test.sh:30` | Installs the submitted backend requirements before grading | By design; failure handling not checked |
| S006 | freight-dispatch-shift `tests/checks.py:22` | `urllib` imports for a local HTTP server in the verifier | False positive |
| S006 | cargo-flight-dispatch `tests/verify.py:47` | `uv pip install` of agent-supplied requirements inside a disposable worker | By design |
| S007 | heat-pump-warranty `tests/checks.py:75` | `time.time()` used to timestamp the result file | False positive |
| S007 | nextjs-performance `tests/run_app.py:42` | 50 ms poll loop while waiting for a process | Normal, low risk |
| S007 | wal-recovery-ordering `tests/_hidden_outputs.py:241` | `sleep(0.01)` between concurrent commits and a crash snapshot; the expected state is recomputed from the snapshot | Possibly intentional; **unverified, no claim made** |

Reviewed: 1 + 1 + 5 + 6 + 5 + 3 = 21 locations across 20 findings and 1 tool bug. I did not review
every finding (for example most S007 and S009 hits, and most S003 hits beyond the four listed).

## What this shows

- **Terminal-Bench tasks are carefully built.** The authors already separate the verifier,
  expose dev data on purpose, and guard infrastructure failures. A static heuristic audit
  finds no defects in the 20 findings I read. I did not find anything to report upstream and
  I am not claiming that any task is broken.
- **The original heuristics were too eager.** Counting thousands of BLOCKER findings and
  blocking all 66 tasks would have made the tool look broken. Reading findings against code
  is what found the problem, and the rule changes below came from that.
- **The route column is only as good as the rules.** After the fixes no task is BLOCKED.

## Rule changes made from this review (each has a regression test)

| Rule | Change |
|---|---|
| S001 | Crash on `RUN` lines ending in `\` fixed. `mkdir` of a `tests` directory no longer counts as a leak. |
| S002 | Skipped when the instruction names the copied destination path (intentional reference data). |
| S003 | Only code files. Severity INFO (specified output contract versus real leak needs a human). |
| S004 | Only assertions inside `test*`/`check*`/`verify*` functions or decorated criteria. |
| S005 | Parses only the `apt-get install` segment; unpinned apt packages are INFO. |
| S006 | Skips lock files, docs, imports and localhost. BLOCKER only when the line visibly reaches an external URL. Installs and unresolved targets are WARN. |

## Limits of this evidence

- Heuristic rules have both false positives and false negatives. This review measured the
  former on one benchmark. It did **not** measure what the rules miss.
- Terminal-Bench is an expert-reviewed public benchmark, the hardest case for a static
  audit. This result says little about draft tasks that have not been through that review.
- 20 reviewed findings is a sample, not a precision estimate.
- No dynamic audit and no calibration were run on this set.
- Not yet verified on my side: the Docker path for the Terminal-Bench format, Ctrl-C cleanup
  on a real interrupted run (this was verified on my machine for the demo task only).
