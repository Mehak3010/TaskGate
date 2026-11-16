"""Deterministic fake agent.

Draws pass/fail from a per-task true pass rate with a seeded RNG and produces
plausible transcripts, so ``taskgate demo`` is free, fast and reproducible.

Everything this module produces is labelled SIMULATED: mock-mode numbers are
simulations and must never be presented as measurements.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..models import Rollout, Task, TranscriptStep
from .pricing import estimate_cost_usd

#: Rough token accounting per agent turn, used for cost simulation.
TOKENS_IN_PER_TURN = 520
TOKENS_OUT_PER_TURN = 90


@dataclass
class MockOutcome:
    index: int
    passed: bool
    status: str = "ok"
    turns: int = 0
    transcript: list[TranscriptStep] = field(default_factory=list)
    verifier_output: str = ""
    verifier_exit_code: int | None = 0
    error: str | None = None


_EXPLORE_STEPS = (
    ("ls -la /app/data", "total 24\ndrwxr-xr-x 2 root root 4096 Jun  2 09:14 .\n-rw-r--r-- 1 root root 2211 Jun  2 09:14 part0.log"),
    ("head -n 12 /app/data/part0.log", "2024-05-01T04:12:07 INFO service=cache msg=\"cache warmed\"\n2024-05-01T04:12:33 WARN service=search msg=\"slow query\""),
    ("wc -l /app/data/*", " 40 /app/data/part0.log\n 40 /app/data/part1.log"),
    ("find /app -maxdepth 2 -type f -name '*.txt'", "/app/requirements.txt"),
)


def _verifier_output(passed: bool, task: Task, profile: str) -> str:
    test_file = "tests/test_output.py"
    tests = sorted(p.name for p in (task.path / "tests").glob("test_*.py")) if task.tests_dir else []
    if tests:
        test_file = f"tests/{tests[0]}"
    if passed:
        return (
            f"$ python -m pytest -q /tests\n...                                                                      [100%]\n"
            f"3 passed in 1.42s"
        )
    if profile == "wrong_path":
        return (
            f"$ python -m pytest -q /tests\nF..                                                                      [100%]\n"
            f"FAILED {test_file}::test_output_exists\n"
            "E   FileNotFoundError: [Errno 2] No such file or directory: '/app/out/errors.json'\n"
            "1 failed, 2 passed in 0.83s"
        )
    if profile == "turn_limit":
        return (
            f"$ python -m pytest -q /tests\nFFFF                                                                     [100%]\n"
            f"FAILED {test_file}\n"
            "E   AssertionError: expected artifact was never written\n"
            "4 failed in 0.61s"
        )
    return (
        f"$ python -m pytest -q /tests\nF.F                                                                      [100%]\n"
        f"FAILED {test_file}::test_shape\n"
        "E   AssertionError: missing key 'hours'\n"
        "1 failed, 2 passed in 0.77s"
    )


def _transcript(rng: random.Random, task: Task, passed: bool, turns: int, profile: str) -> list[TranscriptStep]:
    steps: list[TranscriptStep] = [
        TranscriptStep(turn=1, role="agent", text="Let me inspect the task inputs first."),
        TranscriptStep(turn=1, role="tool", text=_EXPLORE_STEPS[0][1]),
    ]
    budget = max(turns - 2, 1)
    step_index = 1
    while len(steps) < budget and step_index < len(_EXPLORE_STEPS):
        command, output = _EXPLORE_STEPS[step_index]
        steps.append(TranscriptStep(turn=len(steps) + 1, role="agent", text=f"Running `{command}`."))
        steps.append(TranscriptStep(turn=len(steps) + 1, role="tool", text=output))
        step_index += 1
        if len(steps) >= 6:
            break

    if profile == "wrong_path" and not passed:
        steps.append(
            TranscriptStep(
                turn=len(steps) + 1,
                role="agent",
                text="Writing the summary to /app/summary.json and reporting done.",
            )
        )
        steps.append(
            TranscriptStep(turn=len(steps) + 1, role="tool", text="wrote 4 lines to /app/summary.json")
        )
    elif profile == "turn_limit" and not passed:
        steps.append(
            TranscriptStep(
                turn=len(steps) + 1,
                role="agent",
                text="Still exploring: let me check a few more edge cases in the data before writing.",
            )
        )
    elif passed:
        steps.append(
            TranscriptStep(
                turn=len(steps) + 1,
                role="agent",
                text="The output file is written and matches the requested schema. Done.",
            )
        )
    else:
        steps.append(
            TranscriptStep(
                turn=len(steps) + 1,
                role="agent",
                text="I wrote a best-effort output file and I am done.",
            )
        )
    return steps


class MockAgent:
    """Deterministic stand-in for the real agent loop."""

    def __init__(
        self,
        task: Task,
        *,
        seed: int = 1234,
        n_rollouts: int = 20,
        max_turns: int = 30,
        default_pass_rate: float = 0.5,
        infra_error_rate: float = 0.0,
        model: str = "mock",
    ) -> None:
        self.task = task
        self.seed = seed
        self.n_rollouts = n_rollouts
        self.max_turns = max_turns
        self.model = model
        self.demo = task.demo
        self.true_pass_rate = (
            self.demo.true_pass_rate
            if self.demo and self.demo.true_pass_rate is not None
            else default_pass_rate
        )
        self.profile = self.demo.transcript_profile if self.demo else "normal"
        self.infra_error_rate = infra_error_rate
        self._plan: list[MockOutcome] | None = None

    # -- deterministic schedule -------------------------------------------
    def plan(self) -> list[MockOutcome]:
        """Pre-draw every outcome in index order, so concurrency cannot change it.

        Pass/fail and turn counts come from two *independent* streams. Sharing one
        stream would let the number of draws taken for the turn count feed back
        into the outcome schedule, so the realised rate would drift away from the
        declared ``true_pass_rate`` and the demo would stop being predictable.
        """
        if self._plan is not None:
            return self._plan
        outcome_rng = random.Random(f"{self.seed}:{self.task.task_id}:passes")
        detail_rng = random.Random(f"{self.seed}:{self.task.task_id}:turns")
        outcomes: list[MockOutcome] = []
        for index in range(self.n_rollouts):
            if outcome_rng.random() < self.infra_error_rate:
                outcomes.append(
                    MockOutcome(
                        index=index,
                        passed=False,
                        status="infra_error",
                        turns=0,
                        error="simulated transient provider error (429)",
                        verifier_exit_code=None,
                    )
                )
                continue
            passed = outcome_rng.random() < self.true_pass_rate
            turns = self._draw_turns(detail_rng, passed)
            outcomes.append(
                MockOutcome(
                    index=index,
                    passed=passed,
                    turns=turns,
                    transcript=_transcript(detail_rng, self.task, passed, turns, self.profile),
                    verifier_output=_verifier_output(passed, self.task, self.profile),
                    verifier_exit_code=0 if passed else 1,
                )
            )
        self._plan = outcomes
        return outcomes

    def _draw_turns(self, rng: random.Random, passed: bool) -> int:
        if self.profile == "too_easy":
            return rng.randint(1, 2) if passed else rng.randint(2, 5)
        if self.profile == "wrong_path":
            return rng.randint(3, 7)
        if self.profile == "turn_limit":
            return rng.randint(6, 18) if passed else self.max_turns
        return rng.randint(4, 14) if passed else rng.randint(9, 26)

    # -- execution ---------------------------------------------------------
    def outcome(self, index: int) -> MockOutcome:
        plan = self.plan()
        return plan[index % len(plan)] if plan else MockOutcome(index=index, passed=False)

    def rollout(self, index: int) -> Rollout:
        out = self.outcome(index)
        tokens_in = out.turns * TOKENS_IN_PER_TURN
        tokens_out = out.turns * TOKENS_OUT_PER_TURN
        return Rollout(
            index=index,
            status=out.status,  # type: ignore[arg-type]
            passed=out.passed,
            turns=out.turns,
            duration_s=round(out.turns * 1.7 + 0.4, 2),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=estimate_cost_usd(self.model, tokens_in, tokens_out),
            verifier_exit_code=out.verifier_exit_code,
            verifier_output=out.verifier_output,
            error=out.error,
            simulated=True,
            transcript=out.transcript,
        )

    async def run_rollout(self, index: int) -> Rollout:
        return self.rollout(index)
