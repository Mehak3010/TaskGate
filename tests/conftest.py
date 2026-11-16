"""Shared fixtures.

``make_task`` writes a tiny task directory to a temp path and loads it through the
real TaskLoader, so every rule test exercises the same code path a user's task
does (spec parsing, file listing, .dockerignore handling).

``demo_report`` runs the whole pipeline over the generated demo corpus in mock
mode once per session, with Docker forced off, so the integration and report
tests share one deterministic run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from taskgate.demo.build_demo_tasks import build_demo_corpus
from taskgate.loader import TaskLoader, load_tasks
from taskgate.models import RunConfig, Task
from taskgate.pipeline import PipelineOptions, run_pipeline_sync


@pytest.fixture
def make_task(tmp_path: Path):
    def _make(
        files: dict[str, str] | None = None,
        *,
        task_yaml: str = "id: t_demo\n",
        instruction: str | None = None,
        task_id: str = "t_demo",
    ) -> Task:
        task_dir = tmp_path / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.yaml").write_text(task_yaml, encoding="utf-8")
        if instruction is not None:
            (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
        for rel, content in (files or {}).items():
            path = task_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return TaskLoader().load(task_dir)

    return _make


@pytest.fixture(scope="session")
def demo_report(tmp_path_factory):
    """(tasks, report) for the full demo pipeline, in mock mode, without Docker."""
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("taskgate.pipeline.docker_available", lambda *a, **k: False)
    try:
        out = tmp_path_factory.mktemp("demo")
        corpus = out / "tasks"
        build_demo_corpus(corpus, force=True)
        tasks = load_tasks(corpus)
        config = RunConfig(
            model="mock",
            mock=True,
            rollouts=20,
            seed=1234,
            out_dir=str(out),
        )
        report = run_pipeline_sync(tasks, config, options=PipelineOptions())
    finally:
        monkeypatch.undo()
    return tasks, report
