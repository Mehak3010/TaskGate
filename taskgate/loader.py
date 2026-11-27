"""TaskLoader: the only module that knows the on-disk task format.

Two layouts are understood.

TaskGate demo format (also used by this project's own sample corpus)::

    tasks/<task_id>/
      task.yaml            # id, instruction, timeouts, demo metadata
      Dockerfile           # agent image; build context = the task dir
      instruction.md       # what the agent sees
      tests/               # hidden verifier (pytest), mounted at /tests
      solution/            # reference solution, mounted at /solution

Terminal-Bench / Harbor format
(https://github.com/harbor-framework/terminal-bench)::

    tasks/<task_id>/
      task.toml            # TOML: [task] [metadata] [verifier] [agent] [environment]
      instruction.md       # what the agent sees
      environment/
        Dockerfile         # agent image; build context = environment/
        ...                # files the Dockerfile COPYs (relative to environment/)
      tests/
        Dockerfile         # SEPARATE verifier image (environment_mode = "separate")
        test.sh            # verifier entry point (runs RewardKit, writes ctrf.json)
      solution/
        solve.sh           # reference solution, mounted at /solution

The two formats differ in where the agent image's Dockerfile lives (task root
vs ``environment/``) and in how the verifier is invoked (pytest inside the agent
image vs a separate image whose entry point is ``tests/test.sh``). This module
records both facts on the parsed spec; consuming them is the sandbox's job.

If the client's real format differs, this file is the single adapter to change.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import Task, TaskSpec

IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "node_modules"}
MAX_LISTED_FILES = 4000

#: Where the Terminal-Bench agent image and its build context live.
TB_ENVIRONMENT_DIRNAME = "environment"
TB_VERIFIER_ENTRY = "tests/test.sh"


class TaskLoadError(RuntimeError):
    """Raised when a directory cannot be interpreted as a task."""


class TaskLoader:
    spec_filenames = ("task.yaml", "task.yml", "task.toml")
    instruction_filenames = ("instruction.md", "INSTRUCTION.md", "instructions.md")
    tests_dirname = "tests"
    solution_dirname = "solution"

    # -- discovery ---------------------------------------------------------
    def find_spec(self, task_dir: Path) -> Path | None:
        for name in self.spec_filenames:
            candidate = task_dir / name
            if candidate.is_file():
                return candidate
        return None

    def is_task_dir(self, path: Path) -> bool:
        return path.is_dir() and self.find_spec(path) is not None

    def load_many(self, root: Path) -> list[Task]:
        """Load ``root`` itself if it is a task, otherwise its immediate children."""
        root = Path(root)
        if self.is_task_dir(root):
            return [self._load_or_fail(root)]
        if not root.exists():
            raise TaskLoadError(f"path does not exist: {root}")
        if not root.is_dir():
            raise TaskLoadError(f"not a directory: {root}")
        tasks: list[Task] = []
        seen: dict[str, Path] = {}
        for p in sorted(root.iterdir()):
            if not self.is_task_dir(p):
                continue
            task = self._load_or_fail(p)
            if not task.load_error:
                first = seen.get(task.task_id)
                if first is not None:
                    task = self._failed_task(
                        p,
                        f"duplicate task id {task.task_id!r}: also used by {first.name}/. Task ids "
                        "must be unique within a run (change `id:` in task.yaml).",
                        task_id=f"{task.task_id}@{p.name}",
                    )
                else:
                    seen[task.task_id] = p
            tasks.append(task)
        if not tasks:
            raise TaskLoadError(
                f"no task directories found under {root} "
                f"(expected <dir>/{self.spec_filenames[0]})"
            )
        return tasks

    # -- failures are per task ---------------------------------------------
    def _load_or_fail(self, task_dir: Path) -> Task:
        """Load a task; a task that cannot be loaded becomes a BLOCKED LOAD-ERROR task."""
        try:
            return self.load(task_dir)
        except TaskLoadError as exc:
            return self._failed_task(task_dir, str(exc))

    def _failed_task(self, task_dir: Path, message: str, task_id: str | None = None) -> Task:
        tid = task_id or task_dir.name
        return Task(
            task_id=tid,
            path=Path(task_dir).resolve(),
            spec=TaskSpec(id=tid),
            instruction_md="",
            rel_files=self._list_files(Path(task_dir)),
            load_error=message,
        )

    FIELD_EXAMPLES = {
        "id": "id: my_task",
        "instruction": "instruction: Fix the bug in /app/main.py",
        "difficulty": "difficulty: medium",
        "author": "author: your-name",
        "timeout_s": "timeout_s: 900",
        "allow_network": "allow_network: false",
        "expected_output_format": "expected_output_format: text",
        "solution_cmd": "solution_cmd: bash /solution/solve.sh",
        "verifier_cmd": "verifier_cmd: python -m pytest -q /tests",
    }

    @classmethod
    def _format_validation_error(cls, filename: str, exc: ValidationError) -> str:
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            example = cls.FIELD_EXAMPLES.get(str(err["loc"][0])) if err.get("loc") else None
            msg = f"{filename}: field '{loc}' is invalid: {err.get('msg', 'invalid value')}"
            parts.append(msg + (f" (valid example: {example})" if example else ""))
        return "; ".join(parts) or f"{filename} is invalid: {exc}"

    # -- single task -------------------------------------------------------
    def load(self, task_dir: Path) -> Task:
        task_dir = Path(task_dir).resolve()
        spec_path = self.find_spec(task_dir)
        if spec_path is None:
            raise TaskLoadError(f"no {' or '.join(self.spec_filenames)} in {task_dir}")
        raw = self._read_spec(spec_path, task_dir)
        if not isinstance(raw, dict):
            raise TaskLoadError(f"{spec_path.name} must be a mapping, got {type(raw).__name__}")

        raw.setdefault("id", task_dir.name)
        try:
            spec = TaskSpec.model_validate(raw)
        except ValidationError as exc:
            raise TaskLoadError(self._format_validation_error(spec_path.name, exc)) from exc

        instruction_md = spec.instruction
        for name in self.instruction_filenames:
            candidate = task_dir / name
            if candidate.is_file():
                instruction_md = candidate.read_text(encoding="utf-8-sig", errors="replace")
                break

        return Task(
            task_id=spec.id,
            path=task_dir,
            spec=spec,
            instruction_md=instruction_md,
            rel_files=self._list_files(task_dir),
        )

    # -- spec parsing ------------------------------------------------------
    def _read_spec(self, spec_path: Path, task_dir: Path) -> dict:
        """Parse a ``task.yaml``/``task.yml``/``task.toml`` into a spec mapping."""
        text = spec_path.read_text(encoding="utf-8-sig")  # tolerate a BOM (Windows editors add one)
        if spec_path.suffix.lower() == ".toml":
            try:
                raw = tomllib.loads(text)
            except tomllib.TOMLDecodeError as exc:
                raise TaskLoadError(f"{spec_path.name} is not valid TOML: {exc}") from exc
            return self._from_terminal_bench(raw, task_dir)
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TaskLoadError(f"{spec_path.name} is not valid YAML: {exc}") from exc
        return raw or {}

    @staticmethod
    def _as_str(value: object) -> str | None:
        """Coerce a Terminal-Bench metadata value to the string TaskSpec expects.

        ``author_name`` (and occasionally ``difficulty``) is a list of names in a
        handful of tasks; the spec wants a single string, so join the parts.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            parts = [str(v) for v in value if v is not None]
            return ", ".join(parts) or None
        return str(value)

    @staticmethod
    def _from_terminal_bench(raw: dict, task_dir: Path) -> dict:
        """Map a Terminal-Bench ``task.toml`` onto the TaskGate :class:`TaskSpec`.

        Only the fields TaskGate consumes are mapped to first-class spec fields;
        the rest of the metadata (``artifacts``, resource limits, schema version)
        is carried through as extras so a report can quote it verbatim. The two
        layout facts that differ from the demo format are recorded explicitly:

        * ``dockerfile`` / ``build_context`` -- Terminal-Bench builds the agent
          image from ``environment/Dockerfile`` with ``environment/`` as context,
          not from a ``Dockerfile`` at the task root.
        * ``verifier_entry`` / ``environment_mode`` -- the verifier runs from
          ``tests/test.sh`` inside its own image (``environment_mode = "separate"``).
        """
        task_table = raw.get("task") or {}
        metadata = raw.get("metadata") or {}
        verifier = raw.get("verifier") or {}
        agent = raw.get("agent") or {}
        environment = raw.get("environment") or {}

        spec: dict = {
            "instruction": "",
            "author": TaskLoader._as_str(metadata.get("author_name")),
            "difficulty": TaskLoader._as_str(metadata.get("difficulty")),
            "allow_network": (
                str(environment.get("network_mode", "")).lower() == "public"
                or bool(environment.get("allow_internet"))
            ),
        }

        name = str(task_table.get("name") or "")
        if name:
            # "terminal-bench/html-js-filter" -> "html-js-filter"
            spec["id"] = name.rsplit("/", 1)[-1]

        timeout = agent.get("timeout_sec")
        if isinstance(timeout, (int, float)):
            spec["timeout_s"] = int(timeout)

        environment_dir = task_dir / TB_ENVIRONMENT_DIRNAME
        if (environment_dir / "Dockerfile").is_file():
            spec["dockerfile"] = f"{TB_ENVIRONMENT_DIRNAME}/Dockerfile"
            spec["build_context"] = TB_ENVIRONMENT_DIRNAME
        if (task_dir / "solution" / "solve.sh").is_file():
            spec["solution_cmd"] = "bash /solution/solve.sh"
        if (task_dir / TB_VERIFIER_ENTRY).is_file():
            spec["verifier_entry"] = TB_VERIFIER_ENTRY

        # Verbatim metadata, kept for the report but not interpreted here.
        spec.update(
            {
                "schema_version": raw.get("schema_version"),
                "artifacts": list(raw.get("artifacts") or []),
                "environment_mode": verifier.get("environment_mode"),
                "cpus": environment.get("cpus"),
                "memory_mb": environment.get("memory_mb"),
                "docker_image": environment.get("docker_image"),
            }
        )
        return spec

    # -- listing -----------------------------------------------------------
    def _list_files(self, task_dir: Path) -> list[str]:
        out: list[str] = []
        for path in sorted(task_dir.rglob("*")):
            rel = path.relative_to(task_dir)
            if any(part in IGNORED_DIRS for part in rel.parts):
                continue
            if path.is_file():
                out.append(rel.as_posix())
            if len(out) >= MAX_LISTED_FILES:
                break
        return out


def load_tasks(root: Path, loader: TaskLoader | None = None) -> list[Task]:
    return (loader or TaskLoader()).load_many(root)
