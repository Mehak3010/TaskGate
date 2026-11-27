"""Regression tests for the gaps an external 'client' evaluation found.

Each test is named after the finding it pins down. See docs/client_evaluation.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from taskgate.audit.static_checks import (
    check_missing_dockerfile,
    check_s004,
    check_s007,
    check_s009,
    run_static_audit,
)
from taskgate.cli import app
from taskgate.loader import TaskLoader

DOCKERFILE = "FROM python:3.11-slim\n"
runner = CliRunner()


def _ids(findings):
    return [f.rule_id for f in findings]


# ---------------------------------------------------------------- S007
def test_s007_catches_random_uniform_and_gauss(make_task):
    task = make_task(
        {
            "Dockerfile": DOCKERFILE,
            "tests/test_x.py": (
                "import random\n"
                "def test_a():\n"
                "    x = random.uniform(0, 1)\n"
                "    y = random.gauss(0, 1)\n"
                "    assert x != y\n"
            ),
        }
    )
    assert _ids(check_s007(task)).count("S007") == 2


def test_s007_catches_from_random_import_names(make_task):
    task = make_task(
        {
            "Dockerfile": DOCKERFILE,
            "tests/test_x.py": "from random import uniform as u\ndef test_a():\n    assert u(0, 1) >= 0\n",
        }
    )
    assert "S007" in _ids(check_s007(task))


def test_s007_catches_numpy_random(make_task):
    task = make_task(
        {"Dockerfile": DOCKERFILE, "tests/test_x.py": "import numpy as np\ndef test_a():\n    assert np.random.rand(3) is not None\n"}
    )
    assert "S007" in _ids(check_s007(task))


def test_s007_quiet_when_seeded(make_task):
    task = make_task(
        {
            "Dockerfile": DOCKERFILE,
            "tests/test_x.py": "import random\nrandom.seed(7)\ndef test_a():\n    assert random.uniform(0, 1) >= 0\n",
        }
    )
    assert check_s007(task) == []


# ---------------------------------------------------------------- S004
def test_s004_flags_exists_plus_nonempty_size(make_task):
    task = make_task(
        {
            "Dockerfile": DOCKERFILE,
            "tests/test_output.py": (
                "from pathlib import Path\n"
                "OUT = Path('/app/out/result.txt')\n"
                "def test_exists():\n    assert OUT.exists()\n"
                "def test_nonempty():\n    assert OUT.stat().st_size > 0\n"
            ),
        }
    )
    assert "S004" in _ids(check_s004(task))


def test_s004_flags_getsize_and_read_text_truthiness(make_task):
    task = make_task(
        {
            "Dockerfile": DOCKERFILE,
            "tests/test_output.py": (
                "import os\nfrom pathlib import Path\n"
                "def test_a():\n    assert os.path.getsize('/app/out/x') > 0\n"
                "def test_b():\n    assert Path('/app/out/x').read_text()\n"
            ),
        }
    )
    assert "S004" in _ids(check_s004(task))


def test_s004_quiet_when_content_is_checked(make_task):
    task = make_task(
        {
            "Dockerfile": DOCKERFILE,
            "tests/test_output.py": (
                "from pathlib import Path\n"
                "def test_content():\n"
                "    assert Path('/app/out/x').read_text().strip() == 'port=8080'\n"
            ),
        }
    )
    assert check_s004(task) == []


# ---------------------------------------------------------------- S009
def test_s009_input_filename_does_not_hide_a_missing_output(make_task):
    task = make_task(
        {"Dockerfile": DOCKERFILE},
        instruction="There is some sales data in `/app/data/raw_sales.csv`. Clean it up and save the result.\n",
    )
    found = check_s009(task)
    assert len(found) >= 2  # no output path, no format (+ pronoun)
    assert any("output" in f.message.lower() for f in found)


def test_s009_quiet_when_output_path_is_named_even_without_a_format_word(make_task):
    task = make_task(
        {"Dockerfile": DOCKERFILE},
        instruction="Read `/app/data/app.conf`. Write the effective configuration to `/app/out/result.txt`.\n",
    )
    assert check_s009(task) == []


def test_s009_quiet_for_executable_script_output(make_task):
    task = make_task(
        {"Dockerfile": DOCKERFILE},
        instruction="Write an executable shell script to `/app/out/check_port.sh` that prints the port.\n",
    )
    assert check_s009(task) == []


def test_s009_quiet_when_a_rename_names_its_destination(make_task):
    # regression: the demo task t03 was wrongly flagged after the first S009 rewrite
    task = make_task(
        {"Dockerfile": DOCKERFILE},
        instruction=(
            "# Rename a file\n\nRename `/app/data/input.txt` to `/app/data/output.txt`.\n\n"
            "The result must be a plain text file whose bytes are identical to the\n"
            "original file, and `/app/data/input.txt` must no longer exist.\n"
        ),
    )
    assert check_s009(task) == []


def test_s009_quiet_for_copy_and_convert_destinations(make_task):
    for text in (
        "Copy `/app/data/a.csv` to `/app/data/b.csv`.\n",
        "Convert `/app/data/in.csv` into `/app/data/out.json`.\n",
        "Read input.csv and write the totals to totals.json.\n",
    ):
        task = make_task({"Dockerfile": DOCKERFILE}, instruction=text, task_id="t_" + str(abs(hash(text)))[:6])
        assert check_s009(task) == [], text


def test_s009_still_flags_a_read_only_mention(make_task):
    task = make_task(
        {"Dockerfile": DOCKERFILE},
        instruction="Read `/app/data/results.csv` and make it better.\n",
    )
    assert any("output" in f.message.lower() or "file" in f.message.lower() for f in check_s009(task))


# ---------------------------------------------------------------- D000
def test_missing_dockerfile_is_a_blocker_not_a_silent_skip(make_task):
    task = make_task(instruction="Write a file to /app/out/x.txt.\n")
    findings = check_missing_dockerfile(task)
    assert _ids(findings) == ["D000"]
    assert findings[0].severity.value == "BLOCKER"


# ---------------------------------------------------------------- LOAD-ERROR isolation
def _write(root: Path, name: str, spec: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "task.yaml").write_text(spec, encoding="utf-8")
    (d / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (d / "instruction.md").write_text("Write a file to `/app/out/x.json`.\n", encoding="utf-8")


def test_one_invalid_task_does_not_abort_the_folder(tmp_path):
    _write(tmp_path, "good", "id: good\n")
    _write(tmp_path, "bad", "id: bad\nexpected_output_format: txt\n")
    tasks = TaskLoader().load_many(tmp_path)
    assert {t.task_id for t in tasks} == {"good", "bad"}
    bad = next(t for t in tasks if t.task_id == "bad")
    assert bad.load_error and "expected_output_format" in bad.load_error
    assert "valid example" in bad.load_error
    assert _ids(run_static_audit(bad)) == ["LOAD-ERROR"]
    good = next(t for t in tasks if t.task_id == "good")
    assert good.load_error is None


def test_malformed_yaml_becomes_a_load_error(tmp_path):
    _write(tmp_path, "good", "id: good\n")
    _write(tmp_path, "broken", "id: [oops\ninstruction: ::: bad\n")
    tasks = TaskLoader().load_many(tmp_path)
    assert next(t for t in tasks if t.task_id == "broken").load_error


def test_duplicate_task_ids_are_reported(tmp_path):
    _write(tmp_path, "a", "id: same\n")
    _write(tmp_path, "b", "id: same\n")
    tasks = TaskLoader().load_many(tmp_path)
    dup = [t for t in tasks if t.load_error]
    assert len(dup) == 1 and "duplicate task id" in dup[0].load_error
    assert len({t.task_id for t in tasks}) == 2  # ids stay unique for reports and image tags


def test_utf8_bom_in_task_yaml_loads(tmp_path):
    d = tmp_path / "bom"
    d.mkdir()
    (d / "task.yaml").write_bytes(b"\xef\xbb\xbfid: bom\n")
    (d / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    task = TaskLoader().load(d)
    assert task.task_id == "bom" and task.load_error is None


def test_empty_folder_still_errors_clearly(tmp_path):
    result = runner.invoke(app, ["audit", str(tmp_path), "--no-dynamic", "--out", str(tmp_path / "o")])
    assert result.exit_code == 2
    assert "no task directories found" in result.output


# ---------------------------------------------------------------- CI gate and offline mock
def test_fail_on_blocker_exits_1_and_default_exits_0(tmp_path):
    _write(tmp_path / "t", "bad", "id: bad\nexpected_output_format: txt\n")
    base = ["audit", str(tmp_path / "t"), "--no-dynamic", "--out", str(tmp_path / "o")]
    assert runner.invoke(app, base).exit_code == 0
    assert runner.invoke(app, base + ["--fail-on-blocker"]).exit_code == 1


def test_calibrate_mock_is_offline_and_runs_no_container_audit(tmp_path, monkeypatch):
    # Even with a reachable Docker daemon, --mock must not build or run anything.
    monkeypatch.setattr("taskgate.pipeline.docker_available", lambda *a, **k: True)

    class _NoSandbox:
        def __init__(self, *a, **k):
            raise AssertionError("calibrate --mock must not create a sandbox")

    monkeypatch.setattr("taskgate.pipeline.Sandbox", _NoSandbox)
    _write(tmp_path / "t", "good", "id: good\n")
    out = tmp_path / "o"
    result = runner.invoke(app, ["calibrate", str(tmp_path / "t"), "--mock", "--rollouts", "5", "--out", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["dynamic_audit_ran"] is False
    assert "no containers" in result.output


# ---------------------------------------------------------------- terminal-bench build context
def test_terminal_bench_layout_builds_from_environment_dir(tmp_path):
    d = tmp_path / "tb"
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "solution").mkdir()
    (d / "task.toml").write_text('[metadata]\ndifficulty = "easy"\n', encoding="utf-8")
    (d / "instruction.md").write_text("Write `/app/out/x.txt`.\n", encoding="utf-8")
    (d / "environment" / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (d / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (d / "solution" / "solve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    task = TaskLoader().load(d)
    context, dockerfile = task.docker_build_args()
    assert context.name == "environment"
    assert dockerfile == "Dockerfile"  # relative to the context, NOT the task root
    assert check_missing_dockerfile(task) == []  # the environment/ Dockerfile counts


def test_flat_layout_build_args_use_task_root(make_task):
    task = make_task({"Dockerfile": DOCKERFILE})
    context, dockerfile = task.docker_build_args()
    assert context == task.path and dockerfile == "Dockerfile"


# ---------------------------------------------------------------- terminal-bench verifier
def test_terminal_bench_task_is_verified_by_test_sh_not_pytest(tmp_path):
    from taskgate.audit.dynamic_checks import _verifier_cmd

    d = tmp_path / "tb"
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "task.toml").write_text('[task]\nid = "tb"\n', encoding="utf-8")
    (d / "instruction.md").write_text("Write `/app/out/x.txt`.\n", encoding="utf-8")
    (d / "environment" / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (d / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    task = TaskLoader().load(d)
    assert task.verifier_command == "bash /tests/test.sh"
    assert _verifier_cmd(task) == "bash /tests/test.sh"  # the container audit uses it too


def test_flat_layout_keeps_pytest_default_and_explicit_cmd_wins(make_task):
    plain = make_task({"Dockerfile": DOCKERFILE})
    assert plain.verifier_command == "python -m pytest -q /tests"
    custom = make_task({"Dockerfile": DOCKERFILE}, task_yaml="id: t_c\nverifier_cmd: bash /tests/run.sh\n")
    assert custom.verifier_command == "bash /tests/run.sh"
