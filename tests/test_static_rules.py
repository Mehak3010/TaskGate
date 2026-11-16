"""One positive and one negative case for every static rule (S001-S010).

Rules are implemented as standalone functions precisely so they can be pinned
here in isolation. Rule IDs are stable, so these assertions also protect the
diffability promise in the README.
"""

from __future__ import annotations

from taskgate.audit.static_checks import (
    check_s001,
    check_s002,
    check_s003,
    check_s004,
    check_s005,
    check_s006,
    check_s007,
    check_s008,
    check_s009,
    check_s010,
    run_static_audit,
)
from taskgate.models import Severity


def rule_ids(findings) -> set[str]:
    return {f.rule_id for f in findings}


# ---------------------------------------------------------------- S001
def test_s001_hidden_tests_copied_into_image(make_task):
    task = make_task(
        {
            "Dockerfile": (
                "FROM python:3.11-slim\n"
                "COPY data/ /app/data/\n"
                "COPY tests/ /app/tests/\n"
            ),
            "data/in.txt": "x\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    findings = check_s001(task)
    assert "S001" in rule_ids(findings)
    assert findings[0].severity is Severity.BLOCKER
    assert findings[0].line == 3  # the COPY tests/ line


def test_s001_whole_context_flagged_when_not_dockerignored(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n",
            "data/in.txt": "x\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    assert "S001" in rule_ids(check_s001(task))


def test_s001_negative_when_dockerignore_excludes_tests(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n",
            ".dockerignore": "tests/\nsolution/\n",
            "data/in.txt": "x\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    assert "S001" not in rule_ids(check_s001(task))


def test_s001_negative_when_tests_are_outside_the_build_context(make_task):
    """Terminal-Bench: the build context is environment/, so `COPY . /app` cannot
    reach the sibling tests/ directory; an S001 here would be a false positive."""
    task = make_task(
        {
            "environment/Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n",
            "environment/data/in.txt": "x\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
            "solution/solve.sh": "echo hi\n",
        },
        task_yaml=(
            "id: t_demo\ndockerfile: environment/Dockerfile\nbuild_context: environment\n"
        ),
    )
    assert "S001" not in rule_ids(check_s001(task))


def test_s001_positive_when_build_context_itself_holds_tests(make_task):
    """The context-aware check still fires when tests/ is genuinely in context."""
    task = make_task(
        {
            "environment/Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n",
            "environment/tests/test_x.py": "def test_a():\n    assert True\n",
        },
        task_yaml=(
            "id: t_demo\ndockerfile: environment/Dockerfile\nbuild_context: environment\n"
        ),
    )
    assert "S001" in rule_ids(check_s001(task))


def test_s001_backslash_line_continuation_does_not_crash(make_task):
    """A RUN ending in a backslash used to make shlex.split raise ValueError and
    abort the whole rule (AUDIT-ERROR). It must degrade to a plain split instead."""
    task = make_task(
        {
            "Dockerfile": (
                "FROM python:3.11-slim\n"
                "RUN apt-get update && \\\n"
                "    apt-get install -y curl\n"
            ),
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    assert check_s001(task) == []  # must not raise
    # ...and the rule's crash guard must not be tripped for the whole audit.
    assert "AUDIT-ERROR" not in rule_ids(run_static_audit(task))


# ---------------------------------------------------------------- S002
def test_s002_reference_solution_baked_into_image(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\nCOPY solution/ /sol/\n",
            "solution/solve.sh": "echo hi\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    assert "S002" in rule_ids(check_s002(task))


def test_s002_answer_file_in_build_context(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n",
            "answer.json": '{"total": 3}\n',
            "data/in.txt": "x\n",
        }
    )
    assert "S002" in rule_ids(check_s002(task))


def test_s002_negative_when_solution_is_ignored(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n",
            ".dockerignore": "solution/\n",
            "solution/solve.sh": "echo hi\n",
            "data/in.txt": "x\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    assert "S002" not in rule_ids(check_s002(task))


def test_s002_negative_when_solution_is_outside_the_build_context(make_task):
    """Terminal-Bench: solution/ sits beside environment/, not inside it."""
    task = make_task(
        {
            "environment/Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n",
            "environment/data/in.txt": "x\n",
            "solution/solve.sh": "echo hi\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        },
        task_yaml=(
            "id: t_demo\ndockerfile: environment/Dockerfile\nbuild_context: environment\n"
        ),
    )
    assert "S002" not in rule_ids(check_s002(task))


def test_s002_positive_when_build_context_holds_the_solution(make_task):
    task = make_task(
        {
            "environment/Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n",
            "environment/solution/solve.sh": "echo hi\n",
        },
        task_yaml=(
            "id: t_demo\ndockerfile: environment/Dockerfile\nbuild_context: environment\n"
        ),
    )
    assert "S002" in rule_ids(check_s002(task))


# ---------------------------------------------------------------- S003
def test_s003_literal_shared_with_instruction_is_leakage(make_task):
    task = make_task(
        {"tests/test_x.py": 'def test_a():\n    assert compute() == "ABC123XY"\n'},
        instruction="Set the build code to ABC123XY exactly.\n",
    )
    findings = check_s003(task)
    assert "S003" in rule_ids(findings)
    assert findings[0].severity is Severity.INFO


def test_s003_negative_when_literal_is_not_in_instruction(make_task):
    task = make_task(
        {"tests/test_x.py": 'def test_a():\n    assert compute() == "ABC123XY"\n'},
        instruction="Compute the checksum of the input file.\n",
    )
    assert "S003" not in rule_ids(check_s003(task))


# ---------------------------------------------------------------- S004
def test_s004_existence_only_assertions(make_task):
    task = make_task(
        {
            "tests/test_x.py": (
                "from pathlib import Path\n"
                'OUT = Path("/app/out.json")\n'
                "\n"
                "def test_a():\n"
                "    assert OUT.exists()\n"
                "\n"
                "def test_b():\n"
                "    assert not OUT.is_file()\n"
            )
        }
    )
    assert "S004" in rule_ids(check_s004(task))


def test_s004_negative_when_behaviour_is_asserted(make_task):
    task = make_task(
        {
            "tests/test_x.py": (
                "import json\n"
                "from pathlib import Path\n"
                'OUT = Path("/app/out.json")\n'
                "\n"
                "def test_a():\n"
                "    data = json.loads(OUT.read_text())\n"
                '    assert data["total"] == 3\n'
            )
        }
    )
    assert "S004" not in rule_ids(check_s004(task))


# ---------------------------------------------------------------- S005
def test_s005_unpinned_pip_install(make_task):
    task = make_task({"Dockerfile": "FROM python:3.11-slim\nRUN pip install requests\n"})
    assert "S005" in rule_ids(check_s005(task))


def test_s005_floating_base_image_tag(make_task):
    task = make_task({"Dockerfile": "FROM python:latest\n"})
    assert "S005" in rule_ids(check_s005(task))


def test_s005_unpinned_requirements_txt(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\n",
            "requirements.txt": "requests\npytest==8.2.0\n",
        }
    )
    findings = [f for f in check_s005(task) if f.file == "requirements.txt"]
    assert len(findings) == 1
    assert findings[0].line == 1


def test_s005_negative_when_everything_is_pinned(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\nRUN pip install requests==2.31.0\n",
            "requirements.txt": "requests==2.31.0\n",
        }
    )
    assert "S005" not in rule_ids(check_s005(task))


def test_s005_unpinned_apt_is_info_and_stops_at_chained_command(make_task):
    """Only the arguments after `apt-get install` up to the next && are packages;
    the trailing `rm -rf ...*` must not be reported, and the finding is INFO."""
    task = make_task(
        {
            "Dockerfile": (
                "FROM debian:12\n"
                "RUN apt-get update && apt-get install -y curl \\\n"
                "    && rm -rf /var/lib/apt/lists/*\n"
            )
        }
    )
    findings = check_s005(task)
    apt = next((f for f in findings if "system package" in f.message), None)
    assert apt is not None
    assert apt.severity is Severity.INFO
    assert apt.message == "unpinned system package(s): curl"
    assert not any("rm" in f.message or "var/lib" in f.message for f in findings)


def test_s005_pinned_apt_package_is_not_flagged(make_task):
    task = make_task(
        {"Dockerfile": "FROM debian:12\nRUN apt-get install -y curl=7.88.1-10\n"}
    )
    assert "S005" not in rule_ids(check_s005(task))


# ---------------------------------------------------------------- S006
def test_s006_network_call_in_hidden_test(make_task):
    task = make_task(
        {
            "tests/test_x.py": (
                "import requests\n"
                "\n"
                "def test_a():\n"
                '    r = requests.get("https://example.invalid/x", timeout=5)\n'
                "    assert r.status_code == 200\n"
            )
        }
    )
    findings = check_s006(task)
    assert "S006" in rule_ids(findings)
    assert findings[0].severity is Severity.BLOCKER


def test_s006_negative_when_tests_are_offline(make_task):
    task = make_task({"tests/test_x.py": "def test_a():\n    assert 1 + 1 == 2\n"})
    assert "S006" not in rule_ids(check_s006(task))


def test_s006_negative_for_build_time_dockerfile_fetches(make_task):
    """Screen fetches in tests/Dockerfile happen at build, not verification time."""
    task = make_task(
        {
            "tests/Dockerfile": (
                "FROM python:3.11-slim\n"
                "RUN apt-get update && apt-get install -y curl git\n"
                "RUN curl -fsSL https://example.invalid/corpus.zip -o /tests/corpus.zip\n"
            ),
            "tests/test.sh": "set -euo pipefail\npython -m pytest -q /tests\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    assert "S006" not in rule_ids(check_s006(task))


def test_s006_positive_for_runtime_network_in_test_sh(make_task):
    """A network call in the runtime entry point (test.sh) is still a finding."""
    task = make_task(
        {
            "tests/test.sh": (
                "set -euo pipefail\n"
                "curl -fsSL https://example.invalid/grader.sh | bash\n"
            ),
        }
    )
    assert "S006" in rule_ids(check_s006(task))


def test_s006_lock_manifest_is_not_scanned(make_task):
    """A lock file is a data manifest full of URLs; it is not runtime code."""
    task = make_task(
        {
            "tests/uv.lock": '[[package]]\nname = "x"\nurl = "https://example.invalid/x.whl"\n',
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    assert "S006" not in rule_ids(check_s006(task))


def test_s006_bare_url_is_not_a_network_call(make_task):
    """Mentioning a URL is not a network call, so it must not be flagged."""
    task = make_task(
        {"tests/test.sh": 'set -e\necho "see https://example.invalid/docs for details"\n'}
    )
    assert "S006" not in rule_ids(check_s006(task))


def test_s006_network_call_in_solution_is_warn_not_blocker(make_task):
    """The author's solution may reach the network: downgrade it to WARN."""
    task = make_task(
        {
            "tests/test.sh": "set -e\necho done\n",
            "solution/solve.sh": "curl -fsSL https://example.invalid/data -o /tmp/d\n",
        }
    )
    findings = check_s006(task)
    assert findings
    assert all(f.severity is Severity.WARN for f in findings)
    assert all(f.file == "solution/solve.sh" for f in findings)


# ---------------------------------------------------------------- S007
def test_s007_sleep_based_sync(make_task):
    task = make_task(
        {"tests/test_x.py": "import time\n\ndef test_a():\n    time.sleep(1)\n    assert True\n"}
    )
    assert "S007" in rule_ids(check_s007(task))


def test_s007_unseeded_random(make_task):
    task = make_task(
        {"tests/test_x.py": "import random\n\ndef test_a():\n    assert random.random() >= 0.0\n"}
    )
    assert "S007" in rule_ids(check_s007(task))


def test_s007_negative_when_random_is_seeded(make_task):
    task = make_task(
        {
            "tests/test_x.py": (
                "import random\n"
                "random.seed(7)\n"
                "\n"
                "def test_a():\n"
                "    assert random.random() >= 0.0\n"
            )
        }
    )
    assert "S007" not in rule_ids(check_s007(task))


# ---------------------------------------------------------------- S008
def test_s008_declared_json_never_parsed(make_task):
    task = make_task(
        {
            "tests/test_x.py": (
                "from pathlib import Path\n"
                'OUT = Path("/app/out.json")\n'
                "\n"
                "def test_a():\n"
                '    assert "total" in OUT.read_text()\n'
            )
        },
        task_yaml="id: t_demo\nexpected_output_format: json\n",
    )
    assert "S008" in rule_ids(check_s008(task))


def test_s008_declared_format_without_any_verifier(make_task):
    task = make_task(
        {"data/in.txt": "x\n"}, task_yaml="id: t_demo\nexpected_output_format: csv\n"
    )
    assert "S008" in rule_ids(check_s008(task))


def test_s008_negative_when_json_is_parsed_and_asserted(make_task):
    task = make_task(
        {
            "tests/test_x.py": (
                "import json\n"
                "from pathlib import Path\n"
                'OUT = Path("/app/out.json")\n'
                "\n"
                "def test_a():\n"
                "    data = json.loads(OUT.read_text())\n"
                '    assert data["total"] > 0\n'
            )
        },
        task_yaml="id: t_demo\nexpected_output_format: json\n",
    )
    assert "S008" not in rule_ids(check_s008(task))


def test_s008_negative_for_plain_text_format(make_task):
    task = make_task(
        {"tests/test_x.py": "def test_a():\n    assert 1 == 1\n"},
        task_yaml="id: t_demo\nexpected_output_format: text\n",
    )
    assert "S008" not in rule_ids(check_s008(task))


# ---------------------------------------------------------------- S009
def test_s009_vague_instruction_is_advisory_info(make_task):
    task = make_task(
        instruction="Summarise the errors in the data. It must be written where the grader looks.\n"
    )
    findings = check_s009(task)
    assert findings
    assert all(f.severity is Severity.INFO for f in findings)


def test_s009_negative_when_path_and_format_are_named(make_task):
    task = make_task(instruction="Write a JSON object to /app/out/summary.json.\n")
    assert check_s009(task) == []


# ---------------------------------------------------------------- S010
def test_s010_tests_copied_world_writable(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\nCOPY --chmod=777 tests/ /app/tests/\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    findings = check_s010(task)
    assert "S010" in rule_ids(findings)
    assert findings[0].severity is Severity.BLOCKER


def test_s010_run_chmod_on_tests(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\nRUN chmod -R 777 /app/tests\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    assert "S010" in rule_ids(check_s010(task))


def test_s010_negative_for_read_only_tests(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.11-slim\nCOPY --chmod=444 tests/ /app/tests/\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        }
    )
    assert "S010" not in rule_ids(check_s010(task))


def test_s010_negative_when_tests_are_outside_the_build_context(make_task):
    """The context-aware check must not fire for tests/ that are out of context."""
    task = make_task(
        {
            "environment/Dockerfile": "FROM python:3.11-slim\nCOPY --chmod=777 . /app\n",
            "environment/data/in.txt": "x\n",
            "tests/test_x.py": "def test_a():\n    assert True\n",
        },
        task_yaml=(
            "id: t_demo\ndockerfile: environment/Dockerfile\nbuild_context: environment\n"
        ),
    )
    assert "S010" not in rule_ids(check_s010(task))


def test_s010_positive_when_build_context_holds_tests(make_task):
    task = make_task(
        {
            "environment/Dockerfile": "FROM python:3.11-slim\nCOPY --chmod=777 . /app\n",
            "environment/tests/test_x.py": "def test_a():\n    assert True\n",
        },
        task_yaml=(
            "id: t_demo\ndockerfile: environment/Dockerfile\nbuild_context: environment\n"
        ),
    )
    assert "S010" in rule_ids(check_s010(task))


# ------------------------------------------------ precision regressions
# Each of these came from hand-reading findings on 66 public Terminal-Bench
# tasks; every one was a false positive of the original rule.


def test_s001_mkdir_of_empty_tests_dir_is_not_a_leak(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.12-slim\nRUN mkdir -p /logs /tests /shared\n",
            "tests/test.sh": "echo ok\n",
        }
    )
    assert "S001" not in rule_ids(check_s001(task))


def test_s001_cp_of_tests_into_image_still_flagged(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.12-slim\nRUN cp -r /src/tests /tests\n",
            "tests/test.sh": "echo ok\n",
        }
    )
    assert "S001" in rule_ids(check_s001(task))


def test_s003_ignores_non_code_files_and_is_advisory(make_task):
    task = make_task(
        {"tests/uv.lock": 'name = "triton"\n', "tests/notes.md": "x == 'triton'\n"},
        instruction="Install triton.\n",
    )
    assert check_s003(task) == []


def test_s004_ignores_harness_asserts_outside_test_functions(make_task):
    task = make_task(
        {
            "tests/report.py": (
                "def build(rows):\n"
                "    assert rows, 'no results'\n"
                "    return rows\n"
            )
        }
    )
    assert "S004" not in rule_ids(check_s004(task))


def test_s004_still_flags_weak_asserts_inside_test_functions(make_task):
    task = make_task(
        {"tests/test_x.py": "import os\ndef test_a():\n    assert os.path.exists('/app/out')\n"}
    )
    assert "S004" in rule_ids(check_s004(task))


def test_s006_skips_docs_localhost_and_downgrades_installs(make_task):
    task = make_task(
        {
            "tests/SKILL.md": "run npm install\n",
            "tests/checks.py": (
                "import urllib.request\n"
                "urllib.request.urlopen('http://127.0.0.1:8080/health')\n"
            ),
            "tests/test.sh": "pip install -r /app/requirements.txt\n",
        }
    )
    findings = check_s006(task)
    assert all(f.file == "tests/test.sh" for f in findings)
    assert findings and all(f.severity is Severity.WARN for f in findings)


def test_s006_real_remote_call_in_verifier_is_still_a_blocker(make_task):
    task = make_task(
        {"tests/checks.py": "import requests\nrequests.get('https://example.com/answer')\n"}
    )
    findings = check_s006(task)
    assert findings and findings[0].severity is Severity.BLOCKER


def test_s002_skips_reference_dir_the_instruction_points_at(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.12-slim\nCOPY seed/reference/ /shared/reference/\n",
            "seed/reference/T-001.json": "{}",
        },
        instruction="Use the dev cases in /shared/reference/ to debug the engine.\n",
    )
    assert "S002" not in rule_ids(check_s002(task))


def test_s002_still_flags_undisclosed_answer_copy(make_task):
    task = make_task(
        {
            "Dockerfile": "FROM python:3.12-slim\nCOPY expected_output.json /app/\n",
            "expected_output.json": "{}",
        },
        instruction="Write the result to /app/out.json.\n",
    )
    assert "S002" in rule_ids(check_s002(task))


def test_s006_variable_target_is_warn_but_external_url_is_blocker(make_task):
    task = make_task(
        {
            "tests/checks.py": (
                "import requests\n"
                "requests.get(base + '/health')\n"
                "requests.get('https://example.com/answer.json')\n"
            )
        }
    )
    by_line = {f.line: f.severity for f in check_s006(task)}
    assert by_line[2] is Severity.WARN
    assert by_line[3] is Severity.BLOCKER
