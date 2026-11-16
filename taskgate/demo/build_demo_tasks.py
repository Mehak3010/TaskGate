"""Generates the 8-task demo corpus.

Each task is designed to trigger exactly one story. The corpus is written to
disk as ordinary task directories, so it is also the best documentation of the
assumed task format: run ``taskgate demo`` and read the generated files.

The ``demo:`` block in each ``task.yaml`` is demo-only metadata (ground truth
for the mock agent plus the simulated container findings). Real tasks never
carry it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DemoTask:
    task_id: str
    task_yaml: str
    dockerfile: str
    instruction: str
    files: dict[str, str] = field(default_factory=dict)
    planted_issue: str = ""


# --------------------------------------------------------------------------
# synthetic input data (deterministic)
# --------------------------------------------------------------------------
def _access_logs(rng: random.Random) -> dict[str, str]:
    services = ["auth", "billing", "search", "upload"]
    messages = {
        "ERROR": ["token expired", "upstream timeout", "disk full", "write failed"],
        "WARN": ["retrying request", "slow query", "cache miss", "retry budget low"],
        "INFO": ["request accepted", "job started", "cache warmed", "request failed"],
    }
    junk = [
        "2024-05-01T12:00:00 DEBUG heartbeat ok",
        "traceback tail without a level",
        "",
        "{",
        "2024-05-01T12:00:00 ERROR msg=\"missing service field\"",
    ]
    out: dict[str, str] = {}
    for part in range(3):
        lines: list[str] = []
        for _ in range(40):
            if rng.random() < 0.13:
                lines.append(rng.choice(junk))
                continue
            ts = (
                f"2024-05-0{rng.randint(1, 5)}T{rng.randint(0, 23):02d}:"
                f"{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
            )
            level = rng.choices(["ERROR", "WARN", "INFO"], weights=[2, 3, 5])[0]
            lines.append(
                f'{ts} {level} service={rng.choice(services)} msg="{rng.choice(messages[level])}"'
            )
        out[f"data/logs/part{part}.log"] = "\n".join(lines) + "\n"
    return out


def _raw_csv(rng: random.Random) -> dict[str, str]:
    ids = [rng.randint(1000, 1034) for _ in range(72)]
    rows = ["id,name,amount"]
    for index, rid in enumerate(ids):
        first = ids.index(rid) == index
        # duplicates carry a decoy amount: "keep the first occurrence" matters.
        amount = rid * 3 if first else rid * 999
        rows.append(f"{rid},user{rid},{amount}")
    return {"data/raw.csv": "\n".join(rows) + "\n"}


def _latency_events(rng: random.Random) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in range(2):
        lines = []
        for _ in range(120):
            hour = rng.randint(0, 23)
            ts = f"2024-06-02T{hour:02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
            lines.append(f"{ts} route=/v1/{rng.choice(['search', 'feed', 'checkout'])} ms={rng.randint(5, 900)}")
        out[f"data/events_{part}.log"] = "\n".join(lines) + "\n"
    return out


# --------------------------------------------------------------------------
# task definitions
# --------------------------------------------------------------------------
def _demo_block(
    true_pass_rate: float | None,
    profile: str,
    expected_route: str,
    expected_cause: str | None = None,
    simulate_dynamic: tuple[str, ...] = (),
) -> str:
    lines = ["demo:"]
    if true_pass_rate is not None:
        lines.append(f"  true_pass_rate: {true_pass_rate}")
    lines.append(f"  transcript_profile: {profile}")
    if simulate_dynamic:
        lines.append("  simulate_dynamic:")
        lines.extend(f"    - {rule_id}" for rule_id in simulate_dynamic)
    lines.append(f"  expected_route: {expected_route}")
    if expected_cause:
        lines.append(f"  expected_cause: {expected_cause}")
    return "\n".join(lines)


TASKS: list[DemoTask] = [
    # ---------------------------------------------------------------- t01
    DemoTask(
        task_id="t01_log_parser",
        planted_issue="healthy control: a well-formed, mid-difficulty task",
        task_yaml="\n".join(
            [
                "id: t01_log_parser",
                "instruction: Parse structured log lines from /app/data/logs and write a JSON summary.",
                "difficulty: medium",
                "author: demo-corpus",
                "timeout_s: 900",
                "expected_output_format: json",
                "solution_cmd: bash /solution/solve.sh",
                "verifier_cmd: python -m pytest -q /tests",
                _demo_block(0.20, "normal", "SHIP"),
                "",
            ]
        ),
        dockerfile="\n".join(
            [
                "FROM python:3.11-slim",
                "",
                "WORKDIR /app",
                "COPY data/ /app/data/",
                "COPY requirements.txt /app/requirements.txt",
                "RUN pip install --no-cache-dir -r /app/requirements.txt",
                "",
                'CMD ["sleep", "infinity"]',
                "",
            ]
        ),
        instruction=(
            "# Summarise the access logs\n"
            "\n"
            "Read every `*.log` file under `/app/data/logs/`.\n"
            "\n"
            "A well-formed line looks like this:\n"
            "\n"
            "    2024-05-01T12:00:00 ERROR service=auth msg=\"token expired\"\n"
            "\n"
            "Lines that do not have a level, a `service=` field and a `msg=` field are\n"
            "malformed and must be ignored completely.\n"
            "\n"
            "Write a single JSON file to `/app/out/summary.json` with exactly this shape:\n"
            "\n"
            "    {\n"
            "      \"total\": 0,\n"
            "      \"levels\": {\"ERROR\": 0, \"WARN\": 0, \"INFO\": 0},\n"
            "      \"services\": {\"auth\": 0}\n"
            "    }\n"
            "\n"
            "`total` is the number of well-formed lines (so it equals the sum of `levels`),\n"
            "and `services` maps each name from a `service=` field to the number of\n"
            "well-formed lines that mention it.\n"
        ),
        files={
            "requirements.txt": "pytest==8.2.0\n",
            ".dockerignore": "tests/\nsolution/\n.git/\n__pycache__/\n",
            "tests/test_output.py": (
                "import json\n"
                "from pathlib import Path\n"
                "\n"
                'OUT = Path("/app/out/summary.json")\n'
                "\n"
                "\n"
                "def _load():\n"
                '    return json.loads(OUT.read_text(encoding="utf-8"))\n'
                "\n"
                "\n"
                "def test_schema():\n"
                "    data = _load()\n"
                "    assert isinstance(data, dict)\n"
                '    assert set(data) == {"total", "levels", "services"}\n'
                "\n"
                "\n"
                "def test_total_matches_level_sum():\n"
                "    data = _load()\n"
                '    assert data["total"] == sum(data["levels"].values())\n'
                "\n"
                "\n"
                "def test_all_levels_are_reported():\n"
                "    data = _load()\n"
                '    assert set(data["levels"]) == {"ERROR", "WARN", "INFO"}\n'
                '    assert data["levels"]["ERROR"] > 0\n'
                '    assert data["levels"]["WARN"] > 0\n'
                "\n"
                "\n"
                "def test_services_are_counted():\n"
                "    data = _load()\n"
                '    assert data["services"]\n'
                '    assert sum(data["services"].values()) <= data["total"]\n'
            ),
            "solution/solve.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python3 /solution/parse_logs.py\n"
            ),
            "solution/parse_logs.py": (
                "import json\n"
                "import re\n"
                "from pathlib import Path\n"
                "\n"
                "LINE = re.compile(\n"
                "    r'^(?P<ts>\\S+)\\s+(?P<level>ERROR|WARN|INFO)\\s+'\n"
                "    r'service=(?P<service>\\S+)\\s+msg=\".*\"$'\n"
                ")\n"
                "\n"
                "\n"
                "def main() -> None:\n"
                '    levels = {"ERROR": 0, "WARN": 0, "INFO": 0}\n'
                "    services: dict[str, int] = {}\n"
                "    total = 0\n"
                '    for path in sorted(Path("/app/data/logs").glob("*.log")):\n'
                '        for raw in path.read_text(encoding="utf-8").splitlines():\n'
                "            m = LINE.match(raw.strip())\n"
                "            if not m:\n"
                "                continue\n"
                "            total += 1\n"
                '            levels[m.group("level")] += 1\n'
                '            svc = m.group("service")\n'
                "            services[svc] = services.get(svc, 0) + 1\n"
                '    out = Path("/app/out/summary.json")\n'
                "    out.parent.mkdir(parents=True, exist_ok=True)\n"
                "    out.write_text(\n"
                '        json.dumps({"total": total, "levels": levels, "services": services}, indent=2)\n'
                "    )\n"
                "\n"
                "\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
        },
    ),
    # ---------------------------------------------------------------- t02
    DemoTask(
        task_id="t02_csv_dedupe",
        planted_issue="healthy control: a well-formed, mid-difficulty task",
        task_yaml="\n".join(
            [
                "id: t02_csv_dedupe",
                "instruction: Deduplicate /app/data/raw.csv keeping the first occurrence of each id.",
                "difficulty: medium",
                "author: demo-corpus",
                "timeout_s: 900",
                "expected_output_format: csv",
                "solution_cmd: bash /solution/solve.sh",
                "verifier_cmd: python -m pytest -q /tests",
                _demo_block(0.15, "normal", "SHIP"),
                "",
            ]
        ),
        dockerfile="\n".join(
            [
                "FROM python:3.11-slim",
                "",
                "WORKDIR /app",
                "COPY data/ /app/data/",
                "COPY requirements.txt /app/requirements.txt",
                "RUN pip install --no-cache-dir -r /app/requirements.txt",
                "",
                'CMD ["sleep", "infinity"]',
                "",
            ]
        ),
        instruction=(
            "# Deduplicate the rows\n"
            "\n"
            "`/app/data/raw.csv` has the header `id,name,amount` and contains duplicate\n"
            "`id` values. Duplicates must **not** simply be summed or merged: the row that\n"
            "appears first in the file wins and later rows with the same `id` are dropped.\n"
            "\n"
            "Write the deduplicated table as CSV (header included, same column order) to\n"
            "`/app/out/deduplicated.csv`, sorted by `id` as an integer, ascending.\n"
        ),
        files={
            "requirements.txt": "pytest==8.2.0\n",
            ".dockerignore": "tests/\nsolution/\n.git/\n__pycache__/\n",
            "tests/test_output.py": (
                "import csv\n"
                "from pathlib import Path\n"
                "\n"
                'RAW = Path("/app/data/raw.csv")\n'
                'OUT = Path("/app/out/deduplicated.csv")\n'
                "\n"
                "\n"
                "def _rows(path):\n"
                '    with path.open(newline="", encoding="utf-8") as fh:\n'
                "        return list(csv.DictReader(fh))\n"
                "\n"
                "\n"
                "def test_header_is_preserved():\n"
                "    rows = _rows(OUT)\n"
                "    assert rows\n"
                '    assert set(rows[0]) == {"id", "name", "amount"}\n'
                "\n"
                "\n"
                "def test_ids_are_unique_and_sorted():\n"
                "    ids = [int(r['id']) for r in _rows(OUT)]\n"
                "    assert len(ids) == len(set(ids))\n"
                "    assert ids == sorted(ids)\n"
                "\n"
                "\n"
                "def test_first_occurrence_wins():\n"
                "    expected = {}\n"
                "    for row in _rows(RAW):\n"
                '        expected.setdefault(row["id"], row)\n'
                '    got = {r["id"]: r for r in _rows(OUT)}\n'
                "    assert got == expected\n"
            ),
            "solution/solve.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python3 /solution/dedupe.py\n"
            ),
            "solution/dedupe.py": (
                "import csv\n"
                "from pathlib import Path\n"
                "\n"
                "\n"
                "def main() -> None:\n"
                '    with Path("/app/data/raw.csv").open(newline="", encoding="utf-8") as fh:\n'
                "        rows = list(csv.DictReader(fh))\n"
                "    seen = {}\n"
                "    for row in rows:\n"
                '        seen.setdefault(row["id"], row)\n'
                '    ordered = [seen[k] for k in sorted(seen, key=int)]\n'
                '    out = Path("/app/out/deduplicated.csv")\n'
                "    out.parent.mkdir(parents=True, exist_ok=True)\n"
                '    with out.open("w", newline="", encoding="utf-8") as fh:\n'
                '        writer = csv.DictWriter(fh, fieldnames=["id", "name", "amount"])\n'
                "        writer.writeheader()\n"
                "        writer.writerows(ordered)\n"
                "\n"
                "\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
        },
    ),
    # ---------------------------------------------------------------- t03
    DemoTask(
        task_id="t03_trivial_rename",
        planted_issue="degenerately easy: the model solves it almost every time",
        task_yaml="\n".join(
            [
                "id: t03_trivial_rename",
                "instruction: Rename /app/data/input.txt to /app/data/output.txt.",
                "difficulty: trivial",
                "author: demo-corpus",
                "timeout_s: 300",
                "expected_output_format: text",
                "solution_cmd: bash /solution/solve.sh",
                "verifier_cmd: python -m pytest -q /tests",
                _demo_block(0.95, "too_easy", "DROP", "trivially_solvable"),
                "",
            ]
        ),
        dockerfile="\n".join(
            [
                "FROM python:3.11-slim",
                "",
                "WORKDIR /app",
                "COPY data/ /app/data/",
                "COPY requirements.txt /app/requirements.txt",
                "RUN pip install --no-cache-dir -r /app/requirements.txt",
                "",
                'CMD ["sleep", "infinity"]',
                "",
            ]
        ),
        instruction=(
            "# Rename a file\n"
            "\n"
            "Rename `/app/data/input.txt` to `/app/data/output.txt`.\n"
            "\n"
            "The result must be a plain text file whose bytes are identical to the\n"
            "original file, and `/app/data/input.txt` must no longer exist.\n"
        ),
        files={
            "requirements.txt": "pytest==8.2.0\n",
            ".dockerignore": "tests/\nsolution/\n.git/\n__pycache__/\n",
            "tests/test_rename.py": (
                "from pathlib import Path\n"
                "\n"
                'SRC = Path("/app/data/input.txt")\n'
                'DST = Path("/app/data/output.txt")\n'
                'ORIGINAL = (Path(__file__).parent / "original.txt").read_text(encoding="utf-8")\n'
                "\n"
                "\n"
                "def test_source_is_gone():\n"
                "    assert not SRC.exists()\n"
                "\n"
                "\n"
                "def test_target_exists():\n"
                "    assert DST.is_file()\n"
                "\n"
                "\n"
                "def test_bytes_are_unchanged():\n"
                "    assert DST.read_text(encoding=\"utf-8\") == ORIGINAL\n"
            ),
            "solution/solve.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "mv /app/data/input.txt /app/data/output.txt\n"
            ),
        },
    ),
    # ---------------------------------------------------------------- t04
    DemoTask(
        task_id="t04_ambiguous_spec",
        planted_issue="underspecified: the instruction never says where the output goes",
        task_yaml="\n".join(
            [
                "id: t04_ambiguous_spec",
                "instruction: Summarise the errors in /app/data. - intentionally vague",
                "difficulty: unknown",
                "author: demo-corpus",
                "timeout_s: 900",
                "solution_cmd: bash /solution/solve.sh",
                "verifier_cmd: python -m pytest -q /tests",
                _demo_block(0.0, "wrong_path", "REWRITE", "underspecified_instruction"),
                "",
            ]
        ),
        dockerfile="\n".join(
            [
                "FROM python:3.11-slim",
                "",
                "WORKDIR /app",
                "COPY data/ /app/data/",
                "COPY requirements.txt /app/requirements.txt",
                "RUN pip install --no-cache-dir -r /app/requirements.txt",
                "",
                'CMD ["sleep", "infinity"]',
                "",
            ]
        ),
        instruction=(
            "# Error summary\n"
            "\n"
            "Look at the log files under `/app/data` and produce a summary of the errors\n"
            "found there. It must be written where the grader can find it.\n"
            "\n"
            "Only lines whose message contains the word `fail` should be counted.\n"
        ),
        files={
            "requirements.txt": "pytest==8.2.0\n",
            ".dockerignore": "tests/\nsolution/\n.git/\n__pycache__/\n",
            "tests/test_output.py": (
                "import json\n"
                "from pathlib import Path\n"
                "\n"
                'OUT = Path("/app/out/errors.json")\n'
                "\n"
                "\n"
                "def _load():\n"
                '    return json.loads(OUT.read_text(encoding="utf-8"))\n'
                "\n"
                "\n"
                "def test_output_exists():\n"
                "    assert OUT.is_file()\n"
                "\n"
                "\n"
                "def test_total_is_positive():\n"
                "    data = _load()\n"
                "    assert isinstance(data, dict)\n"
                '    assert data["total"] > 0\n'
                "\n"
                "\n"
                "def test_services_are_reported():\n"
                "    data = _load()\n"
                '    assert data["services"]\n'
            ),
            "solution/solve.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python3 /solution/summarise.py\n"
            ),
            "solution/summarise.py": (
                "import json\n"
                "import re\n"
                "from pathlib import Path\n"
                "\n"
                "LINE = re.compile(\n"
                "    r'^(?P<ts>\\S+)\\s+(?P<level>ERROR|WARN|INFO)\\s+'\n"
                "    r'service=(?P<service>\\S+)\\s+msg=\"(?P<msg>.*)\"$'\n"
                ")\n"
                "\n"
                "\n"
                "def main() -> None:\n"
                "    services: dict[str, int] = {}\n"
                "    total = 0\n"
                '    for path in sorted(Path("/app/data/logs").glob("*.log")):\n'
                '        for raw in path.read_text(encoding="utf-8").splitlines():\n'
                "            m = LINE.match(raw.strip())\n"
                "            if not m:\n"
                "                continue\n"
                '            if "fail" not in m.group("msg"):\n'
                "                continue\n"
                "            total += 1\n"
                '            svc = m.group("service")\n'
                "            services[svc] = services.get(svc, 0) + 1\n"
                '    out = Path("/app/out/errors.json")\n'
                "    out.parent.mkdir(parents=True, exist_ok=True)\n"
                '    out.write_text(json.dumps({"total": total, "services": services}, indent=2))\n'
                "\n"
                "\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
        },
    ),
    # ---------------------------------------------------------------- t05
    DemoTask(
        task_id="t05_leaky_tests",
        planted_issue="the hidden verifier is COPYed into the agent's image (S001)",
        task_yaml="\n".join(
            [
                "id: t05_leaky_tests",
                "instruction: Normalise /app/data/raw.csv into /app/out/normalised.csv.",
                "difficulty: medium",
                "author: demo-corpus",
                "timeout_s: 900",
                "expected_output_format: csv",
                "verifier_cmd: python -m pytest -q /app/tests",
                _demo_block(None, "normal", "BLOCKED"),
                "",
            ]
        ),
        dockerfile="\n".join(
            [
                "FROM python:3.11-slim",
                "",
                "WORKDIR /app",
                "COPY data/ /app/data/",
                "COPY tests/ /app/tests/",
                "RUN pip install --no-cache-dir pytest==8.2.0",
                "",
                'CMD ["sleep", "infinity"]',
                "",
            ]
        ),
        instruction=(
            "# Normalise the table\n"
            "\n"
            "Strip whitespace from every field of `/app/data/raw.csv`, lowercase the\n"
            "`name` column, and write the result as CSV to `/app/out/normalised.csv`\n"
            "keeping the header `id,name,amount`.\n"
        ),
        files={
            ".dockerignore": "solution/\n.git/\n__pycache__/\n",
            "tests/test_output.py": (
                "import csv\n"
                "from pathlib import Path\n"
                "\n"
                'OUT = Path("/app/out/normalised.csv")\n'
                "\n"
                "\n"
                "def test_normalised():\n"
                '    with OUT.open(newline="", encoding="utf-8") as fh:\n'
                "        rows = list(csv.DictReader(fh))\n"
                '    assert rows[0]["name"] == rows[0]["name"].lower()\n'
            ),
            "solution/solve.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python3 /solution/normalise.py\n"
            ),
            "solution/normalise.py": (
                "import csv\n"
                "from pathlib import Path\n"
                "\n"
                "\n"
                "def main() -> None:\n"
                '    with Path("/app/data/raw.csv").open(newline="", encoding="utf-8") as fh:\n'
                "        rows = list(csv.DictReader(fh))\n"
                "    for row in rows:\n"
                '        row["name"] = row["name"].strip().lower()\n'
                '        row["amount"] = row["amount"].strip()\n'
                '        row["id"] = row["id"].strip()\n'
                '    out = Path("/app/out/normalised.csv")\n'
                "    out.parent.mkdir(parents=True, exist_ok=True)\n"
                '    with out.open("w", newline="", encoding="utf-8") as fh:\n'
                '        writer = csv.DictWriter(fh, fieldnames=["id", "name", "amount"])\n'
                "        writer.writeheader()\n"
                "        writer.writerows(rows)\n"
                "\n"
                "\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
        },
    ),
    # ---------------------------------------------------------------- t06
    DemoTask(
        task_id="t06_flaky_timing",
        planted_issue="sleep-based sync, unseeded RNG and wall-clock reads in the verifier (S007 / D001)",
        task_yaml="\n".join(
            [
                "id: t06_flaky_timing",
                "instruction: Emit /app/out/report.json describing the events in /app/data.",
                "difficulty: medium",
                "author: demo-corpus",
                "timeout_s: 900",
                "verifier_cmd: python -m pytest -q /tests",
                _demo_block(None, "normal", "BLOCKED", None, ("D001",)),
                "",
            ]
        ),
        dockerfile="\n".join(
            [
                "FROM python:3.11-slim",
                "",
                "WORKDIR /app",
                "COPY data/ /app/data/",
                "COPY requirements.txt /app/requirements.txt",
                "RUN pip install --no-cache-dir -r /app/requirements.txt",
                "",
                'CMD ["sleep", "infinity"]',
                "",
            ]
        ),
        instruction=(
            "# Event report\n"
            "\n"
            "Count the events in every file under `/app/data` and write a JSON object to\n"
            "`/app/out/report.json` with the fields `date`, `events` (per file counts) and\n"
            "`p95_ms`.\n"
        ),
        files={
            "requirements.txt": "pytest==8.2.0\n",
            ".dockerignore": "tests/\nsolution/\n.git/\n__pycache__/\n",
            "tests/test_report.py": (
                "import json\n"
                "import random\n"
                "import time\n"
                "from datetime import datetime\n"
                "from pathlib import Path\n"
                "\n"
                'OUT = Path("/app/out/report.json")\n'
                "\n"
                "\n"
                "def _wait_for_output():\n"
                "    for _ in range(5):\n"
                "        if OUT.exists():\n"
                "            return\n"
                "        time.sleep(1)\n"
                "\n"
                "\n"
                "def test_report_shape():\n"
                "    _wait_for_output()\n"
                "    assert OUT.exists()\n"
                "    data = json.loads(OUT.read_text(encoding=\"utf-8\"))\n"
                '    assert data["date"] == datetime.now().strftime("%Y-%m-%d")\n'
                '    assert data["events"]\n'
                "\n"
                "\n"
                "def test_p95_is_plausible():\n"
                "    data = json.loads(OUT.read_text(encoding=\"utf-8\"))\n"
                '    assert data["p95_ms"] > random.random()\n'
            ),
            "solution/solve.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python3 /solution/report.py\n"
            ),
            "solution/report.py": (
                "import json\n"
                "from datetime import date\n"
                "from pathlib import Path\n"
                "\n"
                "\n"
                "def main() -> None:\n"
                "    events, latencies = {}, []\n"
                '    for path in sorted(Path("/app/data").glob("*.log")):\n'
                "        rows = [r for r in path.read_text(encoding=\"utf-8\").splitlines() if r.strip()]\n"
                "        events[path.name] = len(rows)\n"
                "        for row in rows:\n"
                "            if \" ms=\" in row:\n"
                "                latencies.append(int(row.rsplit(\" ms=\", 1)[1]))\n"
                "    latencies.sort()\n"
                "    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0\n"
                '    out = Path("/app/out/report.json")\n'
                "    out.parent.mkdir(parents=True, exist_ok=True)\n"
                "    out.write_text(\n"
                '        json.dumps({"date": date.today().isoformat(), "events": events, "p95_ms": p95})\n'
                "    )\n"
                "\n"
                "\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
        },
    ),
    # ---------------------------------------------------------------- t07
    DemoTask(
        task_id="t07_network_dep",
        planted_issue="the verifier fetches an expected value over the network at run time (S006 / D003)",
        task_yaml="\n".join(
            [
                "id: t07_network_dep",
                "instruction: Extract the version string from /app/data/meta.txt into /app/out/version.txt.",
                "difficulty: easy",
                "author: demo-corpus",
                "timeout_s: 900",
                "verifier_cmd: python -m pytest -q /tests",
                _demo_block(None, "normal", "BLOCKED", None, ("D003",)),
                "",
            ]
        ),
        dockerfile="\n".join(
            [
                "FROM python:3.11-slim",
                "",
                "WORKDIR /app",
                "COPY data/ /app/data/",
                "RUN pip install --no-cache-dir requests",
                "",
                'CMD ["sleep", "infinity"]',
                "",
            ]
        ),
        instruction=(
            "# Extract the version\n"
            "\n"
            "`/app/data/meta.txt` contains build metadata. Write the version string to\n"
            "`/app/out/version.txt` as a plain text file containing only the version.\n"
        ),
        files={
            ".dockerignore": "tests/\nsolution/\n.git/\n__pycache__/\n",
            "tests/test_version.py": (
                "import requests\n"
                "from pathlib import Path\n"
                "\n"
                'OUT = Path("/app/out/version.txt")\n'
                "\n"
                "\n"
                "def test_version_matches_upstream():\n"
                '    expected = requests.get("https://example.invalid/expected.json", timeout=10)\n'
                "    assert expected.status_code == 200\n"
                '    assert OUT.read_text(encoding="utf-8").strip() == expected.json()["version"]\n'
            ),
            "solution/solve.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "mkdir -p /app/out\n"
                "grep -oE '2\\.[0-9]+\\.[0-9]+' /app/data/meta.txt | head -n 1 > /app/out/version.txt\n"
            ),
        },
    ),
    # ---------------------------------------------------------------- t08
    DemoTask(
        task_id="t08_borderline",
        planted_issue=(
            "borderline: a genuinely hard task whose true solve rate sits just "
            "above the window, so the 90% interval still straddles the boundary"
        ),
        task_yaml="\n".join(
            [
                "id: t08_borderline",
                "instruction: Compute hourly request counts and p95 latency from /app/data.",
                "difficulty: hard",
                "author: demo-corpus",
                "timeout_s: 900",
                "expected_output_format: json",
                "solution_cmd: bash /solution/solve.sh",
                "verifier_cmd: python -m pytest -q /tests",
                _demo_block(0.35, "turn_limit", "NEEDS-MORE-RUNS", "genuinely_hard"),
                "",
            ]
        ),
        dockerfile="\n".join(
            [
                "FROM python:3.11-slim",
                "",
                "WORKDIR /app",
                "COPY data/ /app/data/",
                "COPY requirements.txt /app/requirements.txt",
                "RUN pip install --no-cache-dir -r /app/requirements.txt",
                "",
                'CMD ["sleep", "infinity"]',
                "",
            ]
        ),
        instruction=(
            "# Hourly latency report\n"
            "\n"
            "Every file under `/app/data` contains one line per request:\n"
            "\n"
            "    2024-06-02T13:45:01 route=/v1/search ms=137\n"
            "\n"
            "Write a JSON file to `/app/out/hourly.json` with this exact shape:\n"
            "\n"
            "    {\n"
            "      \"total\": 0,\n"
            "      \"hours\": {\"00\": {\"count\": 0, \"p95_ms\": 0}},\n"
            "      \"busiest_hour\": \"00\"\n"
            "    }\n"
            "\n"
            "`hours` is keyed by the two digit hour of the timestamp, `p95_ms` is the\n"
            "nearest-rank 95th percentile of the `ms` values in that hour, and\n"
            "`busiest_hour` is the hour with the most requests (lowest hour wins ties).\n"
        ),
        files={
            "requirements.txt": "pytest==8.2.0\n",
            ".dockerignore": "tests/\nsolution/\n.git/\n__pycache__/\n",
            "tests/test_output.py": (
                "import json\n"
                "from pathlib import Path\n"
                "\n"
                'OUT = Path("/app/out/hourly.json")\n'
                "\n"
                "\n"
                "def _load():\n"
                '    return json.loads(OUT.read_text(encoding="utf-8"))\n'
                "\n"
                "\n"
                "def test_shape():\n"
                "    data = _load()\n"
                '    assert set(data) == {"total", "hours", "busiest_hour"}\n'
                "    assert isinstance(data[\"hours\"], dict)\n"
                "\n"
                "\n"
                "def test_counts_add_up():\n"
                "    data = _load()\n"
                '    assert data["total"] == sum(h["count"] for h in data["hours"].values())\n'
                '    assert data["total"] > 0\n'
                "\n"
                "\n"
                "def test_p95_is_within_range():\n"
                "    data = _load()\n"
                '    assert all(h["p95_ms"] > 0 for h in data["hours"].values())\n'
                "\n"
                "\n"
                "def test_busiest_hour_agrees_with_counts():\n"
                "    data = _load()\n"
                '    best = max(data["hours"].items(), key=lambda kv: (kv[1]["count"], -int(kv[0])))\n'
                '    assert data["busiest_hour"] == best[0]\n'
            ),
            "solution/solve.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python3 /solution/hourly.py\n"
            ),
            "solution/hourly.py": (
                "import json\n"
                "from pathlib import Path\n"
                "\n"
                "\n"
                "def main() -> None:\n"
                "    buckets: dict[str, list[int]] = {}\n"
                '    for path in sorted(Path("/app/data").glob("*.log")):\n'
                '        for row in path.read_text(encoding="utf-8").splitlines():\n'
                '            if " ms=" not in row or "T" not in row:\n'
                "                continue\n"
                '            stamp, ms = row.rsplit(" ms=", 1)\n'
                '            hour = stamp.split("T", 1)[1][:2]\n'
                "            buckets.setdefault(hour, []).append(int(ms))\n"
                "    hours = {}\n"
                "    total = 0\n"
                "    for hour, values in sorted(buckets.items()):\n"
                "        values.sort()\n"
                "        total += len(values)\n"
                '        hours[hour] = {"count": len(values),'
                " \"p95_ms\": values[max(0, -(-95 * len(values) // 100) - 1)]}\n"
                '    busiest = min(hours, key=lambda h: (-hours[h]["count"], h)) if hours else "00"\n'
                '    out = Path("/app/out/hourly.json")\n'
                "    out.parent.mkdir(parents=True, exist_ok=True)\n"
                "    out.write_text(\n"
                '        json.dumps({"total": total, "hours": hours, "busiest_hour": busiest}, indent=2)\n'
                "    )\n"
                "\n"
                "\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
        },
    ),
]


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------
def _task_payload(task: DemoTask) -> dict[str, str]:
    """All files for one demo task, with generated input data merged in."""
    rng = random.Random(f"taskgate-demo::{task.task_id}")
    files = dict(task.files)
    if task.task_id in ("t01_log_parser", "t04_ambiguous_spec"):
        files.update(_access_logs(rng))
    elif task.task_id == "t02_csv_dedupe":
        files.update(_raw_csv(rng))
    elif task.task_id == "t03_trivial_rename":
        files["data/input.txt"] = "build id: 4177\nstatus: ok\n"
        files["tests/original.txt"] = files["data/input.txt"]
    elif task.task_id == "t05_leaky_tests":
        files.update(_raw_csv(rng))
    elif task.task_id in ("t06_flaky_timing", "t08_borderline"):
        files.update(_latency_events(rng))
    elif task.task_id == "t07_network_dep":
        files["data/meta.txt"] = "name=widget\nversion=2.4.1\nbuild=ci-9931\n"
    files["task.yaml"] = task.task_yaml
    files["Dockerfile"] = task.dockerfile
    files["instruction.md"] = task.instruction
    return files


def build_demo_corpus(dest: Path, force: bool = False) -> list[Path]:
    """Write the 8-task demo corpus into ``dest``. Returns the task directories."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task in TASKS:
        task_dir = dest / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        for rel, content in _task_payload(task).items():
            path = task_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not force:
                pass
            path.write_text(content, encoding="utf-8", newline="\n")
        written.append(task_dir)
    write_manifest(dest)
    return written


def write_manifest(dest: Path) -> Path:
    dest = Path(dest)
    manifest = {
        "generated_by": "taskgate demo",
        "note": (
            "Ground truth for the demo corpus only. The mock agent draws its "
            "pass/fail decisions from true_pass_rate; routes are asserted in "
            "tests/test_integration_demo.py."
        ),
        "tasks": [
            {
                "task_id": t.task_id,
                "planted_issue": t.planted_issue,
                "true_pass_rate": (
                    t.task_yaml.split("true_pass_rate:")[1].split("\n")[0].strip()
                    if "true_pass_rate:" in t.task_yaml
                    else None
                ),
                "expected_route": (
                    t.task_yaml.split("expected_route:")[1].split("\n")[0].strip()
                ),
            }
            for t in TASKS
        ],
    }
    path = dest / "demo_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
