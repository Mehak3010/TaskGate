import json
import random
import time
from datetime import datetime
from pathlib import Path

OUT = Path("/app/out/report.json")


def _wait_for_output():
    for _ in range(5):
        if OUT.exists():
            return
        time.sleep(1)


def test_report_shape():
    _wait_for_output()
    assert OUT.exists()
    data = json.loads(OUT.read_text(encoding="utf-8"))
    assert data["date"] == datetime.now().strftime("%Y-%m-%d")
    assert data["events"]


def test_p95_is_plausible():
    data = json.loads(OUT.read_text(encoding="utf-8"))
    assert data["p95_ms"] > random.random()
