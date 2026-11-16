import json
from pathlib import Path

OUT = Path("/app/out/hourly.json")


def _load():
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_shape():
    data = _load()
    assert set(data) == {"total", "hours", "busiest_hour"}
    assert isinstance(data["hours"], dict)


def test_counts_add_up():
    data = _load()
    assert data["total"] == sum(h["count"] for h in data["hours"].values())
    assert data["total"] > 0


def test_p95_is_within_range():
    data = _load()
    assert all(h["p95_ms"] > 0 for h in data["hours"].values())


def test_busiest_hour_agrees_with_counts():
    data = _load()
    best = max(data["hours"].items(), key=lambda kv: (kv[1]["count"], -int(kv[0])))
    assert data["busiest_hour"] == best[0]
