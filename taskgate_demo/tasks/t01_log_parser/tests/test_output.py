import json
from pathlib import Path

OUT = Path("/app/out/summary.json")


def _load():
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_schema():
    data = _load()
    assert isinstance(data, dict)
    assert set(data) == {"total", "levels", "services"}


def test_total_matches_level_sum():
    data = _load()
    assert data["total"] == sum(data["levels"].values())


def test_all_levels_are_reported():
    data = _load()
    assert set(data["levels"]) == {"ERROR", "WARN", "INFO"}
    assert data["levels"]["ERROR"] > 0
    assert data["levels"]["WARN"] > 0


def test_services_are_counted():
    data = _load()
    assert data["services"]
    assert sum(data["services"].values()) <= data["total"]
