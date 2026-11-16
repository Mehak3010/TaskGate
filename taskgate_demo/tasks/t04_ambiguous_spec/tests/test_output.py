import json
from pathlib import Path

OUT = Path("/app/out/errors.json")


def _load():
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_output_exists():
    assert OUT.is_file()


def test_total_is_positive():
    data = _load()
    assert isinstance(data, dict)
    assert data["total"] > 0


def test_services_are_reported():
    data = _load()
    assert data["services"]
