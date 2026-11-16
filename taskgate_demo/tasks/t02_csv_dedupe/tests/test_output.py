import csv
from pathlib import Path

RAW = Path("/app/data/raw.csv")
OUT = Path("/app/out/deduplicated.csv")


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_header_is_preserved():
    rows = _rows(OUT)
    assert rows
    assert set(rows[0]) == {"id", "name", "amount"}


def test_ids_are_unique_and_sorted():
    ids = [int(r['id']) for r in _rows(OUT)]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_first_occurrence_wins():
    expected = {}
    for row in _rows(RAW):
        expected.setdefault(row["id"], row)
    got = {r["id"]: r for r in _rows(OUT)}
    assert got == expected
