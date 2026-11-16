import csv
from pathlib import Path

OUT = Path("/app/out/normalised.csv")


def test_normalised():
    with OUT.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["name"] == rows[0]["name"].lower()
