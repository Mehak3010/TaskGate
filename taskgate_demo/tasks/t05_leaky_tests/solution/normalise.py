import csv
from pathlib import Path


def main() -> None:
    with Path("/app/data/raw.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["name"] = row["name"].strip().lower()
        row["amount"] = row["amount"].strip()
        row["id"] = row["id"].strip()
    out = Path("/app/out/normalised.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "name", "amount"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
