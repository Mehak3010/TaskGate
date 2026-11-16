import csv
from pathlib import Path


def main() -> None:
    with Path("/app/data/raw.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    seen = {}
    for row in rows:
        seen.setdefault(row["id"], row)
    ordered = [seen[k] for k in sorted(seen, key=int)]
    out = Path("/app/out/deduplicated.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "name", "amount"])
        writer.writeheader()
        writer.writerows(ordered)


if __name__ == "__main__":
    main()
