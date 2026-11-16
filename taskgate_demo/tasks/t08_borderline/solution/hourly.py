import json
from pathlib import Path


def main() -> None:
    buckets: dict[str, list[int]] = {}
    for path in sorted(Path("/app/data").glob("*.log")):
        for row in path.read_text(encoding="utf-8").splitlines():
            if " ms=" not in row or "T" not in row:
                continue
            stamp, ms = row.rsplit(" ms=", 1)
            hour = stamp.split("T", 1)[1][:2]
            buckets.setdefault(hour, []).append(int(ms))
    hours = {}
    total = 0
    for hour, values in sorted(buckets.items()):
        values.sort()
        total += len(values)
        hours[hour] = {"count": len(values), "p95_ms": values[max(0, -(-95 * len(values) // 100) - 1)]}
    busiest = min(hours, key=lambda h: (-hours[h]["count"], h)) if hours else "00"
    out = Path("/app/out/hourly.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"total": total, "hours": hours, "busiest_hour": busiest}, indent=2)
    )


if __name__ == "__main__":
    main()
