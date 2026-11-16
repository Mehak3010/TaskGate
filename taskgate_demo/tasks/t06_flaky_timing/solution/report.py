import json
from datetime import date
from pathlib import Path


def main() -> None:
    events, latencies = {}, []
    for path in sorted(Path("/app/data").glob("*.log")):
        rows = [r for r in path.read_text(encoding="utf-8").splitlines() if r.strip()]
        events[path.name] = len(rows)
        for row in rows:
            if " ms=" in row:
                latencies.append(int(row.rsplit(" ms=", 1)[1]))
    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0
    out = Path("/app/out/report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"date": date.today().isoformat(), "events": events, "p95_ms": p95})
    )


if __name__ == "__main__":
    main()
