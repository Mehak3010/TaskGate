import json
import re
from pathlib import Path

LINE = re.compile(
    r'^(?P<ts>\S+)\s+(?P<level>ERROR|WARN|INFO)\s+'
    r'service=(?P<service>\S+)\s+msg=".*"$'
)


def main() -> None:
    levels = {"ERROR": 0, "WARN": 0, "INFO": 0}
    services: dict[str, int] = {}
    total = 0
    for path in sorted(Path("/app/data/logs").glob("*.log")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            m = LINE.match(raw.strip())
            if not m:
                continue
            total += 1
            levels[m.group("level")] += 1
            svc = m.group("service")
            services[svc] = services.get(svc, 0) + 1
    out = Path("/app/out/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"total": total, "levels": levels, "services": services}, indent=2)
    )


if __name__ == "__main__":
    main()
