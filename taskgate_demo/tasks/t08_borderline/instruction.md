# Hourly latency report

Every file under `/app/data` contains one line per request:

    2024-06-02T13:45:01 route=/v1/search ms=137

Write a JSON file to `/app/out/hourly.json` with this exact shape:

    {
      "total": 0,
      "hours": {"00": {"count": 0, "p95_ms": 0}},
      "busiest_hour": "00"
    }

`hours` is keyed by the two digit hour of the timestamp, `p95_ms` is the
nearest-rank 95th percentile of the `ms` values in that hour, and
`busiest_hour` is the hour with the most requests (lowest hour wins ties).
