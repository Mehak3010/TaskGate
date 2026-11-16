# Summarise the access logs

Read every `*.log` file under `/app/data/logs/`.

A well-formed line looks like this:

    2024-05-01T12:00:00 ERROR service=auth msg="token expired"

Lines that do not have a level, a `service=` field and a `msg=` field are
malformed and must be ignored completely.

Write a single JSON file to `/app/out/summary.json` with exactly this shape:

    {
      "total": 0,
      "levels": {"ERROR": 0, "WARN": 0, "INFO": 0},
      "services": {"auth": 0}
    }

`total` is the number of well-formed lines (so it equals the sum of `levels`),
and `services` maps each name from a `service=` field to the number of
well-formed lines that mention it.
