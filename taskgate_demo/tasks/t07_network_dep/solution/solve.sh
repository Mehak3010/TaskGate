#!/usr/bin/env bash
set -euo pipefail
mkdir -p /app/out
grep -oE '2\.[0-9]+\.[0-9]+' /app/data/meta.txt | head -n 1 > /app/out/version.txt
