import requests
from pathlib import Path

OUT = Path("/app/out/version.txt")


def test_version_matches_upstream():
    expected = requests.get("https://example.invalid/expected.json", timeout=10)
    assert expected.status_code == 200
    assert OUT.read_text(encoding="utf-8").strip() == expected.json()["version"]
