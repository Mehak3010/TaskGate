from pathlib import Path

SRC = Path("/app/data/input.txt")
DST = Path("/app/data/output.txt")
ORIGINAL = (Path(__file__).parent / "original.txt").read_text(encoding="utf-8")


def test_source_is_gone():
    assert not SRC.exists()


def test_target_exists():
    assert DST.is_file()


def test_bytes_are_unchanged():
    assert DST.read_text(encoding="utf-8") == ORIGINAL
