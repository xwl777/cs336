import pytest

from .adapters import run_exact_line_deduplication


def test_exact_line_deduplication_global_counts_and_line_endings(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    first = inputs / "first.txt"
    second = inputs / "second.txt"
    empty = inputs / "empty.txt"
    first.write_bytes("共享\r\n重复\r\n重复\r\n保留 \r\n".encode())
    second.write_bytes("保留\n共享".encode())
    empty.write_bytes(b"")
    output = tmp_path / "nested" / "output"

    run_exact_line_deduplication([first, second, empty], output)

    assert (output / "first.txt").read_bytes() == "保留 \r\n".encode()
    assert (output / "second.txt").read_bytes() == "保留\n".encode()
    assert (output / "empty.txt").read_bytes() == b""
    assert second.read_bytes() == "保留\n共享".encode()


def test_exact_line_deduplication_rejects_input_overwrite(tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("keep me\n")
    with pytest.raises(ValueError, match="overwrite"):
        run_exact_line_deduplication([path], tmp_path)
    assert path.read_text() == "keep me\n"


def test_exact_line_deduplication_rejects_basename_collision(tmp_path):
    first = tmp_path / "one" / "document.txt"
    second = tmp_path / "two" / "document.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first\n")
    second.write_text("second\n")
    with pytest.raises(ValueError, match="basenames"):
        run_exact_line_deduplication([first, second], tmp_path / "output")
