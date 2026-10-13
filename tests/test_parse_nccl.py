from pathlib import Path

from nccl.parse_nccl import format_table, parse

SAMPLE = Path(__file__).resolve().parent.parent / "nccl" / "sample_output.txt"


def test_parses_sample_rows():
    rows = parse(SAMPLE.read_text(encoding="utf-8"))
    assert len(rows) == 7                      # 7 data lines, comments skipped
    assert rows[0].size_bytes == 1024
    assert rows[-1].size_bytes == 268435456


def test_busbw_grows_then_table_reports_peak():
    rows = parse(SAMPLE.read_text(encoding="utf-8"))
    # bus bandwidth rises with message size (small = latency bound)
    assert rows[0].busbw < rows[-1].busbw
    peak = max(r.busbw for r in rows)
    assert abs(peak - 56.86) < 1e-6     # out-of-place busbw of the largest message
    assert "peak busbw" in format_table(rows)
