"""Parse `nccl-tests` all_reduce_perf output into a tidy table.

Run the real benchmark on a GPU box (see run_nccl_tests.sh), then:
    python nccl/parse_nccl.py nccl/sample_output.txt
This parser is verified in CI against the committed sample so the "data plumbing"
is correct even on a machine with no GPU.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class Row:
    size_bytes: int
    count: int
    time_us: float      # out-of-place
    algbw: float        # GB/s
    busbw: float        # GB/s (the number people actually quote)


def parse(text: str) -> list[Row]:
    rows: list[Row] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 8:
            continue
        try:
            rows.append(Row(int(f[0]), int(f[1]), float(f[5]), float(f[6]), float(f[7])))
        except ValueError:
            continue
    return rows


def format_table(rows: list[Row]) -> str:
    out = [f"{'size(B)':>12}{'count':>12}{'time(us)':>12}{'algbw(GB/s)':>14}{'busbw(GB/s)':>14}"]
    for r in rows:
        out.append(f"{r.size_bytes:>12}{r.count:>12}{r.time_us:>12.1f}{r.algbw:>14.2f}{r.busbw:>14.2f}")
    if rows:
        peak = max(rows, key=lambda r: r.busbw)
        out.append(f"\npeak busbw = {peak.busbw:.2f} GB/s @ {peak.size_bytes} B")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    path = argv[0] if argv else "nccl/sample_output.txt"
    with open(path, encoding="utf-8") as fh:
        print(format_table(parse(fh.read())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
