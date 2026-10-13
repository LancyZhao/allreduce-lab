"""Tiny vector helpers (stdlib only — no numpy) and a reference reduction."""
from __future__ import annotations

Vector = list[float]


def reference_allreduce(vectors: list[Vector]) -> Vector:
    """Ground-truth: elementwise sum across all ranks."""
    if not vectors:
        return []
    numel = len(vectors[0])
    assert all(len(v) == numel for v in vectors), "ranks must share vector length"
    out = [0.0] * numel
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    return out


def chunk_ranges(numel: int, parts: int) -> list[tuple[int, int]]:
    """Split [0, numel) into `parts` contiguous ranges, as even as possible."""
    base, extra = divmod(numel, parts)
    ranges, start = [], 0
    for p in range(parts):
        size = base + (1 if p < extra else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def approx_equal(a: Vector, b: Vector, eps: float = 1e-6) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= eps for x, y in zip(a, b))
