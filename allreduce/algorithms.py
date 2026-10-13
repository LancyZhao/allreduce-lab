"""All-reduce algorithms, each returning (result_per_rank, CommStats).

Every algorithm is simulated logically on one process (deterministic, no GPU):
we move the data exactly as the real collective would and count bytes + steps,
then the result on *every* rank must equal the elementwise sum (verified in
tests). This makes the communication-volume formulas falsifiable, not just
asserted in prose.
"""
from __future__ import annotations

import math

from allreduce.simulator import CommStats
from allreduce.tensor import Vector, chunk_ranges


def ring_allreduce(vectors: list[Vector], elem_bytes: int = 4) -> tuple[list[Vector], CommStats]:
    n = len(vectors)
    numel = len(vectors[0])
    bufs = [list(v) for v in vectors]
    rng = chunk_ranges(numel, n)
    cbytes = [(e - s) * elem_bytes for s, e in rng]
    st = CommStats("ring", n, numel, elem_bytes)
    if n == 1:
        return bufs, st

    # reduce-scatter: n-1 steps, each rank sends one chunk to its ring successor
    for step in range(n - 1):
        inbox = {}
        for r in range(n):
            idx = (r - step) % n
            s, e = rng[idx]
            st.rank_send(r, cbytes[idx])
            inbox[(r + 1) % n] = (idx, bufs[r][s:e])
        for dst, (idx, data) in inbox.items():
            s, _ = rng[idx]
            for i, x in enumerate(data):
                bufs[dst][s + i] += x
        st.steps += 1

    # all-gather: n-1 steps, circulate the fully-reduced chunks
    for step in range(n - 1):
        inbox = {}
        for r in range(n):
            idx = (r + 1 - step) % n
            s, e = rng[idx]
            st.rank_send(r, cbytes[idx])
            inbox[(r + 1) % n] = (idx, bufs[r][s:e])
        for dst, (idx, data) in inbox.items():
            s, e = rng[idx]
            bufs[dst][s:e] = data
        st.steps += 1
    return bufs, st


def naive_allreduce(vectors: list[Vector], elem_bytes: int = 4,
                    root: int = 0) -> tuple[list[Vector], CommStats]:
    """Root-centric: everyone sends to root, root sums and broadcasts back.
    Correct and simple, but the root is a bandwidth hotspot (peak = 2(n-1)·D)."""
    n = len(vectors)
    numel = len(vectors[0])
    vec_bytes = numel * elem_bytes
    acc = list(vectors[root])
    st = CommStats("naive", n, numel, elem_bytes)
    if n == 1:
        return [list(acc)], st
    for r in range(n):                     # gather to root
        if r == root:
            continue
        st.rank_send(r, vec_bytes)
        for i in range(numel):
            acc[i] += vectors[r][i]
    st.steps += 1
    for r in range(n):                     # broadcast from root
        if r != root:
            st.rank_send(root, vec_bytes)
    st.steps += 1
    return [list(acc) for _ in range(n)], st


def tree_allreduce(vectors: list[Vector], elem_bytes: int = 4) -> tuple[list[Vector], CommStats]:
    """Binomial-tree reduce to rank 0, then binomial broadcast back.
    Latency is O(log n) steps instead of the ring's O(n)."""
    n = len(vectors)
    numel = len(vectors[0])
    vec_bytes = numel * elem_bytes
    bufs = [list(v) for v in vectors]
    st = CommStats("tree", n, numel, elem_bytes)
    if n == 1:
        return bufs, st
    rounds = math.ceil(math.log2(n))

    # reduce to root 0: higher-of-pair sends up and drops out each round
    alive = set(range(n))
    mask = 1
    while mask < n:
        senders = []
        for r in sorted(alive):
            if r & mask:
                partner = r ^ mask
                if partner in alive:
                    st.rank_send(r, vec_bytes)
                    for i in range(numel):
                        bufs[partner][i] += bufs[r][i]
                    senders.append(r)
        alive.difference_update(senders)
        st.steps += 1
        mask <<= 1

    # broadcast from root 0: mirror the tree outward
    have = {0}
    mask = 1 << (rounds - 1)
    while mask >= 1:
        for r in sorted(have):
            partner = r ^ mask
            if partner < n and partner not in have:
                st.rank_send(r, vec_bytes)
                bufs[partner] = list(bufs[r])
                have.add(partner)
        st.steps += 1
        mask >>= 1
    return bufs, st


ALGORITHMS = {
    "ring": ring_allreduce,
    "tree": tree_allreduce,
    "naive": naive_allreduce,
}
