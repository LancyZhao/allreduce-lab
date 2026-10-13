import math

import pytest

from allreduce.algorithms import naive_allreduce, ring_allreduce, tree_allreduce


def _vectors(n, numel):
    return [[1.0] * numel for _ in range(n)]


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 16])
def test_ring_total_bytes_is_2_n_minus_1_D(n):
    numel, elem = 128, 4
    _, st = ring_allreduce(_vectors(n, numel), elem)
    assert st.total_bytes == 2 * (n - 1) * numel * elem
    assert st.steps == 2 * (n - 1)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 16])
def test_naive_root_is_the_hotspot(n):
    numel, elem = 128, 4
    D = numel * elem
    _, st = naive_allreduce(_vectors(n, numel), elem)
    assert st.steps == 2
    assert st.total_bytes == 2 * (n - 1) * D
    # root sends the whole broadcast => peak = (n-1)*D, far above ring's ~2D
    assert st.peak_rank_bytes == (n - 1) * D


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 16])
def test_tree_latency_is_logarithmic(n):
    _, st = tree_allreduce(_vectors(n, 128), 4)
    assert st.steps == 2 * math.ceil(math.log2(n))


@pytest.mark.parametrize("n", [4, 8, 16])
def test_ring_peak_is_balanced_and_small(n):
    numel, elem = 1024, 4
    D = numel * elem
    _, ring = ring_allreduce(_vectors(n, numel), elem)
    _, naive = naive_allreduce(_vectors(n, numel), elem)
    # ring per-rank peak ~ 2(n-1)/n * D, independent of n (approaches 2D)
    assert ring.peak_rank_bytes <= 2 * D
    assert ring.peak_rank_bytes < naive.peak_rank_bytes
