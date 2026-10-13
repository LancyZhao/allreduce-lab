import pytest

from allreduce.algorithms import ALGORITHMS
from allreduce.tensor import approx_equal, reference_allreduce


def _vectors(n, numel):
    return [[float(((r + 3) * (i + 1)) % 11) for i in range(numel)] for r in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 16])
@pytest.mark.parametrize("numel", [1, 10, 97, 256])
@pytest.mark.parametrize("algo", list(ALGORITHMS))
def test_every_rank_gets_the_sum(algo, n, numel):
    vectors = _vectors(n, numel)
    ref = reference_allreduce(vectors)
    result, stats = ALGORITHMS[algo]([list(v) for v in vectors])
    assert stats.world_size == n
    assert len(result) == n
    for r in range(n):
        assert approx_equal(result[r], ref), f"{algo} rank {r} wrong (n={n}, numel={numel})"
