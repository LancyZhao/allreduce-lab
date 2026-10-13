"""CLI: compare all-reduce algorithms (verifies correctness + prints the model).

    python -m allreduce.cli --world-size 8 --numel 1048576
    python -m allreduce.cli --world-size 8 --regime        # latency vs bandwidth sweep
"""
from __future__ import annotations

import argparse

from allreduce.algorithms import ALGORITHMS
from allreduce.cost_model import CostParams, predict_us, regime_table
from allreduce.tensor import approx_equal, reference_allreduce


def _demo_vectors(n: int, numel: int) -> list[list[float]]:
    return [[float(((r + 1) * (i + 1)) % 7) for i in range(numel)] for r in range(n)]


def compare(n: int, numel: int, params: CostParams) -> None:
    vectors = _demo_vectors(n, numel)
    ref = reference_allreduce(vectors)
    print(f"world_size={n}  numel={numel}  elem=4B  "
          f"(alpha={params.alpha_us}us/step, link={params.link_gbps}GB/s)\n")
    print(f"{'algo':<7}{'steps':>7}{'total_bytes':>14}{'peak_rank_bytes':>18}{'T(us)':>12}  correct")
    for algo, fn in ALGORITHMS.items():
        result, st = fn([list(v) for v in vectors])
        ok = all(approx_equal(result[r], ref) for r in range(n))
        print(f"{algo:<7}{st.steps:>7}{st.total_bytes:>14}{st.peak_rank_bytes:>18}"
              f"{predict_us(st, params):>12.2f}  {ok}")


def regime(n: int, params: CostParams) -> None:
    numels = [1, 1 << 8, 1 << 12, 1 << 16, 1 << 20, 1 << 24]
    rows = regime_table(n, numels, params)
    print(f"world_size={n}  predicted T (us) by message size — lowest wins\n")
    print(f"{'bytes':>12}{'ring_us':>12}{'tree_us':>12}{'naive_us':>12}  best")
    for row in rows:
        best = min(("ring", "tree", "naive"), key=lambda a: row[a + "_us"])
        print(f"{row['bytes']:>12}{row['ring_us']:>12}{row['tree_us']:>12}{row['naive_us']:>12}  {best}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="allreduce", description="all-reduce algorithm lab")
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--numel", type=int, default=1 << 20)
    p.add_argument("--alpha-us", type=float, default=1.0)
    p.add_argument("--link-gbps", type=float, default=100.0)
    p.add_argument("--regime", action="store_true", help="sweep message sizes")
    args = p.parse_args(argv)
    params = CostParams(args.alpha_us, args.link_gbps)
    if args.regime:
        regime(args.world_size, params)
    else:
        compare(args.world_size, args.numel, params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
