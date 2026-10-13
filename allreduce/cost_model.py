"""An alpha-beta cost model, driven by the *simulated* step/byte accounting.

    T = steps · alpha  +  peak_rank_bytes · beta        (beta = 1 / link_bandwidth)

Using the simulator's own `steps` (latency hops) and `peak_rank_bytes` (the
busiest link) keeps the model honest: it predicts the latency- vs
bandwidth-bound regimes directly from how each algorithm moves data.
"""
from __future__ import annotations

from dataclasses import dataclass

from allreduce.algorithms import ALGORITHMS
from allreduce.simulator import CommStats


@dataclass
class CostParams:
    alpha_us: float = 1.0        # per-step link latency (microseconds)
    link_gbps: float = 100.0     # per-link bandwidth (GB/s)


def _beta_us_per_byte(link_gbps: float) -> float:
    return 1e6 / (link_gbps * 1e9)


def predict_us(stats: CommStats, params: CostParams) -> float:
    return stats.steps * params.alpha_us + stats.peak_rank_bytes * _beta_us_per_byte(params.link_gbps)


def stats_for(algo: str, world_size: int, numel: int, elem_bytes: int = 4) -> CommStats:
    vecs = [[1.0] * numel for _ in range(world_size)]
    _, st = ALGORITHMS[algo](vecs, elem_bytes)
    return st


def regime_table(world_size: int, numels: list[int], params: CostParams,
                 elem_bytes: int = 4) -> list[dict[str, object]]:
    rows = []
    for numel in numels:
        row: dict[str, object] = {"numel": numel, "bytes": numel * elem_bytes}
        for algo in ALGORITHMS:
            row[algo + "_us"] = round(predict_us(stats_for(algo, world_size, numel, elem_bytes), params), 3)
        rows.append(row)
    return rows
