# allreduce-lab

[![CI](https://github.com/LancyZhao/allreduce-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/LancyZhao/allreduce-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> A **CPU, dependency-free lab** for collective-communication algorithms —
> **ring / tree / naive all-reduce**. It *simulates the data movement*, verifies
> the textbook **communication-volume formulas**, models the **latency-vs-bandwidth**
> tradeoff, and parses real **`nccl-tests`** output. The point is to make the
> "why NCCL picks ring for big messages and tree for small ones" falsifiable,
> not hand-waved.

No GPU needed to run the simulator and tests — the real-hardware numbers come
from `nccl-tests`, which this repo helps you run and parse.

## Quickstart (no GPU)

```bash
pip install -e ".[dev]"              # or: make dev
pytest -q                            # 134 tests, all CPU
python -m allreduce.cli --world-size 8 --numel 1048576
python -m allreduce.cli --world-size 16 --regime
```

Comparing algorithms on 8 ranks × 1M floats (every rank is verified to hold the
elementwise sum):

```
algo     steps   total_bytes   peak_rank_bytes       T(us)  correct
ring        14      58720256           7340032       87.40  True
tree         6      58720256          12582912      131.83  True
naive        2      58720256          29360128      295.60  True
```

All three move the **same total bytes** `2(n-1)·D`; what differs is **how it is
distributed**. Ring spreads it evenly (small per-link peak → bandwidth-optimal),
naive piles it on the root (`(n-1)·D` hotspot), tree trades a bit of peak for
`O(log n)` latency instead of ring's `O(n)`.

## The algorithms (what the simulator actually does)

- **Ring** = *reduce-scatter* (n−1 steps) + *all-gather* (n−1 steps). Each rank
  only ever talks to its ring successor, sending `2(n-1)/n·D` bytes — which
  approaches `2D` and is **independent of n**, i.e. bandwidth-optimal. Latency
  grows as `2(n-1)` steps.
- **Tree** = binomial reduce to root 0 + binomial broadcast back,
  `2·⌈log₂n⌉` steps. Lower latency than ring; a mild per-rank peak at the root.
- **Naive** = everyone sends to the root, root sums and broadcasts. Only 2 logical
  steps, but the root is a `(n-1)·D` **bandwidth hotspot** — the baseline that
  shows *why* ring/tree exist.

## Latency vs bandwidth (the crossover)

`T = steps · α + peak_rank_bytes · β` (β = 1 / link-bandwidth), driven straight
from the simulated accounting. Sweep message size at 16 ranks:

```
       bytes     ring_us     tree_us    naive_us  best
           4        30.0         8.0       2.001  naive
       16384      30.307       8.655       4.458  naive
      262144      34.915      18.486      41.322  tree
     4194304     108.643     175.772     631.146  ring
    67108864    1288.291    2692.355    10068.33  ring
```

Small messages are **latency-bound** → fewer steps win (this is why NCCL uses a
**tree** for small all-reduce); large messages are **bandwidth-bound** → the
lowest per-link volume wins (**ring**). Same reason real NCCL auto-switches
between Tree and Ring around a size threshold.

## Real hardware: nccl-tests

```bash
./nccl/run_nccl_tests.sh 2 > nccl/my_output.txt   # on a GPU box with nccl-tests built
python nccl/parse_nccl.py nccl/my_output.txt       # -> tidy size/busbw table + peak
```

`busbw` rising with message size (then flattening at the link's roofline) is the
empirical version of the regime table above.

## 对应岗位能力（JD → 实现）

| 岗位要求 | 落点 |
| --- | --- |
| NCCL / 集合通信工作原理 | ring=reduce-scatter+all-gather、tree=binomial、naive=root 热点，公式可验证 |
| 通信量 / 带宽分析（RDMA/IB/RoCE 场景） | `peak_rank_bytes` 建模忙链路，α-β 模型给延迟/带宽拐点 |
| 高性能网络性能评测 | `nccl/` 跑 nccl-tests + 解析 busbw roofline |
| C++/系统工程能力 | 确定性模拟 + 134 测试覆盖正确性/通信量/成本模型，CI 全绿 |

## Honest scope

A **didactic simulator**, not a runtime: it models logical data movement, step
count and per-link volume on a single CPU process. It does **not** model network
contention, compute/communication overlap, protocol (LL/LL128) or real wire
latency — those come from `nccl-tests` on actual hardware. "naive" is a teaching
baseline (its 2-step count ignores root incast serialization), included to make
the root hotspot visible.

## Project layout

```
allreduce/
  tensor.py       vectors + reference reduce + chunking
  simulator.py    CommStats: steps + per-rank bytes accounting
  algorithms.py   ring / tree / naive all-reduce (return result + stats)
  cost_model.py   alpha-beta model + regime sweep
  cli.py          compare + --regime
nccl/             run_nccl_tests.sh, parse_nccl.py, sample_output.txt
tests/            correctness / volume formulas / cost model / parser (134 cases)
```

## License

MIT © LancyZhao

