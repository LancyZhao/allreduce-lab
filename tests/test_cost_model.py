from allreduce.cost_model import CostParams, regime_table, stats_for


def test_large_messages_favor_ring():
    params = CostParams(alpha_us=1.0, link_gbps=100.0)
    rows = regime_table(16, [1 << 24], params)   # 16 MB elements -> big message
    row = rows[0]
    assert row["ring_us"] < row["tree_us"] < row["naive_us"]


def test_tiny_messages_are_latency_bound():
    # at 1 element, cost is dominated by step count: ring (2(n-1)) is worst
    params = CostParams(alpha_us=1.0, link_gbps=100.0)
    rows = regime_table(16, [1], params)
    row = rows[0]
    assert row["ring_us"] > row["tree_us"]


def test_ring_peak_bytes_independent_of_worldsize():
    # ring peak approaches 2*D regardless of n -> bandwidth optimal
    a = stats_for("ring", 4, 1 << 16).peak_rank_bytes
    b = stats_for("ring", 64, 1 << 16).peak_rank_bytes
    D = (1 << 16) * 4
    assert a <= 2 * D and b <= 2 * D
