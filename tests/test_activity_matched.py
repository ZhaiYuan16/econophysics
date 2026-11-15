import numpy as np
import networkx as nx
from local_market.activity_matched import matched_activity, metric_counts, rng_for
from local_market.cb_model import sample_demands_and_active_counts
from local_market.hierarchical_statistics import block_histogram, resampled_counts, select_network_configuration_blocks
from local_market.networks import identify_graph_clusters


def test_k_is_nonempty_component_count_and_activity_is_capped():
    graph = nx.Graph([(0, 1), (2, 3)])
    graph.add_node(4)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(5), format='csr', dtype=np.uint8)
    clusters = identify_graph_clusters(adjacency, np.array([1,1,0,1,1], dtype=bool))
    assert clusters.cluster_count == 3
    assert clusters.sizes.sum() == 4
    assert matched_activity(3, 6) == .5
    assert matched_activity(5, 3) == 1


def test_pilot_and_action_streams_are_independent_and_active_mean_matches():
    pilot = rng_for(71, 120, 0, 0, 0).random(100)
    occupancy = rng_for(71, 130, 0, 0, 0).random(100)
    assert not np.array_equal(pilot, occupancy)
    demands, active = sample_demands_and_active_counts(np.ones(30, dtype=int), 10000, .1, rng_for(71, 140, 0, 0, 0, 0, 0))
    assert abs(active.mean() - 3) < .1
    assert len(demands) == len(active)


def test_hierarchical_resample_keeps_action_block():
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    block_ids = np.repeat(np.arange(4), 2)
    support, matrix = block_histogram(values, block_ids)
    selected = select_network_configuration_blocks(np.random.default_rng(3), 2, 2)
    assert len(selected) == 4
    counts = resampled_counts(matrix, selected)
    assert counts.sum() == 8
    for block in range(4):
        assert counts[2*block] == counts[2*block+1]


def test_common_raw_threshold_and_conditional_crash_probability():
    support = np.array([-6, 0, 1, 2])
    counts = np.array([1, 8, 1, 1])
    result = metric_counts(support, counts, 5, 1)
    assert result['p_r_lt_minus_r0'] == 1/11
    assert result['zero_return_fraction'] == 8/11


def test_deterministic_lattice_bootstrap_resamples_configurations():
    from local_market.activity_matched import select_blocks
    draw = select_blocks(np.random.default_rng(5), 'lattice', 8, 100)
    assert len(draw) == 800
    assert draw.min() >= 0 and draw.max() < 800
