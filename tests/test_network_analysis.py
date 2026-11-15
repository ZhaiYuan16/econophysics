import networkx as nx
import numpy as np

from local_market.network_analysis import _threshold_criteria
from local_market.networks import (
    generate_matched_graph,
    identify_graph_clusters,
)


def test_graph_cluster_identification_on_hand_graph() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (3, 4)])
    graph.add_node(5)
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=range(6), format="csr", dtype=np.uint8
    )
    occupied = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    clusters = identify_graph_clusters(adjacency, occupied)
    assert sorted(clusters.sizes.tolist()) == [1, 2, 2]
    assert clusters.occupied_count == 5
    assert clusters.largest_size == 2


def test_matched_graph_generators_have_expected_size_and_degree() -> None:
    for topology in ("lattice", "er", "ws", "ba"):
        realization = generate_matched_graph(topology, 400, 4, seed=13)
        graph = realization.graph
        assert graph.number_of_nodes() == 400
        mean_degree = sum(dict(graph.degree()).values()) / 400
        assert abs(mean_degree - 4.0) < 0.35
        assert realization.adjacency.shape == (400, 400)


def test_graph_clusters_are_reproducible() -> None:
    graph = generate_matched_graph("ws", 100, 4, seed=8)
    occupied = np.random.default_rng(10).random(100) < 0.4
    first = identify_graph_clusters(graph.adjacency, occupied)
    second = identify_graph_clusters(graph.adjacency, occupied)
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.sizes, second.sizes)


def test_threshold_criteria_recover_synthetic_peaks() -> None:
    p = np.linspace(0.08, 0.36, 29)
    p_infinity = 1.0 / (1.0 + np.exp(-35.0 * (p - 0.22)))
    susceptibility = np.exp(-((p - 0.19) / 0.025) ** 2)
    estimates = _threshold_criteria(p, p_infinity, susceptibility, 0.5)
    assert np.isclose(estimates["p_chi_peak"], 0.19)
    assert abs(estimates["p_smoothed_slope_peak"] - 0.22) <= 0.01
    assert abs(estimates["p_order_parameter_level"] - 0.22) <= 0.01
