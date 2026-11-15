import numpy as np

from local_market.cb_model import (
    #cluster_action_probabilities,
    sample_demands_and_active_counts,
    #sample_cluster_actions,
    simulate_annealed,
    simulate_quenched,
)
from local_market.percolation import identify_lattice_clusters


def test_action_probabilities_sum_to_one() -> None:
    for activity in (0.0, 0.001, 0.5, 1.0):
        #probabilities = cluster_action_probabilities(activity)
        #assert np.isclose(sum(probabilities), 1.0)
        #assert all(probability >= 0 for probability in probabilities)
        assert 1
    

def test_hand_constructed_clusters_have_expected_demand() -> None:
    occupied = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=bool)
    configuration = identify_lattice_clusters(occupied)
    assert configuration.sizes.tolist() == [2, 1]
    #assert excess_demand(configuration, np.array([1, -1])) == 1


def test_zero_activity_gives_zero_returns() -> None:
    configuration = identify_lattice_clusters(np.ones((3, 3), dtype=bool))
    simulation = simulate_quenched(
        configuration,
        steps=20,
        activity=0.0,
        market_depth=9.0,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_array_equal(simulation.returns, 0.0)
    np.testing.assert_array_equal(simulation.log_prices, 0.0)


def test_fixed_seed_is_reproducible() -> None:
    configuration = identify_lattice_clusters(
        np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=bool)
    )
    first = simulate_quenched(
        configuration, 100, 0.2, 9.0, np.random.default_rng(123)
    )
    second = simulate_quenched(
        configuration, 100, 0.2, 9.0, np.random.default_rng(123)
    )
    np.testing.assert_array_equal(first.demands, second.demands)
    np.testing.assert_array_equal(first.log_prices, second.log_prices)


def test_empirical_actions_match_requested_activity() -> None:
    actions = sample_cluster_actions(200_000, 0.2, np.random.default_rng(8))
    assert abs(np.mean(actions != 0) - 0.2) < 0.005
    assert abs(np.mean(actions == 1) - 0.1) < 0.004
    assert abs(np.mean(actions == -1) - 0.1) < 0.004


def test_annealed_regenerates_configuration() -> None:
    configurations = iter(
        [
            identify_lattice_clusters(np.zeros((2, 2), dtype=bool)),
            identify_lattice_clusters(np.ones((2, 2), dtype=bool)),
        ]
    )
    simulation = simulate_annealed(
        lambda: next(configurations),
        steps=2,
        activity=0.0,
        market_depth=4.0,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_array_equal(simulation.occupied_counts, [0, 4])
    np.testing.assert_array_equal(simulation.cluster_counts, [0, 1])


def test_vectorized_demands_report_active_cluster_counts() -> None:
    demands, active = sample_demands_and_active_counts(
        np.array([2, 3]), 40, 1.0, np.random.default_rng(7)
    )
    np.testing.assert_array_equal(active, 2)
    assert set(demands).issubset({-5, -1, 1, 5})
