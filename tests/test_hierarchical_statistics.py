import numpy as np

from local_market.hierarchical_statistics import (
    block_histogram, resampled_counts, select_network_configuration_blocks,
    weighted_pareto_fit, weighted_statistics,
)
from local_market.tail_fitting import select_pareto_tail


def test_weighted_tail_fit_matches_expanded_observations() -> None:
    values = np.abs(np.random.default_rng(77).integers(-20, 21, 5000).astype(float))
    support, inverse = np.unique(values, return_inverse=True)
    counts = np.bincount(inverse)
    quantiles = (.7, .8, .9)
    weighted = weighted_pareto_fit(support, counts, quantiles, 50)
    expanded, _ = select_pareto_tail(values, quantiles, 50)
    np.testing.assert_allclose([weighted["xmin"], weighted["alpha_pdf"], weighted["ks_distance"]],
                               [expanded.xmin, expanded.alpha_pdf, expanded.ks_distance])


def test_resampling_keeps_whole_configuration_blocks() -> None:
    support, matrix = block_histogram(np.array([1., 1., 2., 2., 9., 9.]),
                                      np.array([0, 0, 1, 1, 2, 2]))
    counts = resampled_counts(matrix, np.array([2, 2, 0]))
    np.testing.assert_array_equal(support, [1., 2., 9.])
    np.testing.assert_array_equal(counts, [2, 0, 4])


def test_nested_draw_uses_four_networks_and_sixty_configurations_each() -> None:
    selected = select_network_configuration_blocks(np.random.default_rng(81), 4, 60)
    assert len(selected) == 240
    for network_draw in selected.reshape(4, 60):
        assert np.unique(network_draw // 60).size == 1
        assert np.all((network_draw % 60 >= 0) & (network_draw % 60 < 60))


def test_zero_raw_returns_are_excluded_after_standardization() -> None:
    support = np.array([-10., -5., 0., 2., 5., 10.])
    counts = np.array([10, 10, 100, 10, 10, 10])
    expanded = np.repeat(support, counts)
    z = (expanded - expanded.mean()) / expanded.std(ddof=1)
    expected, _ = select_pareto_tail(np.abs(z[expanded != 0]), (.5, .7), 5)
    actual = weighted_statistics(support, counts, (.5, .7), 5,
                                 exclude_zero_returns=True)
    np.testing.assert_allclose([actual['xmin'], actual['alpha_pdf']],
                               [expected.xmin, expected.alpha_pdf])
