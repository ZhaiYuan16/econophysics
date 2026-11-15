import numpy as np

from local_market.percolation import identify_lattice_clusters
from local_market.statistics import (
    cluster_histogram,
    finite_cluster_sizes,
    susceptibility_from_sizes,
)


def test_largest_cluster_exclusion_and_susceptibility() -> None:
    occupied = np.array(
        [
            [1, 1, 1, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    configuration = identify_lattice_clusters(occupied)
    assert sorted(configuration.sizes.tolist()) == [1, 2, 3]
    finite = finite_cluster_sizes(configuration, "largest")
    assert sorted(finite.tolist()) == [1, 2]
    assert np.isclose(susceptibility_from_sizes(finite), 5 / 3)


def test_spanning_exclusion_removes_all_spanning_clusters() -> None:
    occupied = np.zeros((5, 5), dtype=bool)
    occupied[:, 0] = True
    occupied[:, 4] = True
    occupied[2, 2] = True
    configuration = identify_lattice_clusters(occupied)
    assert len(configuration.spanning_labels) == 2
    np.testing.assert_array_equal(
        finite_cluster_sizes(configuration, "spanning"), np.array([1])
    )


def test_cluster_histogram_counts_clusters_not_agents() -> None:
    histogram = cluster_histogram(np.array([1, 1, 2, 5]))
    np.testing.assert_array_equal(histogram, [0, 2, 1, 0, 0, 1])
