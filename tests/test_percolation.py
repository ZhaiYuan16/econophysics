import numpy as np
import pytest

from local_market.networks import sample_lattice_occupancy
from local_market.percolation import identify_lattice_clusters


def test_cluster_sizes_partition_all_occupied_sites() -> None:
    occupied = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    )
    result = identify_lattice_clusters(occupied)
    assert sorted(result.sizes.tolist()) == [2, 3, 3]
    assert result.sizes.sum() == occupied.sum()
    assert set(np.unique(result.labels)) == {0, 1, 2, 3}


def test_open_boundary_spanning_is_detected() -> None:
    occupied = np.zeros((4, 4), dtype=bool)
    occupied[:, 2] = True
    result = identify_lattice_clusters(occupied, "open")
    assert result.spans
    assert result.spanning_labels == (1,)


def test_periodic_edges_join_without_false_wrapping() -> None:
    occupied = np.zeros((4, 4), dtype=bool)
    occupied[1, 0] = True
    occupied[1, -1] = True
    result = identify_lattice_clusters(occupied, "periodic")
    assert result.sizes.tolist() == [2]
    assert not result.spans


def test_periodic_noncontractible_cycle_wraps() -> None:
    occupied = np.zeros((4, 4), dtype=bool)
    occupied[2, :] = True
    result = identify_lattice_clusters(occupied, "periodic")
    assert result.sizes.tolist() == [4]
    assert result.spans


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_empty_and_full_lattices(boundary: str) -> None:
    empty = identify_lattice_clusters(np.zeros((5, 5), dtype=bool), boundary)
    full = identify_lattice_clusters(np.ones((5, 5), dtype=bool), boundary)
    assert empty.occupied_count == empty.cluster_count == empty.largest_size == 0
    assert empty.largest_occupied_fraction == 0.0
    assert full.occupied_count == 25
    assert full.cluster_count == 1
    assert full.largest_size == 25


def test_occupancy_sampling_validates_inputs() -> None:
    rng = np.random.default_rng(4)
    assert not sample_lattice_occupancy(3, 0.0, rng).any()
    assert sample_lattice_occupancy(3, 1.0, rng).all()
    with pytest.raises(ValueError):
        sample_lattice_occupancy(3, 1.1, rng)


def test_low_p_produces_smaller_clusters_than_near_criticality() -> None:
    """A fixed-seed ensemble gives the expected strong directional signal."""
    rng = np.random.default_rng(2026)
    means = {}
    for p in (0.10, 0.592746):
        largest_sizes = []
        for _ in range(80):
            occupied = sample_lattice_occupancy(32, p, rng)
            largest_sizes.append(identify_lattice_clusters(occupied).largest_size)
        means[p] = np.mean(largest_sizes)
    assert means[0.592746] > 10 * means[0.10]
