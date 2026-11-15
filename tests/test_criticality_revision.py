import numpy as np
import pandas as pd

from local_market.criticality_revision import (
    crossing, finite_powerlaw, pooled_histogram, ratio_grid, scan, summarize,
)
from local_market.percolation import identify_lattice_clusters
from local_market.statistics import finite_cluster_sizes, susceptibility_from_sizes


def test_order_parameter_uses_total_sites_for_every_configuration() -> None:
    empty = identify_lattice_clusters(np.zeros((4, 4), dtype=bool))
    full = identify_lattice_clusters(np.ones((4, 4), dtype=bool))
    partial = np.zeros((4, 4), dtype=bool)
    partial[1, 1:3] = True
    cluster = identify_lattice_clusters(partial)
    assert empty.p_infinity == 0
    assert full.p_infinity == 1
    assert cluster.p_infinity == 2 / 16
    assert cluster.largest_occupied_fraction == 1
    assert not cluster.spans


def test_directional_spanning_and_four_neighbours() -> None:
    horizontal = np.zeros((5, 5), dtype=bool)
    horizontal[2, :] = True
    vertical = horizontal.T.copy()
    for occupied, expected in ((horizontal, (True, False)),
                               (vertical, (False, True))):
        cluster = identify_lattice_clusters(occupied)
        assert (cluster.spans_horizontal, cluster.spans_vertical) == expected
        assert len(cluster.spanning_labels) == 1
    both = horizontal | vertical
    cluster = identify_lattice_clusters(both)
    assert cluster.spans_horizontal and cluster.spans_vertical
    assert cluster.spanning_labels == (1,)
    three_edges = horizontal.copy()
    three_edges[1, 2] = True
    three_edges[0, 2] = True
    cluster = identify_lattice_clusters(three_edges)
    assert cluster.spans_horizontal and not cluster.spans_vertical
    diagonal = np.eye(4, dtype=bool)
    assert identify_lattice_clusters(diagonal).cluster_count == 4


def test_two_exclusion_rules_and_empty_case() -> None:
    occupied = np.zeros((5, 5), dtype=bool)
    occupied[0, :] = True
    occupied[2, 0:2] = True
    occupied[4, 4] = True
    cluster = identify_lattice_clusters(occupied)
    a = finite_cluster_sizes(cluster, "largest")
    b = finite_cluster_sizes(cluster, "spanning")
    assert sorted(a.tolist()) == sorted(b.tolist()) == [1, 2]
    assert int(a.sum()) == 3
    assert int(np.dot(a, a)) == 5
    assert susceptibility_from_sizes(a) == 5 / 3
    empty = identify_lattice_clusters(np.zeros((5, 5), dtype=bool))
    assert susceptibility_from_sizes(finite_cluster_sizes(empty, "spanning")) == 0
    no_span = identify_lattice_clusters(np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=bool))
    assert len(finite_cluster_sizes(no_span, "largest")) == 1
    assert len(finite_cluster_sizes(no_span, "spanning")) == 2
    two_spans = np.zeros((5, 5), dtype=bool)
    two_spans[0, :] = True
    two_spans[4, :] = True
    cluster = identify_lattice_clusters(two_spans)
    assert len(cluster.spanning_labels) == 2
    assert finite_cluster_sizes(cluster, "largest").tolist() == [5]
    assert finite_cluster_sizes(cluster, "spanning").size == 0
    different = np.zeros((8, 8), dtype=bool)
    different[0, :] = True
    different[2:6, 2:6] = True
    cluster = identify_lattice_clusters(different)
    assert finite_cluster_sizes(cluster, "largest").tolist() == [8]
    assert finite_cluster_sizes(cluster, "spanning").tolist() == [16]


def test_scan_configuration_level_and_histogram_normalization(tmp_path) -> None:
    result = scan([4], {4: (0.5, 1.0)}, {4: 5}, 0.592746, 42)
    samples = result.samples
    assert len(samples) == 10
    assert (samples.span_both <= samples.span_x).all()
    assert (samples.span_both <= samples.span_y).all()
    assert (samples.span_any >= samples.span_x).all()
    assert (samples.span_any >= samples.span_y).all()
    assert (samples.p_infinity == samples.largest_size / 16).all()
    summary = summarize(samples, 100, 1)
    assert len(summary) == 2
    assert (summary.chi_largest_excluded >= 0).all()
    pooled = pooled_histogram(result.histograms, samples)
    assert np.allclose(pooled.n_s, pooled.C_s / (pooled.M * pooled.length**2))
    streamed = scan([4], {4: (0.5, 1.0)}, {4: 5}, 0.592746, 42,
                    tmp_path / "counts.csv")
    pd.testing.assert_frame_equal(samples, streamed.samples)
    pd.testing.assert_frame_equal(
        pooled.sort_values(["p_ratio", "exclusion", "cluster_size"]).reset_index(drop=True),
        pooled_histogram(streamed.histograms, streamed.samples, streamed.pooled_counts)
        .sort_values(["p_ratio", "exclusion", "cluster_size"]).reset_index(drop=True))
    assert len(pd.read_csv(tmp_path / "counts.csv")) == len(result.histograms)


def test_finite_power_law_and_grid_and_crossing() -> None:
    values = np.repeat(np.arange(1, 11), np.maximum(2, (100 / np.arange(1, 11)**2).astype(int)))
    alpha, ks, count = finite_powerlaw(values, 1, 10)
    assert np.isfinite(alpha) and 0 <= ks <= 1 and count == len(values)
    assert finite_powerlaw(values, 10, 10)[0] != finite_powerlaw(values, 10, 10)[0]
    grid = ratio_grid([.8, 1, 1.1], 1.06, .03, [.97, 1.03], .002)
    assert .8 in grid and 1 in grid and 1.09 in grid
    assert crossing(np.array([.5, .6]), np.array([.2, .8]),
                    np.array([.8, .2]), .592746) == .55
