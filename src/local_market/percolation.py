"""Cluster identification for square-lattice site percolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

Boundary = Literal["open", "periodic"]


@dataclass(frozen=True)
class ClusterConfiguration:
    """A labelled occupied lattice and its cluster-level observables.

    Positive labels are consecutive cluster identifiers; zero denotes an
    unoccupied site. ``spans`` means open-boundary spanning or periodic
    wrapping, according to ``boundary``.
    """

    labels: np.ndarray
    sizes: np.ndarray
    occupied_count: int
    largest_label: int
    largest_size: int
    spans: bool
    spanning_labels: tuple[int, ...]
    boundary: Boundary
    horizontal_spanning_labels: tuple[int, ...] = ()
    vertical_spanning_labels: tuple[int, ...] = ()

    @property
    def cluster_count(self) -> int:
        return int(self.sizes.size)

    @property
    def largest_occupied_fraction(self) -> float:
        if self.occupied_count == 0:
            return 0.0
        return self.largest_size / self.occupied_count

    @property
    def p_infinity(self) -> float:
        return self.largest_size / self.labels.size

    @property
    def spans_horizontal(self) -> bool:
        if self.boundary != "open":
            raise ValueError("horizontal open-boundary spanning is undefined for periodic lattices")
        return bool(self.horizontal_spanning_labels)

    @property
    def spans_vertical(self) -> bool:
        if self.boundary != "open":
            raise ValueError("vertical open-boundary spanning is undefined for periodic lattices")
        return bool(self.vertical_spanning_labels)


def _empty_configuration(shape: tuple[int, int], boundary: Boundary) -> ClusterConfiguration:
    return ClusterConfiguration(
        labels=np.zeros(shape, dtype=np.int32),
        sizes=np.empty(0, dtype=np.int64),
        occupied_count=0,
        largest_label=0,
        largest_size=0,
        spans=False,
        spanning_labels=(),
        boundary=boundary,
    )


def _open_clusters(occupied: np.ndarray) -> ClusterConfiguration:
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(occupied, structure=structure)
    if count == 0:
        return _empty_configuration(occupied.shape, "open")

    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(np.int64)
    top = set(labels[0, labels[0] > 0].tolist())
    bottom = set(labels[-1, labels[-1] > 0].tolist())
    left = set(labels[labels[:, 0] > 0, 0].tolist())
    right = set(labels[labels[:, -1] > 0, -1].tolist())
    horizontal = tuple(sorted(left & right))
    vertical = tuple(sorted(top & bottom))
    spanning = tuple(sorted(set(horizontal) | set(vertical)))
    largest_label = int(np.argmax(sizes)) + 1
    return ClusterConfiguration(
        labels=labels.astype(np.int32, copy=False),
        sizes=sizes,
        occupied_count=int(sizes.sum()),
        largest_label=largest_label,
        largest_size=int(sizes[largest_label - 1]),
        spans=bool(spanning),
        spanning_labels=spanning,
        boundary="open",
        horizontal_spanning_labels=horizontal,
        vertical_spanning_labels=vertical,
    )


def _periodic_clusters(occupied: np.ndarray) -> ClusterConfiguration:
    """Label periodic clusters and detect non-contractible wrapping cycles."""
    rows, cols = occupied.shape
    flat_occupied = occupied.ravel()
    occupied_nodes = np.flatnonzero(flat_occupied)
    if occupied_nodes.size == 0:
        return _empty_configuration(occupied.shape, "periodic")

    total = rows * cols
    parent = np.arange(total, dtype=np.int64)
    rank = np.zeros(total, dtype=np.int8)
    # displacement from a node to its parent in the unwrapped covering lattice
    dr = np.zeros(total, dtype=np.int64)
    dc = np.zeros(total, dtype=np.int64)
    wraps = np.zeros(total, dtype=bool)

    def find(node: int) -> tuple[int, int, int]:
        if parent[node] == node:
            return node, 0, 0
        root, root_dr, root_dc = find(int(parent[node]))
        dr[node] += root_dr
        dc[node] += root_dc
        parent[node] = root
        return root, int(dr[node]), int(dc[node])

    def union(u: int, v: int, edge_dr: int, edge_dc: int) -> None:
        root_u, u_dr, u_dc = find(u)
        root_v, v_dr, v_dc = find(v)
        if root_u == root_v:
            if (v_dr - u_dr, v_dc - u_dc) != (edge_dr, edge_dc):
                wraps[root_u] = True
            return

        # Position(root_v) - position(root_u), implied by the new edge.
        rel_dr = edge_dr + u_dr - v_dr
        rel_dc = edge_dc + u_dc - v_dc
        combined_wrap = bool(wraps[root_u] or wraps[root_v])
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
            dr[root_u] = -rel_dr
            dc[root_u] = -rel_dc
            wraps[root_v] = combined_wrap
        else:
            parent[root_v] = root_u
            dr[root_v] = rel_dr
            dc[root_v] = rel_dc
            wraps[root_u] = combined_wrap
            if rank[root_u] == rank[root_v]:
                rank[root_u] += 1

    for row in range(rows):
        for col in range(cols):
            u = row * cols + col
            if not flat_occupied[u]:
                continue
            right = row * cols + ((col + 1) % cols)
            down = ((row + 1) % rows) * cols + col
            if flat_occupied[right]:
                union(u, right, 0, 1)
            if flat_occupied[down]:
                union(u, down, 1, 0)

    roots = np.array([find(int(node))[0] for node in occupied_nodes], dtype=np.int64)
    unique_roots, inverse, sizes = np.unique(roots, return_inverse=True, return_counts=True)
    labels_flat = np.zeros(total, dtype=np.int32)
    labels_flat[occupied_nodes] = inverse.astype(np.int32) + 1
    spanning_labels = tuple(
        index + 1
        for index, root in enumerate(unique_roots)
        if wraps[find(int(root))[0]]
    )
    sizes = sizes.astype(np.int64)
    largest_label = int(np.argmax(sizes)) + 1
    return ClusterConfiguration(
        labels=labels_flat.reshape(occupied.shape),
        sizes=sizes,
        occupied_count=int(occupied_nodes.size),
        largest_label=largest_label,
        largest_size=int(sizes[largest_label - 1]),
        spans=bool(spanning_labels),
        spanning_labels=spanning_labels,
        boundary="periodic",
    )


def identify_lattice_clusters(
    occupied: np.ndarray, boundary: Boundary = "open"
) -> ClusterConfiguration:
    """Identify four-neighbour clusters in a two-dimensional Boolean array."""
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 2 or min(occupied.shape) == 0:
        raise ValueError("occupied must be a non-empty two-dimensional array")
    if boundary == "open":
        return _open_clusters(occupied)
    if boundary == "periodic":
        return _periodic_clusters(occupied)
    raise ValueError("boundary must be 'open' or 'periodic'")
