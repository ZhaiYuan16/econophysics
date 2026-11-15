"""Network and occupancy generators used by the market models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

Topology = Literal["lattice", "er", "ws", "ba"]


@dataclass(frozen=True)
class GraphRealization:
    topology: Topology
    graph: nx.Graph
    adjacency: sparse.csr_matrix
    seed: int


@dataclass(frozen=True)
class GraphClusters:
    labels: np.ndarray
    sizes: np.ndarray
    occupied_count: int
    largest_label: int
    largest_size: int

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


def sample_lattice_occupancy(
    length: int, p: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample occupied sites on an ``length x length`` square lattice."""
    if length <= 0:
        raise ValueError("length must be positive")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    return rng.random((length, length)) < p


def generate_matched_graph(
    topology: Topology,
    node_count: int,
    mean_degree: int,
    seed: int,
    ws_rewiring_probability: float = 0.1,
) -> GraphRealization:
    """Generate a graph with approximately matched size and mean degree."""
    if node_count <= 0 or mean_degree <= 0:
        raise ValueError("node_count and mean_degree must be positive")
    if topology == "lattice":
        length = int(round(np.sqrt(node_count)))
        if length * length != node_count:
            raise ValueError("lattice node_count must be a perfect square")
        graph = nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(length, length, periodic=True), ordering="sorted"
        )
    elif topology == "er":
        graph = nx.fast_gnp_random_graph(
            node_count, mean_degree / (node_count - 1), seed=seed
        )
    elif topology == "ws":
        if mean_degree % 2:
            raise ValueError("Watts-Strogatz mean_degree must be even")
        graph = nx.watts_strogatz_graph(
            node_count, mean_degree, ws_rewiring_probability, seed=seed
        )
    elif topology == "ba":
        if mean_degree % 2:
            raise ValueError("BA target mean_degree must be even")
        graph = nx.barabasi_albert_graph(node_count, mean_degree // 2, seed=seed)
    else:
        raise ValueError("unknown topology")
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=range(node_count), format="csr", dtype=np.uint8
    )
    return GraphRealization(topology, graph, sparse.csr_matrix(adjacency), seed)


def identify_graph_clusters(
    adjacency: sparse.csr_matrix, occupied: np.ndarray
) -> GraphClusters:
    """Find components in the occupied-node induced subgraph."""
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 1 or occupied.size != adjacency.shape[0]:
        raise ValueError("occupied must match the adjacency node dimension")
    occupied_nodes = np.flatnonzero(occupied)
    labels = np.zeros(occupied.size, dtype=np.int32)
    if occupied_nodes.size == 0:
        return GraphClusters(labels, np.empty(0, dtype=np.int64), 0, 0, 0)
    induced = adjacency[occupied_nodes][:, occupied_nodes]
    count, component_labels = csgraph.connected_components(
        induced, directed=False, return_labels=True
    )
    labels[occupied_nodes] = component_labels.astype(np.int32) + 1
    sizes = np.bincount(component_labels, minlength=count).astype(np.int64)
    largest_label = int(np.argmax(sizes)) + 1
    return GraphClusters(
        labels=labels,
        sizes=sizes,
        occupied_count=int(occupied_nodes.size),
        largest_label=largest_label,
        largest_size=int(sizes[largest_label - 1]),
    )


def graph_structure_metrics(
    realization: GraphRealization,
    path_sources: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Compute graph metrics, approximating path length from sampled sources."""
    graph = realization.graph
    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=float)
    components = list(nx.connected_components(graph))
    largest_nodes = max(components, key=len)
    largest = graph.subgraph(largest_nodes)
    sources = rng.choice(
        np.fromiter(largest_nodes, dtype=np.int64),
        size=min(path_sources, largest.number_of_nodes()),
        replace=False,
    )
    source_means = []
    for source in sources:
        distances = nx.single_source_shortest_path_length(largest, int(source))
        source_means.append(sum(distances.values()) / max(1, len(distances) - 1))
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "mean_degree": float(degrees.mean()),
        "degree_std": float(degrees.std(ddof=1)),
        "maximum_degree": int(degrees.max()),
        "clustering_coefficient": float(nx.average_clustering(graph)),
        "largest_connected_component_fraction": (
            largest.number_of_nodes() / graph.number_of_nodes()
        ),
        "average_shortest_path_length_approx": float(np.mean(source_means)),
        "path_source_count": int(len(sources)),
    }
