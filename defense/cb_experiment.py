"""Small CB experiments using the published model and separate stage caches."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from local_market.cb_model import sample_demands_and_active_counts
from local_market.networks import (
    generate_matched_graph, identify_graph_clusters, sample_lattice_occupancy,
)
from local_market.percolation import identify_lattice_clusters
from local_market.statistics import cluster_histogram, finite_cluster_sizes


class Cache:
    def __init__(self, root, model_version):
        self.root = root
        self.model_version = model_version
        self.usage = {}

    def get(self, stage, key, compute):
        identity = json.dumps([self.model_version, key], sort_keys=True).encode()
        path = self.root / stage / (hashlib.sha256(identity).hexdigest() + ".npz")
        usage = self.usage.setdefault(stage, {"computed": 0, "reused": 0})
        if path.is_file():
            with np.load(path, allow_pickle=False) as stored:
                value = dict(stored)
            usage["reused"] += 1
        else:
            value = compute()
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    np.savez_compressed(stream, **value)
                    stream.flush()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            usage["computed"] += 1
        return value


def child_seed(seed, *parts):
    return int(np.random.SeedSequence([seed, *parts]).generate_state(1, dtype=np.uint64)[0])


def network_adjacency(config, seed, cache):
    topology = config["topology"]
    n = config["node_count"]
    key = [topology, n, config["degree"], config["rho"], config["rewire"], seed]

    def generate():
        if topology == "er" and config["rho"] == 0:
            adjacency = sparse.csr_matrix((n, n), dtype=np.uint8)
        else:
            degree = config["rho"] * (n - 1) if topology == "er" else config["degree"]
            rewiring = config["rewire"] if topology == "ws" else 0.1
            adjacency = generate_matched_graph(topology, n, degree, seed, rewiring).adjacency
        return dict(data=adjacency.data, indices=adjacency.indices, indptr=adjacency.indptr)

    entries = cache.get("network", key, generate)
    adjacency = sparse.csr_matrix(
        (entries["data"], entries["indices"], entries["indptr"]), shape=(n, n)
    )
    return adjacency, key


def occupancy_block(config, p, graph_seed, network, network_key, cache):
    seed = config["seed"]
    key = [network_key, config["length"], config["boundary"], p, graph_seed, seed,
           config["samples"]]

    def compute():
        all_sizes, samples = [], []
        labels = None
        for sample in range(config["samples"]):
            # Reuse occupation uniforms across p for paired phase-curve comparisons.
            rng = np.random.default_rng(child_seed(seed, 1, graph_seed, sample))
            if network is None:
                occupied = sample_lattice_occupancy(config["length"], p, rng)
                clusters = identify_lattice_clusters(occupied, config["boundary"])
            else:
                occupied = rng.random(config["node_count"]) < p
                clusters = identify_graph_clusters(network, occupied)
            if labels is None:
                labels = clusters.labels
            all_sizes.append(clusters.sizes)
            largest_excluded = finite_cluster_sizes(clusters, "largest")
            spanning_excluded = (finite_cluster_sizes(clusters, "spanning") if network is None
                                 else np.empty(0, dtype=np.int64))
            moments = []
            for sizes in (largest_excluded, clusters.sizes, spanning_excluded):
                moments.extend((sizes.sum(), np.dot(sizes, sizes)))
            spanning = float(clusters.spans) if network is None else np.nan
            span_x = float(clusters.spans_horizontal) if config["boundary"] == "open" else np.nan
            samples.append([clusters.occupied_count, clusters.cluster_count,
                            clusters.p_infinity, spanning, span_x, *moments])
        return dict(observations=np.asarray(samples), sizes=np.concatenate(all_sizes),
                    offsets=np.cumsum([0, *(len(sizes) for sizes in all_sizes)]),
                    labels=labels)

    return cache.get("clusters", key, compute), key


def action_block(config, block, key, graph_seed, cache):
    seed, activity, trades = config["seed"], config["activity"], config["trades"]
    p = key[3]
    # Use the exact binary probability to keep nearby probabilities reproducible.
    p_seed = int(np.float64(p).view(np.uint64))

    def compute():
        sizes = block["sizes"]
        offsets = block["offsets"]
        demands, active_counts = [], []
        for sample, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            rng = np.random.default_rng(child_seed(seed, 2, graph_seed, p_seed, sample))
            for offset in range(0, trades, 128):
                count = min(128, trades - offset)
                demand, active = sample_demands_and_active_counts(sizes[start:end], count, activity, rng)
                demands.append(demand)
                active_counts.append(active)
        return dict(demand=np.concatenate(demands), active_cluster_count=np.concatenate(active_counts))

    return cache.get("actions", [key, activity, trades], compute)


def pooled_chi(first, second, axis=None):
    first = first.sum(axis=axis)
    second = second.sum(axis=axis)
    return np.divide(second, first, out=np.zeros_like(second, dtype=float), where=first != 0)


def phase_statistics(config, observations):
    # Dimensions are p, network realization, occupation sample, and observable.
    shapes = observations.shape
    gi = {"largest": 5, "none": 7, "spanning": 9}[config["exclusion"]]
    phase = pd.DataFrame({
        "p": config["p"],
        "network_replicates": shapes[1],
        "configs_per_network": shapes[2],
        "mean_occupied": observations[..., 0].mean(axis=(1, 2)),
        "mean_cluster_count": observations[..., 1].mean(axis=(1, 2)),
        "p_infinity": observations[..., 2].mean(axis=(1, 2)),
        "chi": pooled_chi(observations[..., gi], observations[..., gi + 1], axis=(1, 2)),
    })
    if config["topology"] == "lattice":
        phase["span_or_wrap_probability"] = observations[..., 3].mean(axis=(1, 2))
        if config["boundary"] == "open":
            phase["span_x_probability"] = observations[..., 4].mean(axis=(1, 2))
    if config["mode"] == "demo":
        return phase, {}

    rng = np.random.default_rng(config["bootstrap_seed"])
    p_boots, chi_boots = [], []
    for _ in range(config["bootstraps"]):
        graphs = rng.integers(shapes[1], size=(shapes[1], 1))
        configs = rng.integers(shapes[2], size=shapes[1:3])
        sampled = observations[:, graphs, configs, :]
        p_boots.append(sampled[..., 2].mean(axis=(1, 2)))
        chi_boots.append(pooled_chi(sampled[..., gi], sampled[..., gi + 1], axis=(1, 2)))
    for name, draws in (("p_infinity", p_boots), ("chi", chi_boots)):
        phase[f"{name}_low"], phase[f"{name}_high"] = np.quantile(draws, [.025, .975], axis=0)
    if len(phase) < 3 or phase.chi.nunique() < 2:
        return phase, {}
    peak_positions = phase.p.to_numpy()[np.asarray(chi_boots).argmax(axis=1)]
    index = int(phase.chi.idxmax())
    return phase, dict(
        p_chi_peak=float(phase.p.iloc[index]),
        p_chi_peak_interval=list(map(float, np.quantile(peak_positions, [.025, .975], method="nearest"))),
        peak_at_grid_edge=index in (0, len(phase) - 1),
    )


def draw_phase(phase, out, config, first_labels, degree_counts, prior):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, field, label in ((axes[0], "p_infinity", "Largest cluster / N"),
                             (axes[1], "chi", "Pooled susceptibility")):
        ax.plot(phase.p, phase[field], "o-", label="current")
        if f"{field}_low" in phase:
            ax.fill_between(phase.p, phase[f"{field}_low"], phase[f"{field}_high"], alpha=.2)
        if prior is not None:
            ax.plot(prior.p, prior[field], "s--", label="previous")
            ax.legend()
        ax.set(xlabel="Occupation probability p", ylabel=label)
        ax.grid(alpha=.2)
    if config["topology"] == "lattice":
        axes[2].imshow(np.where(first_labels, 1 + first_labels % 19, 0), cmap="tab20c",
                       interpolation="nearest", vmin=0, vmax=19)
        axes[2].set(xticks=[], yticks=[], title=f"First configuration, p={phase.p.iloc[0]:g}")
    else:
        axes[2].bar(np.arange(len(degree_counts)), degree_counts)
        axes[2].set(xlabel="Degree", ylabel="Nodes", title="First network")
    fig.savefig(out / "phase.png", dpi=150)
    plt.close(fig)


def simulate_cb(config, cache):
    observations, returns, histograms = [], [], []
    graph_data = []
    for replicate, graph_seed in enumerate(config["network_seeds"]):
        if config["topology"] == "lattice":
            adjacency, key = None, ["lattice"]
        else:
            adjacency, key = network_adjacency(config, graph_seed, cache)
        graph_data.append((adjacency, key, graph_seed))
        if replicate == 0:
            degrees = None if adjacency is None else np.bincount(np.diff(adjacency.indptr))

    first_labels = None
    for p in config["p"]:
        samples_by_graph = []
        for replicate, (adjacency, key, graph_seed) in enumerate(graph_data):
            block, cluster_key = occupancy_block(config, p, graph_seed, adjacency, key, cache)
            if first_labels is None:
                first_labels = block["labels"]
            samples_by_graph.append(block["observations"])
            actions = action_block(config, block, cluster_key, graph_seed, cache)
            returns.append(pd.DataFrame({
                "topology": config["topology"], "p": p, "network_replicate": replicate,
                "configuration": np.repeat(np.arange(config["samples"]), config["trades"]),
                "trade": np.tile(np.arange(config["trades"]), config["samples"]),
                **actions,
            }))
            counts = cluster_histogram(block["sizes"])
            sizes = np.flatnonzero(counts)
            histograms.extend((p, int(size), int(counts[size])) for size in sizes)
        observations.append(samples_by_graph)

    histogram = pd.DataFrame(histograms, columns=["p", "cluster_size", "count"])
    histogram = histogram.groupby(["p", "cluster_size"], as_index=False)["count"].sum()
    denominator = config["node_count"] * config["samples"] * len(graph_data)
    histogram["n_s_all"] = histogram["count"] / denominator
    phase, peak = phase_statistics(config, np.asarray(observations))
    return phase, peak, pd.concat(returns, ignore_index=True), histogram, first_labels, degrees
