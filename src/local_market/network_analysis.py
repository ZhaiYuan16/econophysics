"""Network-topology stress tests for the Cont-Bouchaud model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

from .cb_model import sample_demands_and_active_counts
from .config import BAThresholdConfig, NetworkAnalysisConfig
from .networks import (
    GraphRealization,
    generate_matched_graph,
    graph_structure_metrics,
    identify_graph_clusters,
)
from .statistics import finite_cluster_sizes, return_statistics, standardize_returns


def build_network_ensemble(
    config: NetworkAnalysisConfig,
) -> tuple[dict[str, list[GraphRealization]], pd.DataFrame, pd.DataFrame]:
    ensemble: dict[str, list[GraphRealization]] = {}
    metric_rows: list[dict[str, object]] = []
    degree_rows: list[dict[str, object]] = []
    for topology_index, topology in enumerate(config.topologies):
        realizations = []
        for replicate in range(config.network_replicates):
            seed = int(
                np.random.SeedSequence(
                    [config.seed, 100, topology_index, replicate]
                ).generate_state(1)[0]
            )
            realization = generate_matched_graph(
                topology,
                config.node_count,
                config.target_mean_degree,
                seed,
                config.ws_rewiring_probability,
            )
            realizations.append(realization)
            metric_rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, 101, topology_index, replicate])
            )
            metrics = graph_structure_metrics(
                realization, config.path_sources, metric_rng
            )
            metric_rows.append(
                {
                    "topology": topology,
                    "network_replicate": replicate,
                    "network_seed": seed,
                }
                | metrics
            )
            degrees = np.fromiter(
                (degree for _, degree in realization.graph.degree()), dtype=np.int64
            )
            counts = np.bincount(degrees)
            degree_rows.extend(
                {
                    "topology": topology,
                    "network_replicate": replicate,
                    "degree": degree,
                    "node_count": int(count),
                    "node_fraction": float(count / config.node_count),
                }
                for degree, count in enumerate(counts)
                if count > 0
            )
        ensemble[topology] = realizations
    return ensemble, pd.DataFrame(metric_rows), pd.DataFrame(degree_rows)


def _threshold_criteria(
    p: np.ndarray,
    p_infinity: np.ndarray,
    susceptibility: np.ndarray,
    order_parameter_level: float,
) -> dict[str, float]:
    """Estimate operational thresholds from one uniformly sampled curve."""
    chi_peak = float(p[int(np.argmax(susceptibility))])
    window = min(7, len(p) if len(p) % 2 else len(p) - 1)
    smoothed = savgol_filter(p_infinity, window_length=window, polyorder=2)
    derivative = savgol_filter(
        p_infinity,
        window_length=window,
        polyorder=2,
        deriv=1,
        delta=float(p[1] - p[0]),
    )
    half = window // 2
    interior = np.arange(half, len(p) - half)
    slope_index = int(interior[np.argmax(derivative[interior])])
    monotone = np.maximum.accumulate(smoothed)
    if order_parameter_level <= monotone[0]:
        crossing = float(p[0])
    elif order_parameter_level >= monotone[-1]:
        crossing = float(p[-1])
    else:
        crossing = float(np.interp(order_parameter_level, monotone, p))
    return {
        "p_chi_peak": chi_peak,
        "p_smoothed_slope_peak": float(p[slope_index]),
        "smoothed_slope_peak": float(derivative[slope_index]),
        "p_order_parameter_level": crossing,
    }


def run_ba_threshold_study(
    config: BAThresholdConfig,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a finite-size, dense-grid BA percolation threshold study."""
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    total = (
        len(config.node_counts)
        * config.network_replicates
        * len(config.p_grid)
        * config.occupancy_samples_per_p
    )
    progress = tqdm(total=total, desc="BA threshold", unit="sample", disable=not show_progress)
    for size_index, node_count in enumerate(config.node_counts):
        for network_replicate in range(config.network_replicates):
            network_seed = int(
                np.random.SeedSequence(
                    [config.seed, 300, size_index, network_replicate]
                ).generate_state(1)[0]
            )
            realization = generate_matched_graph(
                "ba",
                node_count,
                config.target_mean_degree,
                network_seed,
            )
            degrees = np.fromiter(
                (degree for _, degree in realization.graph.degree()), dtype=float
            )
            mean_degree = float(degrees.mean())
            mean_degree_squared = float(np.mean(degrees**2))
            denominator = mean_degree_squared - mean_degree
            metric_rows.append(
                {
                    "node_count": node_count,
                    "network_replicate": network_replicate,
                    "network_seed": network_seed,
                    "mean_degree": mean_degree,
                    "mean_degree_squared": mean_degree_squared,
                    "maximum_degree": int(degrees.max()),
                    "molloy_reed_pc": mean_degree / denominator,
                }
            )
            for p_index, p in enumerate(config.p_grid):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 301, size_index, p_index, network_replicate]
                    )
                )
                for occupancy_sample in range(config.occupancy_samples_per_p):
                    occupied = rng.random(node_count) < p
                    clusters = identify_graph_clusters(realization.adjacency, occupied)
                    finite = finite_cluster_sizes(clusters, "largest")
                    rows.append(
                        {
                            "node_count": node_count,
                            "p": p,
                            "network_replicate": network_replicate,
                            "occupancy_sample": occupancy_sample,
                            "occupied_count": clusters.occupied_count,
                            "largest_size": clusters.largest_size,
                            "p_infinity": clusters.p_infinity,
                            "finite_first_moment": int(finite.sum()),
                            "finite_second_moment": int(np.dot(finite, finite)),
                        }
                    )
                    progress.update(1)
    progress.close()
    samples = pd.DataFrame(rows)
    metrics = pd.DataFrame(metric_rows)
    summary = (
        samples.groupby(["node_count", "p"], as_index=False)
        .agg(
            samples=("occupancy_sample", "count"),
            p_infinity=("p_infinity", "mean"),
            p_infinity_sem=("p_infinity", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
            finite_first_moment=("finite_first_moment", "sum"),
            finite_second_moment=("finite_second_moment", "sum"),
        )
    )
    summary["susceptibility_pooled"] = (
        summary["finite_second_moment"] / summary["finite_first_moment"]
    )

    estimate_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for size_index, node_count in enumerate(config.node_counts):
        size_summary = summary[summary["node_count"] == node_count].sort_values("p")
        p_values = size_summary["p"].to_numpy(dtype=float)
        criteria = _threshold_criteria(
            p_values,
            size_summary["p_infinity"].to_numpy(dtype=float),
            size_summary["susceptibility_pooled"].to_numpy(dtype=float),
            config.order_parameter_level,
        )
        criteria["molloy_reed_pc"] = float(
            metrics.loc[metrics["node_count"] == node_count, "molloy_reed_pc"].mean()
        )

        subset = samples[samples["node_count"] == node_count]
        shape = (
            config.network_replicates,
            len(config.p_grid),
            config.occupancy_samples_per_p,
        )
        arrays = {}
        for column in ("p_infinity", "finite_first_moment", "finite_second_moment"):
            arrays[column] = (
                subset.sort_values(["network_replicate", "p", "occupancy_sample"])[column]
                .to_numpy(dtype=float)
                .reshape(shape)
            )
        mr_values = metrics.loc[
            metrics["node_count"] == node_count, "molloy_reed_pc"
        ].to_numpy(dtype=float)
        rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 302, size_index])
        )
        for bootstrap_replicate in range(config.bootstrap_replicates):
            network_indices = rng.integers(
                0, config.network_replicates, size=config.network_replicates
            )
            sampled = {
                name: values[network_indices]
                for name, values in arrays.items()
            }
            occupancy_indices = rng.integers(
                0,
                config.occupancy_samples_per_p,
                size=shape,
            )
            sampled = {
                name: np.take_along_axis(values, occupancy_indices, axis=2)
                for name, values in sampled.items()
            }
            p_inf = sampled["p_infinity"].mean(axis=(0, 2))
            first = sampled["finite_first_moment"].sum(axis=(0, 2))
            second = sampled["finite_second_moment"].sum(axis=(0, 2))
            bootstrap_criteria = _threshold_criteria(
                p_values,
                p_inf,
                second / first,
                config.order_parameter_level,
            )
            bootstrap_rows.append(
                {
                    "node_count": node_count,
                    "bootstrap_replicate": bootstrap_replicate,
                }
                | bootstrap_criteria
                | {"molloy_reed_pc": float(mr_values[network_indices].mean())}
            )

        size_bootstrap = pd.DataFrame(
            row for row in bootstrap_rows if row["node_count"] == node_count
        )
        estimate_row: dict[str, object] = {"node_count": node_count} | criteria
        for criterion in (
            "p_chi_peak",
            "p_smoothed_slope_peak",
            "p_order_parameter_level",
            "molloy_reed_pc",
        ):
            estimate_row[f"{criterion}_ci_low"] = float(
                size_bootstrap[criterion].quantile(0.025)
            )
            estimate_row[f"{criterion}_ci_high"] = float(
                size_bootstrap[criterion].quantile(0.975)
            )
        estimate_rows.append(estimate_row)
    return (
        samples,
        metrics,
        summary,
        pd.DataFrame(estimate_rows),
        pd.DataFrame(bootstrap_rows),
    )


def run_network_criticality_scan(
    config: NetworkAnalysisConfig,
    ensemble: dict[str, list[GraphRealization]],
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    total = sum(len(config.p_grids[name]) for name in config.topologies)
    total *= config.network_replicates * config.occupancy_samples_per_p
    progress = tqdm(total=total, desc="network criticality", unit="sample", disable=not show_progress)
    for topology_index, topology in enumerate(config.topologies):
        for p_index, p in enumerate(config.p_grids[topology]):
            for network_replicate, realization in enumerate(ensemble[topology]):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 110, topology_index, p_index, network_replicate]
                    )
                )
                for occupancy_sample in range(config.occupancy_samples_per_p):
                    occupied = rng.random(config.node_count) < p
                    clusters = identify_graph_clusters(realization.adjacency, occupied)
                    finite = finite_cluster_sizes(clusters, "largest")
                    rows.append(
                        {
                            "topology": topology,
                            "p": p,
                            "network_replicate": network_replicate,
                            "occupancy_sample": occupancy_sample,
                            "occupied_count": clusters.occupied_count,
                            "cluster_count": clusters.cluster_count,
                            "largest_size": clusters.largest_size,
                            "largest_occupied_fraction": clusters.largest_occupied_fraction,
                            "p_infinity_sample": clusters.p_infinity,
                            "finite_first_moment": int(finite.sum()),
                            "finite_second_moment": int(np.dot(finite, finite)),
                        }
                    )
                    progress.update(1)
    progress.close()
    samples = pd.DataFrame(rows)
    summary = (
        samples.groupby(["topology", "p"], as_index=False)
        .agg(
            samples=("occupancy_sample", "count"),
            mean_occupied=("occupied_count", "mean"),
            mean_cluster_count=("cluster_count", "mean"),
            mean_largest_size=("largest_size", "mean"),
            p_infinity=("p_infinity_sample", "mean"),
            p_infinity_sem=(
                "p_infinity_sample",
                lambda x: x.std(ddof=1) / np.sqrt(x.count()),
            ),
            finite_first_moment=("finite_first_moment", "sum"),
            finite_second_moment=("finite_second_moment", "sum"),
        )
    )
    summary["susceptibility_pooled"] = (
        summary["finite_second_moment"] / summary["finite_first_moment"]
    )
    estimate_rows = []
    for topology, group in summary.groupby("topology", sort=False):
        ordered = group.sort_values("p").reset_index(drop=True)
        chi_index = int(ordered["susceptibility_pooled"].idxmax())
        slopes = np.gradient(ordered["p_infinity"], ordered["p"])
        slope_index = int(np.argmax(slopes))
        estimate_rows.append(
            {
                "topology": topology,
                "effective_pc": float(ordered.loc[chi_index, "p"]),
                "pc_criterion": "pooled susceptibility peak, largest cluster removed",
                "chi_peak": float(ordered.loc[chi_index, "susceptibility_pooled"]),
                "pc_max_order_parameter_slope": float(
                    ordered.loc[slope_index, "p"]
                ),
                "max_order_parameter_slope": float(slopes[slope_index]),
                "chi_peak_at_grid_boundary": bool(
                    chi_index in (0, len(ordered) - 1)
                ),
            }
        )
    return samples, summary, pd.DataFrame(estimate_rows)


def run_network_market_comparison(
    config: NetworkAnalysisConfig,
    ensemble: dict[str, list[GraphRealization]],
    estimates: pd.DataFrame,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return_frames: list[pd.DataFrame] = []
    structure_rows: list[dict[str, object]] = []
    total = len(config.topologies) * len(config.relative_p_values) * config.network_replicates
    progress = tqdm(total=total, desc="network markets", unit="replicate", disable=not show_progress)
    pc_lookup = estimates.set_index("topology")["effective_pc"].to_dict()
    for topology_index, topology in enumerate(config.topologies):
        effective_pc = float(pc_lookup[topology])
        for ratio_index, relative_p in enumerate(config.relative_p_values):
            p = min(1.0, effective_pc * relative_p)
            for network_replicate, realization in enumerate(ensemble[topology]):
                occupancy_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 120, topology_index, ratio_index, network_replicate]
                    )
                )
                action_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 121, topology_index, ratio_index, network_replicate]
                    )
                )
                demand_blocks = []
                active_blocks = []
                for sample in range(config.market_configurations_per_network):
                    occupied = occupancy_rng.random(config.node_count) < p
                    clusters = identify_graph_clusters(realization.adjacency, occupied)
                    structure_rows.append(
                        {
                            "topology": topology,
                            "relative_p": relative_p,
                            "p": p,
                            "network_replicate": network_replicate,
                            "configuration": sample,
                            "occupied_count": clusters.occupied_count,
                            "cluster_count": clusters.cluster_count,
                            "largest_size": clusters.largest_size,
                            "largest_occupied_fraction": clusters.largest_occupied_fraction,
                        }
                    )
                    demands, active = sample_demands_and_active_counts(
                        clusters.sizes,
                        config.trades_per_configuration,
                        config.activity,
                        action_rng,
                    )
                    demand_blocks.append(demands)
                    active_blocks.append(active)
                demands = np.concatenate(demand_blocks)
                return_frames.append(
                    pd.DataFrame(
                        {
                            "topology": topology,
                            "relative_p": relative_p,
                            "p": p,
                            "activity": config.activity,
                            "network_replicate": network_replicate,
                            "step": np.arange(demands.size, dtype=np.int64),
                            "configuration": np.repeat(
                                np.arange(config.market_configurations_per_network, dtype=np.int64),
                                config.trades_per_configuration,
                            ),
                            "demand": demands,
                            "active_cluster_count": np.concatenate(active_blocks),
                            "return": demands.astype(float) / config.market_depth,
                        }
                    )
                )
                progress.update(1)
    progress.close()
    return pd.concat(return_frames, ignore_index=True), pd.DataFrame(structure_rows)


def summarize_network_market(
    returns: pd.DataFrame, config: NetworkAnalysisConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = returns.copy()
    data["z"] = np.nan
    overall_rows = []
    replicate_rows = []
    keys = ["topology", "relative_p", "p", "activity"]
    for key, group in data.groupby(keys, sort=False):
        z, mean, standard_deviation = standardize_returns(group["return"].to_numpy())
        data.loc[group.index, "z"] = z
        parameters = dict(zip(keys, key, strict=True))
        row = parameters | return_statistics(
            group["return"].to_numpy(),
            z,
            config.extreme_thresholds,
            group["active_cluster_count"].to_numpy(),
        )
        row["standardization_mean"] = mean
        row["standardization_std"] = standard_deviation
        overall_rows.append(row)
        for replicate, replicate_group in group.groupby("network_replicate"):
            replicate_rows.append(
                parameters
                | {"network_replicate": int(replicate)}
                | return_statistics(
                    replicate_group["return"].to_numpy(),
                    data.loc[replicate_group.index, "z"].to_numpy(),
                    config.extreme_thresholds,
                    replicate_group["active_cluster_count"].to_numpy(),
                )
            )
    overall = pd.DataFrame(overall_rows)
    replicates = pd.DataFrame(replicate_rows)
    dispersion = (
        replicates.groupby(keys, as_index=False)
        .agg(
            replicate_kurtosis_sd=("return_excess_kurtosis", "std"),
            replicate_p_abs_z_gt_5_sd=("p_abs_z_gt_5", "std"),
            replicate_negative_crash_probability_sd=(
                "negative_crash_probability",
                "std",
            ),
        )
    )
    overall = overall.merge(dispersion, on=keys, validate="one_to_one")
    return data, overall, replicates
