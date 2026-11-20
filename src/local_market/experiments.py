"""Small experiment runners shared by command-line entry points."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .cb_model import (
    MarketSimulation,
    sample_demands_and_active_counts,
    simulate_annealed,
    simulate_quenched,
)
from .config import BaselineConfig, CriticalityConfig, MarketAnalysisConfig
from .networks import sample_lattice_occupancy
from .percolation import identify_lattice_clusters
from .statistics import (
    finite_cluster_sizes,
    return_statistics,
    standardize_returns,
    susceptibility_from_sizes,
)


def run_baseline(config: BaselineConfig) -> MarketSimulation:
    """Run one configured baseline experiment with a single RNG stream."""
    rng = np.random.default_rng(config.seed)

    def make_configuration():
        occupied = sample_lattice_occupancy(config.length, config.p, rng)
        return identify_lattice_clusters(occupied, config.boundary)

    if config.mode == "quenched":
        return simulate_quenched(
            make_configuration(),
            config.steps,
            config.activity,
            config.market_depth,
            rng,
        )
    return simulate_annealed(
        make_configuration,
        config.steps,
        config.activity,
        config.market_depth,
        rng,
    )


def _sem(values: pd.Series) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.count()))


def run_criticality_scan(
    config: CriticalityConfig, show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run an ensemble site-percolation scan.

    Returns per-realization observations, grouped summary statistics, pooled
    cluster histograms, and grid-based pseudocritical estimates.
    """
    rng = np.random.default_rng(config.seed)
    sample_rows: list[dict[str, object]] = []
    histogram_counts: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    total = sum(config.samples_per_length[length] for length in config.lengths)
    progress = tqdm(
        total=total * len(config.p_ratios),
        desc="criticality",
        unit="sample",
        disable=not show_progress,
    )

    for length in config.lengths:
        samples = config.samples_per_length[length]
        maximum_size = length * length
        for ratio in config.p_ratios:
            p = config.p_c * ratio
            all_histogram = np.zeros(maximum_size + 1, dtype=np.int64)
            finite_histogram = np.zeros(maximum_size + 1, dtype=np.int64)
            for sample in range(samples):
                occupied = sample_lattice_occupancy(length, p, rng)
                clusters = identify_lattice_clusters(occupied, config.boundary)
                finite_sizes = finite_cluster_sizes(
                    clusters, config.finite_cluster_exclusion
                )
                finite_denominator = int(finite_sizes.sum())
                finite_numerator = int(np.dot(finite_sizes, finite_sizes))
                if clusters.cluster_count:
                    all_histogram += np.bincount(
                        clusters.sizes, minlength=maximum_size + 1
                    )
                if finite_sizes.size:
                    finite_histogram += np.bincount(
                        finite_sizes, minlength=maximum_size + 1
                    )
                sample_rows.append(
                    {
                        "length": length,
                        "p_ratio": ratio,
                        "p": p,
                        "sample": sample,
                        "occupied_count": clusters.occupied_count,
                        "cluster_count": clusters.cluster_count,
                        "largest_size": clusters.largest_size,
                        "largest_occupied_fraction": clusters.largest_occupied_fraction,
                        "p_infinity_sample": clusters.p_infinity,
                        "spans": clusters.spans,
                        "susceptibility": susceptibility_from_sizes(finite_sizes),
                        "finite_first_moment": finite_denominator,
                        "finite_second_moment": finite_numerator,
                    }
                )
                progress.update(1)
            histogram_counts[(length, ratio)] = (all_histogram, finite_histogram)
    progress.close()

    samples_frame = pd.DataFrame(sample_rows)
    summary = (
        samples_frame.groupby(["length", "p_ratio", "p"], as_index=False)
        .agg(
            samples=("sample", "count"),
            mean_occupied=("occupied_count", "mean"),
            occupied_sem=("occupied_count", _sem),
            mean_cluster_count=("cluster_count", "mean"),
            cluster_count_sem=("cluster_count", _sem),
            mean_largest_size=("largest_size", "mean"),
            largest_size_sem=("largest_size", _sem),
            p_infinity=("p_infinity_sample", "mean"),
            p_infinity_sem=("p_infinity_sample", _sem),
            spanning_probability=("spans", "mean"),
            spanning_probability_sem=("spans", _sem),
            susceptibility_mean=("susceptibility", "mean"),
            susceptibility_sem=("susceptibility", _sem),
            finite_first_moment=("finite_first_moment", "sum"),
            finite_second_moment=("finite_second_moment", "sum"),
        )
    )
    summary["susceptibility_pooled"] = (
        summary["finite_second_moment"] / summary["finite_first_moment"]
    ).fillna(0.0)
    bootstrap_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 2]))
    bootstrap_rows: list[dict[str, float | int]] = []
    for (length, ratio, p), group in samples_frame.groupby(
        ["length", "p_ratio", "p"], sort=True
    ):
        first = group["finite_first_moment"].to_numpy(dtype=np.int64)
        second = group["finite_second_moment"].to_numpy(dtype=np.int64)
        estimates = np.empty(config.bootstrap_replicates, dtype=float)
        for replicate in range(config.bootstrap_replicates):
            indices = bootstrap_rng.integers(0, len(group), size=len(group))
            denominator = int(first[indices].sum())
            estimates[replicate] = (
                float(second[indices].sum() / denominator) if denominator else 0.0
            )
        bootstrap_rows.append(
            {
                "length": int(length),
                "p_ratio": float(ratio),
                "p": float(p),
                "susceptibility_bootstrap_se": float(estimates.std(ddof=1)),
                "susceptibility_ci_low": float(np.quantile(estimates, 0.025)),
                "susceptibility_ci_high": float(np.quantile(estimates, 0.975)),
            }
        )
    summary = summary.merge(
        pd.DataFrame(bootstrap_rows), on=["length", "p_ratio", "p"], validate="one_to_one"
    )

    histogram_rows: list[dict[str, object]] = []
    for (length, ratio), (all_counts, finite_counts) in histogram_counts.items():
        samples = config.samples_per_length[length]
        nonzero_sizes = np.flatnonzero((all_counts + finite_counts) > 0)
        for size in nonzero_sizes[nonzero_sizes > 0]:
            histogram_rows.append(
                {
                    "length": length,
                    "p_ratio": ratio,
                    "p": config.p_c * ratio,
                    "cluster_size": int(size),
                    "all_cluster_count": int(all_counts[size]),
                    "finite_cluster_count": int(finite_counts[size]),
                    "all_n_s_per_site": all_counts[size]
                    / (samples * length * length),
                    "finite_n_s_per_site": finite_counts[size]
                    / (samples * length * length),
                }
            )
    histogram_frame = pd.DataFrame(histogram_rows)

    critical_rows: list[dict[str, object]] = []
    for length, group in summary.groupby("length", sort=True):
        ordered = group.sort_values("p").reset_index(drop=True)
        chi_index = int(ordered["susceptibility_pooled"].idxmax())
        slopes = np.gradient(ordered["p_infinity"], ordered["p"])
        slope_index = int(np.argmax(slopes))
        critical_rows.append(
            {
                "length": int(length),
                "p_chi_peak": float(ordered.loc[chi_index, "p"]),
                "p_ratio_chi_peak": float(ordered.loc[chi_index, "p_ratio"]),
                "chi_peak": float(
                    ordered.loc[chi_index, "susceptibility_pooled"]
                ),
                "p_max_order_parameter_slope": float(
                    ordered.loc[slope_index, "p"]
                ),
                "p_ratio_max_order_parameter_slope": float(
                    ordered.loc[slope_index, "p_ratio"]
                ),
                "max_order_parameter_slope": float(slopes[slope_index]),
            }
        )
    critical_frame = pd.DataFrame(critical_rows)
    return samples_frame, summary, histogram_frame, critical_frame


def estimate_finite_size_scaling(
    summary: pd.DataFrame, critical_ratio: float = 1.0
) -> pd.DataFrame:
    """Fit log-log finite-size trends at the configured critical ratio."""
    critical = summary[np.isclose(summary["p_ratio"], critical_ratio)].sort_values(
        "length"
    )
    if len(critical) < 3:
        raise ValueError("at least three system sizes are required for scaling fits")
    rows: list[dict[str, object]] = []
    definitions = (
        ("p_infinity", "p_infinity", -5.0 / 48.0),
        ("susceptibility", "susceptibility_pooled", 43.0 / 24.0),
    )
    log_length = np.log(critical["length"].to_numpy(dtype=float))
    for observable, column, theoretical_exponent in definitions:
        values = critical[column].to_numpy(dtype=float)
        if np.any(values <= 0):
            raise ValueError(f"{column} must be positive for log-log scaling")
        fit = stats.linregress(log_length, np.log(values))
        rows.append(
            {
                "observable": observable,
                "p_ratio": critical_ratio,
                "fitted_exponent": float(fit.slope),
                "exponent_standard_error": float(fit.stderr),
                "theoretical_exponent": theoretical_exponent,
                "intercept": float(fit.intercept),
                "r_squared": float(fit.rvalue**2),
                "size_count": len(critical),
            }
        )
    return pd.DataFrame(rows)


def run_market_ensemble(
    config: MarketAnalysisConfig, show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate return ensembles, reusing configurations across activities."""
    return_frames: list[pd.DataFrame] = []
    structure_rows: list[dict[str, object]] = []
    total = len(config.p_ratios) * config.replicates
    progress = tqdm(total=total, desc="market ensembles", unit="replicate", disable=not show_progress)

    for p_index, ratio in enumerate(config.p_ratios):
        p = config.p_c * ratio
        for replicate in range(config.replicates):
            occupancy_rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, 10, p_index, replicate])
            )
            configurations = []
            for sample in range(config.configurations_per_replicate):
                occupied = sample_lattice_occupancy(config.length, p, occupancy_rng)
                clusters = identify_lattice_clusters(occupied, config.boundary)
                configurations.append(clusters)
                structure_rows.append(
                    {
                        "length": config.length,
                        "p_ratio": ratio,
                        "p": p,
                        "replicate": replicate,
                        "configuration": sample,
                        "occupied_count": clusters.occupied_count,
                        "cluster_count": clusters.cluster_count,
                        "largest_size": clusters.largest_size,
                        "largest_occupied_fraction": clusters.largest_occupied_fraction,
                        "spans": clusters.spans,
                    }
                )

            for activity_index, activity in enumerate(config.activities):
                action_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 20, p_index, replicate, activity_index]
                    )
                )
                demand_blocks = []
                active_blocks = []
                available_blocks = []
                configuration_blocks = []
                for sample, clusters in enumerate(configurations):
                    demands, active_counts = sample_demands_and_active_counts(
                        clusters.sizes,
                        config.trades_per_configuration,
                        activity,
                        action_rng,
                    )
                    demand_blocks.append(demands)
                    active_blocks.append(active_counts)
                    available_blocks.append(
                        np.full(
                            config.trades_per_configuration,
                            clusters.cluster_count,
                            dtype=np.int64,
                        )
                    )
                    configuration_blocks.append(
                        np.full(config.trades_per_configuration, sample, dtype=np.int64)
                    )
                demands = np.concatenate(demand_blocks)
                observations = demands.size
                return_frames.append(
                    pd.DataFrame(
                        {
                            "length": config.length,
                            "p_ratio": ratio,
                            "p": p,
                            "activity": activity,
                            "replicate": replicate,
                            "step": np.arange(observations, dtype=np.int64),
                            "configuration": np.concatenate(configuration_blocks),
                            "available_cluster_count": np.concatenate(available_blocks),
                            "active_cluster_count": np.concatenate(active_blocks),
                            "demand": demands,
                            "return": demands.astype(float) / config.market_depth,
                        }
                    )
                )
            progress.update(1)
    progress.close()
    return pd.concat(return_frames, ignore_index=True), pd.DataFrame(structure_rows)


def summarize_market_returns(
    returns_frame: pd.DataFrame, config: MarketAnalysisConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize per parameter combination and summarize overall and by seed."""
    data = returns_frame.copy()
    data["z"] = np.nan
    overall_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    keys = ["length", "p_ratio", "p", "activity"]
    for key, group in data.groupby(keys, sort=True):
        z, mean, standard_deviation = standardize_returns(group["return"].to_numpy())
        data.loc[group.index, "z"] = z
        overall = dict(zip(keys, key, strict=True))
        overall.update(
            return_statistics(
                group["return"].to_numpy(),
                z,
                config.extreme_thresholds,
                group["active_cluster_count"].to_numpy(),
            )
        )
        overall["standardization_mean"] = mean
        overall["standardization_std"] = standard_deviation
        overall["mean_available_cluster_count"] = float(
            group["available_cluster_count"].mean()
        )
        overall_rows.append(overall)

        for replicate, replicate_group in group.groupby("replicate", sort=True):
            replicate_z = data.loc[replicate_group.index, "z"].to_numpy()
            row = dict(zip(keys, key, strict=True))
            row["replicate"] = int(replicate)
            row.update(
                return_statistics(
                    replicate_group["return"].to_numpy(),
                    replicate_z,
                    config.extreme_thresholds,
                    replicate_group["active_cluster_count"].to_numpy(),
                )
            )
            replicate_rows.append(row)

    overall_frame = pd.DataFrame(overall_rows)
    replicate_frame = pd.DataFrame(replicate_rows)
    dispersion = (
        replicate_frame.groupby(keys, as_index=False)
        .agg(
            replicate_return_std_sd=("return_std", "std"),
            replicate_kurtosis_sd=("return_excess_kurtosis", "std"),
            replicate_p_abs_z_gt_3_sd=("p_abs_z_gt_3", "std"),
            replicate_p_abs_z_gt_5_sd=("p_abs_z_gt_5", "std"),
            replicate_p_abs_z_gt_8_sd=("p_abs_z_gt_8", "std"),
            replicate_negative_crash_probability_sd=(
                "negative_crash_probability",
                "std",
            ),
        )
    )
    overall_frame = overall_frame.merge(dispersion, on=keys, validate="one_to_one")
    return data, overall_frame, replicate_frame
