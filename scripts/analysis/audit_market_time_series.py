#!/usr/bin/env python3
"""Audit saved market ensembles and generate separate CB trajectory artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import chi2, pearsonr

from local_market.cb_model import simulate_quenched
from local_market.config import MarketAnalysisConfig
from local_market.networks import sample_lattice_occupancy
from local_market.percolation import identify_lattice_clusters
from local_market.plotting import plot_market_ensemble_samples, plot_market_time_series


def acf(values: np.ndarray, max_lag: int = 50) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    covariance = fftconvolve(x, x[::-1], mode="full")[len(x)-1:len(x)+max_lag]
    return covariance[1:] / covariance[0]


def ljung_box(values: np.ndarray, max_lag: int = 50) -> float:
    coefficients = acf(values, max_lag)
    n = len(values)
    statistic = n * (n + 2) * np.sum(coefficients**2 / (n - np.arange(1, max_lag + 1)))
    return float(chi2.sf(statistic, max_lag))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/tables/market_returns.csv.gz"))
    parser.add_argument("--config", type=Path, default=Path("configs/market_analysis.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("results/market_time_audit_20261113"))
    args = parser.parse_args()
    config = MarketAnalysisConfig.from_yaml(args.config)
    root = args.output_root
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    columns = ["p_ratio", "activity", "replicate", "step", "configuration", "return", "z"]
    frame = pd.read_csv(args.source, usecols=columns)
    critical = frame[np.isclose(frame.p_ratio, 1.0)].copy()
    plot_market_ensemble_samples(critical, root / "figures/market_ensemble_samples.png")

    # Recreate precisely the stored occupancy streams; compare saved structure IDs below.
    structures = pd.read_csv(args.source.parent / "market_structures.csv")
    rows = []
    selected = critical[critical.replicate == 0]
    for replicate in range(config.replicates):
        occupancy_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 10, 1, replicate]))
        for configuration in range(config.configurations_per_replicate):
            occupied = sample_lattice_occupancy(config.length, config.p_c, occupancy_rng)
            clusters = identify_lattice_clusters(occupied, config.boundary)
            old = structures[(np.isclose(structures.p_ratio, 1.0)) &
                             (structures.replicate == replicate) &
                             (structures.configuration == configuration)].iloc[0]
            assert clusters.cluster_count == old.cluster_count
            assert clusters.largest_size == old.largest_size
            sum_squares = int(np.dot(clusters.sizes, clusters.sizes))
            for activity in config.activities:
                group = critical[(critical.replicate == replicate) &
                                 (np.isclose(critical.activity, activity)) &
                                 (critical.configuration == configuration)]
                assert len(group) == config.trades_per_configuration
                offset = replicate * config.configurations_per_replicate * config.trades_per_configuration
                rows.append({"replicate": replicate, "configuration": configuration,
                             "activity": activity, "start_index": offset + int(group.step.min()),
                             "end_index_inclusive": offset + int(group.step.max()),
                             "stored_step_start": int(group.step.min()),
                             "stored_step_end_inclusive": int(group.step.max()),
                             "cluster_count": clusters.cluster_count,
                             "largest_size": clusters.largest_size,
                             "sum_s_squared": sum_squares,
                             "theoretical_variance": activity * sum_squares / config.market_depth**2,
                             "empirical_variance": float(group["return"].var(ddof=1))})
    variances = pd.DataFrame(rows)
    variances.to_csv(root / "tables/configuration_variances.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, activity in zip(axes.flat, config.activities):
        g = variances[(variances.replicate == 0) & np.isclose(variances.activity, activity)]
        ax.plot(g.configuration, g.theoretical_variance, label="theory", linewidth=1)
        ax.plot(g.configuration, g.empirical_variance, label="500 draws", linewidth=.8, alpha=.75)
        ax.set_title(f"a={activity:g}")
        ax.set_ylabel("Conditional return variance")
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("Configuration ID")
    axes[-1, 1].set_xlabel("Configuration ID")
    fig.tight_layout()
    fig.savefig(root / "figures/configuration_variances.png", dpi=180)
    plt.close(fig)

    fixed_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 50]))
    fixed = identify_lattice_clusters(sample_lattice_occupancy(config.length, config.p_c, fixed_rng), config.boundary)
    trajectories = {}
    tests = []
    acf_rows = []
    for index, activity in enumerate(config.activities):
        trajectory = simulate_quenched(fixed, 3000, activity, config.market_depth,
                    np.random.default_rng(np.random.SeedSequence([config.seed, 51, index])))
        trajectories[activity] = trajectory
        pd.DataFrame({"time": np.arange(1, 3001), "return": trajectory.returns,
                      "log_price": trajectory.log_prices[1:], "activity": activity}).to_csv(
                          root / f"tables/quenched_a_{activity:g}.csv", index=False)
        ordered = selected[np.isclose(selected.activity, activity)].sort_values("step")
        original = ordered["return"].to_numpy()
        shuffled = np.random.default_rng(np.random.SeedSequence([config.seed, 60, index])).permutation(original)
        # The block plot uses shuffled configuration order while retaining within-block draws.
        blocks = original.reshape(config.configurations_per_replicate, config.trades_per_configuration)
        reordered = blocks[np.random.default_rng(np.random.SeedSequence([config.seed, 61, index])).permutation(len(blocks))].ravel()
        for label, data in (("ordered_return", original), ("ordered_absolute", np.abs(original)),
                            ("shuffled_absolute", np.abs(shuffled)),
                            ("fixed_return", trajectory.returns),
                            ("fixed_absolute", np.abs(trajectory.returns))):
            coefficients = acf(data)
            pvalue = ljung_box(data)
            tests.append({"activity": activity, "series": label, "n": len(data),
                          "ljung_box_lag50_p": pvalue, "ci95_halfwidth": 1.96 / np.sqrt(len(data)),
                          "max_abs_acf_lag1_50": float(np.max(np.abs(coefficients)))})
            acf_rows.extend({"activity": activity, "series": label, "lag": lag,
                             "acf": float(value), "ci95_low": -1.96 / np.sqrt(len(data)),
                             "ci95_high": 1.96 / np.sqrt(len(data))}
                            for lag, value in enumerate(coefficients, 1))
        if activity in (0.1, 0.5):
            fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
            for ax, data, title in zip(axes, (original[:3000], reordered[:3000], trajectory.returns),
                                       ("Original configuration order", "Shuffled configuration order", "One fixed configuration, independent actions")):
                ax.plot(np.arange(len(data)), data, linewidth=.55)
                ax.set_title(title)
                ax.set_ylabel("Return")
            axes[-1].set_xlabel("Sample index (top two); simulation time (bottom)")
            fig.tight_layout()
            fig.savefig(root / f"figures/order_comparison_a_{activity:g}.png", dpi=180)
            plt.close(fig)
    plot_market_time_series(trajectories, root / "figures/market_time_series.png")
    pd.DataFrame(acf_rows).to_csv(root / "tables/acf_lags_1_50.csv", index=False)
    pd.DataFrame(tests).to_csv(root / "tables/acf_tests.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    acf_table = pd.DataFrame(acf_rows)
    for ax, activity in zip(axes.flat, config.activities):
        for label in ("ordered_absolute", "shuffled_absolute", "fixed_absolute", "ordered_return"):
            g = acf_table[(np.isclose(acf_table.activity, activity)) & (acf_table.series == label)]
            ax.plot(g.lag, g.acf, label=label, linewidth=1)
        ax.axhspan(-1.96 / np.sqrt(50000), 1.96 / np.sqrt(50000), color="gray", alpha=.15,
                   label="95% pointwise CI (n=50,000)")
        ax.set_title(f"a={activity:g}")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("Lag")
    axes[-1, 1].set_xlabel("Lag")
    fig.tight_layout()
    fig.savefig(root / "figures/acf_diagnostics.png", dpi=180)
    plt.close(fig)

    # Independent seed repeats test the conditional model rather than one favorable path.
    repeat_rows = []
    for seed in (701, 702, 703, 704, 705):
        rng = np.random.default_rng(seed)
        config_seed = identify_lattice_clusters(sample_lattice_occupancy(config.length, config.p_c, rng), config.boundary)
        for activity in (.01, .1, .5):
            sim = simulate_quenched(config_seed, 10000, activity, config.market_depth, rng)
            for label, values in (("return", sim.returns), ("absolute", np.abs(sim.returns))):
                repeat_rows.append({"seed": seed, "activity": activity, "series": label,
                                    "lag50_ljung_box_p": ljung_box(values),
                                    "lag1_acf": acf(values)[0]})
    pd.DataFrame(repeat_rows).to_csv(root / "tables/independent_seed_repeats.csv", index=False)

    # Cluster bootstrap keeps all 500 observations of each configuration together.
    bootstrap_rows = []
    tail_table = pd.read_csv(args.source.parent / "market_tail_fits.csv")
    for (ratio, activity), group in frame.groupby(["p_ratio", "activity"]):
        blocks = group["return"].to_numpy().reshape(-1, config.trades_per_configuration)
        zblocks = group["z"].to_numpy().reshape(blocks.shape)
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 70, round(ratio*10), round(activity*1000)]))
        indices = rng.integers(0, len(blocks), size=(300, len(blocks)))
        means = blocks.mean(axis=1)
        second = (blocks**2).mean(axis=1)
        variance_boot = second[indices].mean(axis=1) - means[indices].mean(axis=1)**2
        extreme_boot = (np.abs(zblocks) > 5).mean(axis=1)[indices].mean(axis=1)
        tail_row = tail_table[(np.isclose(tail_table.p_ratio, ratio)) &
                              (np.isclose(tail_table.activity, activity))].iloc[0]
        xmin = float(tail_row.xmin)
        tail_mask = np.abs(zblocks) >= xmin
        tail_counts = tail_mask.sum(axis=1)
        tail_logs = np.where(tail_mask, np.log(np.maximum(np.abs(zblocks), xmin) / xmin), 0).sum(axis=1)
        alpha_boot = 1 + tail_counts[indices].sum(axis=1) / tail_logs[indices].sum(axis=1)
        bootstrap_rows.append({"p_ratio": ratio, "activity": activity,
                               "return_variance_ci_low": np.quantile(variance_boot, .025),
                               "return_variance_ci_high": np.quantile(variance_boot, .975),
                               "extreme_probability_ci_low": np.quantile(extreme_boot, .025),
                               "extreme_probability_ci_high": np.quantile(extreme_boot, .975),
                               "tail_alpha_fixed_xmin": float(tail_row.alpha_pdf),
                               "tail_alpha_ci_low": np.quantile(alpha_boot, .025),
                               "tail_alpha_ci_high": np.quantile(alpha_boot, .975)})
    pd.DataFrame(bootstrap_rows).to_csv(root / "tables/configuration_bootstrap_ci.csv", index=False)
    pivot = variances[variances.replicate == 0].pivot(index="configuration", columns="activity", values="theoretical_variance")
    correlation = pearsonr(pivot[0.1], pivot[0.5])
    metrics = {"shared_configuration_variance_correlation_a_0.1_0.5": float(correlation.statistic),
               "shared_configuration_correlation_p": float(correlation.pvalue),
               "fixed_cluster_count": fixed.cluster_count,
               "fixed_largest_size": fixed.largest_size,
               "data_source": str(args.source), "theory": "Var(r|s)=a sum(s^2)/lambda^2; zero nonzero-lag conditional covariance"}
    (root / "tables/audit_metadata.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
