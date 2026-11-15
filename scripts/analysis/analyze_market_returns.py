#!/usr/bin/env python3
"""Run stage-3 CB return, activity, extreme-event, and tail analysis."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

from local_market.config import MarketAnalysisConfig
from local_market.experiments import run_market_ensemble, summarize_market_returns
from local_market.plotting import (
    plot_active_clusters_by_size,
    plot_activity_statistics,
    plot_market_time_series,
    plot_market_ensemble_samples,
    plot_return_distributions,
    plot_tail_exponents,
)
from local_market.cb_model import simulate_quenched
from local_market.networks import sample_lattice_occupancy
from local_market.percolation import identify_lattice_clusters
from local_market.statistics import return_distribution_table, nonzero_absolute_standardized_returns
from local_market.hierarchical_statistics import (
    block_histogram, resampled_counts, selection_aware_tail_gof, weighted_statistics,
)
from local_market.tail_fitting import (
    compare_tail_distributions,
    pareto_mle,
    pareto_fixed_xmin_gof_pvalue,
    select_pareto_tail,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _parameter_key(key: tuple[object, ...]) -> dict[str, object]:
    names = ("length", "p_ratio", "p", "activity")
    return dict(zip(names, key, strict=True))


def main() -> None:
    args = parse_args()
    config = MarketAnalysisConfig.from_yaml(args.config)
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    raw_returns, structures = run_market_ensemble(
        config, show_progress=not args.quiet
    )
    returns, summary, replicate_summary = summarize_market_returns(raw_returns, config)

    distribution_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    xmin_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
    group_keys = ["length", "p_ratio", "p", "activity"]
    for group_index, (key, group) in enumerate(returns.groupby(group_keys, sort=True)):
        parameters = _parameter_key(key)
        z = group["z"].to_numpy(dtype=float)
        # A zero raw return is still nonzero after mean centering; remove it
        # before selecting candidate quantiles for the conditional tail.
        absolute_z = nonzero_absolute_standardized_returns(
            group["return"].to_numpy(dtype=float), z
        )

        for row in return_distribution_table(
            z, raw_returns=group["return"].to_numpy(dtype=float)
        ):
            distribution_rows.append(parameters | row)

        best, candidates = select_pareto_tail(
            absolute_z,
            config.tail_xmin_quantiles,
            config.tail_minimum_count,
        )
        bootstrap_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 30, group_index])
        )
        block_ids = (group["replicate"].to_numpy(dtype=int) *
                     config.configurations_per_replicate + group["configuration"].to_numpy(dtype=int))
        support, block_matrix = block_histogram(group["return"].to_numpy(), block_ids)
        selector = lambda rng: np.concatenate([
            replicate * config.configurations_per_replicate
            + rng.integers(0, config.configurations_per_replicate,
                           size=config.configurations_per_replicate)
            for replicate in range(config.replicates)
        ])
        estimates = [weighted_statistics(
            support, resampled_counts(block_matrix, selector(bootstrap_rng)),
            config.tail_xmin_quantiles, config.tail_minimum_count, True,
        ) for _ in range(config.bootstrap_replicates)]
        alpha_values = np.array([estimate["alpha_pdf"] for estimate in estimates])
        xmin_values = np.array([estimate["xmin"] for estimate in estimates])
        alpha_se = float(alpha_values.std(ddof=1))
        alpha_low, alpha_high = np.quantile(alpha_values, [.025, .975])
        gof_pvalue = pareto_fixed_xmin_gof_pvalue(
            best, config.bootstrap_replicates, bootstrap_rng
        )
        selection_gof = selection_aware_tail_gof(
            support, block_matrix, {"xmin": best.xmin, "alpha_pdf": best.alpha_pdf,
                                    "ks_distance": best.ks_distance},
            config.tail_xmin_quantiles, config.tail_minimum_count,
            bootstrap_rng, config.bootstrap_replicates, selector, True,
        )
        comparison = compare_tail_distributions(absolute_z, best.xmin)
        tail_rows.append(
            parameters
            | {
                "tail_definition": "absolute standardized return |z|",
                "xmin": best.xmin,
                "alpha_pdf": best.alpha_pdf,
                "alpha_ccdf": best.alpha_pdf - 1.0,
                "ks_distance": best.ks_distance,
                "tail_count": best.tail_count,
                "nonzero_return_count": int(np.count_nonzero(group["return"].to_numpy())),
                "unique_tail": int(np.unique(absolute_z[absolute_z >= best.xmin]).size),
                "tail_dynamic_range": float(absolute_z.max() / best.xmin),
                "power_law_reliable": bool(
                    selection_gof >= 0.1 and best.tail_count >= 1000
                    and absolute_z.max() / best.xmin >= 10
                    and np.unique(absolute_z[absolute_z >= best.xmin]).size >= 30
                    and comparison["loglikelihood_ratio_power_vs_truncated"] >= -2
                    and comparison["loglikelihood_ratio_power_vs_exponential"] >= 0
                ),
                "tail_fraction_of_nonzero": best.tail_fraction,
                "alpha_bootstrap_se": alpha_se,
                "alpha_ci_low": alpha_low,
                "alpha_ci_high": alpha_high,
                "xmin_ci_low": float(np.quantile(xmin_values, .025)),
                "xmin_ci_high": float(np.quantile(xmin_values, .975)),
                "selection_aware_hierarchical_null_gof_p": selection_gof,
                "pareto_fixed_xmin_gof_p": gof_pvalue,
            }
            | comparison
        )
        xmin_rows.extend(parameters | candidate for candidate in candidates)

        for replicate, replicate_group in group.groupby("replicate", sort=True):
            nonzero = replicate_group["return"].to_numpy(dtype=float) != 0
            alpha, ks, count = pareto_mle(
                np.abs(replicate_group["z"].to_numpy(dtype=float)[nonzero]), best.xmin
            )
            seed_rows.append(
                parameters
                | {
                    "replicate": int(replicate),
                    "common_xmin": best.xmin,
                    "alpha_pdf": alpha,
                    "ks_distance": ks,
                    "tail_count": count,
                }
            )

        nested_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 40, group_index])
        )
        order = nested_rng.permutation(absolute_z.size)
        for fraction in config.length_sensitivity_fractions:
            sample = absolute_z[order[: max(2, int(fraction * absolute_z.size))]]
            length_fit, _ = select_pareto_tail(
                sample,
                config.tail_xmin_quantiles,
                min(config.tail_minimum_count, max(50, sample.size // 20)),
            )
            length_rows.append(
                parameters
                | {
                    "sample_fraction": fraction,
                    "sample_count": sample.size,
                    "xmin": length_fit.xmin,
                    "alpha_pdf": length_fit.alpha_pdf,
                    "ks_distance": length_fit.ks_distance,
                    "tail_count": length_fit.tail_count,
                }
            )

    distributions = pd.DataFrame(distribution_rows)
    tail_fits = pd.DataFrame(tail_rows)
    xmin_sensitivity = pd.DataFrame(xmin_rows)
    seed_sensitivity = pd.DataFrame(seed_rows)
    length_sensitivity = pd.DataFrame(length_rows)

    criticality_path = tables / "criticality_summary.csv"
    if not criticality_path.exists():
        raise FileNotFoundError(
            "stage-2 criticality_summary.csv is required for activity-size scaling"
        )
    criticality = pd.read_csv(criticality_path)
    critical_sizes = criticality[np.isclose(criticality["p_ratio"], 1.0)][
        ["length", "p", "samples", "mean_cluster_count", "cluster_count_sem"]
    ]
    activity_scaling = pd.concat(
        [critical_sizes.assign(activity=activity) for activity in config.activities],
        ignore_index=True,
    )
    activity_scaling["expected_active_cluster_count"] = (
        activity_scaling["activity"] * activity_scaling["mean_cluster_count"]
    )
    activity_scaling["expected_active_cluster_count_sem"] = (
        activity_scaling["activity"] * activity_scaling["cluster_count_sem"]
    )

    paths = {
        "returns": tables / "market_returns.csv.gz",
        "structures": tables / "market_structures.csv",
        "summary": tables / "market_summary.csv",
        "replicates": tables / "market_replicates.csv",
        "distributions": tables / "market_distributions.csv",
        "tail_fits": tables / "market_tail_fits.csv",
        "tail_xmin_sensitivity": tables / "market_tail_xmin_sensitivity.csv",
        "tail_seed_sensitivity": tables / "market_tail_seed_sensitivity.csv",
        "tail_length_sensitivity": tables / "market_tail_length_sensitivity.csv",
        "active_clusters_by_size": tables / "market_active_clusters_by_size.csv",
    }
    returns.to_csv(paths["returns"], index=False, compression="gzip")
    structures.to_csv(paths["structures"], index=False)
    summary.to_csv(paths["summary"], index=False)
    replicate_summary.to_csv(paths["replicates"], index=False)
    distributions.to_csv(paths["distributions"], index=False)
    tail_fits.to_csv(paths["tail_fits"], index=False)
    xmin_sensitivity.to_csv(paths["tail_xmin_sensitivity"], index=False)
    seed_sensitivity.to_csv(paths["tail_seed_sensitivity"], index=False)
    length_sensitivity.to_csv(paths["tail_length_sensitivity"], index=False)
    activity_scaling.to_csv(paths["active_clusters_by_size"], index=False)

    plot_market_ensemble_samples(returns, figures / "market_ensemble_samples.png")
    occupancy_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 50]))
    occupied = sample_lattice_occupancy(config.length, config.p_c, occupancy_rng)
    fixed_configuration = identify_lattice_clusters(occupied, config.boundary)
    trajectories = {
        activity: simulate_quenched(
            fixed_configuration, 3000, activity, config.market_depth,
            np.random.default_rng(np.random.SeedSequence([config.seed, 51, index])),
        )
        for index, activity in enumerate(config.activities)
    }
    plot_market_time_series(trajectories, figures / "market_time_series.png")
    plot_return_distributions(distributions, figures / "market_pdf_ccdf.png")
    plot_activity_statistics(summary, figures / "market_activity_statistics.png")
    plot_tail_exponents(tail_fits, figures / "market_tail_exponents.png")
    plot_active_clusters_by_size(
        activity_scaling, figures / "market_active_clusters_by_size.png"
    )

    metadata = {
        "config": config.to_dict(),
        "sampling": (
            "Independent occupied configurations are generated per p and replicate; "
            "each is reused for multiple independent cluster-action draws and across "
            "activities. The concatenation is an ensemble ordering, not evidence of "
            "temporal volatility clustering."
        ),
        "time_series": (
            "market_time_series.png is a separate 3000-step quenched trajectory at p=pc; "
            "its fixed occupancy seed is [config.seed,50], and each activity has an "
            "independent action stream [config.seed,51,activity_index]. "
            "market_ensemble_samples.png has no price path."
        ),
        "activity_size_scaling": (
            "Expected active cluster counts use E[K_active]=a E[K] with the "
            "p/pc=1 cluster counts produced in stage 2."
        ),
        "standardization": "per (p/p_c, activity), sample standard deviation ddof=1",
        "tail_method": {
            "variable": "absolute standardized return |z|, excluding zeros",
            "exponent": "continuous Pareto PDF exponent alpha",
            "xmin_selection": "minimum KS distance over configured quantile candidates",
            "confidence_interval": "configuration-block bootstrap with xmin reselected in every draw",
            "goodness_of_fit": (
                "legacy conditional parametric KS p-value at fixed xmin is retained; "
                "selection-aware semiparametric null resamples configuration blocks, "
                "preserves their body and tail counts, generates common Pareto tail "
                "magnitudes, and reselects xmin in every draw"
            ),
            "distribution_comparison": (
                "powerlaw package likelihood ratios; positive ratio favors pure power law"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    (tables / "market_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(tail_fits.to_string(index=False))
    print(f"saved stage-3 outputs under {args.output_root}")


if __name__ == "__main__":
    main()
