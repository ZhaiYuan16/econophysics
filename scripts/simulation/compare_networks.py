#!/usr/bin/env python3
"""Run stage-4 network-topology criticality and market stress tests."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from local_market.config import NetworkAnalysisConfig
from local_market.network_analysis import (
    build_network_ensemble,
    run_network_criticality_scan,
    run_network_market_comparison,
    summarize_network_market,
)
from local_market.plotting import (
    plot_network_criticality,
    plot_network_degree_distributions,
    plot_network_market_comparison,
    plot_network_return_ccdfs,
    plot_network_structure_metrics,
    plot_network_tail_exponents,
)
from local_market.statistics import return_distribution_table
from local_market.hierarchical_statistics import (
    block_histogram, resampled_counts, select_network_configuration_blocks,
    selection_aware_tail_gof, weighted_statistics,
)
from local_market.tail_fitting import (
    compare_tail_distributions,
    pareto_fixed_xmin_gof_pvalue,
    pareto_mle,
    select_pareto_tail,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = NetworkAnalysisConfig.from_yaml(args.config)
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    ensemble, network_metrics, degree_distributions = build_network_ensemble(config)
    critical_samples, critical_summary, pc_estimates = run_network_criticality_scan(
        config, ensemble, show_progress=not args.quiet
    )
    raw_returns, market_structures = run_network_market_comparison(
        config, ensemble, pc_estimates, show_progress=not args.quiet
    )
    returns, market_summary, market_replicates = summarize_network_market(
        raw_returns, config
    )

    distribution_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    tail_seed_rows: list[dict[str, object]] = []
    keys = ["topology", "relative_p", "p", "activity"]
    for group_index, (key, group) in enumerate(returns.groupby(keys, sort=False)):
        parameters = dict(zip(keys, key, strict=True))
        absolute_z = np.abs(group["z"].to_numpy(dtype=float))
        for distribution_row in return_distribution_table(group["z"].to_numpy()):
            if distribution_row["distribution"] == "absolute_ccdf":
                distribution_rows.append(parameters | distribution_row)
        fit, _ = select_pareto_tail(
            absolute_z,
            config.tail_xmin_quantiles,
            config.tail_minimum_count,
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 200, group_index])
        )
        block_ids = (group["network_replicate"].to_numpy(dtype=int) *
                     config.market_configurations_per_network + group["configuration"].to_numpy(dtype=int))
        support, block_matrix = block_histogram(group["return"].to_numpy(), block_ids)
        selector = lambda draw_rng: select_network_configuration_blocks(
            draw_rng, config.network_replicates, config.market_configurations_per_network)
        estimates = [weighted_statistics(
            support, resampled_counts(block_matrix, selector(rng)),
            config.tail_xmin_quantiles, config.tail_minimum_count,
        ) for _ in range(config.bootstrap_replicates)]
        alpha_values = np.array([estimate["alpha_pdf"] for estimate in estimates])
        xmin_values = np.array([estimate["xmin"] for estimate in estimates])
        alpha_se = float(alpha_values.std(ddof=1))
        alpha_low, alpha_high = np.quantile(alpha_values, [.025, .975])
        gof = pareto_fixed_xmin_gof_pvalue(
            fit, config.bootstrap_replicates, rng
        )
        selection_gof = selection_aware_tail_gof(
            support, block_matrix, {"xmin": fit.xmin, "alpha_pdf": fit.alpha_pdf,
                                    "ks_distance": fit.ks_distance},
            config.tail_xmin_quantiles, config.tail_minimum_count,
            rng, config.bootstrap_replicates, selector,
        )
        comparison = compare_tail_distributions(absolute_z, fit.xmin)
        tail_rows.append(
            parameters
            | {
                "tail_definition": "absolute standardized return |z|",
                "xmin": fit.xmin,
                "alpha_pdf": fit.alpha_pdf,
                "alpha_ccdf": fit.alpha_pdf - 1.0,
                "ks_distance": fit.ks_distance,
                "tail_count": fit.tail_count,
                "tail_fraction_of_nonzero": fit.tail_fraction,
                "alpha_bootstrap_se": alpha_se,
                "alpha_ci_low": alpha_low,
                "alpha_ci_high": alpha_high,
                "xmin_ci_low": float(np.quantile(xmin_values, .025)),
                "xmin_ci_high": float(np.quantile(xmin_values, .975)),
                "selection_aware_hierarchical_null_gof_p": selection_gof,
                "pareto_fixed_xmin_gof_p": gof,
            }
            | comparison
        )
        for replicate, replicate_group in group.groupby("network_replicate"):
            alpha, ks, count = pareto_mle(
                np.abs(replicate_group["z"].to_numpy()), fit.xmin
            )
            tail_seed_rows.append(
                parameters
                | {
                    "network_replicate": int(replicate),
                    "common_xmin": fit.xmin,
                    "alpha_pdf": alpha,
                    "ks_distance": ks,
                    "tail_count": count,
                }
            )

    distributions = pd.DataFrame(distribution_rows)
    tail_fits = pd.DataFrame(tail_rows)
    tail_seed_sensitivity = pd.DataFrame(tail_seed_rows)
    network_metric_summary = (
        network_metrics.groupby("topology", as_index=False)
        .agg(
            mean_degree=("mean_degree", "mean"),
            mean_degree_sd=("mean_degree", "std"),
            degree_std=("degree_std", "mean"),
            maximum_degree=("maximum_degree", "mean"),
            clustering_coefficient=("clustering_coefficient", "mean"),
            clustering_coefficient_sd=("clustering_coefficient", "std"),
            largest_connected_component_fraction=(
                "largest_connected_component_fraction",
                "mean",
            ),
            average_shortest_path_length_approx=(
                "average_shortest_path_length_approx",
                "mean",
            ),
            average_shortest_path_length_approx_sd=(
                "average_shortest_path_length_approx",
                "std",
            ),
        )
        .merge(pc_estimates, on="topology", validate="one_to_one")
    )

    paths = {
        "network_metrics": tables / "network_metrics.csv",
        "network_metric_summary": tables / "network_metric_summary.csv",
        "degree_distributions": tables / "network_degree_distributions.csv",
        "criticality_samples": tables / "network_criticality_samples.csv",
        "criticality_summary": tables / "network_criticality_summary.csv",
        "effective_pc": tables / "network_effective_pc.csv",
        "market_structures": tables / "network_market_structures.csv",
        "market_returns": tables / "network_market_returns.csv.gz",
        "market_summary": tables / "network_market_summary.csv",
        "market_replicates": tables / "network_market_replicates.csv",
        "market_distributions": tables / "network_market_distributions.csv",
        "tail_fits": tables / "network_tail_fits.csv",
        "tail_seed_sensitivity": tables / "network_tail_seed_sensitivity.csv",
    }
    frames = {
        "network_metrics": network_metrics,
        "network_metric_summary": network_metric_summary,
        "degree_distributions": degree_distributions,
        "criticality_samples": critical_samples,
        "criticality_summary": critical_summary,
        "effective_pc": pc_estimates,
        "market_structures": market_structures,
        "market_summary": market_summary,
        "market_replicates": market_replicates,
        "market_distributions": distributions,
        "tail_fits": tail_fits,
        "tail_seed_sensitivity": tail_seed_sensitivity,
    }
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False)
    returns.to_csv(paths["market_returns"], index=False, compression="gzip")

    plot_network_structure_metrics(network_metrics, figures / "network_metrics.png")
    plot_network_degree_distributions(
        degree_distributions, figures / "network_degree_distributions.png"
    )
    plot_network_criticality(
        critical_summary, pc_estimates, figures / "network_criticality.png"
    )
    plot_network_market_comparison(
        market_summary, figures / "network_extreme_events.png"
    )
    plot_network_return_ccdfs(
        distributions, figures / "network_return_ccdfs.png"
    )
    plot_network_tail_exponents(
        tail_fits, figures / "network_tail_exponents.png"
    )

    metadata = {
        "config": config.to_dict(),
        "comparison": (
            "Networks are matched by node count and target mean degree. Market "
            "comparisons use the same activity and positions relative to each "
            "topology's susceptibility-peak effective threshold."
        ),
        "path_length": (
            "Average shortest path is estimated within the underlying graph's "
            "largest connected component using configured sampled sources."
        ),
        "tail": (
            "Continuous Pareto PDF fit to nonzero |z|; xmin chosen from fixed "
            "quantile candidates by KS distance. Confidence intervals resample "
            "four networks, then configurations within each network, with xmin "
            "reselected. The selection-aware semiparametric GOF retains block "
            "tail counts and refits xmin under a common-Pareto tail null."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "networkx": nx.__version__,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    (tables / "network_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(network_metric_summary.to_string(index=False))
    print(market_summary.to_string(index=False))
    print(tail_fits.to_string(index=False))
    print(f"saved stage-4 outputs under {args.output_root}")


if __name__ == "__main__":
    main()
