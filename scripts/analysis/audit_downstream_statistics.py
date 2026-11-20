#!/usr/bin/env python3
"""Reassess market and network return statistics at their actual sampling levels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from local_market.config import MarketAnalysisConfig, NetworkAnalysisConfig
from local_market.statistics import return_distribution_table, standardize_returns
from local_market.hierarchical_statistics import (
    block_histogram, resampled_counts, select_network_configuration_blocks,
    selection_aware_tail_gof, weighted_statistics,
)


def audit_one_family(
    frame: pd.DataFrame, old_summary: pd.DataFrame, old_tails: pd.DataFrame,
    keys: list[str], quantiles: tuple[float, ...], minimum: int,
    configurations: int, networks: int | None, seed: int, bootstraps: int, gof_draws: int,
    exclude_zero_returns: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, draws_rows = [], []
    for group_index, (key, group) in enumerate(frame.groupby(keys, sort=True)):
        key = key if isinstance(key, tuple) else (key,)
        parameters = dict(zip(keys, key, strict=True))
        if networks is None:
            ids = group.replicate.to_numpy(dtype=int) * configurations + group.configuration.to_numpy(dtype=int)
            count = int(group.replicate.nunique()) * configurations
            assert np.array_equal(np.bincount(ids), np.full(count, len(group) // count))
            selector = lambda rng: rng.integers(0, count, size=count)
        else:
            # Each saved network step spans exactly one configuration block.
            ids = group.network_replicate.to_numpy(dtype=int) * configurations + group.step.to_numpy(dtype=int) // (len(group) // (networks * configurations))
            assert len(np.unique(ids)) == networks * configurations
            assert np.array_equal(np.bincount(ids),
                                  np.full(networks * configurations, len(group) // (networks * configurations)))
            selector = lambda rng: select_network_configuration_blocks(rng, networks, configurations)
        support, matrix = block_histogram(group["return"].to_numpy(dtype=float), ids)
        original_counts = np.asarray(matrix.sum(axis=0)).ravel()
        point = weighted_statistics(support, original_counts, quantiles, minimum,
                                    exclude_zero_returns=exclude_zero_returns)
        old_stat = old_summary
        for column, value in parameters.items():
            old_stat = old_stat[old_stat[column] == value]
        old_stat = old_stat.iloc[0]
        old_tail = old_tails
        for column, value in parameters.items():
            old_tail = old_tail[old_tail[column] == value]
        old_tail = old_tail.iloc[0]
        for column in ("return_excess_kurtosis", "zero_return_fraction", "p_abs_z_gt_5", "negative_crash_probability"):
            assert np.isclose(point[column], old_stat[column], atol=1e-10), (parameters, column)
        for column in ("xmin", "alpha_pdf", "ks_distance"):
            if not np.isclose(point[column], old_tail[column], atol=1e-10):
                print("recomputed tail differs from saved legacy fit", parameters, column,
                      flush=True)
                break
        rng = np.random.default_rng(np.random.SeedSequence([seed, 810, group_index]))
        samples = []
        for replicate in range(bootstraps):
            counts = resampled_counts(matrix, selector(rng))
            result = weighted_statistics(support, counts, quantiles, minimum,
                                         exclude_zero_returns=exclude_zero_returns)
            samples.append(result)
            draws_rows.append(parameters | {"bootstrap_replicate": replicate} | result)
        bootstrap = pd.DataFrame(samples)
        gof_rng = np.random.default_rng(np.random.SeedSequence([seed, 811, group_index]))
        adjusted_gof = selection_aware_tail_gof(support, matrix, point, quantiles, minimum,
                                gof_rng, gof_draws, selector, exclude_zero_returns=exclude_zero_returns)
        row = parameters | {
            "observations": len(group), "independent_networks": networks if networks is not None else np.nan,
            "configuration_blocks": matrix.shape[0], "actions_per_configuration": len(group)//matrix.shape[0],
            "old_xmin": float(old_tail.xmin), "old_alpha_pdf": float(old_tail.alpha_pdf),
            "old_alpha_ci_low": float(old_tail.alpha_ci_low),
            "old_alpha_ci_high": float(old_tail.alpha_ci_high),
            "old_fixed_xmin_gof_p": float(old_tail.pareto_fixed_xmin_gof_p),
            "selection_aware_hierarchical_null_gof_p": adjusted_gof,
        }
        for column in ("return_excess_kurtosis", "zero_return_fraction", "p_abs_z_gt_5",
                       "negative_crash_probability", "xmin", "xmin_quantile", "alpha_pdf", "ks_distance"):
            row[column] = point[column]
            row[column + "_ci_low"] = float(bootstrap[column].quantile(.025))
            row[column + "_ci_high"] = float(bootstrap[column].quantile(.975))
        row["original_xmin_quantile_selection_frequency"] = float(
            np.mean(bootstrap.xmin_quantile == point["xmin_quantile"])
        )
        rows.append(row)
        print("audited", parameters, "alpha", round(point["alpha_pdf"], 4),
              "CI", (round(row["alpha_pdf_ci_low"], 4), round(row["alpha_pdf_ci_high"], 4)),
              "GOF", adjusted_gof, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(draws_rows)


def plot_revised_tails(audit: pd.DataFrame, key: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in audit.groupby(key, sort=True):
        x = group["p_ratio" if key == "activity" else "relative_p"].to_numpy()
        line, = ax.plot(x, group.alpha_pdf, "o-", label=f"{key}={label}")
        for _, row in group.iterrows():
            ax.vlines(row["p_ratio" if key == "activity" else "relative_p"],
                      row.alpha_pdf_ci_low, row.alpha_pdf_ci_high,
                      color=line.get_color(), alpha=.6)
    ax.set_yscale("log")
    ax.set_xlabel("p/pc" if key == "activity" else "p/pc effective")
    ax.set_ylabel("Pareto PDF exponent; hierarchical 95% interval")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_revised_moments(audit: pd.DataFrame, group_key: str, x_key: str, path: Path) -> None:
    columns = (("return_excess_kurtosis", "Excess kurtosis"),
               ("zero_return_fraction", "Zero-return fraction"),
               ("p_abs_z_gt_5", "P(|z|>5)"),
               ("negative_crash_probability", "P(z<-5)"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for label, group in audit.groupby(group_key, sort=True):
        group = group.sort_values(x_key)
        for ax, (column, title) in zip(axes.flat, columns):
            x = group[x_key].to_numpy()
            y = group[column].to_numpy()
            line, = ax.plot(x, y, "o-", label=f"{group_key}={label}")
            ax.vlines(x, group[column + "_ci_low"], group[column + "_ci_high"],
                      color=line.get_color(), alpha=.6)
            ax.set_title(title)
            ax.grid(alpha=.2)
            if x_key == "activity":
                ax.set_xscale("log")
            ax.set_xlabel("Activity a" if x_key == "activity" else "p/pc effective")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Original point estimates with hierarchical 95% bootstrap intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate_saved_distributions(
    returns: pd.DataFrame, saved: pd.DataFrame, keys: list[str], absolute_only: bool
) -> pd.DataFrame:
    rows = []
    for key, group in returns.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        parameters = dict(zip(keys, key, strict=True))
        z, _, _ = standardize_returns(group["return"].to_numpy())
        rebuilt = pd.DataFrame(return_distribution_table(z))
        if absolute_only:
            rebuilt = rebuilt[rebuilt.distribution == "absolute_ccdf"]
        old = saved
        for column, value in parameters.items():
            old = old[old[column] == value]
        rebuilt = rebuilt.sort_values(["distribution", "x"]).reset_index(drop=True)
        old = old.sort_values(["distribution", "x"]).reset_index(drop=True)
        assert len(rebuilt) == len(old)
        assert rebuilt.distribution.tolist() == old.distribution.tolist()
        np.testing.assert_allclose(rebuilt.x, old.x, rtol=1e-12, atol=1e-12)
        maximum = float(np.max(np.abs(rebuilt.value.to_numpy() - old.value.to_numpy())))
        assert maximum < 1e-12
        rows.append(parameters | {"distribution_rows": len(old), "maximum_absolute_value_difference": maximum})
    return pd.DataFrame(rows)


def paired_market_activity_comparisons(
    frame: pd.DataFrame, config: MarketAnalysisConfig, draws: int
) -> pd.DataFrame:
    """Preserve the shared occupied configurations when comparing activities."""
    rows = []
    for ratio, ratio_group in frame.groupby("p_ratio", sort=True):
        matrices = {}
        for activity, group in ratio_group.groupby("activity", sort=True):
            block_ids = group.replicate.to_numpy(dtype=int) * config.configurations_per_replicate + group.configuration.to_numpy(dtype=int)
            matrices[activity] = block_histogram(group["return"].to_numpy(), block_ids)
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 813, round(ratio * 10)]))
        estimates = {activity: [] for activity in matrices}
        block_count = config.replicates * config.configurations_per_replicate
        for _ in range(draws):
            selected = rng.integers(0, block_count, size=block_count)
            for activity, (support, matrix) in matrices.items():
                estimates[activity].append(weighted_statistics(
                    support, resampled_counts(matrix, selected),
                    config.tail_xmin_quantiles, config.tail_minimum_count))
        activities = sorted(matrices)
        for left, right in zip(activities[1:], activities[:-1]):
            for statistic in ("return_excess_kurtosis", "p_abs_z_gt_5", "alpha_pdf"):
                difference = np.array([x[statistic] - y[statistic]
                                       for x, y in zip(estimates[left], estimates[right])])
                rows.append({"p_ratio": ratio, "activity_left": left, "activity_right": right,
                             "statistic": statistic, "difference_ci_low": np.quantile(difference, .025),
                             "difference_ci_high": np.quantile(difference, .975),
                             "probability_left_greater": np.mean(difference > 0)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/downstream_audit_20261114"))
    parser.add_argument("--bootstraps", type=int, default=300)
    parser.add_argument("--gof-draws", type=int, default=100)
    args = parser.parse_args()
    output = args.output_root
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    base = Path("results/tables")
    market_config = MarketAnalysisConfig.from_yaml("configs/market_analysis.yaml")
    market = pd.read_csv(base / "market_returns.csv.gz",
                         usecols=["p_ratio", "activity", "replicate", "configuration", "return"])
    market_summary = pd.read_csv(base / "market_summary.csv")
    market_tails = pd.read_csv(base / "market_tail_fits.csv")
    market_audit, market_draws = audit_one_family(
        market, market_summary, market_tails, ["p_ratio", "activity"],
        market_config.tail_xmin_quantiles, market_config.tail_minimum_count,
        market_config.configurations_per_replicate, None,
        market_config.seed, args.bootstraps, args.gof_draws, exclude_zero_returns=True)
    market_audit.to_csv(output / "tables/market_revised_statistics.csv", index=False)
    market_draws.to_csv(output / "tables/market_bootstrap_draws.csv.gz", index=False, compression="gzip")
    paired_market_activity_comparisons(market, market_config, args.bootstraps).to_csv(
        output / "tables/market_paired_activity_comparisons.csv", index=False)
    validate_saved_distributions(
        market, pd.read_csv(base / "market_distributions.csv"),
        ["p_ratio", "activity"], False,
    ).to_csv(output / "tables/market_distribution_validation.csv", index=False)
    plot_revised_tails(market_audit, "activity", output / "figures/market_tail_exponents_revised.png")
    plot_revised_moments(market_audit, "p_ratio", "activity",
                         output / "figures/market_activity_statistics_revised.png")

    network_config = NetworkAnalysisConfig.from_yaml("configs/network_analysis.yaml")
    network = pd.read_csv(base / "network_market_returns.csv.gz",
                          usecols=["topology", "relative_p", "network_replicate", "step", "return"])
    network_summary = pd.read_csv(base / "network_market_summary.csv")
    network_tails = pd.read_csv(base / "network_tail_fits.csv")
    network_audit, network_draws = audit_one_family(
        network, network_summary, network_tails, ["topology", "relative_p"],
        network_config.tail_xmin_quantiles, network_config.tail_minimum_count,
        network_config.market_configurations_per_network, network_config.network_replicates,
        network_config.seed, args.bootstraps, args.gof_draws)
    network_audit.to_csv(output / "tables/network_revised_statistics.csv", index=False)
    network_draws.to_csv(output / "tables/network_bootstrap_draws.csv.gz", index=False, compression="gzip")
    validate_saved_distributions(
        network, pd.read_csv(base / "network_market_distributions.csv"),
        ["topology", "relative_p"], True,
    ).to_csv(output / "tables/network_distribution_validation.csv", index=False)
    plot_revised_tails(network_audit, "topology", output / "figures/network_tail_exponents_revised.png")
    plot_revised_moments(network_audit, "topology", "relative_p",
                         output / "figures/network_extreme_events_revised.png")
    comparisons = []
    for ratio in network_config.relative_p_values:
        selected = network_draws[np.isclose(network_draws.relative_p, ratio)]
        for left, right in (("ba", "ws"), ("ba", "er"), ("ba", "lattice"),
                            ("ws", "er"), ("er", "lattice")):
            left_draws = selected[selected.topology == left].sort_values("bootstrap_replicate")
            right_draws = selected[selected.topology == right].sort_values("bootstrap_replicate")
            for column in ("alpha_pdf", "p_abs_z_gt_5", "return_excess_kurtosis"):
                difference = left_draws[column].to_numpy() - right_draws[column].to_numpy()
                comparisons.append({"relative_p": ratio, "left": left, "right": right,
                                    "statistic": column, "difference_ci_low": np.quantile(difference, .025),
                                    "difference_ci_high": np.quantile(difference, .975),
                                    "probability_left_greater": np.mean(difference > 0)})
    pd.DataFrame(comparisons).to_csv(output / "tables/network_pairwise_comparisons.csv", index=False)
    sources = [base / name for name in (
        "market_returns.csv.gz", "market_summary.csv", "market_tail_fits.csv",
        "market_distributions.csv", "network_market_returns.csv.gz",
        "network_market_summary.csv", "network_tail_fits.csv",
        "network_market_distributions.csv")]
    metadata = {
        "bootstrap_draws": args.bootstraps,
        "gof_null_draws": args.gof_draws,
        "market_seed": market_config.seed,
        "network_seed": network_config.seed,
        "market_resampling": "400 occupied configurations, whole 500-action blocks; paired across activities for differences",
        "network_resampling": "4 networks, then 60 occupied configurations within each selected network, whole 400-action blocks",
        "standardization": "recomputed on every bootstrap ensemble; tail point estimates follow the selected raw-zero rule",
        "market_tail_zero_rule": "exclude raw return==0 before |z| tail selection; other moments use all returns",
        "xmin": "minimum KS among configured positive-|z| quantiles, reselected in every bootstrap draw",
        "gof": "semiparametric common-Pareto tail null preserving block body and tail counts; xmin reselected",
        "source_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
    }
    (output / "tables/audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
