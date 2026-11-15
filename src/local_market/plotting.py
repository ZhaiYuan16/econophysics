"""Publication-style plotting helpers for simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _finish_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_critical_observables(
    summary: pd.DataFrame, p_c: float, path: Path
) -> None:
    """Plot order parameter, finite-cluster susceptibility, and spanning rate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for length, group in summary.groupby("length", sort=True):
        group = group.sort_values("p_ratio")
        label = f"L={length}"
        axes[0].errorbar(
            group["p_ratio"],
            group["p_infinity"],
            yerr=group["p_infinity_sem"],
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            label=label,
        )
        axes[1].plot(
            group["p_ratio"],
            group["susceptibility_pooled"],
            marker="o",
            markersize=3,
            linewidth=1,
            label=label,
        )
        axes[1].fill_between(
            group["p_ratio"],
            group["susceptibility_ci_low"],
            group["susceptibility_ci_high"],
            alpha=0.12,
        )
        axes[2].plot(
            group["p_ratio"],
            group["spanning_probability"],
            marker="o",
            markersize=3,
            linewidth=1,
            label=label,
        )
    axes[0].set_ylabel(r"$P_\infty=\langle s_{\max}/L^2\rangle$")
    axes[1].set_ylabel(r"$\chi$ (largest cluster removed)")
    axes[2].set_ylabel("Spanning probability")
    for axis in axes:
        axis.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        axis.set_xlabel(r"$p/p_c$")
        axis.grid(alpha=0.25)
    axes[0].legend(title=f"open boundary\n$p_c={p_c:.6f}$", fontsize=8)
    fig.suptitle("Square-lattice site percolation: finite-size observables")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_cluster_distributions(
    histogram: pd.DataFrame,
    selected_ratios: tuple[float, ...],
    exclusion: str,
    path: Path,
) -> None:
    """Plot finite-cluster n_s per lattice site for each system size."""
    lengths = sorted(histogram["length"].unique())
    ncols = 2
    nrows = int(np.ceil(len(lengths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows), squeeze=False)
    for axis, length in zip(axes.flat, lengths, strict=False):
        for ratio in selected_ratios:
            group = histogram[
                (histogram["length"] == length)
                & np.isclose(histogram["p_ratio"], ratio)
                & (histogram["finite_n_s_per_site"] > 0)
            ]
            axis.loglog(
                group["cluster_size"],
                group["finite_n_s_per_site"],
                marker="o",
                linestyle="none",
                markersize=2.2,
                alpha=0.75,
                label=f"p/pc={ratio:g}",
            )
        axis.set_title(f"L={length}")
        axis.set_xlabel("Cluster size s")
        axis.set_ylabel(r"$n_s$ per lattice site")
        axis.grid(which="both", alpha=0.2)
        axis.legend(fontsize=7)
    for axis in axes.flat[len(lengths) :]:
        axis.set_visible(False)
    fig.suptitle(f"Finite-cluster distributions (exclusion: {exclusion})")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_cluster_tail_fits(
    histogram: pd.DataFrame, fits: pd.DataFrame, path: Path
) -> None:
    """Plot critical finite-cluster PMFs with fitted discrete power-law tails."""
    lengths = sorted(fits["length"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), squeeze=False)
    for axis, length in zip(axes.flat, lengths, strict=False):
        fit = fits.loc[fits["length"] == length].iloc[0]
        group = histogram[
            (histogram["length"] == length)
            & np.isclose(histogram["p_ratio"], fit["p_ratio"])
            & (histogram["finite_cluster_count"] > 0)
        ].sort_values("cluster_size")
        sizes = group["cluster_size"].to_numpy(dtype=float)
        density = group["finite_n_s_per_site"].to_numpy(dtype=float)
        axis.loglog(sizes, density, "o", markersize=2.2, alpha=0.65, label="simulation")
        xmin = int(fit["xmin"])
        tail = group[group["cluster_size"] >= xmin]
        anchor = tail.iloc[0]
        fitted = float(anchor["finite_n_s_per_site"]) * (
            tail["cluster_size"].to_numpy(dtype=float) / xmin
        ) ** (-float(fit["alpha_pdf"]))
        axis.loglog(
            tail["cluster_size"],
            fitted,
            "-",
            linewidth=1.4,
            label=rf"fit: $\alpha={fit['alpha_pdf']:.3f}$, $s_{{min}}={xmin}$",
        )
        axis.axvline(xmin, color="black", linestyle=":", linewidth=1)
        axis.set_title(f"L={length}, GOF p={fit['fixed_xmin_gof_p']:.3f}")
        axis.set_xlabel("Finite cluster size s")
        axis.set_ylabel(r"$n_s$ per lattice site")
        axis.grid(which="both", alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(r"Discrete power-law fits at $p/p_c=1$ (largest cluster removed)")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_pseudocritical_estimates(
    estimates: pd.DataFrame, p_c: float, path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    inverse_length = 1.0 / estimates["length"]
    axis.plot(
        inverse_length,
        estimates["p_chi_peak"],
        "o-",
        label=r"peak of $\chi$",
    )
    axis.plot(
        inverse_length,
        estimates["p_max_order_parameter_slope"],
        "s-",
        label=r"max slope of $P_\infty$",
    )
    axis.axhline(p_c, color="black", linestyle="--", label=f"theory {p_c:.6f}")
    axis.set_xlabel(r"$1/L$")
    axis.set_ylabel("Grid-based pseudocritical p")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_finite_size_scaling(
    summary: pd.DataFrame,
    scaling: pd.DataFrame,
    critical_ratio: float,
    path: Path,
) -> None:
    critical = summary[np.isclose(summary["p_ratio"], critical_ratio)].sort_values(
        "length"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    specifications = (
        (axes[0], "p_infinity", r"$P_\infty$"),
        (axes[1], "susceptibility_pooled", r"$\chi$"),
    )
    for axis, column, ylabel in specifications:
        name = "p_infinity" if column == "p_infinity" else "susceptibility"
        fit = scaling.loc[scaling["observable"] == name].iloc[0]
        lengths = critical["length"].to_numpy(dtype=float)
        values = critical[column].to_numpy(dtype=float)
        fitted = np.exp(fit["intercept"]) * lengths ** fit["fitted_exponent"]
        theoretical = values[0] * (lengths / lengths[0]) ** fit["theoretical_exponent"]
        axis.loglog(lengths, values, "o", label="simulation")
        axis.loglog(
            lengths,
            fitted,
            "-",
            label=f"fit exponent={fit['fitted_exponent']:.3f}",
        )
        axis.loglog(
            lengths,
            theoretical,
            "--",
            label=f"2D theory={fit['theoretical_exponent']:.3f}",
        )
        axis.set_xlabel("System length L")
        axis.set_ylabel(ylabel)
        axis.grid(which="both", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(r"Finite-size scaling at $p/p_c=1$")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_market_ensemble_samples(returns: pd.DataFrame, path: Path) -> None:
    """Show ordered ensemble draws without implying a price trajectory."""
    activities = sorted(returns["activity"].unique())
    fig, axes = plt.subplots(len(activities), 1, figsize=(10, 2.4 * len(activities)), squeeze=False)
    for row, activity in enumerate(activities):
        group = returns[
            np.isclose(returns["p_ratio"], 1.0)
            & np.isclose(returns["activity"], activity)
            & (returns["replicate"] == 0)
        ].sort_values("step").head(3000)
        axes[row, 0].plot(group["step"], group["return"], linewidth=0.55)
        axes[row, 0].set_ylabel(f"a={activity:g}\nreturn")
        axes[row, 0].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Ensemble sample index (not time)")
    fig.suptitle("CB ensemble samples at p/pc=1; blocks share an occupied configuration")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_market_time_series(simulations: dict[float, object], path: Path) -> None:
    """Plot only genuine time-ordered MarketSimulation trajectories."""
    from .cb_model import MarketSimulation

    if not simulations or any(not isinstance(x, MarketSimulation) for x in simulations.values()):
        raise TypeError("time-series plots require MarketSimulation trajectories")
    fig, axes = plt.subplots(len(simulations), 2, figsize=(12, 2.8 * len(simulations)), squeeze=False)
    for row, (activity, simulation) in enumerate(sorted(simulations.items())):
        times = np.arange(1, simulation.returns.size + 1)
        axes[row, 0].plot(times, simulation.returns, linewidth=0.55)
        axes[row, 1].plot(np.arange(simulation.log_prices.size), simulation.log_prices, linewidth=0.8)
        axes[row, 0].set_ylabel(f"a={activity:g}\nreturn")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Simulation time t")
    axes[-1, 1].set_xlabel("Simulation time t")
    axes[0, 0].set_title("Log return")
    axes[0, 1].set_title("Cumulative log return")
    fig.suptitle("Quenched CB trajectory at p/pc=1; independent actions, no volatility memory")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_return_distributions(distributions: pd.DataFrame, path: Path) -> None:
    """Plot standardized signed PDFs and absolute-return CCDFs."""
    activities = sorted(distributions["activity"].unique())
    fig, axes = plt.subplots(len(activities), 2, figsize=(12, 3.2 * len(activities)))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, distributions["p_ratio"].nunique()))
    ratios = sorted(distributions["p_ratio"].unique())
    for row, activity in enumerate(activities):
        for color, ratio in zip(colors, ratios, strict=True):
            selected = distributions[
                np.isclose(distributions["activity"], activity)
                & np.isclose(distributions["p_ratio"], ratio)
            ]
            pdf = selected[selected["distribution"] == "signed_pdf"]
            ccdf = selected[selected["distribution"] == "absolute_ccdf"]
            axes[row, 0].semilogy(
                pdf["x"],
                pdf["value"],
                color=color,
                marker=".",
                linestyle="none",
                markersize=2.5,
                label=f"p/pc={ratio:g}",
            )
            axes[row, 1].loglog(
                ccdf["x"], ccdf["value"], color=color, linewidth=1, label=f"p/pc={ratio:g}"
            )
        x_pdf = np.linspace(-6, 6, 300)
        axes[row, 0].semilogy(x_pdf, stats.norm.pdf(x_pdf), "k--", linewidth=0.9, label="Gaussian")
        x_ccdf = np.geomspace(0.05, 8, 200)
        axes[row, 1].loglog(
            x_ccdf, 2 * stats.norm.sf(x_ccdf), "k--", linewidth=0.9, label="Gaussian"
        )
        axes[row, 0].set_xlim(-10, 10)
        axes[row, 0].set_ylim(bottom=1e-7)
        axes[row, 0].set_ylabel(f"a={activity:g}\nPDF")
        axes[row, 1].set_ylabel(r"$P(|z|\geq x)$")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(which="both", alpha=0.2)
        axes[row, 1].legend(fontsize=7)
    axes[-1, 0].set_xlabel("Standardized return z")
    axes[-1, 1].set_xlabel("Absolute standardized return |z|")
    fig.suptitle("CB return distributions: PDF and absolute-return CCDF")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_activity_statistics(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    specifications = (
        ("return_std", "replicate_return_std_sd", "Return standard deviation"),
        ("return_excess_kurtosis", "replicate_kurtosis_sd", "Excess kurtosis"),
        ("p_abs_z_gt_5", "replicate_p_abs_z_gt_5_sd", r"$P(|z|>5)$"),
        ("mean_active_cluster_count", None, "Mean active clusters"),
    )
    for ratio, group in summary.groupby("p_ratio", sort=True):
        group = group.sort_values("activity")
        for axis, (column, error_column, ylabel) in zip(
            axes.flat, specifications, strict=True
        ):
            errors = group[error_column] if error_column is not None else None
            axis.errorbar(
                group["activity"],
                group[column],
                yerr=errors,
                marker="o",
                capsize=2,
                label=f"p/pc={ratio:g}",
            )
            axis.set_xscale("log")
            axis.set_xlabel("Activity a")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Market activity and return statistics")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_tail_exponents(tail_fits: pd.DataFrame, path: Path) -> None:
    """Show fitted exponents as labels; use a marker only for validated fits."""
    if "power_law_reliable" not in tail_fits:
        raise ValueError("tail exponent plot requires power_law_reliable diagnoses")
    alpha_column = "alpha" if "alpha" in tail_fits else "alpha_pdf"
    ratios = sorted(tail_fits.p_ratio.unique())
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for row, ratio in enumerate(ratios):
        group = tail_fits[np.isclose(tail_fits.p_ratio, ratio)]
        for _, result in group.iterrows():
            reliable = bool(result.power_law_reliable)
            color = "tab:blue" if reliable else "crimson"
            axis.scatter(result.activity, row, marker="o" if reliable else "x",
                         color=color, s=70, zorder=3)
            axis.annotate(f"{result[alpha_column]:.2f}", (result.activity, row),
                          xytext=(0, 10), textcoords="offset points",
                          ha="center", fontsize=9, color=color)
    axis.scatter([], [], marker="o", color="tab:blue",
                 label="passes basic pure Pareto diagnostics")
    axis.scatter([], [], marker="x", color="crimson",
                 label="pure Pareto unsupported; alpha descriptive only")
    activities = sorted(tail_fits.activity.unique())
    axis.set(xscale="log", ylim=(-.5, len(ratios)-.5), xlabel="Activity a", ylabel="p/pc")
    axis.set_yticks(range(len(ratios)), [f"{ratio:g}" for ratio in ratios])
    axis.set_xticks(activities, [f"{activity:g}" for activity in activities])
    axis.grid(axis="x", alpha=.3)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.15), fontsize=8)
    axis.set_title("Market tail exponent reliability (labels: descriptive fits)")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_active_clusters_by_size(activity_scaling: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.8))
    for activity, group in activity_scaling.groupby("activity", sort=True):
        group = group.sort_values("length")
        axis.errorbar(
            group["length"],
            group["expected_active_cluster_count"],
            yerr=group["expected_active_cluster_count_sem"],
            marker="o",
            capsize=3,
            label=f"a={activity:g}",
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("System length L")
    axis.set_ylabel(r"Expected active clusters $a\langle K\rangle$")
    axis.grid(which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_network_structure_metrics(metrics: pd.DataFrame, path: Path) -> None:
    columns = (
        ("mean_degree", "Mean degree"),
        ("clustering_coefficient", "Clustering coefficient"),
        ("largest_connected_component_fraction", "Largest component fraction"),
        ("average_shortest_path_length_approx", "Approx. mean shortest path"),
    )
    order = ["lattice", "er", "ws", "ba"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for axis, (column, label) in zip(axes.flat, columns, strict=True):
        grouped = metrics.groupby("topology")[column]
        means = grouped.mean().reindex(order)
        errors = grouped.std().reindex(order).fillna(0)
        axis.bar(order, means, yerr=errors, capsize=3)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Matched network structure metrics")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_network_degree_distributions(degrees: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.8))
    aggregated = (
        degrees.groupby(["topology", "degree"], as_index=False)["node_count"].sum()
    )
    totals = aggregated.groupby("topology")["node_count"].transform("sum")
    aggregated["fraction"] = aggregated["node_count"] / totals
    for topology, group in aggregated.groupby("topology", sort=False):
        axis.loglog(group["degree"], group["fraction"], "o-", markersize=3, label=topology.upper())
    axis.set_xlabel("Degree k")
    axis.set_ylabel("Node fraction")
    axis.grid(which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_network_criticality(
    summary: pd.DataFrame, estimates: pd.DataFrame, path: Path
) -> None:
    order = ["lattice", "er", "ws", "ba"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, topology in zip(axes.flat, order, strict=True):
        group = summary[summary["topology"] == topology].sort_values("p")
        pc = float(
            estimates.loc[estimates["topology"] == topology, "effective_pc"].iloc[0]
        )
        axis.plot(group["p"], group["p_infinity"], "o-", color="tab:blue", label=r"$P_\infty$")
        axis.set_xlabel("Occupation probability p")
        axis.set_ylabel(r"$P_\infty$", color="tab:blue")
        twin = axis.twinx()
        twin.plot(
            group["p"],
            group["susceptibility_pooled"],
            "s-",
            color="tab:red",
            markersize=3,
            label=r"$\chi$",
        )
        twin.set_ylabel(r"$\chi$", color="tab:red")
        axis.axvline(pc, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{topology.upper()}, effective pc={pc:.3f}")
        axis.grid(alpha=0.2)
    fig.suptitle("Numerical site-percolation thresholds by topology")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_ba_threshold_sensitivity(
    summary: pd.DataFrame, estimates: pd.DataFrame, path: Path
) -> None:
    """Plot dense BA critical curves and criterion-dependent thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for node_count, group in summary.groupby("node_count", sort=True):
        group = group.sort_values("p")
        axes[0].plot(group["p"], group["p_infinity"], "o-", markersize=2.5, label=f"N={node_count}")
        axes[1].plot(
            group["p"],
            group["susceptibility_pooled"],
            "o-",
            markersize=2.5,
            label=f"N={node_count}",
        )
    axes[0].set_ylabel(r"$P_\infty$")
    axes[1].set_ylabel(r"$\chi$ (largest cluster removed)")
    for axis in axes[:2]:
        axis.set_xlabel("Occupation probability p")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    criteria = (
        ("p_chi_peak", "susceptibility peak", "o"),
        ("p_smoothed_slope_peak", r"max smoothed $dP_\infty/dp$", "s"),
        ("p_order_parameter_level", r"$P_\infty=0.10$", "^"),
        ("molloy_reed_pc", "degree-moment estimate", "D"),
    )
    for column, label, marker in criteria:
        values = estimates[column].to_numpy(dtype=float)
        lower = values - estimates[f"{column}_ci_low"].to_numpy(dtype=float)
        upper = estimates[f"{column}_ci_high"].to_numpy(dtype=float) - values
        axes[2].errorbar(
            estimates["node_count"],
            values,
            yerr=np.vstack([lower, upper]),
            marker=marker,
            capsize=3,
            label=label,
        )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Node count N")
    axes[2].set_ylabel("Operational threshold p")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=7)
    fig.suptitle("BA site percolation: finite-size and criterion sensitivity")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_network_market_comparison(summary: pd.DataFrame, path: Path) -> None:
    specifications = (
        ("return_excess_kurtosis", "Excess kurtosis"),
        ("p_abs_z_gt_5", r"$P(|z|>5)$"),
        ("p_abs_z_gt_8", r"$P(|z|>8)$"),
        ("negative_crash_probability", r"$P(z<-5)$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    for topology, group in summary.groupby("topology", sort=False):
        group = group.sort_values("relative_p")
        for axis, (column, ylabel) in zip(axes.flat, specifications, strict=True):
            axis.plot(group["relative_p"], group[column], "o-", label=topology.upper())
            axis.set_xlabel(r"$p/p_c^{eff}$")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Network topology and standardized extreme events")
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_network_return_ccdfs(distributions: pd.DataFrame, path: Path) -> None:
    ratios = sorted(distributions["relative_p"].unique())
    fig, axes = plt.subplots(1, len(ratios), figsize=(5 * len(ratios), 4.5), squeeze=False)
    for axis, ratio in zip(axes.flat, ratios, strict=True):
        selected = distributions[np.isclose(distributions["relative_p"], ratio)]
        for topology, group in selected.groupby("topology", sort=False):
            axis.loglog(group["x"], group["value"], label=topology.upper())
        gaussian_x = np.geomspace(0.05, 8, 200)
        axis.loglog(gaussian_x, 2 * stats.norm.sf(gaussian_x), "k--", label="Gaussian")
        axis.set_title(f"p/pc_eff={ratio:g}")
        axis.set_xlabel("|z|")
        axis.set_ylabel(r"$P(|z|\geq x)$")
        axis.grid(which="both", alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_network_tail_exponents(tails: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.8))
    for topology, group in tails.groupby("topology", sort=False):
        group = group.sort_values("relative_p")
        axis.errorbar(
            group["relative_p"],
            group["alpha_pdf"],
            yerr=np.vstack(
                [
                    group["alpha_pdf"] - group["alpha_ci_low"],
                    group["alpha_ci_high"] - group["alpha_pdf"],
                ]
            ),
            marker="o",
            capsize=3,
            label=topology.upper(),
        )
    axis.set_yscale("log")
    axis.set_xlabel(r"$p/p_c^{eff}$")
    axis.set_ylabel("Absolute-return PDF tail exponent alpha")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    _finish_figure(fig, path)


def plot_soc_dynamics(
    paths: pd.DataFrame,
    summary: pd.DataFrame,
    burn_in: int,
    critical_reference: float,
    critical_band_relative_width: float,
    path: Path,
) -> None:
    """Plot adaptive occupancy, stationary distributions, and event sizes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    lower = critical_reference * (1.0 - critical_band_relative_width)
    upper = critical_reference * (1.0 + critical_band_relative_width)
    for initial, group in paths.groupby("initial_occupancy", sort=True):
        selected = group[group["replicate"] == 0].sort_values("step")
        stride = max(1, len(selected) // 4000)
        axes[0, 0].plot(
            selected["step"].iloc[::stride],
            selected["occupancy_after"].iloc[::stride],
            linewidth=0.8,
            label=f"initial p={initial:g}",
        )
    axes[0, 0].axhspan(lower, upper, color="grey", alpha=0.18, label="critical band")
    axes[0, 0].axvline(burn_in, color="black", linestyle=":", linewidth=1)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Occupied fraction p(t)")
    axes[0, 0].legend(fontsize=8)

    stationary = paths[paths["step"] >= burn_in]
    bins = np.linspace(0.0, 1.0, 81)
    for initial, group in stationary.groupby("initial_occupancy", sort=True):
        axes[0, 1].hist(
            group["occupancy_after"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.4,
            label=f"initial p={initial:g}",
        )
    axes[0, 1].axvline(critical_reference, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Stationary occupied fraction")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend(fontsize=8)

    events = stationary[stationary["removed_nodes"] > 0]
    for initial, group in events.groupby("initial_occupancy", sort=True):
        sizes = np.sort(group["removed_nodes"].to_numpy(dtype=float))
        ccdf = (sizes.size - np.arange(sizes.size)) / sizes.size
        axes[1, 0].loglog(sizes, ccdf, ".", markersize=3, label=f"initial p={initial:g}")
    axes[1, 0].set_xlabel("Nodes removed per event")
    axes[1, 0].set_ylabel("CCDF")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].errorbar(
        summary["initial_occupancy"],
        summary["mean_occupancy"],
        yerr=summary["mean_occupancy_sd"],
        marker="o",
        capsize=3,
        label="stationary mean",
    )
    axes[1, 1].plot([0, 1], [0, 1], "k:", label="no convergence")
    axes[1, 1].axhspan(lower, upper, color="grey", alpha=0.18)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Initial occupied fraction")
    axes[1, 1].set_ylabel("Stationary mean occupied fraction")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("Driven-dissipative CB extension")
    fig.tight_layout()
    _finish_figure(fig, path)
