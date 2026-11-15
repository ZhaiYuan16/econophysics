"""Figures for the revised square-lattice criticality scan."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_figures(summary: pd.DataFrame, peaks: pd.DataFrame,
                 crossings: pd.DataFrame, histogram: pd.DataFrame,
                 fits: pd.DataFrame, p_c: float, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(path / name, dpi=180)
        plt.close(fig)
    def lines(ax, column, x="p"):
        for length, group in summary.groupby("length"):
            group = group.sort_values(x)
            ax.plot(group[x], group[column], "o-", markersize=2, label=f"L={length}")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col, label in zip(axes.flat,
            ("p_infinity_mean", "chi_largest_excluded", "chi_all_spanning_excluded", "span_x_mean"),
            (r"$P_\infty=\langle s_{max}/L^2\rangle$", r"$\chi$: one largest excluded",
             r"$\chi$: all spanning excluded", "$P_x$: left to right")):
        lines(ax, col); ax.set_ylabel(label); ax.set_xlabel("p")
    save(fig, "criticality_observables_revised.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for rule, group in peaks.groupby("exclusion"):
        ax.errorbar(group.length + (0 if rule == "largest_excluded" else 5), group.p_grid_peak,
            yerr=[np.maximum(0, group.p_grid_peak - group.p_ci_low),
                  np.maximum(0, group.p_ci_high - group.p_grid_peak)],
            fmt="o-", capsize=2, label=rule)
    ax.axhline(p_c, color="black", ls="--", label="theoretical $p_c$")
    ax.set(xlabel="L", ylabel="grid peak p"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, "criticality_pseudocritical_revised.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for length, group in summary.groupby("length"):
        group = group.sort_values("p")
        ax.plot(group.p, group.span_x_mean, "o-", ms=2, label=f"L={length}")
        ax.fill_between(group.p, group.span_x_ci_low, group.span_x_ci_high, alpha=.1)
    for value in crossings.p_crossing.dropna() if not crossings.empty else []:
        ax.axvline(value, color="gray", alpha=.25)
    ax.axvline(p_c, color="black", ls="--"); ax.set(xlabel="p", ylabel="$P_x$: left to right")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, "criticality_spanning_probability_revised.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    lines(ax, "span_y_mean")
    ax.set(xlabel="p", ylabel="$P_y$: top to bottom")
    save(fig, "criticality_direction_symmetry_revised.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for length, group in summary.groupby("length"):
        x = (group.p - p_c) * length**.75
        ax.plot(x, group.span_x_mean, "o", ms=3, label=f"L={length}")
    ax.set(xlabel=r"$(p-p_c)L^{3/4}$, theoretical $p_c$", ylabel="$P_x$")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, "criticality_spanning_collapse_revised.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for length, group in summary.groupby("length"):
        x = (group.p - p_c) * length**.75
        for ax, col, power in zip(axes,
                ("p_infinity_mean", "chi_largest_excluded", "chi_all_spanning_excluded"),
                (5/48, -43/24, -43/24)):
            ax.plot(x, group[col] * length**power, "o", ms=2, label=f"L={length}")
    for ax, label in zip(axes, (r"$P_\infty L^{5/48}$", r"$\chi_A L^{-43/24}$", r"$\chi_B L^{-43/24}$")):
        ax.set(xlabel=r"$(p-p_c)L^{3/4}$", ylabel=label); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    save(fig, "criticality_finite_size_scaling_revised.png")

    critical = histogram[np.isclose(histogram.p_ratio, 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, rule in zip(axes, ("largest_excluded", "all_spanning_excluded")):
        for length, group in critical[critical.exclusion == rule].groupby("length"):
            ax.loglog(group.cluster_size, group.n_s, ".", ms=2, label=f"L={length}")
        ax.set(xlabel="cluster size s", ylabel=r"$n_s=C_s/(ML^2)$", title=rule)
        ax.legend(fontsize=7)
    save(fig, "criticality_cluster_distributions_revised.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, rule in zip(axes, ("largest_excluded", "all_spanning_excluded")):
        subset = critical[critical.exclusion == rule]
        for length, group in subset.groupby("length"):
            ax.loglog(group.cluster_size, group.C_s, ".", ms=2, label=f"L={length}")
        for row in fits[(fits.exclusion == rule) & (fits.count_threshold == 10) &
                        (fits.status == "ok")].itertuples():
            ax.axvline(row.s_max_fit, color="gray", alpha=.25)
            support = np.arange(int(row.s_min), int(row.s_max_fit) + 1)
            if len(support) > 1:
                weights = support ** (-row.alpha_pdf)
                ax.loglog(support, row.tail_count * weights / weights.sum(), "-", lw=1)
            observed = subset[subset.length == row.length]
            if not observed.empty and np.isfinite(row.cutoff_s_c):
                upper = int(observed.cluster_size.max())
                support = np.arange(int(row.s_min), upper + 1)
                weights = support ** (-row.cutoff_alpha_pdf) * np.exp(-support / row.cutoff_s_c)
                tail_count = observed[observed.cluster_size >= row.s_min].C_s.sum()
                ax.loglog(support, tail_count * weights / weights.sum(), "--", lw=1)
        ax.axhline(10, color="black", ls="--", alpha=.5)
        ax.set(xlabel="cluster size s", ylabel="raw count $C_s$",
               title=f"{rule}: solid finite power law, dashed cutoff")
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    save(fig, "criticality_cluster_tail_fits_revised.png")
