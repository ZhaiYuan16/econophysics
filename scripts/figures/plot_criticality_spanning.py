"""Main-report criticality figures using the spanning-cluster exclusion rule."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[2] / "results"
DATA = HERE / "criticality_revision_20261113/full/tables"
VALIDATION = HERE / "criticality_validation_20261113"
FIGURES = HERE / "figures"
P_C = 0.592746
RULE = "all_spanning_excluded"

FIGURES.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(DATA / "summary.csv")
peaks = pd.read_csv(DATA / "pseudocritical.csv")
fits = pd.read_csv(VALIDATION / "pseudocritical_extrapolation_gridaware.csv")
histogram = pd.read_csv(DATA / "cluster_histogram.csv")
correlation = pd.read_csv(HERE / "tables/correlation_length_preview.csv")
peaks = peaks[peaks.exclusion == RULE].sort_values("length")
fits = fits[fits.exclusion == RULE]
histogram = histogram[(histogram.exclusion == RULE) & (histogram.p_ratio == 1)]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11})
colors = dict(zip(sorted(summary.length.unique()), plt.cm.tab10.colors[:5]))


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_curves(ax, column):
    for length, group in summary.groupby("length", sort=True):
        group = group.sort_values("p")
        ax.plot(group.p, group[column], "o-", ms=2.1, lw=1.25,
                color=colors[length], label=rf"$L={length}$")
    ax.axvline(P_C, color="#555555", ls="--", lw=1)
    ax.set_xlabel("Occupancy probability $p$")
    ax.grid(alpha=0.22)


fig, axes = plt.subplots(2, 2, figsize=(9, 6.0), layout="constrained")
for ax, column, ylabel in (
    (axes[0, 0], "p_infinity_mean", r"$P_\infty=\langle s_{\max}/L^2\rangle$"),
    (axes[0, 1], "chi_all_spanning_excluded", r"$\chi_{\rm span}$"),
    (axes[1, 1], "span_x_mean", r"$P_x$"),
):
    draw_curves(ax, column)
    ax.set_ylabel(ylabel)
for length, group in correlation.groupby("L", sort=True):
    group = group.sort_values("p")
    axes[1, 0].plot(group.p, group.xi, "o-", ms=2.8, lw=1.3,
                    color=colors[length])
axes[1, 0].set(xlabel="Occupancy probability $p$",
               ylabel=r"Correlation length $\xi$ (lattice sites)",
               xlim=(summary.p.min() - .005, summary.p.max() + .005),
               yscale="log", ylim=(1, 160))
axes[1, 0].axvline(P_C, color="#555555", ls="--", lw=1)
axes[1, 0].grid(alpha=0.22)
axes[0, 0].legend(fontsize=8, ncol=2)
save(fig, "criticality_observables.png")


fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
x = peaks.length.to_numpy(float) ** (-0.75)
y = peaks.p_grid_peak.to_numpy(float)
yerr = np.vstack([y - peaks.p_ci_low.to_numpy(float),
                  peaks.p_ci_high.to_numpy(float) - y])
ax.errorbar(x, y, yerr=yerr, fmt="o", color="#246394", capsize=3,
            label="Spanning-excluded peak (95% CI)")
xx = np.linspace(0, max(x) * 1.05, 200)
for subset, color, label in (
    ("all_5", "#cc6946", "All five sizes"),
    ("large_3", "#488656", r"$L\geq128$"),
):
    row = fits[fits.subset == subset].iloc[0]
    ax.plot(xx, row.point_intercept + row.point_slope * xx,
            color=color, lw=1.8, label=label)
ax.axhline(P_C, color="black", ls="--", lw=1.2,
           label=rf"Theory $p_c={P_C}$")
ax.set(xlabel=r"$L^{-3/4}$", ylabel=r"Susceptibility peak $p_\chi^*(L)$")
ax.grid(alpha=0.22)
ax.legend(fontsize=8, loc="upper left")
save(fig, "criticality_pseudocritical.png")


fig = plt.figure(figsize=(9, 6.2), layout="constrained")
grid = fig.add_gridspec(2, 2)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :])]
for length, group in summary.groupby("length", sort=True):
    x = (group.p.to_numpy(float) - P_C) * length ** 0.75
    axes[0].plot(x, group.p_infinity_mean * length ** (5 / 48),
                 "o", ms=3, color=colors[length], label=rf"$L={length}$")
    axes[1].plot(x, group.chi_all_spanning_excluded * length ** (-43 / 24),
                 "o", ms=3, color=colors[length])
    geometry = correlation[correlation.L == length].sort_values("p")
    x_geometry = (geometry.p.to_numpy(float) - P_C) * length ** 0.75
    axes[2].plot(x_geometry, geometry.xi_over_L, "o-", ms=3, lw=1.2,
                 color=colors[length])
axes[0].set_ylabel(r"$P_\infty L^{5/48}$")
axes[1].set_ylabel(r"$\chi_{\rm span}L^{-43/24}$")
axes[2].set_ylabel(r"$\xi/L$")
for ax in axes:
    ax.set_xlabel(r"$(p-p_c)L^{3/4}$")
    ax.set_xlim(-3, 3)
    ax.axvline(0, color="#555555", ls="--", lw=.8)
    ax.grid(alpha=0.22)
axes[0].legend(fontsize=8)
save(fig, "criticality_finite_size_scaling.png")


fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
for length, group in histogram.groupby("length", sort=True):
    ax.loglog(group.cluster_size, group.n_s, ".", ms=2.2,
              color=colors[length], label=rf"$L={length}$")
ax.set(xlabel="Finite-cluster size $s$", ylabel=r"$n_s=C_s/(ML^2)$")
ax.grid(which="major", alpha=0.22)
ax.legend(fontsize=8)
save(fig, "criticality_cluster_distributions.png")

print("Saved B-rule criticality figures to", FIGURES)
