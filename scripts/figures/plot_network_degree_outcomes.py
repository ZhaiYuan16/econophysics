"""Plot crash frequencies and the saved piecewise log-log OLS exponents."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results/tables/network_metric_summary.csv"
STATS = ROOT / "results/downstream_audit_20261114/tables/network_revised_statistics.csv"
OLS = ROOT / "results/loglog_ols_comparison/ols_summary_v2.csv"
OUTPUT = ROOT / "results/figures/network_degree_outcomes.png"

with METRICS.open(newline="") as fh:
    metrics = {row["topology"]: row for row in csv.DictReader(fh)}
with STATS.open(newline="") as fh:
    stats = {
        row["topology"]: row
        for row in csv.DictReader(fh)
        if float(row["relative_p"]) == 1.0
    }
with OLS.open(newline="") as fh:
    fits = {
        row["label"].split()[0].lower(): row
        for row in csv.DictReader(fh)
        if row["kind"] == "network" and row["label"].endswith("p/pc=1")
    }

names = ["lattice", "ws", "er", "ba"]
labels = ["Lattice", "WS", "ER", "BA"]
x = np.arange(len(names))
colors = ["#315f8f", "#43835c", "#ca6131", "#9555a4"]
degree_sd = [float(metrics[name]["degree_std"]) for name in names]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")
for i, name in enumerate(names):
    row = stats[name]
    crash = 100 * float(row["negative_crash_probability"])
    crash_low = 100 * float(row["negative_crash_probability_ci_low"])
    crash_high = 100 * float(row["negative_crash_probability_ci_high"])
    fit = fits[name]
    tau = float(fit["pdf_tau"])
    axes[0].errorbar(
        i, crash, yerr=[[crash - crash_low], [crash_high - crash]],
        fmt="o", color=colors[i], capsize=4, markersize=8, lw=1.7,
    )
    axes[1].plot(
        i, tau, "o", color=colors[i], markersize=9,
        markerfacecolor="white" if fit["tier"] == "local" else colors[i],
        markeredgewidth=2,
    )
    axes[1].text(i, tau + 0.055, f"{float(fit['span_decades']):.2f} dec.",
                 ha="center", va="bottom", fontsize=8)

for ax in axes:
    ax.set_xticks(x, [f"{label}\n$\\sigma_k$={sd:.2f}" for label, sd in zip(labels, degree_sd)])
    ax.grid(axis="y", color="#dddddd", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.5, 3.5)
axes[0].set_ylabel(r"Negative crash frequency $P(z<-5)$ (%)")
axes[1].set_ylabel(r"OLS effective PDF exponent $\tau=1-b$")
axes[0].set_ylim(0.28, 0.57)
axes[1].set_ylim(1.5, 3.25)
axes[1].text(0.98, 0.04, "Open marker: short local fit (BA)",
             transform=axes[1].transAxes, ha="right", fontsize=8)
fig.suptitle(
    r"Fixed $a=0.01$, each network at its own $p_\chi^*$",
    fontsize=13,
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=220)
print(OUTPUT)
