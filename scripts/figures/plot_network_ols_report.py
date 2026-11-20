"""Replot saved network CCDF interval OLS at each topology's critical position."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FITS = ROOT / "results/loglog_ols_comparison/ols_summary_v2.csv"
RETURNS = ROOT / "results/tables/network_market_returns.csv.gz"
OUTPUT = ROOT / "results/figures/network_ols_fits_pc.png"

with FITS.open(newline="") as fh:
    fit_rows = {
        row["label"].split()[0].lower(): row
        for row in csv.DictReader(fh)
        if row["kind"] == "network" and row["label"].endswith("p/pc=1")
    }

raw = pd.read_csv(RETURNS, usecols=["topology", "relative_p", "return", "z"])
fig, axes = plt.subplots(2, 2, figsize=(9.3, 7.3), layout="constrained")
for ax, name in zip(axes.flat, ["lattice", "ws", "er", "ba"]):
    group = raw.loc[
        (raw["topology"] == name) & (raw["relative_p"] == 1.0)
    ]
    z = np.abs(group.loc[group["return"] != 0, "z"].to_numpy())
    z = z[np.isfinite(z) & (z > 0)]
    x, count = np.unique(z, return_counts=True)
    ccdf = np.cumsum(count[::-1])[::-1] / len(z)
    sampled = np.unique(np.linspace(0, len(x) - 1, min(len(x), 1600)).astype(int))
    ax.loglog(x[sampled], ccdf[sampled], ".", ms=2.3, color="#436c92",
              label="Empirical CCDF")

    fit = fit_rows[name]
    xmin, xmax = float(fit["xmin"]), float(fit["xmax"])
    b, intercept = float(fit["slope"]), float(fit["intercept"])
    grid = np.geomspace(xmin, xmax, 100)
    ax.loglog(grid, 10**intercept * grid**b, color="#bd4939", lw=2,
              label="Interval OLS")
    ax.axvspan(xmin, xmax, color="#e4a399", alpha=0.18)
    ax.set_title(
        f"{name.upper()}: [{xmin:.3g}, {xmax:.3g}], "
        + rf"$\tau={1-b:.3f}$, $R^2={float(fit['r2']):.3f}$",
        fontsize=10,
    )
    ax.text(0.04, 0.05,
            f"{float(fit['span_decades']):.2f} decades; {fit['tier']} fit",
            transform=ax.transAxes, fontsize=8)
    ax.set_xlabel(r"Absolute standardized return $|z|$")
    ax.set_ylabel(r"$P(|z|\geq x\mid r\neq 0)$")
    ax.grid(which="major", color="#dddddd", lw=0.6)
axes.flat[0].legend(fontsize=8, loc="upper right")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=210)
print(OUTPUT)
