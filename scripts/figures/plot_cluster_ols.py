"""Finite-cluster CCDF interval OLS for the CBmodel report."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] / "results"
SOURCE = ROOT / "criticality_revision_20261113/full/tables/cluster_histogram.csv"
FIGURE = ROOT / "figures/criticality_cluster_ols_intervals.png"
TABLE = ROOT / "tables/cluster_ols_intervals.csv"

FIGURE.parent.mkdir(parents=True, exist_ok=True)
TABLE.parent.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(SOURCE)
data = data[(data.p_ratio == 1) & (data.exclusion == "all_spanning_excluded")]
rows = []
fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
panel = 0

for L, group in data.groupby("length", sort=True):
    x = group.cluster_size.to_numpy(float)
    counts = group.C_s.to_numpy(float)
    remaining = np.cumsum(counts[::-1])[::-1]
    ccdf = remaining / counts.sum()
    eligible = (x >= 5) & (remaining >= 100)
    xe, ye, re = x[eligible], ccdf[eligible], remaining[eligible]
    indices = np.unique(np.clip(
        np.searchsorted(xe, np.geomspace(xe[0], xe[-1], 120)),
        0, len(xe) - 1,
    ))
    xs, ys, rs = xe[indices], ye[indices], re[indices]
    lx, ly = np.log10(xs), np.log10(ys)
    candidates = []
    for i in range(len(xs) - 11):
        for j in range(i + 11, len(xs)):
            span = lx[j] - lx[i]
            if span < 0.5 or rs[i] - rs[j] < 100:
                continue
            slope, intercept = np.polyfit(lx[i:j + 1], ly[i:j + 1], 1)
            fitted = intercept + slope * lx[i:j + 1]
            target = ly[i:j + 1]
            r2 = 1 - np.sum((target - fitted) ** 2) / np.sum((target - target.mean()) ** 2)
            mid = (i + j) // 2
            left = np.polyfit(lx[i:mid + 1], ly[i:mid + 1], 1)[0]
            right = np.polyfit(lx[mid:j + 1], ly[mid:j + 1], 1)[0]
            drift = abs(left - right) / abs(slope)
            if slope < 0 and r2 >= 0.98 and drift <= 0.25:
                candidates.append((span, r2, -drift, i, j, slope, intercept))
    if not candidates:
        raise RuntimeError(f"No admissible interval for L={L}")
    span, r2, negative_drift, i, j, slope, intercept = max(candidates)
    low, high = xs[i], xs[j]
    tau = 1 - slope
    rows.append(dict(
        L=L, p=0.592746, exclusion="all_spanning_excluded", n_clusters=int(counts.sum()),
        xmin=low, xmax=high, span_decades=span, n_ols_points=j - i + 1,
        n_interval=int(counts[(x >= low) & (x <= high)].sum()),
        n_above_upper=int(rs[j]),
        ccdf_slope=slope, pdf_tau_effective=tau, r2=r2,
        half_slope_relative_difference=-negative_drift,
    ))

    if L == 32:
        continue  # Keep this result in the table so the four larger-size panels stay legible.
    ax = axes.flat[panel]
    panel += 1
    sampled = np.unique(np.linspace(0, len(x) - 1, min(len(x), 1100)).astype(int))
    ax.loglog(x[sampled], ccdf[sampled], ".", ms=2, color="#426e9b",
              label="Empirical CCDF")
    line_x = np.geomspace(low, high, 100)
    ax.loglog(line_x, 10 ** intercept * line_x ** slope,
              color="#c84b42", lw=2, label="Interval OLS")
    ax.axvspan(low, high, color="#e5a69e", alpha=0.18)
    ax.set_title(rf"$L={L}$, $\tau_{{\rm eff}}={tau:.3f}$", fontsize=12)
    ax.set_xlabel("Finite-cluster size $s$", fontsize=11)
    ax.set_ylabel(r"$P(S\geq s)$", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(which="major", color="#dddddd", lw=0.6)

axes.flat[0].legend(fontsize=8, loc="upper right")
fig.savefig(FIGURE, dpi=210)
with TABLE.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(FIGURE)
print(TABLE)
