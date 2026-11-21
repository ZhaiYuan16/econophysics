"""Exploratory second-moment correlation length for open site percolation.

Recreates the original occupancy configurations where the p grid overlaps the
formal scan, then measures each nonspanning cluster's squared radius of
gyration. The CSV also supplies the correlation-length panel in the report's
finite-size scaling figure; the four-panel PNG remains an exploratory view.
"""

from pathlib import Path
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.local_market.networks import sample_lattice_occupancy
from src.local_market.percolation import identify_lattice_clusters

P_C = 0.592746
SEED = 20261113
RATIOS = (.3, .5, .8, .9, .94, .98, .99, .994, .998, 1.0,
          1.002, 1.006, 1.01, 1.02, 1.04, 1.06, 1.1, 1.2, 1.4)
SAMPLES = {32: 300, 64: 250, 128: 200, 256: 200, 512: 200}
OUT_CSV = ROOT / "results/tables/correlation_length_preview.csv"
OUT_PNG = ROOT / "results/figures/correlation_length_preview.png"


def geometry_moments(configuration):
    """Return Σ s_i² R_g,i², Σ s_i² and Σ s_i over nonspanning clusters."""
    labels = configuration.labels
    sizes = configuration.sizes.astype(np.float64)
    if not sizes.size:
        return 0.0, 0.0, 0.0
    row, col = np.nonzero(labels)
    ids = labels[row, col]
    n = len(sizes) + 1
    mean_row = np.bincount(ids, weights=row, minlength=n)[1:] / sizes
    mean_col = np.bincount(ids, weights=col, minlength=n)[1:] / sizes
    mean_radius2 = np.bincount(
        ids, weights=row.astype(float) ** 2 + col.astype(float) ** 2,
        minlength=n,
    )[1:] / sizes
    gyration2 = np.maximum(mean_radius2 - mean_row**2 - mean_col**2, 0)
    keep = np.ones(len(sizes), dtype=bool)
    if configuration.spanning_labels:
        keep[np.asarray(configuration.spanning_labels) - 1] = False
    finite_sizes = sizes[keep]
    weight = finite_sizes**2
    return float(np.dot(weight, gyration2[keep])), float(weight.sum()), float(finite_sizes.sum())


observed = pd.read_csv(
    ROOT / "results/criticality_revision_20261113/full/tables/configuration_observables.csv",
    usecols=["length", "p_ratio", "configuration_id", "largest_size", "span_x",
             "all_spanning_excluded_m1", "all_spanning_excluded_m2"],
)
observed["ratio_key"] = np.rint(observed.p_ratio * 1_000_000).astype(int)
observed = observed.set_index(["length", "ratio_key", "configuration_id"])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
rows = []
checked = 0
for length, count in SAMPLES.items():
    for ratio in RATIOS:
        p = P_C * ratio
        config_values = []
        for sample in range(count):
            key = (length, round(ratio * 1_000_000), sample)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, *key]))
            config = identify_lattice_clusters(
                sample_lattice_occupancy(length, p, rng), boundary="open"
            )
            numerator, second, first = geometry_moments(config)
            if key in observed.index:
                reference = observed.loc[key]
                assert config.largest_size == int(reference.largest_size)
                assert int(config.spans_horizontal) == int(reference.span_x)
                assert int(first) == int(reference.all_spanning_excluded_m1)
                assert int(second) == int(reference.all_spanning_excluded_m2)
                checked += 1
            config_values.append((numerator, second, first,
                                  config.p_infinity, int(config.spans_horizontal)))

        values = np.asarray(config_values, dtype=float)
        numerator, second, first = values[:, :3].sum(axis=0)
        xi = float(np.sqrt(numerator / second)) if second > 0 else np.nan
        chi = float(second / first) if first > 0 else np.nan
        bootstrap = np.random.default_rng(
            np.random.SeedSequence([SEED, length, round(ratio * 1_000_000), 999])
        )
        indices = bootstrap.integers(0, count, size=(400, count))
        boot_num = values[indices, 0].sum(axis=1)
        boot_den = values[indices, 1].sum(axis=1)
        boot_xi = np.sqrt(np.divide(
            boot_num, boot_den, out=np.full(400, np.nan), where=boot_den > 0
        ))
        valid = boot_xi[np.isfinite(boot_xi)]
        low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        rows.append(dict(
            L=length, p_ratio=ratio, p=p, configurations=count,
            xi=xi, xi_ci_low=low, xi_ci_high=high, xi_over_L=xi / length,
            chi_spanning_excluded=chi, p_infinity=values[:, 3].mean(),
            span_x_probability=values[:, 4].mean(),
        ))
    print(f"Finished L={length} ({count} configurations per p)", flush=True)

with OUT_CSV.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

data = pd.DataFrame(rows)
summary = pd.read_csv(ROOT / "results/criticality_revision_20261113/full/tables/summary.csv")
colors = dict(zip(SAMPLES, plt.cm.tab10.colors[:5]))
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for length, group in data.groupby("L", sort=True):
    color = colors[length]
    axes[0, 0].plot(group.p, group.xi, "o-", ms=3, lw=1.5,
                    color=color, label=rf"$L={length}$")
    axes[0, 0].fill_between(group.p, group.xi_ci_low, group.xi_ci_high,
                            color=color, alpha=.11)
    axes[0, 1].plot(group.p, group.xi_over_L, "o-", ms=3, lw=1.5,
                    color=color, label=rf"$L={length}$")

for ax, ylabel in zip(axes[0],
                      (r"Correlation length $\xi$ (lattice sites)", r"$\xi/L$")):
    ax.set(xlabel="Occupancy probability $p$", ylabel=ylabel, xlim=(0.15, 0.85))
    ax.axvline(P_C, color="#555555", ls="--", lw=1)
    ax.grid(alpha=.22)
axes[0, 0].legend(fontsize=8, ncol=2)

for ax, length in zip(axes[1], (128, 256)):
    group = data[data.L == length]
    formal = summary[summary.length == length].sort_values("p")
    ax.plot(group.p, group.xi / group.xi.max(), "o-", ms=3, lw=1.8,
            label=r"$\xi/\max\xi$")
    ax.plot(formal.p, formal.chi_all_spanning_excluded / formal.chi_all_spanning_excluded.max(),
            "-", lw=1.7, label=r"$\chi/\max\chi$")
    ax.plot(formal.p, formal.p_infinity_mean, "-", lw=1.6, label=r"$P_\infty$")
    ax.plot(formal.p, formal.span_x_mean, "-", lw=1.6, label=r"$P_x$")
    ax.axvline(P_C, color="#555555", ls="--", lw=1)
    ax.set(xlabel="Occupancy probability $p$", ylabel="Normalized value / probability",
           title=rf"Comparison at $L={length}$", xlim=(0.48, 0.66), ylim=(-.04, 1.08))
    ax.grid(alpha=.22)
    ax.legend(fontsize=8)

fig.savefig(OUT_PNG, dpi=210)
plt.close(fig)
print(f"Checked {checked} reconstructed configurations against the original scan")
print(OUT_PNG)
print(OUT_CSV)
