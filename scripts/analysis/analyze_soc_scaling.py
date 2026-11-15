#!/usr/bin/env python3
"""Independent finite-size avalanche scaling analysis from saved tail fits."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import linregress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    durations = pd.read_csv(root/"duration_diagnostics.csv")
    qualified = durations[durations.status == "ok"] if "status" in durations else pd.DataFrame()
    pd.DataFrame([dict(status="insufficient_duration_dynamic_range" if len(qualified) < 3
        else "duration_cutoff_fit_required", qualified_sizes=len(qualified), required=3)]).to_csv(
        root/"duration_scaling.csv", index=False)
    fits = pd.read_csv(root/"tail_fits.csv")
    if "model" not in fits:
        cuts = pd.DataFrame()
    else:
        cuts = fits[(fits.model == "cutoff") & (fits.status == "ok") &
                    (fits.boundary == False)].sort_values("length")
    if len(cuts) < 3:
        pd.DataFrame([dict(status="insufficient_identified_cutoffs", size_count=len(cuts),
            required=3)]).to_csv(root/"size_scaling.csv", index=False)
        return
    reg = linregress(np.log(cuts.length), np.log(cuts.cutoff))
    pd.DataFrame([dict(status="preliminary_three_size_fit", D_s=reg.slope,
        standard_error=reg.stderr, r2=reg.rvalue**2, size_count=len(cuts))]).to_csv(
            root/"size_scaling.csv", index=False)
    figures = root/"figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(cuts.length, cuts.cutoff, "o")
    ax.plot(cuts.length, np.exp(reg.intercept)*cuts.length**reg.slope)
    ax.set(xscale="log", yscale="log", xlabel="L", ylabel="S_c")
    fig.savefig(figures/"size_scaling.png", dpi=160)
    plt.close(fig)
    cfg = yaml.safe_load((root/"config.yaml").read_text())
    events = pd.read_csv(root/"cascade"/"events.csv.gz")
    pure = fits[fits.model == "power"].set_index("length")
    tau = pure.loc[cuts.length, "exponent"].mean()
    fig, ax = plt.subplots()
    for row in cuts.itertuples():
        burn = cfg.get("burn_in_by_length", {}).get(int(row.length), cfg["burn_in"])
        sub = events[(events.length == row.length) & (events.step >= burn)]
        counts = sub["size"].value_counts().sort_index()
        x = counts.index.to_numpy()
        y = counts.to_numpy()/len(sub)
        ax.plot(x/row.length**reg.slope, x**tau*y, ".", label=f"L={row.length}")
    ax.set(xscale="log", yscale="log", xlabel="S/L^Ds", ylabel="S^tau P(S;L)")
    ax.legend()
    fig.savefig(figures/"size_collapse_diagnostic.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
