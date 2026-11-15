#!/usr/bin/env python3
"""Recompute SOC diagnostics from versioned path and event CSV files."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, linregress
from local_market.soc_tail import bootstrap_paths, fit_tail, select_xmin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    cfg = yaml.safe_load((root/"config.yaml").read_text())
    boot_count = cfg["bootstrap_replicates"]
    if "smoke" in root.name:
        boot_count = 10
    win = pd.read_csv(root/"critical_windows.csv")
    recomputed = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/"paths.csv.gz").exists():
            continue
        raw = pd.read_csv(folder/"paths.csv.gz")
        for pid, group in raw.groupby("path_id", sort=False):
            length = int(group.length.iloc[0])
            burn = 20 if "smoke" in root.name else cfg.get("burn_in_by_length", {}).get(length, cfg["burn_in"])
            post = group[group.step >= burn]
            p = post.occupancy_after.to_numpy()
            half = len(p)//2
            bounds = win.set_index("length").loc[length]
            recomputed.append(dict(condition=folder.name, length=length,
                initial_occupancy=float(group.initial_occupancy.iloc[0]),
                replicate=int(group.replicate.iloc[0]), path_id=pid,
                seed=int(group.seed.iloc[0]), steps=len(group), burn_in=burn,
                mean_occupancy=p.mean(), occupancy_sd=p.std(ddof=1),
                drift=p[half:].mean()-p[:half].mean(),
                window_fraction=np.mean((p >= bounds.window_low)&(p <= bounds.window_high)),
                event_rate=np.mean(post.duration > 0),
                event_count=int((post.duration > 0).sum()),
                mean_event_size=post.loc[post.removed_nodes > 0, "removed_nodes"].mean()))
    summary = pd.DataFrame(recomputed)
    summary.to_csv(root/"path_summary_recomputed.csv", index=False)
    figure_dir = root/"figures"
    figure_dir.mkdir(exist_ok=True)
    summary.groupby(["condition", "length", "initial_occupancy"]).agg(
        paths=("path_id", "nunique"), mean_p=("mean_occupancy", "mean"),
        sd_p=("mean_occupancy", "std"), window_fraction=("window_fraction", "mean"),
        drift=("drift", "mean"), event_rate=("event_rate", "mean")
    ).reset_index().to_csv(root/"ensemble_summary.csv", index=False)
    distribution_rows = []
    tail_rows, scan_rows, boot_rows, duration_rows = [], [], [], []
    rng = np.random.default_rng(cfg["seed"]+771)
    for condition in ("baseline", "cascade"):
        folder = root/condition
        if not folder.exists():
            continue
        all_paths = pd.read_csv(folder/"paths.csv.gz")
        paths = all_paths
        events = pd.read_csv(folder/"events.csv.gz")
        if "step" not in events:
            events["step"] = pd.Series(dtype=int)
            events["length"] = pd.Series(dtype=int)
            events["size"] = pd.Series(dtype=int)
        paths = paths[paths.step >= paths.length.map(
            lambda l: 20 if "smoke" in root.name else cfg.get("burn_in_by_length", {}).get(int(l), cfg["burn_in"]))]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for l in sorted(paths.length.unique()):
            group = paths[paths.length == l]
            for p0 in sorted(group.initial_occupancy.unique()):
                sub = group[group.initial_occupancy == p0]
                axes[0].hist(sub.occupancy_after, bins=40, density=True, histtype="step",
                             label=f"L={l}, p0={p0}")
            for a in sorted(group.initial_occupancy.unique()):
                for b in sorted(group.initial_occupancy.unique()):
                    if a >= b:
                        continue
                    va = group[group.initial_occupancy == a].occupancy_after
                    vb = group[group.initial_occupancy == b].occupancy_after
                    distribution_rows.append(dict(condition=condition, length=l, initial_a=a,
                        initial_b=b, ks_descriptive=ks_2samp(va, vb).statistic,
                        note="pooled time points dependent; no iid p-value"))
            local_burn = 20 if "smoke" in root.name else cfg.get("burn_in_by_length", {}).get(int(l), cfg["burn_in"])
            ev = events[(events.length == l)&(events.step >= local_burn)]
            ev = ev[ev["size"] > 0]
            if len(ev):
                x = np.sort(ev["size"].to_numpy())
                axes[1].step(x, 1-np.arange(len(x))/len(x), label=f"L={l}")
                axes[2].hist(x, bins=np.geomspace(1, max(x)+1, 25), density=True,
                             histtype="step", label=f"L={l}")
            if condition != "cascade" or len(ev) == 0:
                continue
            duration_min, _, duration_status = select_xmin(ev.duration, 1000)
            duration_rows.append(dict(length=l, event_count=len(ev),
                max_duration=int(ev.duration.max()), status=duration_status,
                xmin=duration_min))
            upper = l*l
            xmin, scan, status = select_xmin(ev["size"], upper)
            for row in scan:
                scan_rows.append(dict(condition=condition, length=l, **row))
            if status != "ok":
                tail_rows.append(dict(condition=condition, length=l, status=status,
                                      positive_events=len(ev)))
                continue
            fits = [fit_tail(ev["size"], xmin, upper, model)
                    for model in ("power", "cutoff", "exponential")]
            for row in fits:
                tail_rows.append(dict(condition=condition, length=l, status=status,
                                      positive_events=len(ev), **row))
            path_values = {pid: ev.loc[ev.path_id == pid, "size"].to_numpy(dtype=int)
                           for pid in group.path_id.unique()}
            path_groups = {pid: float(sub.initial_occupancy.iloc[0])
                           for pid, sub in group.groupby("path_id")}
            for row in bootstrap_paths(path_values, upper, boot_count, rng,
                                       path_groups=path_groups):
                boot_rows.append(dict(condition=condition, length=l, **row))
        axes[0].set_xlabel("occupancy p")
        axes[1].set(xscale="log", yscale="log", xlabel="S", ylabel="CCDF")
        axes[2].set(xscale="log", yscale="log", xlabel="S", ylabel="PDF")
        for ax in axes:
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(figure_dir/f"{condition}_distributions.png", dpi=160)
        plt.close(fig)
        sample = all_paths[(all_paths.length == all_paths.length.max()) & (all_paths.replicate == 0)]
        fig, ax = plt.subplots(figsize=(9,3))
        for p0, group in sample.groupby("initial_occupancy"):
            ax.plot(group.step, group.occupancy_after, lw=.6, label=f"p0={p0}")
        w = win[win.length == paths.length.max()].iloc[0]
        ax.axhspan(w.window_low, w.window_high, color="gray", alpha=.2)
        ax.set(xlabel="true step", ylabel="occupancy p")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir/f"{condition}_trajectories.png", dpi=160)
        plt.close(fig)
    pd.DataFrame(distribution_rows).to_csv(root/"distribution_distances.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(root/"tail_fits.csv", index=False)
    pd.DataFrame(scan_rows).to_csv(root/"xmin_scan.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(root/"tail_bootstrap.csv", index=False)
    pd.DataFrame(duration_rows).to_csv(root/"duration_diagnostics.csv", index=False)
    if duration_rows:
        ev_all = pd.read_csv(root/"cascade"/"events.csv.gz")
        fig, ax = plt.subplots()
        for l, group in ev_all.groupby("length"):
            local_burn = cfg.get("burn_in_by_length", {}).get(int(l), cfg["burn_in"])
            durations = np.sort(group.loc[group.step >= local_burn, "duration"].to_numpy())
            if len(durations):
                ax.step(durations, 1-np.arange(len(durations))/len(durations), label=f"L={l}")
        ax.set(xlabel="relaxation rounds T", ylabel="empirical CCDF")
        ax.legend()
        fig.savefig(figure_dir/"duration_ccdf.png", dpi=160)
        plt.close(fig)
    if boot_rows:
        boot = pd.DataFrame(boot_rows)
        intervals = []
        for l, group in boot[boot.status == "ok"].groupby("length"):
            intervals.append(dict(length=l, successful=len(group),
                tau_low=group.tau.quantile(.025), tau_high=group.tau.quantile(.975),
                cutoff_low=group.cutoff.quantile(.025), cutoff_high=group.cutoff.quantile(.975),
                cutoff_boundary_fraction=group.cutoff_boundary.mean()))
        pd.DataFrame(intervals).to_csv(root/"tail_intervals.csv", index=False)
        fig, ax = plt.subplots()
        for l, group in boot[boot.status == "ok"].groupby("length"):
            ax.hist(group.tau, bins=25, histtype="step", label=f"L={l}")
        ax.set(xlabel="PDF exponent tau", ylabel="path bootstrap count")
        ax.legend()
        fig.savefig(figure_dir/"bootstrap_tau.png", dpi=160)
        plt.close(fig)
    if scan_rows:
        fig, ax = plt.subplots()
        for l, group in pd.DataFrame(scan_rows).groupby("length"):
            ax.plot(group.xmin, group.ks, label=f"L={l}")
        ax.set(xlabel="S_min", ylabel="CCDF KS distance")
        ax.legend()
        fig.savefig(figure_dir/"xmin_scan.png", dpi=160)
        plt.close(fig)
    fits = pd.DataFrame(tail_rows)
    if len(fits) and "model" in fits:
        for l, subset in fits[(fits.condition == "cascade") & (fits.status == "ok")].groupby("length"):
            if len(subset) != 3:
                continue
            xmin = int(subset.xmin.iloc[0])
            upper = int(l*l)
            raw = pd.read_csv(root/"cascade"/"events.csv.gz")
            x = raw[(raw.length == l) & (raw.step >= cfg.get("burn_in_by_length", {}).get(int(l), cfg["burn_in"]))]
            x = x.loc[x["size"] >= xmin, "size"].to_numpy(dtype=int)
            if len(x) == 0:
                continue
            support = np.arange(xmin, upper+1)
            fig, ax = plt.subplots()
            observed = np.sort(x)
            ax.step(observed, 1-np.arange(len(observed))/len(observed), label="empirical")
            for row in subset.itertuples():
                rate = 0 if row.model == "power" else 1/row.cutoff
                weights = support.astype(float)**(-row.exponent)*np.exp(-rate*(support-xmin))
                probs = weights/weights.sum()
                ax.plot(support, np.cumsum(probs[::-1])[::-1], label=row.model)
            ax.set(xscale="log", yscale="log", xlabel="avalanche size S", ylabel="CCDF")
            ax.legend()
            fig.savefig(figure_dir/f"L{l}_tail_models.png", dpi=160)
            plt.close(fig)
        cuts = fits[(fits.model == "cutoff") & (~fits.boundary.astype(bool))]
        if len(cuts) >= 3:
            reg = linregress(np.log(cuts.length), np.log(cuts.cutoff))
            pd.DataFrame([dict(exponent_Ds=reg.slope, r2=reg.rvalue**2,
                size_count=len(cuts), note="preliminary three-size regression")]).to_csv(
                    root/"size_scaling.csv", index=False)
            fig, ax = plt.subplots()
            ax.plot(cuts.length, cuts.cutoff, "o-")
            ax.set(xscale="log", yscale="log", xlabel="L", ylabel="S_c")
            fig.savefig(figure_dir/"size_scaling.png", dpi=160)
            plt.close(fig)
            exponent = fits[(fits.model == "power") & (fits.length.isin(cuts.length))].exponent.mean()
            fig, ax = plt.subplots()
            raw = pd.read_csv(root/"cascade"/"events.csv.gz")
            for row in cuts.itertuples():
                vals = raw[(raw.length == row.length) &
                    (raw.step >= cfg.get("burn_in_by_length", {}).get(int(row.length), cfg["burn_in"]))]["size"]
                counts = vals.value_counts().sort_index()
                x = counts.index.to_numpy()
                y = counts.to_numpy()/len(vals)
                ax.plot(x/row.length**reg.slope, x**exponent*y, ".", label=f"L={row.length}")
            ax.set(xscale="log", yscale="log", xlabel="S/L^Ds", ylabel="S^tau P(S;L)")
            ax.legend()
            fig.savefig(figure_dir/"size_collapse_diagnostic.png", dpi=160)
            plt.close(fig)
        else:
            pd.DataFrame([dict(status="insufficient_identified_cutoffs",
                identified_sizes=len(cuts), required_sizes=3)]).to_csv(
                    root/"size_scaling.csv", index=False)
    else:
        pd.DataFrame([dict(status="no_valid_tail_fits", identified_sizes=0,
            required_sizes=3)]).to_csv(root/"size_scaling.csv", index=False)


if __name__ == "__main__":
    main()
