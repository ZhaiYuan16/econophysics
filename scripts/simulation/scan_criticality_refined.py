#!/usr/bin/env python3
"""Run revised square-lattice criticality smoke, pilot, or full scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import yaml

from local_market.criticality_revision import (
    crossings, pooled_histogram, pseudocritical, ratio_grid, scan, summarize,
    tail_fits,
)
from local_market.criticality_revision_plots import make_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/criticality_refined.yaml"))
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/criticality_revision_20261113"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    p_c = float(config["p_c"])
    seed = int(config["seed"])
    step = float(config["ratio_step"])
    low, high = map(float, config["spanning_window"])
    common = [round(float(x), 6) for x in np.arange(low, high + step / 2, step)]
    if 1.0 not in common:
        raise ValueError("common window must contain p_c")
    if args.mode == "smoke":
        lengths = [8, 12]
        ratios = {length: (0.98, 1.0, 1.02) for length in lengths}
        counts = {length: 4 for length in lengths}
        common_used = list(ratios[lengths[0]])
        bootstraps = 100
    elif args.mode == "pilot":
        lengths = [512]
        common_used = list(map(float, config["pilot_ratios"]))
        ratios = {512: tuple(common_used)}
        counts = {512: int(config["pilot_samples"])}
        bootstraps = int(config["bootstrap_replicates"])
    else:
        lengths = list(map(int, config["lengths"]))
        estimates = pd.read_csv(config["baseline_estimates"])
        centers = {int(row.length): float(row.p_ratio_chi_peak) for row in estimates.itertuples()}
        centers[512] = float(config["peak_center_512"])
        ratios = {length: ratio_grid(config["coarse_ratios"], centers[length],
            float(config["peak_half_width"]), config["spanning_window"], step)
            for length in lengths}
        counts = {int(key): int(value) for key, value in config["samples_per_length"].items()}
        common_used = common
        bootstraps = int(config["bootstrap_replicates"])
    out = args.output_root / args.mode
    out.mkdir(parents=True, exist_ok=False)
    tables, figures, metadata = (out / name for name in ("tables", "figures", "metadata"))
    for path in (tables, figures, metadata):
        path.mkdir()
    (metadata / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    started = datetime.now(timezone.utc)
    streamed_histogram = tables / "configuration_cluster_counts.csv" if args.mode == "full" else None
    scanned = scan(lengths, ratios, counts, p_c, seed, streamed_histogram)
    extension_notes = []
    if args.mode == "full":
        local_bounds = {length: [centers[length] - float(config["peak_half_width"]),
                                 centers[length] + float(config["peak_half_width"])]
                        for length in lengths}
        for _ in range(3):
            additions = {}
            for length in lengths:
                group = scanned.samples[scanned.samples.length == length]
                pooled = group.groupby("p_ratio").agg({
                    "largest_excluded_m1": "sum", "largest_excluded_m2": "sum",
                    "all_spanning_excluded_m1": "sum", "all_spanning_excluded_m2": "sum"})
                left, right = local_bounds[length]
                desired = set()
                for rule in ("largest_excluded", "all_spanning_excluded"):
                    curve = pooled[f"{rule}_m2"] / pooled[f"{rule}_m1"].replace(0, np.nan)
                    peak = float(curve.fillna(0).idxmax())
                    if peak <= left + 1.5 * step:
                        desired.update(round(float(x), 6) for x in np.arange(left - .02, left, step))
                    if peak >= right - 1.5 * step:
                        desired.update(round(float(x), 6) for x in np.arange(right + step, right + .02 + step / 2, step))
                additions[length] = tuple(sorted(desired - set(ratios[length])))
            if not any(additions.values()):
                break
            extra_lengths = [length for length in lengths if additions[length]]
            extra = scan(extra_lengths, additions, counts, p_c, seed, streamed_histogram)
            scanned.samples = pd.concat([scanned.samples, extra.samples], ignore_index=True)
            scanned.histograms = pd.concat([scanned.histograms, extra.histograms], ignore_index=True)
            scanned.pooled_counts = pd.concat([scanned.pooled_counts, extra.pooled_counts], ignore_index=True)
            scanned.runtime_seconds += extra.runtime_seconds
            for length in extra_lengths:
                ratios[length] = tuple(sorted(set(ratios[length]) | set(additions[length])))
                local_bounds[length][0] = min(local_bounds[length][0], min(additions[length]))
                local_bounds[length][1] = max(local_bounds[length][1], max(additions[length]))
                extension_notes.append({"length": length, "added_ratios": list(additions[length])})
    summary = summarize(scanned.samples, bootstraps, seed + 1)
    peaks = pseudocritical(scanned.samples, summary, bootstraps, seed + 2)
    cross = crossings(scanned.samples, lengths, common_used, p_c, bootstraps, seed + 3)
    histogram = pooled_histogram(scanned.histograms, scanned.samples, scanned.pooled_counts)
    if args.mode == "full":
        fits = tail_fits(scanned.histograms, 1.0, config["tail_count_thresholds"],
                         int(config["tail_bootstrap_replicates"]), seed + 4)
    else:
        fits = pd.DataFrame(columns=["length", "exclusion", "count_threshold", "status"])
    frames = {"configuration_observables": scanned.samples,
              "configuration_cluster_counts": scanned.histograms,
              "summary": summary, "pseudocritical": peaks, "crossings": cross,
              "cluster_histogram": histogram, "cluster_tail_fits": fits}
    for name, frame in frames.items():
        if name == "configuration_cluster_counts" and streamed_histogram is not None:
            continue
        frame.to_csv(tables / f"{name}.csv", index=False)
    make_figures(summary, peaks, cross, histogram, fits, p_c, figures)
    ended = datetime.now(timezone.utc)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_files = [Path(__file__), Path("src/local_market/criticality_revision.py"),
                    Path("src/local_market/criticality_revision_plots.py"),
                    Path("src/local_market/percolation.py"), Path("src/local_market/statistics.py")]
    meta = {"mode": args.mode, "started_utc": started.isoformat(), "ended_utc": ended.isoformat(),
            "scan_wall_seconds": scanned.runtime_seconds,
            "total_wall_seconds": (ended - started).total_seconds(),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "sample_count": len(scanned.samples), "grid_points_by_length":
            {str(length): len(ratios[length]) for length in lengths},
            "peak_window_extensions": extension_notes,
            "seed": seed, "bootstrap_replicates": bootstraps, "code_head": revision,
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in source_files},
            "definitions": {"p_infinity": "s_max/L**2, all configurations",
                "chi_largest_excluded": "pooled sum(s^2)/sum(s), one largest cluster per configuration excluded",
                "chi_all_spanning_excluded": "pooled sum(s^2)/sum(s), all open-boundary spanning clusters excluded",
                "span_x": "left-to-right", "span_y": "top-to-bottom",
                "span_any": "x or y", "span_both": "x and y",
                "n_s": "C_s/(M*L**2)"},
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__}}
    (metadata / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    manifest = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(out.rglob("*")) if path.is_file()}
    (metadata / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(out), "scan_seconds": scanned.runtime_seconds,
                      "total_seconds": meta["total_wall_seconds"],
                      "max_rss_kib": meta["max_rss_kib"], "sample_count": len(scanned.samples)}, indent=2))


if __name__ == "__main__":
    main()
