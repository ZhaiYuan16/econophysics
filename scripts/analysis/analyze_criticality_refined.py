#!/usr/bin/env python3
"""Add scaling and diagnostic tables to a completed refined scan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def analyze(root: Path) -> None:
    tables = root / "tables"
    samples = pd.read_csv(tables / "configuration_observables.csv")
    summary = pd.read_csv(tables / "summary.csv")
    histogram = pd.read_csv(tables / "cluster_histogram.csv")
    fits = pd.read_csv(tables / "cluster_tail_fits.csv")

    critical = summary[np.isclose(summary.p_ratio, 1.0)].sort_values("length")
    if len(critical) < 3:
        raise ValueError("at least three lattice sizes at p_c are needed for scaling")
    scaling_rows = []
    for name, column, theory in (
        ("p_infinity", "p_infinity_mean", -5 / 48),
        ("chi_largest_excluded", "chi_largest_excluded", 43 / 24),
        ("chi_all_spanning_excluded", "chi_all_spanning_excluded", 43 / 24),
    ):
        regression = stats.linregress(np.log(critical.length), np.log(critical[column]))
        scaling_rows.append({"observable": name, "p": float(critical.p.iloc[0]),
            "fitted_exponent": regression.slope, "exponent_standard_error": regression.stderr,
            "theoretical_exponent": theory, "r_squared": regression.rvalue**2,
            "size_count": len(critical), "fit_method": "log-log OLS at theoretical p_c"})
    pd.DataFrame(scaling_rows).to_csv(tables / "scaling_exponents.csv", index=False)

    rng = np.random.default_rng(20261113 + 5)
    symmetry_rows = []
    for (length, ratio, p), group in samples.groupby(["length", "p_ratio", "p"]):
        difference = (group.span_x - group.span_y).to_numpy(dtype=float)
        boot = difference[rng.integers(0, len(group), size=(1000, len(group)))].mean(axis=1)
        low, high = np.quantile(boot, [.025, .975])
        symmetry_rows.append({"length": length, "p_ratio": ratio, "p": p,
            "p_x_minus_p_y": difference.mean(), "ci_low": low, "ci_high": high,
            "configurations": len(group), "bootstrap_replicates": 1000})
    pd.DataFrame(symmetry_rows).to_csv(tables / "direction_symmetry.csv", index=False)

    maxima = histogram[np.isclose(histogram.p_ratio, 1.0)].groupby(
        ["length", "exclusion"]).cluster_size.max()
    diagnostics = []
    for row in fits.itertuples():
        observed_max = int(maxima.loc[(row.length, row.exclusion)])
        upper_bound = observed_max * 100
        diagnostics.append({"length": row.length, "exclusion": row.exclusion,
            "count_threshold": row.count_threshold,
            "full_tail_observed_max": observed_max,
            "cutoff_scale_optimizer_upper_bound": upper_bound,
            "cutoff_scale_at_upper_bound": bool(row.cutoff_s_c >= .99 * upper_bound),
            "cutoff_scale_status": "not_identified" if row.cutoff_s_c >= .99 * upper_bound
                                   else "finite_optimum"})
    pd.DataFrame(diagnostics).to_csv(tables / "cutoff_diagnostics.csv", index=False)

    metadata = root / "metadata"
    provenance = {"analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "figures_regenerated_after_scan": True,
                  "figure_source_sha256": hashlib.sha256(
                      Path("src/local_market/criticality_revision_plots.py").read_bytes()).hexdigest(),
                  "inputs": ["configuration_observables.csv", "summary.csv",
                             "cluster_histogram.csv", "cluster_tail_fits.csv"],
                  "direction_bootstrap_unit": "independent occupancy configuration",
                  "cutoff_flag_rule": "fitted s_c >= 0.99 * optimizer upper bound (100 * maximum observed cluster size)"}
    (metadata / "postanalysis.json").write_text(json.dumps(provenance, indent=2) + "\n")
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*")) if path.is_file() and path.name != "sha256_manifest.json"}
    (metadata / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(scaling_rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("results/criticality_revision_20261113/full"))
    analyze(parser.parse_args().root)
