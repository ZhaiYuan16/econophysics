"""Read-only audit of completed square-lattice criticality results.

Writes only to results/criticality_validation_20261113. No occupancy sampling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path("results/criticality_revision_20261113/full")
OUT = Path("results/criticality_validation_20261113")
PC = 0.592746
LENGTHS = (32, 64, 128, 256, 512)
RULES = ("largest_excluded", "all_spanning_excluded")
B = 1000
RNG = np.random.default_rng(20261113 + 810)
COLS = ("p_infinity", "chi_largest_excluded", "chi_all_spanning_excluded", "span_x", "span_y")


def get_inputs():
    tables = ROOT / "tables"
    return (pd.read_csv(tables / name) for name in (
        "configuration_observables.csv", "summary.csv", "cluster_histogram.csv",
        "cluster_tail_fits.csv", "pseudocritical.csv", "direction_symmetry.csv"))


def bootstrap_curves(samples: pd.DataFrame, summary: pd.DataFrame):
    """Resample configurations at each (L,p), keeping all observables paired."""
    curves = {}
    for length in LENGTHS:
        group = samples[samples.length == length]
        ratios = np.sort(group.p_ratio.unique())
        point = np.empty((len(ratios), len(COLS)))
        boots = np.empty((B, len(ratios), len(COLS)))
        for j, ratio in enumerate(ratios):
            subset = group[np.isclose(group.p_ratio, ratio)].sort_values("configuration_id")
            n = len(subset)
            idx = RNG.integers(0, n, size=(B, n))
            pinf = subset.p_infinity.to_numpy()
            sx = subset.span_x.to_numpy()
            sy = subset.span_y.to_numpy()
            boots[:, j, 0] = pinf[idx].mean(axis=1)
            boots[:, j, 3] = sx[idx].mean(axis=1)
            boots[:, j, 4] = sy[idx].mean(axis=1)
            point[j, (0, 3, 4)] = (pinf.mean(), sx.mean(), sy.mean())
            for k, rule in ((1, "largest_excluded"), (2, "all_spanning_excluded")):
                m1 = subset[f"{rule}_m1"].to_numpy()
                m2 = subset[f"{rule}_m2"].to_numpy()
                numerator = m2[idx].sum(axis=1)
                denominator = m1[idx].sum(axis=1)
                boots[:, j, k] = np.divide(numerator, denominator,
                    out=np.zeros(B), where=denominator != 0)
                point[j, k] = m2.sum() / m1.sum() if m1.sum() else 0
        curves[length] = {"ratio": ratios, "p": PC * ratios, "point": point, "boot": boots}
    return curves


def wls(x, y, sigma):
    X = np.column_stack((np.ones(len(x)), x))
    weight = 1 / np.square(sigma)
    operator = np.linalg.solve(X.T @ (weight[:, None] * X), X.T * weight)
    beta = operator @ y
    residual = y - X @ beta
    chi2 = float(np.sum((residual / sigma) ** 2))
    weighted_mean = np.average(y, weights=weight)
    denominator = np.sum(weight * (y - weighted_mean) ** 2)
    r2 = 1 - np.sum(weight * residual**2) / denominator if denominator else np.nan
    return beta, operator, chi2, float(r2)


def pseudocritical_extrapolation(curves, peaks):
    peak_boot = np.empty((B, len(LENGTHS), 2))
    height_boot = np.empty_like(peak_boot)
    point_rows = []
    for i, length in enumerate(LENGTHS):
        data = curves[length]
        for k, rule in enumerate(RULES):
            column = k + 1
            indices = np.argmax(data["boot"][:, :, column], axis=1)
            peak_boot[:, i, k] = data["p"][indices]
            height_boot[:, i, k] = data["boot"][np.arange(B), indices, column]
            stored = peaks[(peaks.length == length) & (peaks.exclusion == rule)].iloc[0]
            point_rows.append({"length": length, "exclusion": rule,
                "stored_grid_peak": stored.p_grid_peak,
                "stored_quadratic_peak": stored.p_quadratic_peak,
                "stored_ci_low": stored.p_ci_low, "stored_ci_high": stored.p_ci_high,
                "rebootstrap_ci_low": np.quantile(peak_boot[:, i, k], .025),
                "rebootstrap_ci_high": np.quantile(peak_boot[:, i, k], .975),
                "rebootstrap_sd": np.std(peak_boot[:, i, k], ddof=1),
                "grid_spacing_min": stored.minimum_grid_spacing})
    pd.DataFrame(point_rows).to_csv(OUT / "peak_bootstrap_audit.csv", index=False)
    pd.DataFrame({f"L{length}_{rule}": peak_boot[:, i, k]
        for i, length in enumerate(LENGTHS) for k, rule in enumerate(RULES)}).to_csv(
            OUT / "bootstrap_peak_replicates.csv", index=False)

    fits = []
    for k, rule in enumerate(RULES):
        for subset_name, indices in (("all_5", np.arange(5)), ("large_3", np.arange(2, 5))):
            lengths = np.array(LENGTHS)[indices]
            x = lengths.astype(float)**(-.75)
            p = np.array([peaks[(peaks.length == length) &
                         (peaks.exclusion == rule)].p_grid_peak.iloc[0] for length in lengths])
            spread = peak_boot[:, indices, k].std(axis=0, ddof=1)
            grid = np.array([peaks[(peaks.length == length) &
                         (peaks.exclusion == rule)].minimum_grid_spacing.iloc[0]
                         for length in lengths])
            sigma = np.sqrt(spread**2 + grid**2 / 12)
            beta, operator, chi2, r2 = wls(x, p, sigma)
            boot_beta = (operator @ peak_boot[:, indices, k].T).T
            ci = np.quantile(boot_beta, [.025, .975], axis=0)
            fits.append({"exclusion": rule, "subset": subset_name, "size_count": len(lengths),
                "intercept_pc_infinite": beta[0], "intercept_ci_low": ci[0, 0],
                "intercept_ci_high": ci[1, 0], "slope": beta[1],
                "slope_ci_low": ci[0, 1], "slope_ci_high": ci[1, 1],
                "chi_squared": chi2, "reduced_chi_squared": chi2 / (len(x) - 2),
                "weighted_r_squared": r2, "theory_pc": PC,
                "grid_floor_applied": True,
                "sigma_min": sigma.min(), "sigma_max": sigma.max(),
                "estimator": "stored discrete grid peaks; configuration bootstrap"})
    pd.DataFrame(fits).to_csv(OUT / "pseudocritical_extrapolation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, rule in zip(axes, RULES):
        audit = pd.DataFrame(point_rows).query("exclusion == @rule").sort_values("length")
        x = audit.length.to_numpy(dtype=float)**(-.75)
        y = audit.stored_grid_peak.to_numpy()
        error = np.vstack((y - audit.rebootstrap_ci_low.to_numpy(),
                           audit.rebootstrap_ci_high.to_numpy() - y))
        ax.errorbar(x, y, yerr=np.maximum(error, 0), fmt="o", capsize=3, label="grid peak, 95% CI")
        for row in fits:
            if row["exclusion"] == rule:
                xx = np.linspace(0, max(x), 100)
                ax.plot(xx, row["intercept_pc_infinite"] + row["slope"] * xx,
                        label=row["subset"])
        ax.axhline(PC, color="black", ls="--", label="theory")
        ax.set(xlabel=r"$L^{-3/4}$", title=rule)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$p_c(L)$, discrete susceptibility peak")
    fig.tight_layout(); fig.savefig(OUT / "pseudocritical_extrapolation.png", dpi=180); plt.close(fig)
    return peak_boot, height_boot


def crossing_candidates(p, left, right):
    d = left - right
    records = []
    for i in range(len(p) - 1):
        if abs(d[i]) < 1e-14:
            records.append((float(p[i]), "exact_equal", i))
        elif d[i] * d[i + 1] < 0:
            records.append((float(p[i] - d[i] * (p[i + 1] - p[i]) /
                                  (d[i + 1] - d[i])), "sign_change", i))
    if abs(d[-1]) < 1e-14:
        records.append((float(p[-1]), "exact_equal", len(p) - 1))
    return records


def crossing_audit(curves):
    common = np.round(np.arange(.97, 1.030001, .002), 6)
    candidate_rows, summary_rows, boot_rows = [], [], []
    for left, right in zip(LENGTHS[:-1], LENGTHS[1:]):
        a, b = curves[left], curves[right]
        ia = np.array([np.flatnonzero(np.isclose(a["ratio"], q))[0] for q in common])
        ib = np.array([np.flatnonzero(np.isclose(b["ratio"], q))[0] for q in common])
        p = PC * common
        selected_boot = {}
        for direction, col in (("x", 3), ("y", 4)):
            candidates = crossing_candidates(p, a["point"][ia, col], b["point"][ib, col])
            chosen = min(candidates, key=lambda value: abs(value[0] - PC)) if candidates else None
            for value, kind, index in candidates:
                candidate_rows.append({"length_left": left, "length_right": right,
                    "direction": direction, "candidate_p": value, "kind": kind,
                    "bracket_left_p": p[index],
                    "bracket_right_p": p[min(index + 1, len(p) - 1)],
                    "selected": chosen is not None and value == chosen[0]})
            boot = np.full(B, np.nan)
            counts = np.zeros(B, dtype=int)
            for i in range(B):
                possibilities = crossing_candidates(p, a["boot"][i, ia, col],
                                                   b["boot"][i, ib, col])
                counts[i] = len(possibilities)
                if possibilities:
                    boot[i] = min(possibilities, key=lambda value: abs(value[0] - PC))[0]
                boot_rows.append({"length_left": left, "length_right": right,
                    "direction": direction, "replicate": i, "selected_p": boot[i],
                    "candidate_count": counts[i]})
            selected_boot[direction] = boot
            finite = boot[np.isfinite(boot)]
            ci = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
            summary_rows.append({"length_left": left, "length_right": right,
                "direction": direction, "selected_p": chosen[0] if chosen else np.nan,
                "candidate_count": len(candidates), "ci_low": ci[0], "ci_high": ci[1],
                "bootstrap_valid_fraction": len(finite) / B,
                "bootstrap_multiple_fraction": np.mean(counts > 1),
                "selection_rule": "candidate nearest theoretical pc"})
        paired = selected_boot["x"] - selected_boot["y"]
        paired = paired[np.isfinite(paired)]
        ci = np.quantile(paired, [.025, .975]) if len(paired) else [np.nan, np.nan]
        for row in summary_rows[-2:]:
            row["x_minus_y_crossing_ci_low"] = ci[0]
            row["x_minus_y_crossing_ci_high"] = ci[1]
    pd.DataFrame(candidate_rows).to_csv(OUT / "crossing_all_candidates.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "crossing_audit.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap_crossing_replicates.csv", index=False)


def symmetry_audit(samples, stored):
    rows = []
    for length in LENGTHS:
        subset = samples[(samples.length == length) & np.isclose(samples.p_ratio, 1.0)]
        xonly = int(((subset.span_x == 1) & (subset.span_y == 0)).sum())
        yonly = int(((subset.span_x == 0) & (subset.span_y == 1)).sum())
        row = stored[(stored.length == length) & np.isclose(stored.p_ratio, 1.0)].iloc[0]
        rows.append({"length": length, "samples": len(subset), "p_x": subset.span_x.mean(),
            "p_y": subset.span_y.mean(), "difference": row.p_x_minus_p_y,
            "paired_bootstrap_ci_low": row.ci_low, "paired_bootstrap_ci_high": row.ci_high,
            "x_only": xonly, "y_only": yonly,
            "mcnemar_exact_p": binomtest(xonly, xonly + yonly, .5).pvalue})
    frame = pd.DataFrame(rows)
    order = np.argsort(frame.mcnemar_exact_p.to_numpy())
    adjusted = np.zeros(len(frame))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(frame) - rank) * frame.mcnemar_exact_p.iloc[index])
        adjusted[index] = min(1., running)
    frame["holm_adjusted_p"] = adjusted
    frame.to_csv(OUT / "direction_symmetry_pc.csv", index=False)


def tail_audit(histogram, fits):
    from local_market.criticality_revision import finite_powerlaw
    rows = []
    for row in fits.itertuples():
        subset = histogram[(histogram.length == row.length) &
            np.isclose(histogram.p_ratio, 1.0) & (histogram.exclusion == row.exclusion)]
        counts = np.zeros(int(subset.cluster_size.max()) + 1, dtype=int)
        counts[subset.cluster_size.to_numpy(dtype=int)] = subset.C_s.to_numpy(dtype=int)
        upper = int(row.s_max_fit)
        lower = int(row.s_min)
        eligible = np.flatnonzero(counts >= row.count_threshold)
        previous = int(eligible[-2]) if len(eligible) >= 2 else 0
        close_eligible = [int(s) for s in eligible if s >= lower and
            np.count_nonzero(counts[max(lower, s - 10):s + 1] >= row.count_threshold) >= 3]
        conservative_upper = max(close_eligible) if close_eligible else 0
        values = np.repeat(subset.cluster_size.to_numpy(dtype=int),
                           subset.C_s.to_numpy(dtype=int))
        alt_alpha, _, alt_count = finite_powerlaw(values, lower, conservative_upper)
        rows.append({"length": row.length, "exclusion": row.exclusion,
            "count_threshold": row.count_threshold, "s_min": lower, "s_max_fit": upper,
            "tail_count": row.tail_count, "alpha_pdf": row.alpha_pdf,
            "alpha_ci_low": row.alpha_ci_low, "alpha_ci_high": row.alpha_ci_high,
            "ks_distance": row.ks_distance, "fixed_interval_gof_p": row.fixed_interval_gof_p,
            "gof_replicates": row.gof_replicates,
            "delta_aic_cutoff_minus_powerlaw": row.cutoff_aic_same_interval - row.powerlaw_aic,
            "delta_aic_exponential_minus_powerlaw": row.exponential_aic_same_interval - row.powerlaw_aic,
            "observed_max": int(subset.cluster_size.max()),
            "upper_count": counts[upper], "previous_eligible_s": previous,
            "eligible_gap": upper - previous,
            "preceding_10_below_threshold": int(np.count_nonzero(
                counts[max(lower, upper - 10):upper] < row.count_threshold)),
            "zero_bin_fraction_in_support": float(np.mean(counts[lower:upper + 1] == 0)),
            "conservative_upper": conservative_upper if conservative_upper else np.nan,
            "conservative_alpha": alt_alpha, "conservative_tail_count": alt_count,
            "distance_to_theory": row.alpha_pdf - 187 / 91})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "tail_fit_audit.csv", index=False)
    return frame


def _slope_bootstrap(lengths, values, boots):
    x = np.log(np.asarray(lengths, dtype=float))
    centered = x - x.mean()
    denom = centered @ centered
    slope = float(centered @ np.log(values) / denom)
    slopes = np.log(boots) @ centered / denom
    low, high = np.quantile(slopes, [.025, .975])
    return slope, float(low), float(high)


def susceptibility_scaling(curves, peaks, height_boot):
    rows = []
    exponent_rows = []
    for k, rule in enumerate(RULES):
        for i, length in enumerate(LENGTHS):
            data = curves[length]
            at_pc = int(np.flatnonzero(np.isclose(data["ratio"], 1.0))[0])
            stored = peaks[(peaks.length == length) & (peaks.exclusion == rule)].iloc[0]
            factor = length**(-43 / 24)
            scaled = height_boot[:, i, k] * factor
            rows.append({"length": length, "exclusion": rule,
                "chi_peak": stored.chi_peak, "chi_peak_scaled_theory": stored.chi_peak * factor,
                "scaled_peak_ci_low": np.quantile(scaled, .025),
                "scaled_peak_ci_high": np.quantile(scaled, .975),
                "chi_at_pc": data["point"][at_pc, k + 1],
                "chi_at_pc_scaled_theory": data["point"][at_pc, k + 1] * factor})
        for subset, indices in (("all_5", np.arange(5)), ("large_3", np.arange(2, 5))):
            lengths = np.array(LENGTHS)[indices]
            peak_values = np.array([peaks[(peaks.length == length) &
                (peaks.exclusion == rule)].chi_peak.iloc[0] for length in lengths])
            peak_boot_values = height_boot[:, indices, k]
            point_pc = []
            boot_pc = []
            for length in lengths:
                data = curves[length]
                j = int(np.flatnonzero(np.isclose(data["ratio"], 1.0))[0])
                point_pc.append(data["point"][j, k + 1])
                boot_pc.append(data["boot"][:, j, k + 1])
            for location, values, boots in (
                ("susceptibility_peak", peak_values, peak_boot_values),
                ("theoretical_pc", np.array(point_pc), np.column_stack(boot_pc))):
                slope, low, high = _slope_bootstrap(lengths, values, boots)
                exponent_rows.append({"exclusion": rule, "subset": subset,
                    "location": location, "fitted_gamma_over_nu": slope,
                    "ci_low": low, "ci_high": high,
                    "theoretical_gamma_over_nu": 43 / 24,
                    "bootstrap_replicates": B})
    pd.DataFrame(rows).to_csv(OUT / "susceptibility_scaled_peaks.csv", index=False)
    pd.DataFrame(exponent_rows).to_csv(OUT / "susceptibility_exponent_subsets.csv", index=False)
    return pd.DataFrame(exponent_rows)


def collapse_audit(curves, exponent_frame):
    """Dimensionless between-size RMS on one fixed common critical x-window."""
    xgrid = np.linspace(-.20, .20, 41)
    definitions = (("p_infinity", 0, 5 / 48),
                   ("chi_largest_excluded", 1, -43 / 24),
                   ("chi_all_spanning_excluded", 2, -43 / 24),
                   ("span_x", 3, 0.0))
    rows = []

    def evaluate(name, column, exponent, subset):
        point_curves, bootstrap_curves = [], []
        for length in subset:
            data = curves[length]
            select = (data["ratio"] >= .97 - 1e-9) & (data["ratio"] <= 1.03 + 1e-9)
            x = (data["p"][select] - PC) * length**.75
            if x.min() > xgrid.min() or x.max() < xgrid.max():
                raise ValueError(f"x window not covered at L={length}")
            factor = length**exponent
            point_curves.append(np.interp(xgrid, x, data["point"][select, column]) * factor)
            bootstrap_curves.append(np.stack([
                np.interp(xgrid, x, data["boot"][i, select, column]) * factor
                for i in range(B)]))
        point = np.stack(point_curves)
        boot = np.stack(bootstrap_curves, axis=1)
        normalizer = float(np.mean(point[:, len(xgrid) // 2]))
        score = float(np.sqrt(np.mean((point - point.mean(axis=0))**2)) / abs(normalizer))
        boot_scores = np.sqrt(np.mean((boot - boot.mean(axis=1, keepdims=True))**2,
                                      axis=(1, 2))) / abs(normalizer)
        return score, boot_scores, normalizer

    all_scores = {}
    for subset_name, subset in (("all_5", LENGTHS),
                                ("without_32", LENGTHS[1:]),
                                ("large_3", LENGTHS[2:])):
        for name, column, exponent in definitions:
            score, boot_scores, normalizer = evaluate(name, column, exponent, subset)
            all_scores[(name, subset_name)] = boot_scores
            low, high = np.quantile(boot_scores, [.025, .975])
            rows.append({"observable": name, "subset": subset_name,
                "exponent_source": "theory", "vertical_L_power": exponent,
                "dimensionless_rms": score, "bootstrap_ci_low": low,
                "bootstrap_ci_high": high, "normalizer_at_x0": normalizer,
                "common_x_min": xgrid.min(), "common_x_max": xgrid.max(),
                "x_grid_points": len(xgrid), "bootstrap_replicates": B,
                "probability_large3_better_than_all5": float(np.mean(
                    all_scores[(name, "large_3")] < all_scores[(name, "all_5")]))
                    if subset_name == "large_3" else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "collapse_quality_theory.csv", index=False)

    sensitivity = []
    for subset_name, subset in (("all_5", LENGTHS), ("without_32", LENGTHS[1:]),
                                ("large_3", LENGTHS[2:])):
        for name, column in (("chi_largest_excluded", 1),
                             ("chi_all_spanning_excluded", 2)):
            for fit_subset in ("all_5", "large_3"):
                fit = exponent_frame[(exponent_frame.exclusion == name.removeprefix("chi_")) &
                    (exponent_frame.subset == fit_subset) &
                    (exponent_frame.location == "theoretical_pc")].iloc[0]
                exponent = -fit.fitted_gamma_over_nu
                score, _, _ = evaluate(name, column, exponent, subset)
                sensitivity.append({"observable": name, "curve_subset": subset_name,
                    "exponent_fit_subset": fit_subset, "vertical_L_power": exponent,
                    "dimensionless_rms": score,
                    "interpretation": "diagnostic only; fitted exponent is not a theory replacement"})
    pd.DataFrame(sensitivity).to_csv(OUT / "collapse_exponent_sensitivity.csv", index=False)

    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(3)
    for j, name in enumerate(("p_infinity", "chi_largest_excluded",
                              "chi_all_spanning_excluded", "span_x")):
        group = frame[frame.observable == name].set_index("subset").loc[
            ["all_5", "without_32", "large_3"]]
        ax.plot(x, group.dimensionless_rms, "o-", label=name)
    ax.set_xticks(x, ["all five", "L≥64", "L≥128"])
    ax.set(ylabel="dimensionless between-size RMS", title=r"Theory-exponent collapse on $|x|\leq0.20$")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "collapse_quality.png", dpi=180); plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing validation output: {OUT}")
    OUT.mkdir(parents=True)
    samples, summary, histogram, fits, peaks, stored_symmetry = get_inputs()
    curves = bootstrap_curves(samples, summary)
    peak_boot, height_boot = pseudocritical_extrapolation(curves, peaks)
    crossing_audit(curves)
    symmetry_audit(samples, stored_symmetry)
    tail_audit(histogram, fits)
    exponent_frame = susceptibility_scaling(curves, peaks, height_boot)
    collapse_audit(curves, exponent_frame)
    metadata = {"source_root": str(ROOT), "theoretical_pc": PC,
        "bootstrap_replicates": B, "bootstrap_unit": "independent occupancy configuration",
        "bootstrap_seed": 20261113 + 810,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            (ROOT / "metadata/sha256_manifest.json").read_bytes()).hexdigest(),
        "no_new_occupancy_sampling": True}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
                if p.is_file() and p.name != "sha256_manifest.json"}
    (OUT / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("saved", len(manifest), "validation artifacts to", OUT)


if __name__ == "__main__":
    main()
