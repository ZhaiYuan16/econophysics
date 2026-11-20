#!/usr/bin/env python3
"""CCDF interval OLS for the market and network results in the submitted report."""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "results/loglog_ols_comparison"


def absolute_returns(group: pd.DataFrame, exclude_raw_zeros: bool) -> np.ndarray:
    """Keep the raw-return zero rule separate from log-axis positivity."""
    raw = group["return"].to_numpy(dtype=float)
    z = group["z"].to_numpy(dtype=float)
    keep = np.isfinite(raw) & np.isfinite(z)
    if exclude_raw_zeros:
        keep &= raw != 0
    return np.abs(z[keep])


def empirical(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if not len(v):
        return None
    x, counts = np.unique(v, return_counts=True)
    remaining = np.cumsum(counts[::-1])[::-1]
    return x, remaining / remaining[0], remaining


def market_interval_fit(x, ccdf, remaining):
    """Preserve the original market fit: 70th-percentile search and wide near-best R²."""
    if len(x) < 14:
        return None, "too_few_distinct_points"
    ix = np.unique(np.clip(np.searchsorted(x, np.geomspace(x[0], x[-1], 100)), 0, len(x)-1))
    xx, yy, rr = x[ix], ccdf[ix], remaining[ix]
    good = (rr >= 20) & (yy > 0) & (xx > 0)
    xx, yy, rr = xx[good], yy[good], rr[good]
    if len(xx) < 14:
        return None, "too_few_tail_points"
    lx, ly = np.log10(xx), np.log10(yy)
    candidates = []
    for i in range(len(lx)-13):
        for j in range(i+13, len(lx)):
            if lx[j]-lx[i] < 1 or rr[i]-rr[j] < 100:
                continue
            X, Y = lx[i:j+1], ly[i:j+1]
            slope, intercept = np.polyfit(X, Y, 1)
            fitted = intercept+slope*X
            sst = np.sum((Y-Y.mean())**2)
            r2 = 1-np.sum((Y-fitted)**2)/sst if sst > 0 else -np.inf
            candidates.append((r2, lx[j]-lx[i], i, j, intercept, slope))
    if not candidates:
        return None, "no_interval_meets_width_and_count"
    best_r2 = max(row[0] for row in candidates)
    eligible = [row for row in candidates if row[0] >= best_r2-.002]
    r2, width, i, j, intercept, slope = max(eligible, key=lambda row: (row[1], row[0]))
    mid = (i+j)//2
    left = np.polyfit(lx[i:mid+1], ly[i:mid+1], 1)[0]
    right = np.polyfit(lx[mid:j+1], ly[mid:j+1], 1)[0]
    drift = abs(left-right)/max(abs(slope), .01)
    status = "visible_linear_candidate" if r2 >= .98 and slope < 0 and drift <= .25 else "no_clear_linear_scaling"
    return dict(xmin=xx[i], xmax=xx[j], n_points=j-i+1, n_interval=int(rr[i]-rr[j]),
                slope=slope, intercept=intercept, r2=r2, ccdf_alpha=-slope,
                pdf_tau=1-slope, span_decades=width,
                slope_half_relative_difference=drift, status=status), None


def market_fit(values):
    e = empirical(values)
    if e is None:
        return {"status": "no_positive_data"}
    x, ccdf, remaining = e
    lower = np.flatnonzero(remaining <= .30*remaining[0])
    floor = x[lower[0]] if len(lower) else x[-1]
    keep = x >= floor
    fit, reason = market_interval_fit(x[keep], ccdf[keep], remaining[keep])
    return {"scanned_from": floor, "curve_points": len(x),
            **(fit if fit is not None else {"status": reason})}


def network_candidates(x, ccdf, remaining):
    """Preserve the original refined network interval selection."""
    ix = np.unique(np.clip(np.searchsorted(x, np.geomspace(x[0], x[-1], 145)), 0, len(x)-1))
    xx, yy, rr = x[ix], ccdf[ix], remaining[ix]
    good = (rr >= 10) & (yy > 0)
    xx, yy, rr = xx[good], yy[good], rr[good]
    if len(xx) < 7:
        return [], "too_few_curve_points"
    lx, ly = np.log10(xx), np.log10(yy)
    all_rows = []
    for i in range(len(xx)-6):
        if yy[i] > .98:
            continue
        for j in range(i+6, len(xx)):
            span, nobs, drop = lx[j]-lx[i], rr[i]-rr[j], ly[i]-ly[j]
            if span < .65 or nobs < 100 or drop < .15:
                continue
            X, Y = lx[i:j+1], ly[i:j+1]
            slope, intercept = np.polyfit(X, Y, 1)
            if slope >= 0:
                continue
            fitted = intercept+slope*X
            sst = np.sum((Y-Y.mean())**2)
            r2 = 1-np.sum((Y-fitted)**2)/sst if sst > 0 else -np.inf
            one, two = i+(j-i)//3, i+2*(j-i)//3
            slopes, drops = [], []
            for a, b in ((i, one), (one, two), (two, j)):
                if b-a < 2:
                    break
                slopes.append(np.polyfit(lx[a:b+1], ly[a:b+1], 1)[0])
                drops.append(ly[a]-ly[b])
            if len(slopes) < 3 or min(drops) < .015:
                continue
            drift = (max(slopes)-min(slopes))/max(abs(slope), .02)
            stable = all(s < 0 for s in slopes) and drift < .35
            tier = "standard" if span >= 1 and len(X) >= 14 and r2 >= .98 and stable else "local"
            if tier == "local" and (r2 < .965 or drift > .85):
                continue
            all_rows.append(dict(xmin=xx[i], xmax=xx[j], n_points=len(X),
                n_interval=int(nobs), n_above_upper=int(rr[j]), start_ccdf=yy[i], end_ccdf=yy[j],
                log_drop=drop, span_decades=span, slope=slope, intercept=intercept, r2=r2,
                ccdf_alpha=-slope, pdf_tau=1-slope, third_slope_spread=drift, tier=tier))
    if not all_rows:
        available = np.log10(xx[-1]/xx[0]) if xx[-1] > xx[0] else 0
        if available < .65:
            return [], "insufficient_scale_range"
        populated = any(lx[j]-lx[i] >= .65 and rr[i]-rr[j] >= 100
                        for i in range(len(lx)) for j in range(i+1, len(lx)))
        return [], "no_stable_decline" if populated else "insufficient_events_in_window"
    all_rows.sort(key=lambda row: (row["tier"] != "standard", -row["r2"], -row["span_decades"]))
    chosen = []
    for row in all_rows:
        if any(abs(np.log10(row["xmin"]/other["xmin"])) < .22 and
               abs(np.log10(row["xmax"]/other["xmax"])) < .22 for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) == 5:
            break
    return chosen, None


def network_fit(values):
    e = empirical(values)
    if e is None:
        return {"status": "no_positive_data"}, []
    x, ccdf, remaining = e
    candidates, reason = network_candidates(x, ccdf, remaining)
    result = {"curve_points": len(x), "candidate_count": len(candidates)}
    if candidates:
        best = candidates[0]
        result.update(best)
        result["status"] = "standard_candidate" if best["tier"] == "standard" else "exploratory_local_only"
    else:
        result["status"] = reason
    return result, candidates


def main():
    rows, candidate_rows = [], []
    families = (
        ("market", "results/tables/market_returns.csv.gz", ["p_ratio", "activity"], False),
        ("network", "results/tables/network_market_returns.csv.gz", ["topology", "relative_p"], True),
    )
    for kind, source, keys, exclude_zeros in families:
        data = pd.read_csv(BASE/source, usecols=keys+["return", "z"])
        for key, group in data.groupby(keys, sort=True):
            label = (f"p/pc={key[0]:g}, a={key[1]:g}" if kind == "market"
                     else f"{key[0].upper()} p/pc={key[1]:g}")
            values = absolute_returns(group, exclude_zeros)
            base = dict(model="CB_market" if kind == "market" else "CB_network_market",
                kind=kind, condition=label, fitted_distribution="CCDF", label=label,
                n_observations=len(group), zero_return_count=int((group["return"] == 0).sum()),
                n_raw=int(np.count_nonzero(np.isfinite(values) & (values > 0))),
                zero_rule="exclude_raw_zero" if exclude_zeros else "include_raw_zero",
                source=source)
            if kind == "market":
                rows.append(base | market_fit(values))
            else:
                fit, candidates = network_fit(values)
                rows.append(base | fit)
                candidate_rows.extend(dict(kind=kind, label=label, rank=rank, **candidate)
                                      for rank, candidate in enumerate(candidates, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/"ols_summary_v2.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(OUT/"candidate_intervals_v2.csv", index=False)
    print(summary[["kind", "label", "status", "xmin", "xmax", "slope", "pdf_tau", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
