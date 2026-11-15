"""Finite-support discrete avalanche tail likelihoods and path bootstrap."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, minimize_scalar


def fit_tail(values, xmin: int, upper: int, model: str):
    x = np.asarray(values, dtype=int)
    x = x[(x >= xmin) & (x <= upper)]
    if not len(x):
        raise ValueError("empty tail")
    support = np.arange(xmin, upper+1)
    logs = np.log(support)
    def objective(par):
        exponent, rate = par if model == "cutoff" else (par[0], 0.) if model == "power" else (0., par[0])
        lw = -exponent*logs-rate*support
        peak = lw.max()
        logz = peak + np.log(np.exp(lw-peak).sum())
        return float(len(x)*logz + exponent*np.log(x).sum() + rate*x.sum())
    if model == "power":
        result = minimize_scalar(lambda a: objective([a]), bounds=(.1, 8), method="bounded")
        exponent, rate = result.x, 0.
        npar = 1
    elif model == "exponential":
        result = minimize_scalar(lambda r: objective([r]), bounds=(1e-9, 2), method="bounded")
        exponent, rate = 0., result.x
        npar = 1
    else:
        result = minimize(objective, [1.5, .01], bounds=[(.01, 8), (1e-9, 2)], method="L-BFGS-B")
        exponent, rate = result.x
        npar = 2
    weights = support.astype(float)**(-exponent)*np.exp(-rate*(support-xmin))
    probs = weights/weights.sum()
    observed = np.sort(x)
    ecdf = np.searchsorted(observed, support, side="right")/len(observed)
    ks = float(np.max(np.abs(ecdf-np.cumsum(probs))))
    return dict(model=model, xmin=xmin, upper=upper, tail_count=len(x),
        exponent=float(exponent), cutoff=float(1/rate) if rate else np.inf,
        log_likelihood=-float(result.fun), aic=2*npar+2*float(result.fun), ks=ks,
        boundary=bool(rate < 1e-7 or 1/rate > 10*upper) if model == "cutoff" else False)


def select_xmin(values, upper: int, minimum_count=100, minimum_range=10):
    x = np.asarray(values, dtype=int)
    x = x[(x > 0) & (x <= upper)]
    if len(x) < minimum_count:
        return None, [], "sample_insufficient"
    candidates = np.unique(x)
    candidates = candidates[(candidates <= x.max()/minimum_range)]
    if len(candidates) > 35:
        candidates = np.unique(candidates[np.linspace(0, len(candidates)-1, 35).astype(int)])
    rows = []
    for xmin in candidates:
        if (x >= xmin).sum() < minimum_count:
            continue
        rows.append(fit_tail(x, int(xmin), upper, "power"))
    if not rows:
        return None, [], "dynamic_range_or_tail_insufficient"
    best = min(rows, key=lambda row: row["ks"])
    return int(best["xmin"]), rows, "ok"


def bootstrap_paths(path_values, upper: int, count: int, rng, minimum_count=100,
                    path_groups=None):
    """Resample whole paths within each initial-condition group and refit."""
    draws = []
    ids = list(path_values)
    if path_groups is None:
        path_groups = {pid: "all" for pid in ids}
    groups = {group: [pid for pid in ids if path_groups[pid] == group]
              for group in set(path_groups.values())}
    for index in range(count):
        chosen = np.concatenate([rng.choice(members, size=len(members), replace=True)
                                 for members in groups.values()])
        values = np.concatenate([path_values[pid] for pid in chosen])
        xmin, _, status = select_xmin(values, upper, minimum_count)
        if status != "ok":
            draws.append(dict(bootstrap=index, status=status))
            continue
        pure = fit_tail(values, xmin, upper, "power")
        cutoff = fit_tail(values, xmin, upper, "cutoff")
        draws.append(dict(bootstrap=index, status="ok", xmin=xmin,
            tau=pure["exponent"], cutoff_tau=cutoff["exponent"],
            cutoff=cutoff["cutoff"], cutoff_boundary=cutoff["boundary"]))
    return draws
