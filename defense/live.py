#!/usr/bin/env python3
"""Run reproducible, headless Project 6 defense experiments."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy.stats import linregress
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cb_experiment import Cache, child_seed, draw_phase, simulate_cb
from local_market.soc_research import ResearchConfig, simulate
from local_market.soc_tail import fit_tail
from local_market.statistics import return_statistics, standardize_returns


DEFAULT_SEED = 20261121
MARKET_RATIOS = (0.9, 1.0, 1.1)
ACTIVITIES = (.001, .01, .1, .5)
SOC_MECHANISMS = ("baseline", "cascade", "no_dissipation", "random_dissipation")
VERSIONS = {
    "python": platform.python_version(), "numpy": np.__version__,
    "scipy": scipy.__version__, "networkx": nx.__version__, "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
}


def finite_number(value):
    result = float(value)
    if not np.isfinite(result):
        raise argparse.ArgumentTypeError("expected a finite number")
    return result


def probability(value):
    result = finite_number(value)
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("expected a probability in [0, 1]")
    return result


def positive_number(value):
    result = finite_number(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return result


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return result


def archived_selection(parser):
    parser.add_argument("--ratio", type=float, choices=MARKET_RATIOS, default=1.0,
                        help="archived p / reference p")
    parser.add_argument("--activity", type=probability, default=.01)
    parser.add_argument("--topology", choices=("lattice", "er", "ws", "ba"), default="ba")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cb = commands.add_parser("cb", help="occupation scan and cluster trades on lattice or graph")
    cb.add_argument("--mode", choices=("demo", "stats"), default="demo")
    cb.add_argument("--topology", choices=("lattice", "er", "ws", "ba"), default="lattice")
    cb.add_argument("--p", nargs="+", type=probability, help="one or more absolute probabilities")
    cb.add_argument("--length", type=positive_int)
    cb.add_argument("--nodes", type=positive_int)
    cb.add_argument("--boundary", choices=("open", "periodic"))
    cb.add_argument("--degree", type=positive_int, help="target degree for ER, k for WS, 2m for BA")
    cb.add_argument("--rho", type=probability, help="ER link probability")
    cb.add_argument("--k", type=positive_int, help="WS neighbor count")
    cb.add_argument("--rewire", type=probability, help="WS edge rewiring probability")
    cb.add_argument("--m", type=positive_int, help="BA edges attached per new node")
    cb.add_argument("--activity", "--a", type=probability, default=.01)
    cb.add_argument("--depth", "--lambda", type=positive_number)
    cb.add_argument("--trades", type=positive_int, help="actions per configuration; use 1 to redraw every trade")
    cb.add_argument("--samples", type=positive_int, help="occupation configurations per p and network")
    cb.add_argument("--networks", type=positive_int)
    cb.add_argument("--bootstraps", type=positive_int)
    cb.add_argument("--seed", type=nonnegative_int, default=DEFAULT_SEED)
    cb.add_argument("--exclusion", choices=("spanning", "largest", "none"))
    cb.add_argument("--compare", type=Path, help="previous cb run directory containing phase.csv")

    depth = commands.add_parser("depth", help="reprice archived demands; skip the simulation")
    depth.add_argument("--source", choices=("market", "network", "quenched"), default="quenched")
    depth.add_argument("--depth", "--lambda", type=positive_number, required=True)
    depth.add_argument("--threshold", type=positive_number, default=.01,
                       help="fixed threshold in raw log-return units")
    depth.add_argument("--initial-log-price", type=finite_number, default=0.,
                       help="initial log price of a quenched path")
    archived_selection(depth)

    fit = commands.add_parser("fit", help="refit an interval on archived data or a previous CB run")
    fit.add_argument("--source", choices=("market", "network", "cluster", "soc", "run"), default="market")
    fit.add_argument("--xmin", type=positive_number, required=True)
    fit.add_argument("--xmax", type=positive_number, required=True)
    fit.add_argument("--length", type=positive_int)
    fit.add_argument("--p", type=probability, help="absolute p for a previous CB run")
    fit.add_argument("--run", type=Path, help="previous CB or SOC run directory")
    fit.add_argument("--burn-in", type=nonnegative_int, help="override the SOC tail burn-in")
    fit.add_argument("--exclusion", choices=("spanning", "largest", "none"), default="spanning")
    fit.add_argument("--mechanism", choices=("baseline", "cascade"), default="cascade")
    archived_selection(fit)

    soc = commands.add_parser("soc", help="run the published SOC driver and dissipation rules")
    soc.add_argument("--mode", choices=("demo", "stats"), default="demo")
    soc.add_argument("--length", type=positive_int, default=32)
    soc.add_argument("--p0", nargs="+", type=probability)
    soc.add_argument("--steps", type=positive_int)
    soc.add_argument("--burn-in", type=nonnegative_int)
    soc.add_argument("--replicates", type=positive_int)
    soc.add_argument("--activity", "--a", type=probability, default=.01)
    soc.add_argument("--theta", type=probability, default=.08)
    soc.add_argument("--q", type=probability, default=.25)
    soc.add_argument("--drive-nodes", type=positive_int, default=1)
    soc.add_argument("--mechanism", choices=SOC_MECHANISMS, default="baseline")
    soc.add_argument("--max-rounds", type=positive_int, default=1000)
    soc.add_argument("--seed", type=nonnegative_int, default=DEFAULT_SEED)
    return parser, parser.parse_args()


def cb_config(args):
    topology = args.topology
    stats = args.mode == "stats"
    lattice = topology == "lattice"
    if lattice and any(v is not None for v in (args.nodes, args.degree, args.rho, args.k, args.rewire, args.m)):
        raise ValueError("use --length and --boundary for a lattice")
    if not lattice and (args.length is not None or args.boundary is not None):
        raise ValueError("use --nodes for ER, WS or BA")
    if ((args.rho is not None and topology != "er") or
        (args.m is not None and topology != "ba") or
        ((args.k is not None or args.rewire is not None) and topology != "ws")):
        raise ValueError("--rho is ER only; --m is BA only; --k and --rewire are WS only")
    if args.degree is not None and any(v is not None for v in (args.rho, args.k, args.m)):
        raise ValueError("--degree and the topology-specific parameter are alternatives")
    length = (args.length or (64 if stats else 32)) if lattice else None
    n = length**2 if lattice else args.nodes or (1024 if stats else 512)
    boundary = (args.boundary or "open") if lattice else None
    rho, degree, rewiring = None, None, None
    if topology == "er":
        if n < 2:
            raise ValueError("ER requires at least two nodes")
        rho = args.rho if args.rho is not None else (args.degree or 4) / (n - 1)
        if not 0 <= rho <= 1:
            raise ValueError("ER link probability must be in [0, 1]")
    if topology == "ws":
        degree = args.k or args.degree or 4
        rewiring = args.rewire if args.rewire is not None else .1
        if degree % 2 or not 2 <= degree <= n:
            raise ValueError("WS k must be even and satisfy 2 <= k <= nodes")
    if topology == "ba":
        degree = 2 * args.m if args.m is not None else args.degree or 4
        if degree % 2 or not 1 <= degree // 2 < n:
            raise ValueError("BA requires integer m with 1 <= m < nodes and degree = 2m")
    exclusion = args.exclusion or ("spanning" if lattice else "largest")
    if not lattice and exclusion == "spanning":
        raise ValueError("spanning exclusion requires a lattice")
    networks = args.networks or (1 if lattice or not stats else 3)
    if lattice and networks != 1:
        raise ValueError("the lattice is deterministic; use --samples for repeats")
    grid_file = ROOT / "configs/network_analysis.yaml"
    grids = yaml.safe_load(grid_file.read_text())["p_grids"]
    grid = args.p if args.p is not None else grids[topology]
    grid = sorted(set(grid))
    if args.p is None and not stats:
        grid = [grid[i] for i in np.linspace(0, len(grid) - 1, 7, dtype=int)]
    samples = args.samples or (24 if stats else 2)
    if stats and samples < 2:
        raise ValueError("stats mode needs at least two occupation configurations per network")
    return dict(
        topology=topology, mode=args.mode, p=grid, length=length, node_count=n, boundary=boundary,
        degree=degree, rho=rho, rewire=rewiring, m=degree // 2 if topology == "ba" else None,
        activity=args.activity, depth=args.depth or float(n), exclusion=exclusion,
        samples=samples, trades=args.trades or (128 if stats else 64), seed=args.seed,
        bootstraps=(args.bootstraps or 200) if stats else 0,
        network_seeds=[child_seed(args.seed, 0, i) for i in range(networks)],
        bootstrap_seed=child_seed(args.seed, 3) if stats else None,
        compare=str(args.compare.resolve()) if args.compare else None,
    )


def ccdf(values, counts=None):
    if counts is None:
        values, counts = np.unique(values, return_counts=True)
    keep = (values > 0) & (counts > 0)
    x, counts = values[keep], counts[keep]
    remaining = np.cumsum(counts[::-1])[::-1]
    y = remaining / counts.sum() if counts.size else np.empty(0, dtype=float)
    return x, y, remaining


def render_returns(frame, summary, out, price=False):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    if price:
        prices = frame["log_price"].to_numpy()
        scale = float(np.max(np.abs(prices))) or 1.0
        axes[0].plot(frame["time"], prices / scale)
        axes[0].set(xlabel="Trade", ylabel=f"Cumulative log price / {scale:.3g}")
    else:
        sigmas = summary["return_std"]
        scale = float(sigmas.max()) or 1.0
        axes[0].plot(summary["p"], sigmas / scale, "o-")
        axes[0].set(xlabel="Occupation probability p", ylabel=f"Return std / {scale:.3g}")
    scale = float(frame["return"].abs().max()) or 1.0
    for p, group in frame.groupby("p"):
        x, y, _ = ccdf(group["return"].abs().to_numpy() / scale)
        if x.size:
            axes[1].loglog(x, y, label=f"p={p:.4g}")
    axes[1].set(xlabel=f"Nonzero |r| / {scale:.3g}", ylabel="Conditional CCDF")
    if axes[1].lines:
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(.5, .5, "All returns are zero", ha="center", transform=axes[1].transAxes)
    fig.savefig(out / "returns.png", dpi=150)
    plt.close(fig)


def return_table(frame, depth, out, threshold=.01, initial_log_price=None):
    price = initial_log_price is not None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        frame["return"] = frame["demand"].to_numpy(dtype=float) / depth
        if price:
            frame["log_price"] = initial_log_price + frame["return"].cumsum()
    if not np.isfinite(frame["return"]).all() or (price and not np.isfinite(frame["log_price"]).all()):
        raise ValueError("market depth is too small for finite returns or log prices")
    pieces = []
    rows = []
    for p, group in frame.groupby("p"):
        demand = group["demand"].to_numpy(dtype=float)
        if len(demand) > 1 and np.min(demand) != np.max(demand):
            z, _, _ = standardize_returns(demand)
            metrics = return_statistics(demand, z, (3., 5., 8.), group.get("active_cluster_count"))
        else:
            z = np.full(len(demand), np.nan)
            metrics = dict(
                observations=len(group), zero_return_fraction=np.mean(demand == 0),
                return_std=0., return_skewness=np.nan, return_excess_kurtosis=np.nan,
                negative_crash_probability=np.nan,
                p_abs_z_gt_3=np.nan, p_abs_z_gt_5=np.nan, p_abs_z_gt_8=np.nan,
            )
            if "active_cluster_count" in group:
                metrics["mean_active_cluster_count"] = group["active_cluster_count"].mean()
        group = group.assign(z=z)
        pieces.append(group)
        returns = group["return"].to_numpy()
        rows.append(dict(
            metrics, p=p, return_mean=float(demand.mean() / depth),
            return_std=float(metrics["return_std"] / depth), raw_threshold=threshold,
            raw_negative_frequency=float(np.mean(returns < -threshold)),
            raw_extreme_frequency=float(np.mean(np.abs(returns) > threshold)),
        ))
    if not all(np.isfinite(row["return_std"]) for row in rows):
        raise ValueError("market depth is too small for a finite sample standard deviation")
    summary = pd.DataFrame(rows)
    result = pd.concat(pieces, ignore_index=True)
    summary.to_csv(out / "returns_summary.csv", index=False)
    result.to_csv(out / "returns.csv.gz", index=False)
    render_returns(result, summary, out, price)


def archive_returns(args):
    tables = ROOT / "results/tables"
    if args.source == "quenched":
        if args.activity not in ACTIVITIES or args.ratio != 1:
            raise ValueError("quenched paths have ratio 1 and activity .001, .01, .1 or .5")
        path = (ROOT / "results/market_time_audit_20261113/tables"
                / f"quenched_a_{args.activity:g}.csv")
        source_depth = json.loads((tables / "market_metadata.json").read_text())["config"]["market_depth"]
        frame = pd.read_csv(path, usecols=["time", "return"])
        frame["demand"] = np.rint(frame["return"].to_numpy() * source_depth).astype(np.int64)
        frame["p"] = 0.592746
        return frame, path, source_depth
    if args.source == "market":
        if args.activity not in ACTIVITIES:
            raise ValueError("archived market activity must be .001, .01, .1 or .5")
        path = tables / "market_returns.csv.gz"
        keys = ["p", "activity", "demand", "replicate", "configuration", "step", "active_cluster_count"]
        depth = json.loads((tables / "market_metadata.json").read_text())["config"]["market_depth"]
        ratio = "p_ratio"
    else:
        if args.activity != .01:
            raise ValueError("archived network activity is fixed at .01")
        path = tables / "network_market_returns.csv.gz"
        keys = ["p", "activity", "demand", "topology", "network_replicate", "step", "active_cluster_count"]
        depth = json.loads((tables / "network_metadata.json").read_text())["config"]["market_depth"]
        ratio = "relative_p"
    if args.command == "fit":
        keys += ["return", "z"]
    chunks = []
    for frame in pd.read_csv(path, usecols=keys + [ratio], chunksize=150_000):
        keep = (frame[ratio] == args.ratio) & (frame["activity"] == args.activity)
        if args.source == "network":
            keep &= frame["topology"] == args.topology
        chunks.append(frame.loc[keep])
    result = pd.concat(chunks, ignore_index=True)
    if result.empty:
        raise ValueError("no archived returns match the request")
    return result, path, depth


def fit_input(args):
    if args.source in ("market", "network"):
        frame, path, _ = archive_returns(args)
        if args.source == "network":
            frame = frame[frame["return"] != 0]
        return ccdf(frame["z"].abs().dropna().to_numpy()), path, "Absolute standardized return", {}
    if args.source == "run":
        if args.run is None:
            raise ValueError("use --run with --source run")
        path = args.run / "returns.csv.gz"
        frame = pd.read_csv(path)
        if args.p is None and frame.p.nunique() > 1:
            raise ValueError("use --p to select one probability from the previous run")
        if args.p is not None:
            frame = frame[np.isclose(frame.p, args.p, rtol=1e-12, atol=1e-15)]
        frame = frame[(frame.topology == "lattice") | (frame["return"] != 0)]
        return ccdf(frame.z.abs().dropna().to_numpy()), path, "Absolute standardized return", {}
    length = args.length or (32 if args.source == "soc" else 128)
    if args.source == "soc":
        if args.run:
            record = json.loads((args.run / "run.json").read_text())
            if record.get("command") != "soc" or record.get("status") != "ok":
                raise ValueError("use a completed SOC run with --source soc --run")
            config = record["parameters"]
            if args.length is not None and args.length != config["length"]:
                raise ValueError("length does not match the previous SOC run")
            path = args.run / "events.csv.gz"
            frame = pd.read_csv(path, usecols=["step", "size"])
            steps, burn = config["steps"], config["burn_in"]
            settings = dict(length=config["length"], mechanism=config["mode"], seed=config["seed"])
        else:
            root = ROOT / "results/soc_research_v2_full"
            path = root / args.mechanism / "events.csv.gz"
            config = yaml.safe_load((root / "config.yaml").read_text())
            if length not in config["burn_in_by_length"]:
                raise ValueError("archived SOC sizes are 32, 64 and 128")
            frame = pd.read_csv(path, usecols=["length", "step", "size"])
            frame = frame[frame.length == length]
            steps, burn = config["steps_by_length"][length], config["burn_in_by_length"][length]
            settings = dict(length=length, mechanism=args.mechanism, seed=config["seed"])
        burn = args.burn_in if args.burn_in is not None else burn
        if burn >= steps:
            raise ValueError("burn-in must be smaller than number of recorded steps")
        values = frame.loc[(frame.step >= burn) & (frame["size"] > 0), "size"].to_numpy()
        return ccdf(values), path, "Removed nodes", dict(settings, burn_in=burn)

    exclusion = {"spanning": "all_spanning_excluded", "largest": "largest_excluded", "none": "all"}
    path = ROOT / "results/criticality_revision_20261113/full/tables/cluster_histogram.csv"
    frame = pd.read_csv(path, usecols=["length", "p_ratio", "exclusion", "cluster_size", "C_s"])
    group = frame[(frame.length == length) & (frame.p_ratio == args.ratio)
                  & (frame.exclusion == exclusion[args.exclusion])].sort_values("cluster_size")
    return ccdf(group.cluster_size.to_numpy(), group.C_s.to_numpy()), path, "Cluster size", {"length": length}


def run_fit(args, out, metadata):
    if args.xmin >= args.xmax:
        raise ValueError("xmin must be smaller than xmax")
    if args.run and args.source not in ("run", "soc"):
        raise ValueError("--run requires --source run or --source soc")
    if args.burn_in is not None and args.source != "soc":
        raise ValueError("--burn-in requires --source soc")
    (x, y, remaining), source, label, settings = fit_input(args)
    keep = (x >= args.xmin) & (x <= args.xmax)
    if keep.sum() < 3:
        raise ValueError("the requested interval needs at least three distinct positive data points")
    regression = linregress(np.log10(x[keep]), np.log10(y[keep]))
    weights = -np.diff(np.append(remaining, 0))
    result = dict(xmin=args.xmin, xmax=args.xmax, observed_min=float(x[keep][0]),
                  observed_max=float(x[keep][-1]), distinct_points=int(keep.sum()),
                  observations=int(weights[keep].sum()), slope=regression.slope,
                  pdf_exponent=1 - regression.slope, r_squared=regression.rvalue**2)
    pd.DataFrame([result]).to_csv(out / "fit.csv", index=False)
    pd.DataFrame(dict(x=x, ccdf=y, remaining=remaining, fitted_interval=keep)).to_csv(
        out / "ccdf.csv", index=False)
    if args.source == "soc":
        events = np.repeat(x, weights)
        rows = [fit_tail(events, int(np.ceil(args.xmin)), int(np.floor(args.xmax)), model)
                for model in ("power", "cutoff", "exponential")]
        pd.DataFrame(rows).to_csv(out / "soc_tail_models.csv", index=False)
    fig, ax = plt.subplots(layout="constrained")
    ax.loglog(x, y, ".", label="Empirical CCDF")
    ax.loglog(x[keep], 10**(regression.intercept + regression.slope * np.log10(x[keep])),
              label="Interval OLS")
    ax.axvspan(x[keep][0], x[keep][-1], alpha=.15)
    ax.set(xlabel=label, ylabel="CCDF", title="Descriptive tail fit")
    ax.legend()
    fig.savefig(out / "fit.png", dpi=150)
    plt.close(fig)
    metadata.update(source_path=str(source.resolve()), reused=["existing observations"], result=result,
                    effective_settings=settings)
    if not args.run and args.source != "soc":
        metadata["archive_seed"] = {
            "market": 20261006, "network": 20261006, "cluster": 20261113,
        }[args.source]


def run_soc(args, out, metadata):
    stats = args.mode == "stats"
    steps = args.steps or (1200 if stats else 240)
    burn = args.burn_in if args.burn_in is not None else steps // 3
    replicates = args.replicates or (5 if stats else 1)
    if stats and replicates < 2:
        raise ValueError("SOC stats mode needs at least two paths per initial occupancy")
    p0s = sorted(set(args.p0 or ([.20, .50, .85] if stats else [.50])))
    config = ResearchConfig(args.length, steps, burn, args.activity, args.theta, args.q,
                            args.drive_nodes, args.mechanism, args.max_rounds)
    archive = ROOT / "results/soc_research_v2_full/critical_windows.csv"
    window_row = pd.read_csv(archive).set_index("length")
    window = window_row.loc[args.length, ["window_low", "window_high"]].tolist() if args.length in window_row.index else None
    paths, events, rounds, summaries = [], [], [], []
    for p0 in p0s:
        identity = int(np.float64(p0).view(np.uint64))
        for index in range(replicates):
            seed = child_seed(args.seed, 4, identity, index)
            trace, event, round_trace = simulate(config, p0, seed, f"p{identity}_r{index}")
            for frame in (trace, event, round_trace):
                frame["initial_occupancy"] = p0
                frame["path_seed"] = seed
            paths.append(trace)
            events.append(event)
            rounds.append(round_trace)
            retained = trace[trace.step >= burn]
            events_after = event[event.step >= burn]
            occupancy = retained.occupancy_after
            summaries.append(dict(
                path_id=trace.path_id.iloc[0], initial_occupancy=p0, path_seed=seed,
                mean_occupancy=occupancy.mean(), window_fraction=(
                    occupancy.between(*window).mean() if window is not None else np.nan),
                event_rate=(retained.duration > 0).mean(),
                positive_event_rate=(retained.removed_nodes > 0).mean(),
                mean_drive=retained.added_nodes.mean(), mean_removed=retained.removed_nodes.mean(),
                mean_positive_event_size=events_after.loc[events_after["size"] > 0, "size"].mean(),
                multi_round_events=int((events_after.duration > 1).sum()),
                capped_events=int((events_after.end_reason == "max_rounds").sum()),
            ))
    paths = pd.concat(paths, ignore_index=True)
    events = pd.concat(events, ignore_index=True)
    rounds = pd.concat(rounds, ignore_index=True)
    for name, frame in (("paths", paths), ("events", events), ("rounds", rounds)):
        frame.to_csv(out / f"{name}.csv.gz", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "soc_summary.csv", index=False)
    if stats:
        metrics = ["mean_occupancy", "window_fraction", "event_rate", "positive_event_rate",
                   "mean_drive", "mean_removed"]
        summary.groupby("initial_occupancy")[metrics].agg(["mean", "std"]).to_csv(out / "soc_ensemble.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for _, group in paths.groupby("path_id"):
        axes[0].plot(group.step, group.occupancy_after, alpha=.5)
    if window is not None:
        axes[0].axhspan(*window, alpha=.2, color="gray", label="Static reference window")
        axes[0].legend(fontsize=7)
    axes[0].axvline(burn, color="black", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="Slow step", ylabel="Occupied fraction")
    x, y, _ = ccdf(events.loc[events.step >= burn, "size"].to_numpy())
    if x.size:
        axes[1].loglog(x, y, ".")
    else:
        axes[1].text(.5, .5, "No positive events after burn-in", ha="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="Removed nodes", ylabel="Positive event size CCDF")
    fig.savefig(out / "soc.png", dpi=150)
    plt.close(fig)
    metadata.update(parameters=dict(asdict(config), experiment_mode=args.mode, p0=p0s,
                                    replicates=replicates, seed=args.seed,
                                    path_seeds=[row["path_seed"] for row in summaries]),
                    static_window=window, recalculated=["complete SOC paths", "event statistics"])


def main():
    parser, args = arguments()
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_root = ROOT / "outputs"
    out = output_root / f"{args.command}-{stamp}-{uuid4().hex[:8]}"
    metadata = dict(command=args.command, started_utc=stamp, software=VERSIONS,
                    parameters={key: str(value.resolve()) if isinstance(value, Path) else value
                                for key, value in vars(args).items() if key != "command"})
    try:
        out.mkdir(parents=True)
        if args.command == "cb":
            config = cb_config(args)
            prior = pd.read_csv(args.compare / "phase.csv") if args.compare else None
            if prior is not None and not {"p", "chi", "p_infinity"}.issubset(prior.columns):
                raise ValueError("comparison needs a previous CB phase.csv")
            if args.compare:
                previous = json.loads((args.compare / "run.json").read_text())["parameters"]
                different = [key for key in ("p", "exclusion", "node_count", "boundary")
                             if previous.get(key) != config[key]]
                if different:
                    metadata["comparison_note"] = "Different comparison settings: " + ", ".join(different)
            inputs = [ROOT / "defense/cb_experiment.py"]
            inputs.extend(ROOT / "src/local_market" / f"{name}.py"
                          for name in ("cb_model", "networks", "percolation", "statistics"))
            version = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in inputs}
            cache = Cache(output_root / ".cache", [version, VERSIONS])
            phase, peak, returns, histogram, labels, degrees = simulate_cb(config, cache)
            return_table(returns, config["depth"], out)
            phase.to_csv(out / "phase.csv", index=False)
            histogram.to_csv(out / "clusters.csv", index=False)
            np.savez_compressed(out / "example_configuration.npz", labels=labels,
                                p=phase.p.iloc[0], graph_seed=config["network_seeds"][0])
            draw_phase(phase, out, config, labels, degrees, prior)
            metadata.update(parameters=config, stage_cache=cache.usage, peak=peak,
                            code_hashes=version, recalculated=["returns", "statistics", "figures"])
        elif args.command == "depth":
            if args.source != "quenched" and args.initial_log_price != 0:
                raise ValueError("an initial log price requires a continuous quenched path")
            frame, source, old_depth = archive_returns(args)
            initial = args.initial_log_price if args.source == "quenched" else None
            return_table(frame, args.depth, out, args.threshold, initial_log_price=initial)
            metadata.update(source_path=str(source.relative_to(ROOT)), original_depth=old_depth,
                            archive_p=float(frame.p.iloc[0]), archive_seed=20261006,
                            reused=["archived excess demand"],
                            recalculated=["returns", "statistics", "figures"])
        elif args.command == "fit":
            run_fit(args, out, metadata)
        else:
            run_soc(args, out, metadata)
    except (ValueError, OSError, OverflowError) as error:
        if out.is_dir():
            metadata.update(status="failed", error=str(error), seconds=time.perf_counter() - started)
            (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        parser.exit(2, f"error: {error}\n")
    metadata.update(status="ok", seconds=round(time.perf_counter() - started, 3))
    (out / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in metadata.items() if key != "code_hashes"},
                     indent=2, ensure_ascii=False))
    print(f"Files saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
