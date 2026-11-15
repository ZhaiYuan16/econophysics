#!/usr/bin/env python3
"""Run versioned SOC paths and derive finite-size windows from static P_x."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from local_market.soc_research import ResearchConfig, simulate


def windows(summary: pd.DataFrame, lengths: list[int]) -> pd.DataFrame:
    out = []
    for length in lengths:
        group = summary[summary.length == length].sort_values("p")
        x = group.p.to_numpy()
        y = group.span_x_mean.to_numpy()
        if not (y.min() <= .1 and y.max() >= .9):
            raise ValueError(f"static P_x does not bracket both endpoints for L={length}")
        # Isotonic envelope removes minor binomial noise before inversion.
        y = np.maximum.accumulate(y)
        low, high = np.interp([.1, .9], y, x)
        peak = group.loc[group.chi_all_spanning_excluded.idxmax(), "p"]
        out.append(dict(length=length, window_low=low, window_high=high,
                        susceptibility_peak=peak, method="P_x isotonic linear interpolation"))
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/soc_research_v1.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(args.config.read_text())
    (args.output / "config.yaml").write_text(args.config.read_text())
    lengths = [32, 64, 128] if not args.smoke else [32, 64, 128]
    win = windows(pd.read_csv(cfg["static_summary"]), lengths)
    win.to_csv(args.output / "critical_windows.csv", index=False)
    steps = 60 if args.smoke else cfg["steps"]
    burn = 20 if args.smoke else cfg["burn_in"]
    reps = 1 if args.smoke else cfg["replicates"]
    start = time.perf_counter()
    summary_rows = []
    for condition in cfg["conditions"]:
        if args.smoke and condition["name"] not in ("baseline", "cascade"):
            continue
        name = condition["name"]
        folder = args.output / name
        folder.mkdir()
        trace_files = []
        event_files = []
        round_files = []
        for l in condition.get("lengths", lengths):
            for initial in condition.get("initial_occupancies", cfg["initial_occupancies"]):
                for rep in range(reps):
                    seed = int(np.random.SeedSequence([cfg["seed"],
                        cfg["conditions"].index(condition), l, int(initial*100), rep]).generate_state(1)[0])
                    path_id = f"{name}_L{l}_p{int(initial*100):02d}_r{rep:02d}"
                    local_steps = 60 if args.smoke else int(cfg.get("steps_by_length", {}).get(l, steps))
                    local_burn = 20 if args.smoke else int(cfg.get("burn_in_by_length", {}).get(l, burn))
                    conf = ResearchConfig(l, local_steps, local_burn, cfg["activity"],
                        condition.get("theta", cfg["theta"]), condition.get("q", cfg["q"]),
                        condition.get("drive_nodes", cfg["drive_nodes"]), condition["mode"])
                    trace, event, rounds = simulate(conf, initial, seed, path_id)
                    for frame in (trace, event, rounds):
                        frame["condition"] = name
                        frame["length"] = l
                        frame["initial_occupancy"] = initial
                        frame["replicate"] = rep
                        frame["seed"] = seed
                    trace_files.append(trace)
                    event_files.append(event)
                    round_files.append(rounds)
                    ss = trace[trace.step >= local_burn]
                    low, high = win.set_index("length").loc[l, ["window_low", "window_high"]]
                    p = ss.occupancy_after.to_numpy()
                    half = len(p)//2
                    summary_rows.append(dict(condition=name, length=l, initial_occupancy=initial,
                        replicate=rep, path_id=path_id, seed=seed, steps=local_steps,
                        burn_in=local_burn, mean_occupancy=p.mean(),
                        occupancy_sd=p.std(ddof=1), drift=p[half:].mean()-p[:half].mean(),
                        window_fraction=np.mean((p >= low)&(p <= high)),
                        event_rate=np.mean(ss.duration > 0), event_count=int((ss.duration > 0).sum()),
                        mean_event_size=ss.loc[ss.removed_nodes > 0, "removed_nodes"].mean()))
        for filename, parts in (("paths.csv.gz", trace_files), ("events.csv.gz", event_files),
                                ("rounds.csv.gz", round_files)):
            pd.concat(parts, ignore_index=True).to_csv(folder / filename, index=False, compression="gzip")
        print(f"{name}: {len(trace_files)} paths, {time.perf_counter()-start:.1f}s", flush=True)
    pd.DataFrame(summary_rows).to_csv(args.output / "path_summary.csv", index=False)
    (args.output / "run.json").write_text(json.dumps(dict(smoke=args.smoke, seconds=time.perf_counter()-start,
        steps_by_length={str(l): 60 if args.smoke else int(cfg.get("steps_by_length", {}).get(l, steps))
                         for l in lengths},
        burn_in_by_length={str(l): 20 if args.smoke else int(cfg.get("burn_in_by_length", {}).get(l, burn))
                           for l in lengths},
        replicates=reps, config=str(args.config)), indent=2))


if __name__ == "__main__":
    main()
