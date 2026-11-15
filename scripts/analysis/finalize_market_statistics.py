#!/usr/bin/env python3
"""Build the market chapter's four core figures from archived observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from local_market.config import MarketAnalysisConfig
from local_market.experiments import summarize_market_returns
from local_market.plotting import plot_return_distributions, plot_tail_exponents
from local_market.statistics import return_distribution_table


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/tables/market_returns.csv.gz"))
    parser.add_argument("--time-tables", type=Path,
                        default=Path("results/market_time_audit_20261113/tables"))
    parser.add_argument("--tail-table", type=Path,
                        default=Path("results/market_tail_audit_20261114/tables/market_tail_diagnostics.csv"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/market_statistics_final_20261114"))
    return parser.parse_args()


def plot_archived_time_paths(tables: Path, activities: tuple[float, ...], path: Path) -> None:
    fig, axes = plt.subplots(len(activities), 3, figsize=(15, 2.75*len(activities)), squeeze=False)
    for row, activity in enumerate(activities):
        trace = pd.read_csv(tables / f"quenched_a_{activity:g}.csv")
        assert np.array_equal(trace.time.to_numpy(), np.arange(1,len(trace)+1))
        np.testing.assert_allclose(trace["return"].cumsum(), trace.log_price, atol=1e-12)
        t = trace.time.to_numpy()
        axes[row,0].plot(t,trace["return"],lw=.5)
        axes[row,1].plot(t,np.abs(trace["return"]),lw=.5,color="tab:orange")
        axes[row,2].plot(t,trace.log_price,lw=.8,color="tab:green")
        for axis in axes[row]:
            axis.grid(alpha=.2)
            axis.set_xlabel("simulation time t")
        axes[row,0].set_ylabel(f"a={activity:g}\nr(t)")
        axes[row,1].set_ylabel("|r(t)|")
        axes[row,2].set_ylabel("cumulative log return")
    for axis,title in zip(axes[0],("log return","absolute log return","log price relative to t=0")):
        axis.set_title(title)
    fig.suptitle("One fixed p/pc=1 occupancy; independent cluster actions each step")
    fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)


def plot_market_activity(summary: pd.DataFrame, path: Path) -> None:
    specs=(("return_mean","Mean r"),("return_variance","Var(r)"),
           ("mean_absolute_return","Mean |r|"),("p_abs_z_gt_5",r"P(|z|>5)"),
           ("return_excess_kurtosis","Excess kurtosis"),
           ("mean_active_cluster_count","Mean active clusters"))
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for ratio,group in summary.groupby("p_ratio",sort=True):
        group=group.sort_values("activity")
        for ax,(column,label) in zip(axes.flat,specs):
            ax.plot(group.activity,group[column],"o-",label=f"p/pc={ratio:g}")
            ax.set(xscale="log",xlabel="activity a",ylabel=label)
            ax.grid(alpha=.25)
    axes[0,0].axhline(0,color="gray",lw=.8)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("CB ensemble statistics: 400 configurations and 500 actions each per condition")
    fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args=arguments()
    config=MarketAnalysisConfig.from_yaml("configs/market_analysis.yaml")
    figures=args.output_root/"figures"
    tables=args.output_root/"tables"
    figures.mkdir(parents=True,exist_ok=True)
    tables.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.source)
    returns,summary,_=summarize_market_returns(raw,config)
    np.testing.assert_allclose(raw.z.to_numpy(),returns.z.to_numpy(),atol=1e-12)
    assert len(summary)==len(config.p_ratios)*len(config.activities)==12
    assert (summary.observations==config.replicates*config.configurations_per_replicate*
            config.trades_per_configuration).all()
    distribution_rows=[]
    absolute_means=[]
    for key,group in returns.groupby(["length","p_ratio","p","activity"],sort=True):
        raw_return=group["return"].to_numpy(dtype=float)
        z=group.z.to_numpy(dtype=float)
        absolute_means.append({"p_ratio":key[1],"activity":key[3],
                               "mean_absolute_return":float(np.abs(raw_return).mean()),
                               "return_variance":float(np.var(raw_return,ddof=1)),
                               "nonzero_returns":int(np.count_nonzero(raw_return))})
        distribution_rows.extend(dict(zip(("length","p_ratio","p","activity"),key,strict=True)) | row
                                 for row in return_distribution_table(z,raw_returns=raw_return))
    summary=summary.merge(pd.DataFrame(absolute_means),on=["p_ratio","activity"],validate="one_to_one")
    np.testing.assert_allclose(summary.return_variance,summary.return_std**2,rtol=1e-12)
    distribution=pd.DataFrame(distribution_rows)
    tail=pd.read_csv(args.tail_table)
    assert len(tail)==12 and set(tail.power_law_reliable)=={False}
    for _,row in tail.iterrows():
        check=summary[np.isclose(summary.p_ratio,row.p_ratio)&
                      np.isclose(summary.activity,row.activity)].iloc[0]
        assert row.n_nonzero==check.nonzero_returns
    summary.to_csv(tables/"market_activity_statistics.csv",index=False)
    distribution.to_csv(tables/"market_pdf_ccdf.csv",index=False)
    tail.to_csv(tables/"market_tail_diagnostics.csv",index=False)
    plot_archived_time_paths(args.time_tables,config.activities,figures/"market_time_series.png")
    plot_market_activity(summary,figures/"market_activity_statistics.png")
    plot_return_distributions(distribution,figures/"market_pdf_ccdf.png")
    plot_tail_exponents(tail,figures/"market_tail_exponents.png")
    inputs=[args.source,args.tail_table]+[
        args.time_tables/f"quenched_a_{activity:g}.csv" for activity in config.activities]
    metadata={"input_sha256":{str(path):sha256(path) for path in inputs},
              "ensemble":"4 repeats x 100 independent configurations x 500 actions per condition",
              "time_paths":"archived separate quenched paths at p/pc=1; not the ordered ensemble",
              "normalization":"r=D/N, N=16384; z=(r-mean_all)/std_all(ddof=1)",
              "zero_rule":"signed PDF and unconditional CCDF include raw zero returns; tail fits exclude raw return==0 before |z|",
              "tail_fit":"audited 120 configuration-bootstrap intervals and 100-draw GOF are reused without recomputation"}
    (tables/"provenance.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n")
    print(f"Wrote 4 core figures and 3 tables to {args.output_root}")


if __name__=="__main__":main()
