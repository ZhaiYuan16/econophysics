#!/usr/bin/env python3
"""Reproduce visual and numerical diagnostics for the archived CB market tails."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import halfnorm

from local_market.tail_fitting import select_pareto_tail
from local_market.hierarchical_statistics import block_histogram, resampled_counts, weighted_pareto_fit
from local_market.statistics import nonzero_absolute_standardized_returns
from local_market.plotting import plot_tail_exponents

ROOT = Path("results/market_tail_audit_20261114")
FIG = ROOT / "figures"
TAB = ROOT / "tables"
QUANTILES = (.70, .80, .85, .90, .925, .95)


def truncated_fit(tail: np.ndarray, xmin: float) -> tuple[float, float, float]:
    """Continuous x^-alpha exp[-lambda(x-xmin)] density, conditional x>=xmin."""
    y = tail / xmin
    logy_sum = np.log(y).sum()
    excess_sum = (y - 1).sum()
    n = len(y)

    def objective(theta: np.ndarray) -> float:
        alpha, rate = theta
        norm = quad(lambda u: np.exp(-alpha*u-rate*(np.exp(u)-1)+u),
                    0, min(40., np.log(max(1e3, 100/rate))), epsrel=2e-6)[0]
        return alpha*logy_sum + rate*excess_sum + n*np.log(norm*xmin)

    fit = minimize(objective, [min(2., 1+len(y)/max(logy_sum, 1)), .05],
                   bounds=[(0, 250), (1e-8, 250)], method="L-BFGS-B")
    return float(fit.x[0]), float(fit.x[1]/xmin), float(-fit.fun)


def model_curves(x: np.ndarray, xmin: float, alpha: float,
                 truncated_alpha: float, rate: float, exp_rate: float):
    pareto = (x/xmin)**(1-alpha)
    exponential = np.exp(-exp_rate*(x-xmin))
    # Integrate on log x, retaining the actual fitted continuous cutoff model.
    def integral(lower):
        return quad(lambda u: np.exp((1-truncated_alpha)*u-rate*xmin*(np.exp(u)-1)),
                    np.log(lower/xmin), min(40., np.log(max(1e3,100/max(rate*xmin,1e-9)))),
                    epsrel=2e-5)[0]
    norm = integral(xmin)
    truncated = np.array([integral(v)/norm for v in x])
    return pareto, truncated, exponential


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG/name, dpi=170)
    plt.close(fig)


def corrected_uncertainty(group: pd.DataFrame, best, seed: int):
    """Configuration bootstrap, excluding exact zero raw returns before each fit."""
    ids = group.replicate.to_numpy(dtype=int)*100+group.configuration.to_numpy(dtype=int)
    support, blocks = block_histogram(group["return"].to_numpy(dtype=float), ids)
    rng = np.random.default_rng(seed)
    estimates = []
    model_wins = {"pure power law":0,"truncated power law":0,"exponential":0}
    for _ in range(120):
        picked = np.concatenate([r*100+rng.integers(0,100,100) for r in range(4)])
        weights = resampled_counts(blocks,picked)
        mean = np.dot(support,weights)/weights.sum()
        std = np.sqrt(np.dot((support-mean)**2,weights)/(weights.sum()-1))
        absolute = np.abs((support[support!=0]-mean)/std)
        fit = weighted_pareto_fit(absolute,weights[support!=0],QUANTILES,500)
        estimates.append(fit["alpha_pdf"])
        tail = np.repeat(absolute[absolute>=fit["xmin"]],
                         weights[support!=0][absolute>=fit["xmin"]])
        n,cut,alpha = len(tail),fit["xmin"],fit["alpha_pdf"]
        exp_rate = 1/np.mean(tail-cut)
        _,_,ll_trunc = truncated_fit(tail,cut)
        ll_pure = n*np.log((alpha-1)/cut)-alpha*np.log(tail/cut).sum()
        ll_exp = n*np.log(exp_rate)-exp_rate*(tail-cut).sum()
        aics={"pure power law":2-2*ll_pure,
              "truncated power law":4-2*ll_trunc,"exponential":2-2*ll_exp}
        model_wins[min(aics,key=aics.get)]+=1
    ci = np.quantile(estimates,[.025,.975])
    # Semiparametric null: resample configuration tail counts, replace tail
    # magnitudes under the fitted Pareto, and reselect xmin in each simulation.
    original = np.asarray(blocks.sum(axis=0)).ravel()
    mean = np.dot(support,original)/original.sum()
    std = np.sqrt(np.dot((support-mean)**2,original)/(original.sum()-1))
    absz = np.abs((support-mean)/std)
    eligible = support!=0
    body = eligible & (absz<best.xmin)
    tail_mask = eligible & (absz>=best.xmin)
    exceed = 0
    for _ in range(100):
        picked = np.concatenate([r*100+rng.integers(0,100,100) for r in range(4)])
        counts = resampled_counts(blocks,picked)
        ntail = int(counts[tail_mask].sum())
        synthetic_tail = best.xmin*(1-rng.random(ntail))**(-1/(best.alpha_pdf-1))
        synthetic = weighted_pareto_fit(np.concatenate([absz[body],synthetic_tail]),
                      np.concatenate([counts[body],np.ones(ntail,dtype=np.int64)]),QUANTILES,500)
        exceed += synthetic["ks_distance"]>=best.ks_distance
    return ci,(exceed+1)/101,{key:value/120 for key,value in model_wins.items()}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv("results/tables/market_returns.csv.gz")
    old = pd.read_csv("results/tables/market_tail_fits.csv")
    revised = pd.read_csv("results/downstream_audit_20261114/tables/market_revised_statistics.csv")
    scan_rows = []
    panels = {kind: plt.subplots(3, 4, figsize=(19, 13)) for kind in
              ("ccdf", "zipf", "discreteness", "xmin_scan")}
    rows = []
    anomaly = []
    for index, ((ratio, activity), group) in enumerate(data.groupby(["p_ratio", "activity"], sort=True)):
        oldrow = old[(np.isclose(old.p_ratio,ratio)) & (np.isclose(old.activity,activity))].iloc[0]
        newrow = revised[(np.isclose(revised.p_ratio,ratio)) & (np.isclose(revised.activity,activity))].iloc[0]
        raw = group["return"].to_numpy(dtype=float)
        x = nonzero_absolute_standardized_returns(raw, group.z.to_numpy(dtype=float))
        best,candidates = select_pareto_tail(x,QUANTILES,500)
        scan_rows.extend({"p_ratio":ratio,"activity":activity,**c} for c in candidates)
        ci,gof,model_freq = corrected_uncertainty(group,best,20261114+index)
        xmin = float(best.xmin)
        tail = np.sort(x[x>=xmin])
        unique, count = np.unique(tail, return_counts=True)
        alpha = float(best.alpha_pdf)
        exrate = 1 / np.mean(tail-xmin)
        ta, rate, ll_trunc = truncated_fit(tail, xmin)
        ll_pure = len(tail)*np.log((alpha-1)/xmin)-alpha*np.log(tail/xmin).sum()
        ll_exp = len(tail)*np.log(exrate)-exrate*(tail-xmin).sum()
        aics = {"pure power law":2-2*ll_pure, "truncated power law":4-2*ll_trunc,
                "exponential":2-2*ll_exp}
        preferred = min(aics, key=aics.get)
        dynamic = tail[-1]/xmin
        # A failure on any criterion means that an exponent is descriptive only.
        reliable = bool(gof>=.1 and
                        len(tail)>=1000 and dynamic>=10 and len(unique)>=30 and
                        aics["pure power law"]<=min(aics.values())+2)
        rows.append({"p_ratio":ratio,"activity":activity,"x_min":xmin,"x_max":tail[-1],
                     "n_nonzero":len(x),"n_tail":len(tail),"unique_tail":len(unique),
                     "x_max/x_min":dynamic,"alpha":alpha,
                     "old_alpha_CI":f"[{oldrow.alpha_ci_low:.3g}, {oldrow.alpha_ci_high:.3g}]",
                     "alpha_CI":f"[{ci[0]:.3g}, {ci[1]:.3g}]",
                     "KS":best.ks_distance,"GOF_p":gof,
                     "old_alpha":oldrow.alpha_pdf,"old_xmin":oldrow.xmin,
                     "preferred_model":preferred,"power_law_reliable":reliable,
                     "aic_pure":aics["pure power law"],"aic_truncated":aics["truncated power law"],
                     "aic_exponential":aics["exponential"],"truncated_alpha":ta,
                     "truncated_lambda":rate,"exponential_lambda":exrate,
                     "max_value_frequency":int(count[-1]),"largest_frequency_fraction":count.max()/len(tail)})
        rows[-1].update({"bootstrap_preferred_"+k.replace(" ","_"):v for k,v in model_freq.items()})
        title = f"p/pc={ratio:g}, a={activity:g}"
        ix = np.unravel_index(index,(3,4))
        ax = panels["ccdf"][1][ix]
        survival = np.cumsum(count[::-1])[::-1]/len(x)
        ax.step(unique, survival, where="post", color="black",lw=1,label="empirical")
        grid = np.geomspace(xmin,tail[-1],120)
        curves = model_curves(grid,xmin,alpha,ta,rate,exrate)
        for curve,label,color in zip(curves,("Pareto","truncated","exponential"),("tab:blue","tab:orange","tab:green")):
            ax.plot(grid,len(tail)/len(x)*curve,color=color,lw=1.3,label=label)
        ax.axvline(xmin,color="gray",ls=":",label="xmin")
        ax.set(xscale="log",yscale="log",title=title,xlabel="|z|",ylabel="P(|z| >= x)")
        ax.text(.03,.04,f"alpha={alpha:.2f}, xmin={xmin:.3g}\nn={len(tail)}, GOF p={gof:.3g}\nrange={dynamic:.2f}x",transform=ax.transAxes,fontsize=8,
                bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
        ax = panels["zipf"][1][ix]
        rank = np.arange(1,len(tail)+1)
        ax.plot(rank,tail[::-1],".",ms=1,color="black",label="raw ordered tail")
        ref = xmin*(len(tail)/rank)**(1/max(alpha-1,1e-8))
        ax.plot(rank,ref,color="tab:blue",lw=1,label="Pareto reference")
        ax.set(xscale="log",yscale="log",title=title,xlabel="rank k",ylabel="x(k)")
        ax.text(.03,.04,f"n={len(tail)}, unique={len(unique)}",transform=ax.transAxes,fontsize=8)
        ax = panels["discreteness"][1][ix]
        ax.scatter(unique,count,s=7,color="tab:purple")
        ax.set(xscale="log",yscale="log",title=title,xlabel="distinct |z|",ylabel="frequency")
        ax.text(.03,.04,f"unique_tail={len(unique)}\nn_tail={len(tail)}",transform=ax.transAxes,fontsize=8,
                bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
        ax = panels["xmin_scan"][1][ix]
        c = pd.DataFrame(candidates).sort_values("xmin")
        ax.plot(c.xmin,c.ks_distance,"o-",color="tab:blue",label="KS")
        ax2=ax.twinx(); ax2.plot(c.xmin,c.alpha_pdf,"s--",color="tab:orange",ms=3,label="alpha")
        ax3=ax.twinx(); ax3.spines["right"].set_position(("axes",1.13))
        ax3.plot(c.xmin,c.tail_count,"^:",color="tab:green",ms=3,label="n tail")
        ax.axvline(xmin,color="gray",ls=":")
        ax.set(title=title,xlabel="candidate xmin",ylabel="KS",xscale="log")
        ax2.set_ylabel("alpha",color="tab:orange",fontsize=8)
        ax3.set_ylabel("n tail",color="tab:green",fontsize=8)
        if alpha>10: anomaly.append((title,x,tail,xmin,alpha,ta,rate,exrate))
        print(title,"alpha",round(alpha,2),"range",round(dynamic,2),"unique",len(unique),"model",preferred,flush=True)
    for kind,(fig,axes) in panels.items():
        if kind=="ccdf": axes[0,0].legend(fontsize=7)
        if kind=="zipf": axes[0,0].legend(fontsize=7)
        fig.suptitle({"ccdf":"Tail CCDF and conditional fits","zipf":"Rank-size: raw repeated observations remain visible",
                      "discreteness":"Exact tail value frequencies","xmin_scan":"xmin selection sensitivity: KS, alpha, n tail"}[kind])
        save(fig,f"market_tail_{kind}.png")
    if anomaly:
        fig,axs=plt.subplots(len(anomaly),2,figsize=(13,4*len(anomaly)),squeeze=False)
        for row,(title,x,tail,xmin,alpha,ta,rate,exrate) in enumerate(anomaly):
            bins=np.geomspace(max(x.min(),xmin/10),x.max(),70)
            for col in range(2):
                ax=axs[row,col]
                ax.hist(x,bins=bins,density=True,histtype="step",color="black",label="observed")
                grid=np.geomspace(xmin,x.max(),150)
                norm=quad(lambda u: np.exp((1-ta)*u-rate*xmin*(np.exp(u)-1)),0,40,epsrel=1e-5)[0]*xmin
                curves=((alpha-1)/xmin*(grid/xmin)**(-alpha),
                        (grid/xmin)**(-ta)*np.exp(-rate*(grid-xmin))/norm,
                        exrate*np.exp(-exrate*(grid-xmin)),
                        halfnorm.pdf(grid,scale=1))
                for y,label in zip(curves,("Pareto tail","truncated tail","exponential tail","half normal reference")):
                    ax.plot(grid,y*len(tail)/len(x) if label!="half normal reference" else y,lw=1,label=label)
                ax.axvline(xmin,color="gray",ls=":")
                ax.set(xscale="log",yscale="log" if col else "linear",title=f"{title}: {'log-log' if col else 'log-linear'}",xlabel="|z|",ylabel="density")
                if row==0: ax.legend(fontsize=7)
        save(fig,"market_tail_high_alpha_histograms.png")
    summary=pd.DataFrame(rows)
    summary.to_csv(TAB/"market_tail_diagnostics.csv",index=False)
    pd.DataFrame(scan_rows).to_csv(TAB/"market_tail_xmin_scan.csv",index=False)
    plot_tail_exponents(summary,FIG/"market_tail_exponents.png")
    (TAB/"method.json").write_text(json.dumps({"source":"archived 200000 returns per combination",
      "continuous_approximation":"|z| is discrete because D is integer; continuous model curves are diagnostic approximations",
      "gof":"selection-aware, within-replicate configuration-block semiparametric null, 100 draws; zeros removed before xmin selection",
      "confidence_interval":"120 bootstrap draws: within each of 4 replicates resample 100 configuration blocks; restandardize and reselect xmin; zeros excluded; refit all three conditional tail models and record AIC winner each draw",
      "reliability_rule":"GOF p>=0.1, n_tail>=1000, range>=10, unique_tail>=30, pure Pareto AIC within 2 of best",
      "preferred_model":"minimum continuous conditional AIC at original selected xmin; interpret as descriptive when discrete peaks appear"},indent=2)+"\n")

if __name__=="__main__": main()
