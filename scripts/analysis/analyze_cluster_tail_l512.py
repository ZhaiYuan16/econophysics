#!/usr/bin/env python3
"""Configuration-level, finite-interval analysis of the L=512 critical cluster tail."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq, minimize

SOURCE = Path("results/criticality_revision_20261113/full/tables/configuration_cluster_counts.csv")
HIST = Path("results/criticality_revision_20261113/full/tables/cluster_histogram.csv")
OUT = Path("results/criticality_revision_20261114/cluster_tail_L512_A")
SEED = 20261114
XMIN = (1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 70, 100)


def model(c, lo, hi, tau):
    k = np.arange(lo, hi + 1, dtype=float)
    z = np.exp(-tau * np.log(k))
    return z / z.sum()


def fit(c, lo, hi):
    k = np.arange(lo, hi + 1, dtype=float)
    n = c[lo:hi + 1].sum()
    if n < 100 or hi <= lo:
        return None
    logk = np.log(k)
    observed = np.dot(c[lo:hi + 1], logk) / n
    def score(t):
        z = np.exp(-t * logk)
        return np.dot(z, logk) / z.sum() - observed
    tau = brentq(score, .01, 8)
    p = model(c, lo, hi, tau)
    empirical = np.cumsum(c[lo:hi + 1]) / n
    ks = np.max(np.abs(empirical - np.cumsum(p)))
    ll = np.dot(c[lo:hi + 1], np.log(p))
    return dict(tau=tau, ks=ks, n=int(n), ll=ll, aic=2 - 2 * ll)


def upper(c, threshold, width=50, fraction=.6):
    eligible = np.flatnonzero(c >= threshold)
    if len(eligible) == 0:
        return None
    if width == 0:
        return int(eligible[-1])
    good = [s for s in eligible if s >= width and np.mean(c[s-width + 1:s + 1] >= threshold) >= fraction]
    return int(good[-1]) if good else None


def choose(c, hi):
    rows = []
    for lo in XMIN:
        if hi / lo < 10:
            continue
        f = fit(c, lo, hi)
        if f and f['n'] >= 10000:
            rows.append((lo, f))
    if not rows:
        return None
    viable = [(lo, f) for lo, f in rows if lo >= 10]
    floor = min(f['ks'] for _, f in viable)
    plateau = np.median([f['tau'] for lo, f in viable if 20 <= lo <= 50])
    accepted = [(lo, f) for lo, f in viable if f['ks'] <= 1.25 * floor and abs(f['tau'] - plateau) <= .035]
    return (accepted or viable)[0][0]


def cutoff_fit(c, lo, hi, pure):
    k = np.arange(lo, hi + 1, dtype=float)
    n = c[lo:hi + 1].sum()
    logk = np.log(k)
    def negll(x):
        tau, rate = x
        v = -tau * logk - rate * k
        vmax = v.max()
        logz = vmax + np.log(np.exp(v - vmax).sum())
        return tau * np.dot(c[lo:hi + 1], logk) + rate * np.dot(c[lo:hi + 1], k) + n * logz
    # A nonnegative rate nests the pure power law. A zero fitted rate gives
    # only a lower bound on s_c, not an identified cutoff scale.
    opt = minimize(negll, [pure['tau'], .0001], bounds=[(.01, 8), (0, 1)], method='L-BFGS-B',
                   options={'ftol': 1e-12, 'maxiter': 500})
    if not opt.success:
        # Retry from the nested pure model.
        opt = minimize(negll, [pure['tau'], 0], bounds=[(.01, 8), (0, 1)], method='L-BFGS-B')
    tau, rate = opt.x
    return dict(cutoff_tau=tau, rate=rate, s_c=(1 / rate if rate > 1e-8 else np.inf),
                cutoff_aic=4 + 2 * opt.fun, delta_aic=2 + 2 * (opt.fun + pure['ll']),
                identifiable=bool(rate > 1e-8 and rate < .999 and rate * hi >= .1 and opt.success), optimizer_success=bool(opt.success))


def load():
    required = {'length', 'p_ratio', 'p', 'configuration_id', 'exclusion', 'cluster_size', 'count'}
    chunks = []
    for chunk in pd.read_csv(SOURCE, chunksize=500000):
        if not required <= set(chunk):
            raise ValueError(f'Missing columns: {required - set(chunk)}')
        x = chunk[(chunk.length == 512) & np.isclose(chunk.p, .592746, atol=1e-12, rtol=0)
                  & np.isclose(chunk.p_ratio, 1, atol=1e-12, rtol=0)
                  & (chunk.exclusion == 'largest_excluded')]
        if len(x):
            chunks.append(x)
    df = pd.concat(chunks, ignore_index=True)
    if df.empty or df.duplicated(['configuration_id', 'cluster_size']).any() or (df['count'] <= 0).any():
        raise ValueError('Invalid or duplicate configuration counts')
    ids = sorted(df.configuration_id.unique())
    c = np.zeros((len(ids), int(df.cluster_size.max()) + 1), dtype=np.int32)
    idmap = {v: i for i, v in enumerate(ids)}
    c[[idmap[v] for v in df.configuration_id], df.cluster_size] = df['count']
    pooled = c.sum(axis=0)
    h = pd.read_csv(HIST)
    h = h[(h.length == 512) & np.isclose(h.p_ratio, 1) & (h.exclusion == 'largest_excluded')]
    if not np.array_equal(pooled[h.cluster_size.to_numpy()], h.C_s.to_numpy()):
        raise ValueError('Configuration counts disagree with official histogram')
    return c, ids, df


def plot_all(c, rows, main, boots, gof, out):
    s = np.arange(len(c))
    pos = c > 0
    lo, hi = main['s_min'], main['s_max']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].loglog(s[pos], c[pos], '.', ms=2); ax[0].axvspan(20, 684, color='gray', alpha=.15, label='historical [20,684]')
    ax[0].axvspan(lo, hi, color='tab:green', alpha=.14, label='selected interval')
    ax[0].set(xlabel='size s', ylabel='raw count C_s'); ax[0].legend(fontsize=8)
    ax[1].loglog(s[pos], c[pos] / (200 * 512**2), '.', ms=2); ax[1].set(xlabel='size s', ylabel='density n_s')
    tail = np.cumsum(c[::-1])[::-1]
    ax[2].loglog(s[1:], tail[1:] / tail[1], lw=1); ax[2].set(xlabel='size s', ylabel='P(S ≥ s) (all retained clusters)')
    for a in ax: a.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'cluster_distribution.png', dpi=180); plt.close(fig)

    # Log-spaced bins; fit local slope of density per integer size over 7 bins.
    edges = np.unique(np.ceil(np.geomspace(1, min(len(c), 5000), 55)).astype(int))
    mid, density = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a: continue
        mid.append(np.sqrt(a * (b - 1))); density.append(c[a:b].sum() / (b - a))
    mid, density = np.array(mid), np.array(density)
    centers, eff = [], []
    for j in range(3, len(mid) - 3):
        x, y = mid[j-3:j+4], density[j-3:j+4]
        if np.all(y > 0):
            centers.append(mid[j]); eff.append(-np.polyfit(np.log(x), np.log(y), 1)[0])
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.semilogx(centers, eff, 'o-', ms=3)
    ax.axhline(187/91, ls='--', color='black', label='187/91'); ax.axvspan(lo, hi, alpha=.12, color='green')
    ax.set(xlabel='size s (7 log-bin window)', ylabel='local effective PDF exponent'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'local_effective_exponent.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].loglog(s[pos], c[pos], '.', ms=2, color='.45')
    ax[0].axvspan(1, lo, color='tab:orange', alpha=.15, label='small sizes excluded')
    ax[0].axvspan(lo, hi, color='tab:green', alpha=.18, label='main interval')
    ax[0].axvspan(hi, len(c), color='tab:red', alpha=.12, label='large sizes excluded')
    for t, color in [(5,'C0'),(10,'C1'),(20,'C2')]:
        for w, ls in [(0,':'),(25,'--'),(50,'-'),(100,'-.')]:
            u = upper(c,t,w)
            if u: ax[0].axvline(u, color=color, ls=ls, alpha=.7)
    ax[0].set(xlabel='size s', ylabel='raw C_s', xlim=(1, 3000), ylim=(.7, None)); ax[0].legend(fontsize=7)
    r = pd.DataFrame(rows); r = r[(r.threshold == 10) & (r.width == 50)]
    ax[1].plot(r.s_min, r.tau, 'o-'); ax[1].axhline(187/91, ls='--', color='black')
    ax[1].axvline(lo, color='green'); ax[1].set(xlabel='candidate lower bound', ylabel='finite discrete MLE tau')
    fig.tight_layout(); fig.savefig(out / 'fit_interval_diagnostics.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5)); ax.loglog(s[pos], c[pos], '.', ms=2, color='.6')
    k = np.arange(lo, hi+1); p = model(c, lo, hi, main['tau']); ax.loglog(k, main['n']*p, color='C0', label='pure discrete power law')
    rate = main['rate']; q = np.exp(-main['cutoff_tau']*np.log(k)-rate*k);q /= q.sum()
    ax.loglog(k, main['n']*q, color='C3', ls='--', label='truncated discrete power law')
    ax.set(xlabel='size s', ylabel='raw C_s', xlim=(1, 3000));ax.legend();fig.tight_layout();fig.savefig(out/'powerlaw_fits.png',dpi=180);plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5)); b = boots[np.isfinite(boots)]
    ax.hist(b, bins=40, color='C0', alpha=.7);ax.axvline(main['tau'], color='C3', label='original MLE')
    ax.axvline(187/91, color='black', ls='--', label='187/91');ax.set(xlabel='bootstrap PDF exponent',ylabel='replicates');ax.legend()
    fig.tight_layout();fig.savefig(out/'bootstrap_distribution.png',dpi=180);plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4.5)); r = pd.DataFrame(rows)
    for (t,w), group in r.groupby(['threshold','width']):
        if w in (0,50): ax.plot(group.s_min,group.tau,'o-',ms=3,label=f'C≥{t}, w={w}')
    ax.axhline(187/91,ls='--',color='black');ax.set(xlabel='lower bound',ylabel='PDF exponent',ylim=(1.7,2.2));ax.legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(out/'sensitivity_analysis.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--bootstrap',type=int,default=1000);parser.add_argument('--gof',type=int,default=1000)
    args=parser.parse_args();start=time.time();out=args.output
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Output directory is not empty: {out}')
    out.mkdir(parents=True,exist_ok=True)
    matrix,ids,df=load();c=matrix.sum(axis=0);m=len(ids)
    pd.DataFrame({'cluster_size':np.arange(1,len(c)), 'C_s':c[1:], 'n_s':c[1:]/(m*512**2)}).to_csv(out/'raw_cluster_counts.csv',index=False)
    rows=[]
    for t in (5,10,20):
        for w in (0,25,50,100):
            hi=upper(c,t,w)
            for lo in XMIN:
                if hi is None or hi/lo<10:continue
                f=fit(c,lo,hi)
                if f:rows.append(dict(threshold=t,width=w,s_min=lo,s_max=hi,dynamic_range=hi/lo,**f))
    pd.DataFrame(rows).to_csv(out/'fit_interval_candidates.csv',index=False)
    hi=upper(c,10,50);lo=choose(c,hi)
    if lo is None:raise RuntimeError('No interval passes width and sample-count requirements')
    pure=fit(c,lo,hi);cut=cutoff_fit(c,lo,hi,pure)
    mainrow=dict(s_min=lo,s_max=hi,threshold=10,width=50,**pure,**cut)
    rng=np.random.default_rng(SEED)
    boot_rows=[]; schemes=[(t,w,x) for t in (5,10,20) for w in (0,50) for x in ('auto',10,20,30,50)]
    scheme_tau={v:[] for v in schemes}
    for i in range(args.bootstrap):
        sample=matrix[rng.integers(m,size=m)].sum(axis=0)
        u=upper(sample,10,50);l=choose(sample,u) if u else None
        f=fit(sample,l,u) if l is not None else None
        boot_rows.append(dict(replicate=i,s_min=l,s_max=u,tau=f['tau'] if f else np.nan,success=bool(f)))
        for t,w,x in schemes:
            u=upper(sample,t,w)
            l=choose(sample,u) if x=='auto' and u else (x if x!='auto' else None)
            f=fit(sample,l,u) if l is not None and u is not None and u/l>=10 else None
            scheme_tau[(t,w,x)].append(f['tau'] if f else np.nan)
    boot=pd.DataFrame(boot_rows);boot.to_csv(out/'bootstrap_exponents.csv',index=False)
    valid=boot.tau.dropna().to_numpy();ci=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
    summary=dict(**mainrow,bootstrap_mean=float(np.mean(valid)),bootstrap_median=float(np.median(valid)),ci_low=ci[0],ci_high=ci[1],bootstrap_failure_fraction=1-len(valid)/args.bootstrap,theory=187/91,relative_deviation=(pure['tau']/(187/91)-1))
    pd.DataFrame([summary]).to_csv(out/'powerlaw_fit_summary.csv',index=False)
    pd.DataFrame([dict(s_min=lo,s_max=hi,pure_aic=pure['aic'],**cut)]).to_csv(out/'model_comparison.csv',index=False)

    # Conditional parametric GOF: observed n and selected interval are fixed.
    p=model(c,lo,hi,pure['tau']);k=np.arange(lo,hi+1)
    ks_sim=np.empty(args.gof)
    for i in range(args.gof):
        sim=np.zeros_like(c);sim[lo:hi+1]=rng.multinomial(pure['n'],p)
        ks_sim[i]=fit(sim,lo,hi)['ks']
    gofp=(1+np.sum(ks_sim>=pure['ks']))/(args.gof+1)
    pd.DataFrame([dict(test='conditional_fixed_interval',observed_ks=pure['ks'],p_value=gofp,replicates=args.gof,simulated_ks_mean=np.mean(ks_sim),selection_aware=False)]).to_csv(out/'gof_results.csv',index=False)

    sens=[]
    for t,w,x in schemes:
        u=upper(c,t,w);l=choose(c,u) if x=='auto' else x
        if l is None or u/l<10:continue
        f=fit(c,l,u)
        if not f:continue
        cf=cutoff_fit(c,l,u,f);b=np.array(scheme_tau[(t,w,x)]);b=b[np.isfinite(b)]
        q=np.quantile(b,[.025,.975]) if len(b) else [np.nan,np.nan]
        sens.append(dict(scheme=f'C>={t}; '+('basic' if w==0 else 'continuity50')+f'; xmin={x}',threshold=t,width=w,s_min=l,s_max=u,n=f['n'],tau=f['tau'],ci_low=q[0],ci_high=q[1],ks=f['ks'],gof_p=gofp if (t,w,x)==(10,50,'auto') else np.nan,gof_type='conditional_fixed_interval' if (t,w,x)==(10,50,'auto') else 'not_run',delta_aic=cf['delta_aic'],cutoff_tau=cf['cutoff_tau'],s_c=cf['s_c'],cutoff_identifiable=cf['identifiable'],bootstrap_valid=len(b)/args.bootstrap))
    pd.DataFrame(sens).to_csv(out/'sensitivity_analysis.csv',index=False)
    plot_all(c,rows,summary,valid,gofp,out)
    meta=dict(seed=SEED,bootstrap_replicates=args.bootstrap,gof_replicates=args.gof,source=str(SOURCE),configurations=m,configuration_ids=[int(v) for v in ids],total_clusters=int(c.sum()),filter={'length':512,'p':.592746,'p_ratio':1.,'exclusion':'largest_excluded'},selection='C_s >= 10; upper endpoint in >=60% qualifying preceding 50 integer sizes; lower first candidate >=10 with KS <=1.25*best KS and tau within .035 of median tau at lower bounds 20,30,40,50; at least 10x range and 10000 clusters',local_slope='7 adjacent log-spaced bins, approximately 54 bins from s=1 to 5000; unweighted OLS of log(count per integer size) on log bin center',software={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},runtime_seconds=time.time()-start)
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    report=f'''# L=512 critical cluster tail, exclusion rule A

## Data and distribution

Source: `{SOURCE}`. Filter: `length=512`, `p=0.592746`, `p_ratio=1.0`, `exclusion=largest_excluded`. There are {m} independent configurations and {int(c.sum()):,} retained clusters. The configuration counts reproduce the official histogram exactly. `raw_cluster_counts.csv` retains every integer size through {len(c)-1}, including zeros; density is `C_s/(M L²)`. Counts at sizes 1, 2, 3, 5, 10, 20 are {', '.join(str(int(c[j])) for j in (1,2,3,5,10,20))}. The CCDF plot is descriptive and its slope is not the PDF exponent.

## Interval selection

The historical interval [20,684] uses the largest isolated `C_s >= 10` point. Under a continuity rule requiring the endpoint to meet the threshold and 60% of the preceding 50 integer sizes to meet it, the upper endpoint is {hi}. For thresholds 5, 10, 20, basic upper endpoints are {', '.join(str(upper(c,t,0)) for t in (5,10,20))}; 50-size continuity endpoints are {', '.join(str(upper(c,t,50)) for t in (5,10,20))}. These count rules diagnose sampling sparsity, not the physical cutoff. Window lengths 25 and 100 are also tabulated. The local exponent uses seven adjacent logarithmic bins and an unweighted regression of log count density against log bin center. The lower candidate is selected among {XMIN}: first `s_min >= 10` with KS within 25% of the candidate minimum, exponent within 0.035 of the median for lower bounds 20–50, at least a decade of sizes, and at least 10,000 clusters. This gives [{lo},{hi}] ({hi/lo:.1f}×, {pure['n']:,} clusters). The selection rule is diagnostic and its numerical tolerances are analysis choices.

## Discrete finite-interval MLE and model comparison

For integer sizes `k` including zero-observation sizes, `Z(tau)=sum_(k=lo)^hi k^(-tau)` and `log L=-tau sum_s C_s log(s)-n log Z`. Solving the score equation gives PDF exponent **{pure['tau']:.6f}**. The empirical versus model discrete CDF has KS **{pure['ks']:.6f}**. On exactly the same observations, the truncated model has `P(s) ∝ s^(-tau) exp(-s/s_c)` and finite-interval normalization. Its exponent is {cut['cutoff_tau']:.6f}, cutoff scale {cut['s_c']:.1f}, and `ΔAIC=AIC_truncated-AIC_pure={cut['delta_aic']:.3f}`. Cutoff identifiable under optimizer bounds: {cut['identifiable']}. AIC is descriptive after interval selection.

## Uncertainty and goodness of fit

The {args.bootstrap} nonparametric replicates resample whole configurations and reselect both endpoints each time. Valid fraction {len(valid)/args.bootstrap:.3f}; mean {np.mean(valid):.6f}, median {np.median(valid):.6f}; percentile 95% CI **[{ci[0]:.6f}, {ci[1]:.6f}]**. This interval covers configuration variation and the stated selection algorithm, but not alternative selection rules. The {args.gof}-replicate parametric KS test simulates from the fitted finite-interval pure law and refits tau on each replicate. Conditional fixed-interval GOF p = **{gofp:.4f}**. This is conditional on the chosen bounds and says nothing about the entire distribution. A full selection-aware GOF is unavailable without a credible generating model outside the selected range; it has not been claimed.

## Sensitivity and theory

`sensitivity_analysis.csv` lists threshold 5/10/20, basic and continuity upper rules, and auto or fixed lower candidates 10/20/30/50. Each row has a configuration-bootstrap percentile CI and a same-interval truncated-model AIC comparison. GOF was run for the primary interval only; blanks in other rows mean not tested. Relative difference from 187/91 is {100*summary['relative_deviation']:.2f}%; theoretical value in the primary bootstrap CI: {bool(ci[0] <= 187/91 <= ci[1])}. The interval-choice spread and conditional GOF must be considered before treating a close slope or a high p value as confirmation of universality. A single L=512 data set cannot isolate asymptotic finite-size corrections from microscopic corrections.

## Physical conclusion and limitations

**There is a credible approximate finite-interval power-law regime at s={lo}–{hi}, but the data do not support the asymptotic two-dimensional exponent.** The interval spans {hi/lo:.1f} in size, contains {pure['n']:,} clusters, has conditional KS p={gofp:.3f}, and the truncated alternative does not improve AIC (ΔAIC={cut['delta_aic']:.2f}). The local effective exponent rises gradually from roughly 1.9 near s=15 to about 2.0 near s=150, so it is not a perfectly flat plateau. Threshold-10 continuity fits with lower bounds 10, 20 and 30 give approximately 1.934, 1.952 and 1.954; small-size corrections are material. Moving the threshold-10 upper bound from the continuity endpoint {hi} to the isolated-point endpoint 684 changes the s_min=20 estimate from about 1.952 to 1.951, a smaller effect here. The main fitted cutoff scale {cut['s_c']:.0f} exceeds the fitted upper size by {cut['s_c']/hi:.1f} times, so it is practically weakly identified despite an interior optimizer solution. The conditional GOF treats cluster draws as independent and fixes the selected interval; correlations within configurations and interval selection can affect its calibration. The configuration bootstrap addresses dependence for the confidence interval, not for this GOF. No full selection-aware GOF is claimed. The theoretical value 187/91 is outside the reported configuration-bootstrap CI, with a {100*summary['relative_deviation']:.2f}% point-estimate difference. One system size cannot determine whether the difference comes from finite-size or microscopic corrections.
'''
    (out/'cluster_tail_analysis_L512_A.md').write_text(report)
    print(json.dumps({'interval':[lo,hi],'tau':pure['tau'],'ci':ci.tolist(),'gof_p':gofp,'delta_aic':cut['delta_aic'],'runtime_seconds':meta['runtime_seconds']},indent=2))

if __name__=='__main__':main()
