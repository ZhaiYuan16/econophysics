"""Redraw chapter 6's twelve CCDF panels from saved returns and OLS table."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
raw = pd.read_csv(ROOT / 'results/tables/market_returns.csv.gz', usecols=['p_ratio','activity','z'])
fits = pd.read_csv(ROOT / 'results/loglog_ols_comparison/ols_summary_v2.csv')
fits = fits[fits.kind == 'market'].set_index('label')
fig, axes = plt.subplots(4, 3, figsize=(15, 14), layout='constrained')
for i, activity in enumerate((.001, .01, .1, .5)):
    for j, ratio in enumerate((.9, 1., 1.1)):
        ax = axes[i, j]
        z = np.abs(raw[(np.isclose(raw.p_ratio, ratio)) & (np.isclose(raw.activity, activity))].z.to_numpy())
        z = z[np.isfinite(z) & (z > 0)]
        x, n = np.unique(z, return_counts=True)
        y = np.cumsum(n[::-1])[::-1] / len(z)
        ix = np.unique(np.linspace(0, len(x)-1, min(1600, len(x))).astype(int))
        ax.loglog(x[ix], y[ix], '.', ms=1.5, label='empirical CCDF')
        label = f'p/pc={ratio:g}, a={activity:g}'
        fit = fits.loc[label]
        if np.isfinite(fit.xmin) and np.isfinite(fit.xmax):
            xx = np.geomspace(fit.xmin, fit.xmax, 100)
            ax.loglog(xx, 10**fit.intercept * xx**fit.slope, color='red', lw=1.4, label='interval OLS')
            ax.axvline(fit.xmin, color='green', ls='--', lw=.7)
            ax.axvline(fit.xmax, color='green', ls='--', lw=.7)
        ax.set(title=label, xlabel='Absolute standardized return |z|', ylabel='P(|z| ≥ x)')
        if i == j == 0: ax.legend(fontsize=7)
out = ROOT / 'results/figures/market_loglog_ols.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180)
print(out)
