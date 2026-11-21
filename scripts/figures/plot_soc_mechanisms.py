"""Recreate the SOC comparison used in report chapter 8 from archived v2 paths."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
for row, mechanism in enumerate(('baseline', 'cascade')):
    base = ROOT / 'results/soc_research_v2_full' / mechanism
    paths = pd.read_csv(base / 'paths.csv.gz', usecols=['length', 'step', 'initial_occupancy', 'occupancy_after'])
    events = pd.read_csv(base / 'events.csv.gz', usecols=['length', 'step', 'size'])
    for length, color in zip((32, 64, 128), ('#4778a8', '#de8437', '#4b9862')):
        for initial, style in zip(sorted(paths.initial_occupancy.unique()), ('-', '--', ':')):
            vals = paths[(paths.length == length) & (paths.initial_occupancy == initial) & (paths.step >= {32: 1200, 64: 3000, 128: 7500}[length])].occupancy_after
            if len(vals):
                axes[row, 0].hist(vals, bins=70, density=True, histtype='step', color=color,
                                  linestyle=style, label=f'L={length}, p0={initial:g}')
        sizes = events[(events.length == length) & (events.step >= {32: 1200, 64: 3000, 128: 7500}[length])]["size"].to_numpy()
        sizes = sizes[sizes > 0]
        if len(sizes):
            x = np.unique(sizes)
            ccdf = np.searchsorted(np.sort(sizes), x, side='left')
            axes[row, 1].loglog(x, 1 - ccdf / len(sizes), color=color, label=f'L={length}')
    axes[row, 0].set(xlabel='Occupancy p', ylabel='Probability density', title=mechanism.capitalize())
    axes[row, 1].set(xlabel='Removed agents S', ylabel='P(size >= S)')
    axes[row, 0].legend(fontsize=6, ncol=2)
    axes[row, 1].legend(fontsize=8)
    axes[row, 0].grid(alpha=.2)
    axes[row, 1].grid(alpha=.2)
out = ROOT / 'results/figures/soc_mechanisms_comparison.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(out)
