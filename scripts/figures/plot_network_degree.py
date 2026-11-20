"""Recreate the saved network degree distribution chart from original observations."""
from pathlib import Path
import pandas as pd
from local_market.plotting import plot_network_degree_distributions

ROOT = Path(__file__).resolve().parents[2]
data = pd.read_csv(ROOT / 'results/tables/network_degree_distributions.csv')
out = ROOT / 'results/figures/network_degree_distributions.png'
out.parent.mkdir(parents=True, exist_ok=True)
plot_network_degree_distributions(data, out)
