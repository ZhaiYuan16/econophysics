"""Matched active-cluster network experiment; all random stages have separate streams."""
from __future__ import annotations

import gzip
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .cb_model import sample_demands_and_active_counts
from .hierarchical_statistics import block_histogram, resampled_counts, select_network_configuration_blocks, weighted_pareto_fit
from .networks import generate_matched_graph, identify_graph_clusters, graph_structure_metrics
from .statistics import finite_cluster_sizes
from .tail_fitting import select_pareto_tail, compare_tail_distributions, pareto_fixed_xmin_gof_pvalue


def rng_for(seed, *parts):
    return np.random.default_rng(np.random.SeedSequence([seed, *parts]))


def select_blocks(rng, topology, networks, configurations):
    if topology == 'lattice':
        # A periodic regular lattice is a single deterministic graph. Resample its
        # independent configurations directly; network IDs are simulation batches.
        return rng.integers(0, networks * configurations, size=networks * configurations)
    return select_network_configuration_blocks(rng, networks, configurations)


def matched_activity(target, mean_k):
    if mean_k <= 0:
        return 1.0
    return min(1.0, target / mean_k)


def metric_counts(support, counts, r0, depth):
    n = int(counts.sum())
    x = support / depth
    mean = float(np.dot(x, counts) / n)
    delta = x - mean
    m2 = float(np.dot(delta**2, counts) / n)
    std = float(np.sqrt(m2 * n / (n - 1)))
    z = delta / std if std else np.zeros_like(delta)
    m3 = float(np.dot(delta**3, counts) / n)
    m4 = float(np.dot(delta**4, counts) / n)
    nonzero = int(counts[support != 0].sum())
    result = {'observations': n, 'return_mean': mean, 'return_std': std,
              'return_skewness': float(m3 / m2**1.5) if m2 else np.nan,
              'return_kurtosis': float(m4 / m2**2) if m2 else np.nan,
              'return_excess_kurtosis': float(m4 / m2**2 - 3) if m2 else np.nan,
              'zero_return_fraction': float(1 - nonzero / n),
              'p_z_lt_minus_5': float(counts[z < -5].sum() / n),
              'p_z_lt_minus_5_given_nonzero': float(counts[(z < -5) & (support != 0)].sum() / nonzero) if nonzero else np.nan,
              'p_r_lt_minus_r0': float(counts[x < -r0].sum() / n)}
    for t in (3, 5, 8):
        result[f'p_abs_z_gt_{t}'] = float(counts[np.abs(z) > t].sum() / n)
    return result


def scan_one(graph, p_grid, samples, seed, ti, ni, bootstrap, allow_boundary=False):
    rows = []
    first = np.empty((len(p_grid), samples), dtype=float)
    second = np.empty_like(first)
    for pi, p in enumerate(p_grid):
        rng = rng_for(seed, 110, ti, ni, pi)
        for si in range(samples):
            clusters = identify_graph_clusters(graph.adjacency, rng.random(graph.adjacency.shape[0]) < p)
            finite = finite_cluster_sizes(clusters, 'largest')
            f1, f2 = int(finite.sum()), int(np.dot(finite, finite))
            first[pi, si], second[pi, si] = f1, f2
            rows.append((pi, si, clusters.cluster_count, clusters.largest_size, f1, f2))
    chi = np.divide(second.sum(axis=1), first.sum(axis=1), out=np.zeros(len(p_grid)), where=first.sum(axis=1)>0)
    peak = int(np.argmax(chi))
    if not allow_boundary and peak in (0, len(p_grid)-1):
        raise ValueError(f'susceptibility peak on scan boundary: {graph.topology} network {ni}')
    rng = rng_for(seed, 111, ti, ni)
    peaks = []
    for _ in range(bootstrap):
        indices = rng.integers(0, samples, size=(len(p_grid), samples))
        f1 = np.take_along_axis(first, indices, axis=1).sum(axis=1)
        f2 = np.take_along_axis(second, indices, axis=1).sum(axis=1)
        peaks.append(p_grid[int(np.argmax(np.divide(f2, f1, out=np.zeros(len(p_grid)), where=f1>0)))])
    return rows, chi, float(p_grid[peak]), np.quantile(peaks, [.025, .975])


def reference_sigma(path):
    frame = pd.read_csv(path)
    row = frame[(frame.topology == 'lattice') & np.isclose(frame.relative_p, 1) & np.isclose(frame.activity, .01)]
    if len(row) != 1 or not np.isfinite(row.iloc[0].return_std):
        raise ValueError('original lattice reference sigma is unavailable or ambiguous')
    return float(row.iloc[0].return_std)


def run(config_path, output, smoke=False):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if smoke:
        cfg.update(node_count=256, network_replicates=2, scan_samples_per_p=5,
                   scan_bootstrap=20, pilot_configurations=8,
                   market_configurations_per_network=3, trades_per_configuration=10,
                   bootstrap_replicates=20, tail_bootstrap_replicates=0)
        cfg['p_grids'] = {k: np.linspace(.05, .90, 18).tolist() for k in cfg['p_grids']}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output directory must be empty: {output}')
    (output / 'tables').mkdir(parents=True, exist_ok=True)
    (output / 'returns').mkdir(exist_ok=True)
    start = time.monotonic()
    seed = cfg['seed']
    nnet = cfg['network_replicates']
    nc = cfg['market_configurations_per_network']
    nt = cfg['trades_per_configuration']
    sigma_ref = reference_sigma(Path('results/tables/network_market_summary.csv'))
    r0 = 5 * sigma_ref
    scan_rows, pc_rows, metric_rows, pilot_rows, activity_rows = [], [], [], [], []
    structures, block_rows, group_paths = [], [], []
    for ti, topology in enumerate(cfg['topologies']):
        for ni in range(nnet):
            network_seed = int(np.random.SeedSequence([seed, 100, ti, ni]).generate_state(1)[0])
            graph = generate_matched_graph(topology, cfg['node_count'], cfg['target_mean_degree'], network_seed, cfg['ws_rewiring_probability'])
            metrics = graph_structure_metrics(graph, 8 if smoke else 16, rng_for(seed, 101, ti, ni))
            metric_rows.append(dict(topology=topology, network_id=ni, network_seed=network_seed, **metrics))
            grid = np.asarray(cfg['p_grids'][topology], dtype=float)
            rows, chi, pc, ci = scan_one(graph, grid, cfg['scan_samples_per_p'], seed, ti, ni, cfg['scan_bootstrap'], allow_boundary=smoke)
            for pi, si, k, largest, f1, f2 in rows:
                scan_rows.append(dict(topology=topology, network_id=ni, p=grid[pi], configuration_id=si,
                                      cluster_count=k, largest_size=largest, finite_first_moment=f1, finite_second_moment=f2))
            for pi, p in enumerate(grid):
                pc_rows.append(dict(topology=topology, network_id=ni, network_seed=network_seed, p=p,
                                    susceptibility_pooled=chi[pi], effective_pc=pc, pc_ci_low=ci[0], pc_ci_high=ci[1]))
            for ri, ratio in enumerate(cfg['relative_p_values']):
                p = float(min(1, pc * ratio))
                pilot_rng = rng_for(seed, 120, ti, ni, ri)
                pilot_k = []
                for si in range(cfg['pilot_configurations']):
                    cl = identify_graph_clusters(graph.adjacency, pilot_rng.random(cfg['node_count']) < p)
                    pilot_k.append(cl.cluster_count)
                    pilot_rows.append(dict(topology=topology, network_id=ni, relative_p=ratio, p=p, pilot_configuration_id=si, cluster_count=cl.cluster_count))
                mean_k = float(np.mean(pilot_k))
                conditions = [('fixed', cfg['fixed_activity'], cfg['fixed_activity'])] + [(f'c{c}', c, matched_activity(c, mean_k)) for c in cfg['targets']]
                for condition, target, activity in conditions:
                    activity_rows.append(dict(topology=topology, network_id=ni, relative_p=ratio, p=p,
                                              condition=condition, target_active=target, pilot_mean_k=mean_k,
                                              pilot_sem_k=float(np.std(pilot_k, ddof=1)/np.sqrt(len(pilot_k))),
                                              activity=activity, theoretical_active=activity*mean_k,
                                              capped=bool(condition != 'fixed' and activity == 1 and mean_k < target)))
                occupancy_rng = rng_for(seed, 130, ti, ni, ri)
                handles = {}
                try:
                    for condition, _, _ in conditions:
                        path = output / 'returns' / f'{topology}_r{ri}_{condition}_n{ni}.csv.gz'
                        handles[condition] = gzip.open(path, 'wt', newline='')
                        handles[condition].write('topology,network_id,relative_p,p,condition,configuration_id,action_id,demand,return,active_cluster_count\n')
                        group_paths.append(dict(topology=topology, network_id=ni, relative_p=ratio, condition=condition, path=str(path)))
                    for ci in range(nc):
                        cl = identify_graph_clusters(graph.adjacency, occupancy_rng.random(cfg['node_count']) < p)
                        structures.append(dict(topology=topology, network_id=ni, relative_p=ratio, p=p, configuration_id=ci,
                                               cluster_count=cl.cluster_count, occupied_count=cl.occupied_count, largest_size=cl.largest_size))
                        for ai, (condition, target, activity) in enumerate(conditions):
                            action_rng = rng_for(seed, 140, ti, ni, ri, ci, ai)
                            demands, active = sample_demands_and_active_counts(cl.sizes, nt, activity, action_rng)
                            block_rows.append(dict(topology=topology, network_id=ni, relative_p=ratio,
                                                   condition=condition, configuration_id=ci, mean_active=float(active.mean()),
                                                   mean_return=float(demands.mean()/cfg['market_depth'])))
                            handle = handles[condition]
                            handle.writelines(f'{topology},{ni},{ratio},{p},{condition},{ci},{j},{int(d)},{d/cfg["market_depth"]},{int(a)}\n'
                                              for j, (d, a) in enumerate(zip(demands, active)))
                finally:
                    for handle in handles.values():
                        handle.close()
            print(f'generated {topology} network {ni+1}/{nnet} in {time.monotonic()-start:.1f}s', flush=True)
    tables = output / 'tables'
    frames = dict(network_metrics=metric_rows, criticality_scan_samples=scan_rows, criticality_curves=pc_rows,
                  pilot_configurations=pilot_rows, activity_calibration=activity_rows,
                  market_configurations=structures, configuration_summary=block_rows, return_files=group_paths)
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(tables / f'{name}.csv', index=False)
    metadata = dict(config=cfg, smoke=smoke, sigma_ref=sigma_ref, sigma_ref_source='results/tables/network_market_summary.csv, lattice relative_p=1 a=0.01, 96000 returns',
                    r0=r0, standardization='Each topology × relative_p × condition pooled across its networks and configurations; sample std ddof=1',
                    pc_definition='peak of pooled susceptibility, largest occupied cluster removed per configuration',
                    seed_streams={'100':'network','101':'structure metrics','110':'criticality scan','111':'criticality bootstrap','120':'pilot occupancy','130':'market occupancy','140':'actions'},
                    git_commit=subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True).stdout.strip(),
                    python=platform.python_version(), elapsed_seconds=time.monotonic()-start)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return output


def analyze(output, smoke=False):
    """Analyze saved raw files. Bootstrap draws networks, configurations, then keeps actions intact."""
    output = Path(output)
    meta = json.loads((output / 'metadata.json').read_text())
    cfg = meta['config']
    tables = output / 'tables'
    manifest = pd.read_csv(tables / 'return_files.csv')
    calibration = pd.read_csv(tables / 'activity_calibration.csv')
    blocks = pd.read_csv(tables / 'configuration_summary.csv')
    summary_rows, network_rows, boot_rows, tail_rows, dist_rows = [], [], [], [], []
    nnet, nc = cfg['network_replicates'], cfg['market_configurations_per_network']
    for gi, (key, files) in enumerate(manifest.groupby(['topology', 'relative_p', 'condition'], sort=False)):
        topology, ratio, condition = key
        frames = [pd.read_csv(path) for path in files.sort_values('network_id').path]
        data = pd.concat(frames, ignore_index=True)
        support, matrix = block_histogram(data.demand.to_numpy(),
            data.network_id.to_numpy(dtype=int)*nc + data.configuration_id.to_numpy(dtype=int))
        counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
        base = metric_counts(support, counts, meta['r0'], cfg['market_depth'])
        base['mean_active_cluster_count'] = float(data.active_cluster_count.mean())
        parameters = dict(topology=topology, relative_p=ratio, condition=condition)
        cal = calibration[(calibration.topology == topology) & (calibration.relative_p == ratio) & (calibration.condition == condition)]
        base['mean_theoretical_active'] = float(cal.theoretical_active.mean())
        base['mean_activity'] = float(cal.activity.mean())
        base['capped_networks'] = int(cal.capped.sum())
        summary_rows.append(parameters | base)
        mean, std = base['return_mean'], base['return_std']
        data['z'] = (data['return'] - mean) / std
        standardized_path = output / 'returns' / f'{topology}_r{cfg["relative_p_values"].index(ratio)}_{condition}_standardized.csv.gz'
        data.to_csv(standardized_path, index=False, compression='gzip')
        for ni in range(nnet):
            subcounts = np.asarray(matrix[ni*nc:(ni+1)*nc].sum(axis=0)).ravel()
            network_rows.append(parameters | dict(network_id=ni) | metric_counts(support, subcounts, meta['r0'], cfg['market_depth']) |
                                dict(mean_active_cluster_count=float(data.loc[data.network_id == ni, 'active_cluster_count'].mean())))
        rng = rng_for(cfg['seed'], 160, gi)
        active_by_block = blocks[(blocks.topology == topology) & (blocks.relative_p == ratio) & (blocks.condition == condition)].sort_values(['network_id','configuration_id']).mean_active.to_numpy()
        for bi in range(cfg['bootstrap_replicates']):
            indices = select_blocks(rng, topology, nnet, nc)
            resampled = resampled_counts(matrix, indices)
            row = metric_counts(support, resampled, meta['r0'], cfg['market_depth'])
            row['mean_active_cluster_count'] = float(active_by_block[indices].mean())
            boot_rows.append(parameters | dict(bootstrap_id=bi) | row)
        # Empirical PDF and CCDF, standardized using the full condition sample.
        z = data.z.to_numpy()
        hist, edges = np.histogram(z, bins=np.linspace(-12, 12, 241), density=False)
        hist = hist / len(z) / np.diff(edges)
        for left, right, h in zip(edges[:-1], edges[1:], hist):
            dist_rows.append(parameters | dict(distribution='pdf_z', x=(left+right)/2, value=h))
        for x in np.geomspace(.05, max(12, float(np.abs(z).max())), 100):
            dist_rows.append(parameters | dict(distribution='ccdf_abs_z', x=x, value=float(np.mean(np.abs(z) > x))))
            dist_rows.append(parameters | dict(distribution='ccdf_z_negative', x=x, value=float(np.mean(z < -x))))
        if not smoke:
            nonzero = np.abs(z[data.demand.to_numpy() != 0])
            try:
                fit, _ = select_pareto_tail(nonzero, tuple(cfg['tail_xmin_quantiles']), cfg['tail_minimum_count'])
                tail_rng = rng_for(cfg['seed'], 170, gi)
                tail_alpha, tail_xmin = [], []
                for _ in range(cfg['tail_bootstrap_replicates']):
                    selected = select_blocks(tail_rng, topology, nnet, nc)
                    resampled = resampled_counts(matrix, selected)
                    m = metric_counts(support, resampled, meta['r0'], cfg['market_depth'])
                    absolute = np.abs((support / cfg['market_depth'] - m['return_mean']) / m['return_std'])
                    tail = weighted_pareto_fit(absolute[support != 0], resampled[support != 0],
                                               tuple(cfg['tail_xmin_quantiles']), cfg['tail_minimum_count'])
                    tail_alpha.append(tail['alpha_pdf'])
                    tail_xmin.append(tail['xmin'])
                comparison = compare_tail_distributions(nonzero, fit.xmin)
                gof = pareto_fixed_xmin_gof_pvalue(fit, cfg['tail_gof_replicates'], tail_rng)
                tail_rows.append(parameters | dict(status='fit', xmin=fit.xmin, alpha_pdf=fit.alpha_pdf,
                               tail_count=fit.tail_count, ks_distance=fit.ks_distance,
                               alpha_ci_low=np.quantile(tail_alpha,.025), alpha_ci_high=np.quantile(tail_alpha,.975),
                               xmin_ci_low=np.quantile(tail_xmin,.025), xmin_ci_high=np.quantile(tail_xmin,.975),
                               fixed_xmin_gof_p=gof, gof_note='fixed xmin parametric null; xmin reselected for CI') | comparison)
            except (ValueError, RuntimeError) as exc:
                tail_rows.append(parameters | dict(status=f'insufficient_or_failed: {exc}', tail_count=len(nonzero)))
        print(f'analyzed {gi+1}/{len(manifest.groupby(["topology","relative_p","condition"]))}: {key}', flush=True)
    summary = pd.DataFrame(summary_rows)
    boot = pd.DataFrame(boot_rows)
    metric_names = ['mean_active_cluster_count','return_mean','return_std','return_skewness','return_kurtosis',
                    'return_excess_kurtosis','zero_return_fraction','p_abs_z_gt_3','p_abs_z_gt_5','p_abs_z_gt_8',
                    'p_z_lt_minus_5','p_z_lt_minus_5_given_nonzero','p_r_lt_minus_r0']
    for metric in metric_names:
        bounds = boot.groupby(['topology','relative_p','condition'])[metric].quantile([.025,.975]).unstack()
        bounds.columns = [f'{metric}_ci_low', f'{metric}_ci_high']
        summary = summary.merge(bounds.reset_index(), on=['topology','relative_p','condition'])
    diffs = []
    for (ratio, condition), subset in boot.groupby(['relative_p','condition']):
        names = list(cfg['topologies'])
        for i, a in enumerate(names):
            for b in names[i+1:]:
                aa = subset[subset.topology == a].sort_values('bootstrap_id')
                bb = subset[subset.topology == b].sort_values('bootstrap_id')
                for metric in ('p_z_lt_minus_5','p_z_lt_minus_5_given_nonzero','p_r_lt_minus_r0','return_excess_kurtosis'):
                    delta = aa[metric].to_numpy() - bb[metric].to_numpy()
                    diffs.append(dict(relative_p=ratio, condition=condition, topology_a=a, topology_b=b,
                                      metric=metric, difference=float(summary[(summary.topology == a)&(summary.relative_p == ratio)&(summary.condition == condition)][metric].iloc[0] -
                                                                     summary[(summary.topology == b)&(summary.relative_p == ratio)&(summary.condition == condition)][metric].iloc[0]),
                                      ci_low=float(np.quantile(delta,.025)), ci_high=float(np.quantile(delta,.975))))
    outputs = dict(overall_summary=summary, network_summary=pd.DataFrame(network_rows),
                   bootstrap_replicates=boot, topology_differences=pd.DataFrame(diffs),
                   tail_fits=pd.DataFrame(tail_rows), distributions=pd.DataFrame(dist_rows))
    for name, frame in outputs.items():
        frame.to_csv(tables / f'{name}.csv', index=False)
    return outputs
