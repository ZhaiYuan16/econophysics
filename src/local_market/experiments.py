"""Small experiment runners shared by command-line entry points."""

from __future__ import annotations  # 延后求值类型注解，使返回类型可以引用当前配置类。

import numpy as np  # 第三方 NumPy 库，基准入口用 random.default_rng 按种子创建随机数生成器。
import pandas as pd  # 第三方 Pandas 库，供本模块的扫描和系综实验构造、分组及合并表格。
from scipy import stats  # 第三方 SciPy 统计模块，linregress 用于本模块的有限尺寸标度回归。
from tqdm import tqdm  # 第三方进度显示工具，tqdm 包装采样进度并支持逐次更新。

from .cb_model import (  # 本地 src/local_market/cb_model.py，提供交易模拟和结果结构。
    MarketSimulation,  # 保存需求、收益率、对数价格、格点占据状态和团簇计数的返回对象。
    sample_demands_and_active_counts,  # 为本模块的市场系综接口批量抽取需求和活跃团簇数。
    simulate_annealed,  # 每步更新团簇配置的模拟。
    simulate_quenched,  # 固定团簇配置的模拟。
)
from .config import BaselineConfig, CriticalityConfig, MarketAnalysisConfig  # 本地 src/local_market/config.py，分别由 scripts/run_baseline.py, scripts/run_criticality_scan.py, scripts/run_market_analysis.py 调用。
from .networks import sample_lattice_occupancy  # 本地 src/local_market/networks.py，按占据概率生成布尔格点。
from .percolation import identify_lattice_clusters  # 本地 src/local_market/percolation.py，按边界识别四邻域团簇。
from .statistics import (  # 本地 src/local_market/statistics.py，供其他实验接口进行团簇与收益率统计。
    finite_cluster_sizes,  # 根据排除规则取得有限团簇的规模数组。
    return_statistics,  # 计算收益率的描述统计与极端事件概率。
    standardize_returns,  # 用样本均值和样本标准差计算标准化收益率。
    susceptibility_from_sizes,  # 由规模数组计算有限团簇易感性 Σs²/Σs。
)

def run_baseline(config: BaselineConfig) -> MarketSimulation:
    # 组织一次基准实验，按 mode(quenched/annealed) 选择固定配置或逐步更新配置的交易方式。
    # 由 scripts/run_baseline.py 的 main 调用。
    # 输入：
    #       config的8个字段：length、p、activity、market_depth、boundary、mode、steps 和 seed。
    # 输出：
    #       MarketSimulation对象，来自 simulate_quenched() 或 simulate_annealed()。

    """Run one configured baseline experiment with a single RNG stream."""
    rng = np.random.default_rng(config.seed)  # 占据抽样和交易抽样共用此随机流，种子用于复现实验。

    def make_configuration():
        # 生成一次团簇配置。quenched 时由 run_baseline 调用一次，annealed 时由 simulate_annealed 每步回调。
        # 无显式参数，通过闭包读取外层 config 与 rng。
        # 输出：ClusterConfiguration，identify_lattice_clusters()的输出格式，保存格点团簇标签与规模。

        occupied = sample_lattice_occupancy(config.length, config.p, rng)   # 来自 networks.py，根据 p 抽取占据格点
        return identify_lattice_clusters(occupied, config.boundary)     # 来自 percolation.py，根据 open/periodic 标记团簇。

    if config.mode == "quenched":
        return simulate_quenched(   # 来自 src/local_market/cb_model.py，执行固定团簇配置的交易模拟。
            make_configuration(),   # 带括号的 make_configuration() 先执行，将固定配置交给 cb_model.py 批量交易。
            config.steps,
            config.activity,
            config.market_depth,
            rng,
        )
    
    return simulate_annealed(       # 来自 src/local_market/cb_model.py，执行逐步更新团簇配置的交易模拟。
        make_configuration,         # 将 make_configuration 函数对象交给 cb_model.py，由它在每个交易步重新生成配置。
        config.steps,
        config.activity,
        config.market_depth,
        rng,
    )


def _sem(values: pd.Series) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.count()))


def run_criticality_scan(
    config: CriticalityConfig, show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run an ensemble site-percolation scan.

    Returns per-realization observations, grouped summary statistics, pooled
    cluster histograms, and grid-based pseudocritical estimates.
    """
    rng = np.random.default_rng(config.seed)
    sample_rows: list[dict[str, object]] = []
    histogram_counts: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    total = sum(config.samples_per_length[length] for length in config.lengths)
    progress = tqdm(
        total=total * len(config.p_ratios),
        desc="criticality",
        unit="sample",
        disable=not show_progress,
    )

    for length in config.lengths:
        samples = config.samples_per_length[length]
        maximum_size = length * length
        for ratio in config.p_ratios:
            p = config.p_c * ratio
            all_histogram = np.zeros(maximum_size + 1, dtype=np.int64)
            finite_histogram = np.zeros(maximum_size + 1, dtype=np.int64)
            for sample in range(samples):
                occupied = sample_lattice_occupancy(length, p, rng)
                clusters = identify_lattice_clusters(occupied, config.boundary)
                finite_sizes = finite_cluster_sizes(
                    clusters, config.finite_cluster_exclusion
                )
                finite_denominator = int(finite_sizes.sum())
                finite_numerator = int(np.dot(finite_sizes, finite_sizes))
                if clusters.cluster_count:
                    all_histogram += np.bincount(
                        clusters.sizes, minlength=maximum_size + 1
                    )
                if finite_sizes.size:
                    finite_histogram += np.bincount(
                        finite_sizes, minlength=maximum_size + 1
                    )
                sample_rows.append(
                    {
                        "length": length,
                        "p_ratio": ratio,
                        "p": p,
                        "sample": sample,
                        "occupied_count": clusters.occupied_count,
                        "cluster_count": clusters.cluster_count,
                        "largest_size": clusters.largest_size,
                        "largest_occupied_fraction": clusters.largest_occupied_fraction,
                        "p_infinity_sample": clusters.p_infinity,
                        "spans": clusters.spans,
                        "susceptibility": susceptibility_from_sizes(finite_sizes),
                        "finite_first_moment": finite_denominator,
                        "finite_second_moment": finite_numerator,
                    }
                )
                progress.update(1)
            histogram_counts[(length, ratio)] = (all_histogram, finite_histogram)
    progress.close()

    samples_frame = pd.DataFrame(sample_rows)
    summary = (
        samples_frame.groupby(["length", "p_ratio", "p"], as_index=False)
        .agg(
            samples=("sample", "count"),
            mean_occupied=("occupied_count", "mean"),
            occupied_sem=("occupied_count", _sem),
            mean_cluster_count=("cluster_count", "mean"),
            cluster_count_sem=("cluster_count", _sem),
            mean_largest_size=("largest_size", "mean"),
            largest_size_sem=("largest_size", _sem),
            p_infinity=("p_infinity_sample", "mean"),
            p_infinity_sem=("p_infinity_sample", _sem),
            spanning_probability=("spans", "mean"),
            spanning_probability_sem=("spans", _sem),
            susceptibility_mean=("susceptibility", "mean"),
            susceptibility_sem=("susceptibility", _sem),
            finite_first_moment=("finite_first_moment", "sum"),
            finite_second_moment=("finite_second_moment", "sum"),
        )
    )
    summary["susceptibility_pooled"] = (
        summary["finite_second_moment"] / summary["finite_first_moment"]
    ).fillna(0.0)
    bootstrap_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 2]))
    bootstrap_rows: list[dict[str, float | int]] = []
    for (length, ratio, p), group in samples_frame.groupby(
        ["length", "p_ratio", "p"], sort=True
    ):
        first = group["finite_first_moment"].to_numpy(dtype=np.int64)
        second = group["finite_second_moment"].to_numpy(dtype=np.int64)
        estimates = np.empty(config.bootstrap_replicates, dtype=float)
        for replicate in range(config.bootstrap_replicates):
            indices = bootstrap_rng.integers(0, len(group), size=len(group))
            denominator = int(first[indices].sum())
            estimates[replicate] = (
                float(second[indices].sum() / denominator) if denominator else 0.0
            )
        bootstrap_rows.append(
            {
                "length": int(length),
                "p_ratio": float(ratio),
                "p": float(p),
                "susceptibility_bootstrap_se": float(estimates.std(ddof=1)),
                "susceptibility_ci_low": float(np.quantile(estimates, 0.025)),
                "susceptibility_ci_high": float(np.quantile(estimates, 0.975)),
            }
        )
    summary = summary.merge(
        pd.DataFrame(bootstrap_rows), on=["length", "p_ratio", "p"], validate="one_to_one"
    )

    histogram_rows: list[dict[str, object]] = []
    for (length, ratio), (all_counts, finite_counts) in histogram_counts.items():
        samples = config.samples_per_length[length]
        nonzero_sizes = np.flatnonzero((all_counts + finite_counts) > 0)
        for size in nonzero_sizes[nonzero_sizes > 0]:
            histogram_rows.append(
                {
                    "length": length,
                    "p_ratio": ratio,
                    "p": config.p_c * ratio,
                    "cluster_size": int(size),
                    "all_cluster_count": int(all_counts[size]),
                    "finite_cluster_count": int(finite_counts[size]),
                    "all_n_s_per_site": all_counts[size]
                    / (samples * length * length),
                    "finite_n_s_per_site": finite_counts[size]
                    / (samples * length * length),
                }
            )
    histogram_frame = pd.DataFrame(histogram_rows)

    critical_rows: list[dict[str, object]] = []
    for length, group in summary.groupby("length", sort=True):
        ordered = group.sort_values("p").reset_index(drop=True)
        chi_index = int(ordered["susceptibility_pooled"].idxmax())
        slopes = np.gradient(ordered["p_infinity"], ordered["p"])
        slope_index = int(np.argmax(slopes))
        critical_rows.append(
            {
                "length": int(length),
                "p_chi_peak": float(ordered.loc[chi_index, "p"]),
                "p_ratio_chi_peak": float(ordered.loc[chi_index, "p_ratio"]),
                "chi_peak": float(
                    ordered.loc[chi_index, "susceptibility_pooled"]
                ),
                "p_max_order_parameter_slope": float(
                    ordered.loc[slope_index, "p"]
                ),
                "p_ratio_max_order_parameter_slope": float(
                    ordered.loc[slope_index, "p_ratio"]
                ),
                "max_order_parameter_slope": float(slopes[slope_index]),
            }
        )
    critical_frame = pd.DataFrame(critical_rows)
    return samples_frame, summary, histogram_frame, critical_frame


def estimate_finite_size_scaling(
    summary: pd.DataFrame, critical_ratio: float = 1.0
) -> pd.DataFrame:
    """Fit log-log finite-size trends at the configured critical ratio."""
    critical = summary[np.isclose(summary["p_ratio"], critical_ratio)].sort_values(
        "length"
    )
    if len(critical) < 3:
        raise ValueError("at least three system sizes are required for scaling fits")
    rows: list[dict[str, object]] = []
    definitions = (
        ("p_infinity", "p_infinity", -5.0 / 48.0),
        ("susceptibility", "susceptibility_pooled", 43.0 / 24.0),
    )
    log_length = np.log(critical["length"].to_numpy(dtype=float))
    for observable, column, theoretical_exponent in definitions:
        values = critical[column].to_numpy(dtype=float)
        if np.any(values <= 0):
            raise ValueError(f"{column} must be positive for log-log scaling")
        fit = stats.linregress(log_length, np.log(values))
        rows.append(
            {
                "observable": observable,
                "p_ratio": critical_ratio,
                "fitted_exponent": float(fit.slope),
                "exponent_standard_error": float(fit.stderr),
                "theoretical_exponent": theoretical_exponent,
                "intercept": float(fit.intercept),
                "r_squared": float(fit.rvalue**2),
                "size_count": len(critical),
            }
        )
    return pd.DataFrame(rows)


def run_market_ensemble(
    config: MarketAnalysisConfig, show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate return ensembles, reusing configurations across activities."""
    return_frames: list[pd.DataFrame] = []
    structure_rows: list[dict[str, object]] = []
    total = len(config.p_ratios) * config.replicates
    progress = tqdm(total=total, desc="market ensembles", unit="replicate", disable=not show_progress)

    for p_index, ratio in enumerate(config.p_ratios):
        p = config.p_c * ratio
        for replicate in range(config.replicates):
            occupancy_rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, 10, p_index, replicate])
            )
            configurations = []
            for sample in range(config.configurations_per_replicate):
                occupied = sample_lattice_occupancy(config.length, p, occupancy_rng)
                clusters = identify_lattice_clusters(occupied, config.boundary)
                configurations.append(clusters)
                structure_rows.append(
                    {
                        "length": config.length,
                        "p_ratio": ratio,
                        "p": p,
                        "replicate": replicate,
                        "configuration": sample,
                        "occupied_count": clusters.occupied_count,
                        "cluster_count": clusters.cluster_count,
                        "largest_size": clusters.largest_size,
                        "largest_occupied_fraction": clusters.largest_occupied_fraction,
                        "spans": clusters.spans,
                    }
                )

            for activity_index, activity in enumerate(config.activities):
                action_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 20, p_index, replicate, activity_index]
                    )
                )
                demand_blocks = []
                active_blocks = []
                available_blocks = []
                configuration_blocks = []
                for sample, clusters in enumerate(configurations):
                    demands, active_counts = sample_demands_and_active_counts(
                        clusters.sizes,
                        config.trades_per_configuration,
                        activity,
                        action_rng,
                    )
                    demand_blocks.append(demands)
                    active_blocks.append(active_counts)
                    available_blocks.append(
                        np.full(
                            config.trades_per_configuration,
                            clusters.cluster_count,
                            dtype=np.int64,
                        )
                    )
                    configuration_blocks.append(
                        np.full(config.trades_per_configuration, sample, dtype=np.int64)
                    )
                demands = np.concatenate(demand_blocks)
                observations = demands.size
                return_frames.append(
                    pd.DataFrame(
                        {
                            "length": config.length,
                            "p_ratio": ratio,
                            "p": p,
                            "activity": activity,
                            "replicate": replicate,
                            "step": np.arange(observations, dtype=np.int64),
                            "configuration": np.concatenate(configuration_blocks),
                            "available_cluster_count": np.concatenate(available_blocks),
                            "active_cluster_count": np.concatenate(active_blocks),
                            "demand": demands,
                            "return": demands.astype(float) / config.market_depth,
                        }
                    )
                )
            progress.update(1)
    progress.close()
    return pd.concat(return_frames, ignore_index=True), pd.DataFrame(structure_rows)


def summarize_market_returns(
    returns_frame: pd.DataFrame, config: MarketAnalysisConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize per parameter combination and summarize overall and by seed."""
    data = returns_frame.copy()
    data["z"] = np.nan
    overall_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    keys = ["length", "p_ratio", "p", "activity"]
    for key, group in data.groupby(keys, sort=True):
        z, mean, standard_deviation = standardize_returns(group["return"].to_numpy())
        data.loc[group.index, "z"] = z
        overall = dict(zip(keys, key, strict=True))
        overall.update(
            return_statistics(
                group["return"].to_numpy(),
                z,
                config.extreme_thresholds,
                group["active_cluster_count"].to_numpy(),
            )
        )
        overall["standardization_mean"] = mean
        overall["standardization_std"] = standard_deviation
        overall["mean_available_cluster_count"] = float(
            group["available_cluster_count"].mean()
        )
        overall_rows.append(overall)

        for replicate, replicate_group in group.groupby("replicate", sort=True):
            replicate_z = data.loc[replicate_group.index, "z"].to_numpy()
            row = dict(zip(keys, key, strict=True))
            row["replicate"] = int(replicate)
            row.update(
                return_statistics(
                    replicate_group["return"].to_numpy(),
                    replicate_z,
                    config.extreme_thresholds,
                    replicate_group["active_cluster_count"].to_numpy(),
                )
            )
            replicate_rows.append(row)

    overall_frame = pd.DataFrame(overall_rows)
    replicate_frame = pd.DataFrame(replicate_rows)
    dispersion = (
        replicate_frame.groupby(keys, as_index=False)
        .agg(
            replicate_return_std_sd=("return_std", "std"),
            replicate_kurtosis_sd=("return_excess_kurtosis", "std"),
            replicate_p_abs_z_gt_3_sd=("p_abs_z_gt_3", "std"),
            replicate_p_abs_z_gt_5_sd=("p_abs_z_gt_5", "std"),
            replicate_p_abs_z_gt_8_sd=("p_abs_z_gt_8", "std"),
            replicate_negative_crash_probability_sd=(
                "negative_crash_probability",
                "std",
            ),
        )
    )
    overall_frame = overall_frame.merge(dispersion, on=keys, validate="one_to_one")
    return data, overall_frame, replicate_frame
