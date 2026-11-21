"""Small experiment runners shared by command-line entry points."""
# 组织格点基线、渗流扫描与市场收益率实验。
# run_baseline() 按配置运行单条交易轨迹，run_criticality_scan() 和 estimate_finite_size_scaling() 输出渗流统计与标度。
# run_market_ensemble() 生成共用团簇的多活跃率系综，run_quenched_market_trajectories() 生成固定格点轨迹，summarize_market_returns() 汇总收益。
#
# 命令行脚本先加载实验参数，再在这里串联占据抽样、团簇识别和 Cont-Bouchaud 交易，将结果交给后续统计与绘图。

from __future__ import annotations  # 延后求值类型注解。

import numpy as np  # NumPy 用于随机数流、矩统计、向量化交易和标度计算。
import pandas as pd  # Pandas 用于组织逐配置、逐笔交易和汇总表。
from scipy import stats  # SciPy 统计模块用于拟合有限尺寸趋势。
from tqdm import tqdm  # tqdm 用于显示样本或重复层进度。

from .cb_model import (  # src/local_market/cb_model.py，提供 Cont-Bouchaud 交易和结果对象。
    MarketSimulation,  # 承载单次模拟的需求、收益、价格和配置计数。
    sample_demands_and_active_counts,  # 为固定规模批量生成每笔需求与活跃团簇数。
    simulate_annealed,  # 在逐交易步更新占据配置的模式下交易。
    simulate_quenched,  # 在固定团簇配置上重复交易。
)
from .config import BaselineConfig, CriticalityConfig, MarketAnalysisConfig  # src/local_market/config.py，分别定义基线、格点渗流和市场系综配置。
from .networks import sample_lattice_occupancy  # src/local_market/networks.py，独立抽取格点占据状态。
from .percolation import ClusterConfiguration, identify_lattice_clusters  # src/local_market/percolation.py，定义格点团簇结果并按边界识别团簇。
from .statistics import (  # src/local_market/statistics.py，提供渗流与收益的统计量。
    finite_cluster_sizes,  # 按排除规则选出用于有限团簇统计的规模。
    return_statistics,  # 汇总收益矩和极端概率。
    standardize_returns,  # 按样本均值和标准差计算 z。
)


def run_baseline(config: BaselineConfig) -> MarketSimulation:
    """Run one configured baseline experiment with a single RNG stream."""
    # 根据配置运行固定团簇或逐步重建团簇的基线交易。
    # 引用
    #     scripts/simulation/run_baseline.py 的 main() 用它生成并保存逐步交易数据。
    # 输入
    #     config 为 BaselineConfig，指定格点、边界、交易模式、步数、活跃率、深度和种子。
    # 输出 MarketSimulation，封装需求、收益、对数价格及每步团簇配置计数。
    rng = np.random.default_rng(config.seed)  # 建立占据抽样与交易行动共用的可复现随机流。

    def make_configuration():
        # 抽取一份格点占据状态并识别四邻域团簇。
        # 引用
        #     run_baseline() 在 quenched 分支直接调用一次，在 annealed 分支将其传给 src/local_market/cb_model.py 的 simulate_annealed() 逐步调用。
        # 输入无显式参数，读取外层 config 并推进 rng 的状态。
        # 输出 ClusterConfiguration，包含占据标签、团簇规模和相应边界统计。
        occupied = sample_lattice_occupancy(config.length, config.p, rng)  # 用配置的格点边长和占据概率抽取布尔阵列。
        return identify_lattice_clusters(occupied, config.boundary)  # 按配置边界划分团簇，作为交易前结构。

    if config.mode == "quenched":
        # 只抽一次配置，在同一组团簇上重复交易。
        return simulate_quenched(
            make_configuration(),  # 立即生成固定团簇配置。
            config.steps,
            config.activity,
            config.market_depth,
            rng,
        )
    # 将配置工厂交给逐步更新模式。
    return simulate_annealed(
        make_configuration,  # 传入函数本身，每步由交易模拟调用。
        config.steps,
        config.activity,
        config.market_depth,
        rng,
    )


def _sem(values: pd.Series) -> float:
    # 用样本标准差除以非缺失样本数的平方根估计均值标准误。
    # 引用
    #     run_criticality_scan() 将它作为 Pandas 分组聚合回调，由 Pandas 用于占据数、团簇数、最大规模、序参量、贯通和易感性列。
    # 输入
    #     values 为一个分组的一维 Series，缺失值不计入样本数。
    # 输出 浮点标准误，有效观测不足两个时通常为 NaN。
    return float(values.std(ddof=1) / np.sqrt(values.count()))  # 样本标准差使用 ddof=1，计数使用非缺失观测数。


def run_criticality_scan(
    config: CriticalityConfig, show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run an ensemble site-percolation scan.

    Returns per-realization observations, grouped summary statistics, pooled
    cluster histograms, and grid-based pseudocritical estimates.
    """
    # 为渗流临界性质实验按边长和占据概率扫描格点，测量序参量、贯通概率与易感性。
    # 引用
    #     scripts/simulation/scan_criticality.py 的 main() 保存样本、统计和直方图，并从汇总表估计有限尺寸标度。
    # 输入
    #     config 为 CriticalityConfig，含扫描网格、排除规则、样本数、bootstrap 次数和种子，峰位斜率计算需要至少两个概率点。
    #     show_progress 默认 True，显示逐配置扫描进度。
    # 输出依次为
    #     逐实现样本表。
    #     边长和概率分组汇总表，含均值、标准误和池化易感性置信区间。
    #     按规模汇总的全部与有限团簇计数和每格点密度。
    #     各边长的易感性峰与序参量最大斜率的网格估计。
    rng = np.random.default_rng(config.seed)  # 初始化贯穿整个渗流扫描的占据随机流。
    sample_rows: list[dict[str, object]] = []
    histogram_counts: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    total = sum(config.samples_per_length[length] for length in config.lengths)  # 累加每个边长在单个概率下的配置数。
    # 进度按每份占据配置计数。
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
            p = config.p_c * ratio  # 把扫描的无量纲 p/p_c 还原为实际占据概率。
            all_histogram = np.zeros(maximum_size + 1, dtype=np.int64)  # 为所有团簇建立规模为下标的计数表，保留索引 0。
            finite_histogram = np.zeros(maximum_size + 1, dtype=np.int64)  # 为有限团簇建立采用相同下标的独立计数表。
            for sample in range(samples):  # 逐个生成当前尺寸与概率点的独立配置。
                occupied = sample_lattice_occupancy(length, p, rng)  # 在当前概率独立抽取一个格点配置。
                clusters = identify_lattice_clusters(occupied, config.boundary)  # 按实验边界划分团簇。
                # 应用配置指定的排除规则，得到用于有限团簇统计的规模。
                finite_sizes = finite_cluster_sizes(
                    clusters, config.finite_cluster_exclusion
                )
                finite_denominator = int(finite_sizes.sum())  # Σs 为该配置有限团簇的占据节点数。
                finite_numerator = int(np.dot(finite_sizes, finite_sizes))  # Σs² 为规模平方和。
                susceptibility = (
                    finite_numerator / finite_denominator if finite_denominator else 0.0  # 逐配置易感性用 Σs²/Σs，分母为零时取零。
                )
                if clusters.cluster_count:
                    # 按规模累计当前配置的所有团簇频数。
                    all_histogram += np.bincount(
                        clusters.sizes, minlength=maximum_size + 1
                    )
                if finite_sizes.size:
                    # 按规模累计当前配置筛选后的团簇频数。
                    finite_histogram += np.bincount(
                        finite_sizes, minlength=maximum_size + 1
                    )
                # 保存一个配置的渗流观测和有限团簇矩。
                sample_rows.append(
                    {
                        "length": length,
                        "p_ratio": ratio,
                        "p": p,
                        "sample": sample,
                        "occupied_count": clusters.occupied_count,
                        "cluster_count": clusters.cluster_count,
                        "largest_size": clusters.largest_size,
                        "largest_occupied_fraction": clusters.largest_occupied_fraction,  # 最大团簇规模除以已占据格点数。
                        "p_infinity_sample": clusters.p_infinity,  # 序参量以所有格点为分母。
                        "spans": clusters.spans,
                        "susceptibility": susceptibility,
                        "finite_first_moment": finite_denominator,
                        "finite_second_moment": finite_numerator,
                    }
                )
                progress.update(1)  # 标记当前配置完成。
            histogram_counts[(length, ratio)] = (all_histogram, finite_histogram)
    progress.close()  # 结束扫描进度显示。

    samples_frame = pd.DataFrame(sample_rows)  # 组装一行对应一个配置的原始样本表。
    # 按边长和概率汇总结构均值、均值标准误和有限团簇矩。
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
            p_infinity=("p_infinity_sample", "mean"),  # 对各配置的 P∞ 取均值。
            p_infinity_sem=("p_infinity_sample", _sem),
            spanning_probability=("spans", "mean"),  # 对各配置是否贯通或绕行的布尔值取均值。
            spanning_probability_sem=("spans", _sem),
            susceptibility_mean=("susceptibility", "mean"),  # 先在每个配置取矩比，再按配置平均。
            susceptibility_sem=("susceptibility", _sem),
            finite_first_moment=("finite_first_moment", "sum"),  # 跨配置汇总 Σs，供池化分母使用。
            finite_second_moment=("finite_second_moment", "sum"),  # 跨配置汇总 Σs²，供池化分子使用。
        )
    )
    # 池化易感性由全部配置矩之和取比，空分母填 0。
    summary["susceptibility_pooled"] = (
        summary["finite_second_moment"] / summary["finite_first_moment"]
    ).fillna(0.0)
    bootstrap_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 2]))  # 给 bootstrap 使用独立于占据扫描的随机流。
    bootstrap_rows: list[dict[str, float | int]] = []
    # 按相同尺寸和概率取出各组可重抽的配置。
    for (length, ratio, p), group in samples_frame.groupby(
        ["length", "p_ratio", "p"], sort=True
    ):
        first = group["finite_first_moment"].to_numpy(dtype=np.int64)  # 提取每个配置的有限团簇规模和。
        second = group["finite_second_moment"].to_numpy(dtype=np.int64)  # 提取每个配置的有限团簇规模平方和。
        estimates = np.empty(config.bootstrap_replicates, dtype=float)  # 为每次重抽后的池化易感性留出空间。
        for replicate in range(config.bootstrap_replicates):  # 逐轮计算一次重抽后的池化易感性。
            indices = bootstrap_rng.integers(0, len(group), size=len(group))  # 在相同尺寸和概率内有放回抽取配置索引。
            denominator = int(first[indices].sum())  # 池化分母保留同一个配置被重复抽到的次数。
            estimates[replicate] = (
                float(second[indices].sum() / denominator) if denominator else 0.0  # 用相同抽样索引对平方和取池化比值。
            )
        # 保存本组 bootstrap 的标准误与百分位区间。
        bootstrap_rows.append(
            {
                "length": int(length),
                "p_ratio": float(ratio),
                "p": float(p),
                "susceptibility_bootstrap_se": float(estimates.std(ddof=1)),  # 以各轮估计的样本标准差作为 bootstrap 标准误。
                "susceptibility_ci_low": float(np.quantile(estimates, 0.025)),  # 取重抽分布的 2.5% 分位数为下界。
                "susceptibility_ci_high": float(np.quantile(estimates, 0.975)),  # 取重抽分布的 97.5% 分位数为上界。
            }
        )
    # 按尺寸和概率一对一合并重抽样区间。
    summary = summary.merge(
        pd.DataFrame(bootstrap_rows), on=["length", "p_ratio", "p"], validate="one_to_one"
    )

    histogram_rows: list[dict[str, object]] = []
    for (length, ratio), (all_counts, finite_counts) in histogram_counts.items():  # 分别整理各尺寸和概率比下的全部与有限规模频数。
        samples = config.samples_per_length[length]
        nonzero_sizes = np.flatnonzero((all_counts + finite_counts) > 0)  # 定位两类直方图至少有一类出现的规模。
        for size in nonzero_sizes[nonzero_sizes > 0]:
            # 每个正规模写一行计数和按总格点机会归一化的频数。
            histogram_rows.append(
                {
                    "length": length,
                    "p_ratio": ratio,
                    "p": config.p_c * ratio,
                    "cluster_size": int(size),
                    "all_cluster_count": int(all_counts[size]),  # 取所有团簇的频数。
                    "finite_cluster_count": int(finite_counts[size]),  # 取有限团簇的频数。
                    "all_n_s_per_site": all_counts[size]
                    / (samples * length * length),  # 除以配置数乘单配置格点数。
                    "finite_n_s_per_site": finite_counts[size]
                    / (samples * length * length),  # 有限团簇频数采用同一分母。
                }
            )
    histogram_frame = pd.DataFrame(histogram_rows)  # 将稀疏的规模记录汇为长表。

    critical_rows: list[dict[str, object]] = []
    for length, group in summary.groupby("length", sort=True):  # 每个边长估计一组网格峰位。
        ordered = group.sort_values("p").reset_index(drop=True)  # 按实际占据概率排序并重置位置索引。
        chi_index = int(ordered["susceptibility_pooled"].idxmax())  # 在扫描网格上找池化易感性最大的位置。
        slopes = np.gradient(ordered["p_infinity"], ordered["p"])  # 用实际 p 间距近似 P∞ 对 p 的导数。
        slope_index = int(np.argmax(slopes))  # 找序参量斜率最大的网格索引。
        # 同时记录两种伪临界估计的概率和峰值。
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
    critical_frame = pd.DataFrame(critical_rows)  # 汇总每个边长的一条伪临界记录。
    return samples_frame, summary, histogram_frame, critical_frame


def estimate_finite_size_scaling(
    summary: pd.DataFrame, critical_ratio: float = 1.0
) -> pd.DataFrame:
    """Fit log-log finite-size trends at the configured critical ratio."""
    # 为渗流临界性质实验在指定 p/p_c 对序参量和池化易感性拟合有限尺寸幂律。
    # 引用
    #     scripts/simulation/scan_criticality.py 的 main() 保存拟合表并绘制标度趋势。
    # 输入
    #     summary 为 run_criticality_scan() 的汇总表，使用 length、p_ratio、p_infinity 和 susceptibility_pooled。
    #     critical_ratio 为所选无量纲占据概率比，默认 1.0。
    # 输出 每个观测量一行的回归表，含指数、标准误、理论指数、截距、R² 和尺寸数。少于三个尺寸或观测量非正时抛出 ValueError。
    # 用浮点近似相等取出目标概率比，并按边长排序。
    critical = summary[np.isclose(summary["p_ratio"], critical_ratio)].sort_values(
        "length"
    )
    if len(critical) < 3:  # 根据拟合所需的尺寸数检查筛选结果。
        raise ValueError("at least three system sizes are required for scaling fits")
    rows: list[dict[str, object]] = []
    definitions = (
        ("p_infinity", "p_infinity", -5.0 / 48.0),  # 二维渗流序参量理论斜率为 −β/ν = −5/48。
        ("susceptibility", "susceptibility_pooled", 43.0 / 24.0),  # 二维渗流易感性理论斜率为 γ/ν = 43/24。
    )
    log_length = np.log(critical["length"].to_numpy(dtype=float))  # 以自然对数边长作为两个回归的共同横坐标。
    for observable, column, theoretical_exponent in definitions:
        values = critical[column].to_numpy(dtype=float)  # 取得当前观测量在各尺寸下的值。
        if np.any(values <= 0):  # 自然对数拟合要求每个观测值严格为正。
            raise ValueError(f"{column} must be positive for log-log scaling")
        fit = stats.linregress(log_length, np.log(values))  # 将观测量取对数并做线性回归，斜率是幂律指数。
        # 保存拟合参数和对应的理论参照。
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
    return pd.DataFrame(rows)  # 返回每个观测量一行的表。


def run_market_ensemble(
    config: MarketAnalysisConfig, show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate return ensembles, reusing configurations across activities."""
    # 为市场收益率的影响因素实验在同一占据配置组上模拟不同活跃率的交易系综。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 取得逐笔交易和配置结构，后续做标准化、尾部和活动统计。
    # 输入
    #     config 为 MarketAnalysisConfig，含格点和交易参数、各层样本数及种子。
    #     show_progress 默认 True，按占据概率下的重复数显示进度。
    # 输出依次为
    #     逐笔需求、收益和活跃团簇等组成的交易表。
    #     每份占据配置一行的结构表，供所有活跃率共用。
    return_frames: list[pd.DataFrame] = []
    structure_rows: list[dict[str, object]] = []
    total = len(config.p_ratios) * config.replicates  # 进度总数为概率比数乘重复数。
    progress = tqdm(total=total, desc="market ensembles", unit="replicate", disable=not show_progress)  # 每完成一个概率比和重复的组合推进一次。

    for p_index, ratio in enumerate(config.p_ratios):  # 保留占据概率比的索引以派生随机流。
        p = config.p_c * ratio
        for replicate in range(config.replicates):  # 同一 p 下逐次生成独立格点系综。
            # 为每个概率与重复组合派生独立的占据随机流。
            occupancy_rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, 10, p_index, replicate])  # 用实验种子、占据流标记和组合索引生成子种子。
            )
            configurations = []
            for sample in range(config.configurations_per_replicate):  # 为本次重复生成全部可跨活跃率复用的占据配置。
                occupied = sample_lattice_occupancy(config.length, p, occupancy_rng)  # 用共用占据随机流独立抽取一个格点样本。
                clusters = identify_lattice_clusters(occupied, config.boundary)  # 按指定边界识别这份样本的团簇。
                configurations.append(clusters)  # 缓存配置供当前重复的所有活跃率使用。
                # 每份配置只记录一次结构，避免随活跃率重复。
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

            for activity_index, activity in enumerate(config.activities):  # 遍历共用配置上的各活跃率，索引用于分离交易随机流。
                # 为每个概率、重复和活跃率组合派生交易随机流。
                action_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 20, p_index, replicate, activity_index]  # 交易流使用独立标记，并按活跃率索引区分。
                    )
                )
                demand_blocks = []
                active_blocks = []
                available_blocks = []
                configuration_blocks = []
                for sample, clusters in enumerate(configurations):  # 逐配置累积等长交易区块，保留配置编号。
                    # 在该固定规模配置上批量抽取多笔交易。
                    demands, active_counts = sample_demands_and_active_counts(
                        clusters.sizes,
                        config.trades_per_configuration,
                        activity,
                        action_rng,
                    )
                    demand_blocks.append(demands)  # 按配置顺序累积逐笔超额需求。
                    active_blocks.append(active_counts)  # 保留相同配置顺序的活跃团簇计数。
                    # 将配置的团簇总数复制到该配置的每笔交易。
                    available_blocks.append(
                        np.full(
                            config.trades_per_configuration,
                            clusters.cluster_count,
                            dtype=np.int64,
                        )
                    )
                    # 将配置编号重复到该配置的每条交易记录。
                    configuration_blocks.append(
                        np.full(config.trades_per_configuration, sample, dtype=np.int64)
                    )
                demands = np.concatenate(demand_blocks)  # 沿交易顺序拼接配置的需求区块。
                observations = demands.size
                # 为当前参数组合装配逐笔交易表。
                return_frames.append(
                    pd.DataFrame(
                        {
                            "length": config.length,
                            "p_ratio": ratio,
                            "p": p,
                            "activity": activity,
                            "replicate": replicate,
                            "step": np.arange(observations, dtype=np.int64),  # 步号从每次重复的 0 开始连续编号。
                            "configuration": np.concatenate(configuration_blocks),  # 将每笔交易映射回其占据配置。
                            "available_cluster_count": np.concatenate(available_blocks),  # 拼接逐笔可用团簇数。
                            "active_cluster_count": np.concatenate(active_blocks),  # 拼接逐笔实际参加交易的团簇数。
                            "demand": demands,
                            "return": demands.astype(float) / config.market_depth,  # 将整数净需求换算为对数收益率 r = D/λ。
                        }
                    )
                )
            progress.update(1)  # 本次重复的所有活跃率处理完成。
    progress.close()  # 结束市场系综进度显示。
    return pd.concat(return_frames, ignore_index=True), pd.DataFrame(structure_rows)  # 统一交易行索引，另将配置结构转为表。


def run_quenched_market_trajectories(
    config: MarketAnalysisConfig,
) -> tuple[ClusterConfiguration, dict[float, MarketSimulation]]:
    """Generate the dedicated 3000-step trajectories on one lattice at p_c."""
    # 为市场收益率的影响因素实验在同一个 p_c 格点上生成各活跃率的 3000 步固定配置轨迹。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 生成独立示例轨迹。
    #     scripts/analysis/audit_market_time_series.py 的 main() 加载同一生成逻辑以审计时间序列。
    # 输入
    #     config 为 MarketAnalysisConfig，使用边长、p_c、边界、活跃率、市场深度和种子。
    # 输出依次为
    #     各活跃率共用的 ClusterConfiguration。
    #     从活跃率映射到 MarketSimulation 的字典，每条轨迹有 3000 笔交易。
    occupancy_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 50]))  # 用独立随机流生成用于展示的固定格点。
    occupied = sample_lattice_occupancy(config.length, config.p_c, occupancy_rng)  # 在参照临界概率 p_c 抽取单个占据样本。
    fixed_configuration = identify_lattice_clusters(occupied, config.boundary)  # 只识别一次团簇，供所有活跃率共用。
    trajectories = {}
    for index, activity in enumerate(config.activities):  # 逐活跃率生成固定格点的独立交易序列。
        action_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 51, index]))  # 为每个活跃率用索引派生独立交易随机流。
        # 对同一个固定格点重复模拟 3000 笔交易。
        trajectories[activity] = simulate_quenched(
            fixed_configuration, 3000, activity, config.market_depth, action_rng,
        )
    return fixed_configuration, trajectories


def summarize_market_returns(
    returns_frame: pd.DataFrame, config: MarketAnalysisConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize per parameter combination and summarize overall and by seed."""
    # 为市场收益率的影响因素实验按参数组合统一标准化收益，并统计池化结果和重复间差异。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 为主分析准备 z 和收益汇总。
    #     scripts/analysis/finalize_market_statistics.py 的 main() 汇总重新标准化的最终结果。
    # 输入
    #     returns_frame 为 run_market_ensemble() 的逐笔交易表，需参数列、重复索引、原始收益和两种团簇计数。
    #     config 为 MarketAnalysisConfig，使用极端事件阈值。
    # 输出依次为
    #     新增 z 的逐笔交易表副本。
    #     每组参数一行的池化统计及重复间标准差。
    #     每组参数和重复一行的统计，极端概率沿用池化标准化。
    data = returns_frame.copy()  # 复制逐笔表，使写入 z 不影响调用方数据。
    data["z"] = np.nan
    overall_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    keys = ["length", "p_ratio", "p", "activity"]
    for key, group in data.groupby(keys, sort=True):  # 按格点长度、占据概率与活跃率分组，跨重复池化。
        z, mean, standard_deviation = standardize_returns(group["return"].to_numpy())  # 从组内全部收益计算共同的均值、标准差和 z。
        data.loc[group.index, "z"] = z
        overall = dict(zip(keys, key, strict=True))  # 将当前分组键还原为结果表的参数列。
        # 为该组全部交易计算收益矩、极端概率与活跃团簇均值。
        overall.update(
            return_statistics(
                group["return"].to_numpy(),
                z,
                config.extreme_thresholds,
                group["active_cluster_count"].to_numpy(),  # 提供实际活跃团簇数供统计。
            )
        )
        overall["standardization_mean"] = mean
        overall["standardization_std"] = standard_deviation
        overall["mean_available_cluster_count"] = float(
            group["available_cluster_count"].mean()  # 另外计算配置中可参与交易的团簇数的逐笔均值。
        )
        overall_rows.append(overall)  # 将当前参数的池化统计纳入输出。

        for replicate, replicate_group in group.groupby("replicate", sort=True):  # 在当前参数内进一步按重复划分。
            replicate_z = data.loc[replicate_group.index, "z"].to_numpy()  # 读取整组参数共同标准化的 z。
            row = dict(zip(keys, key, strict=True))  # 还原参数列，供重复统计使用。
            row["replicate"] = int(replicate)
            # 逐重复统计原始收益矩，极端概率仍用整组参数的 z。
            row.update(
                return_statistics(
                    replicate_group["return"].to_numpy(),
                    replicate_z,
                    config.extreme_thresholds,
                    replicate_group["active_cluster_count"].to_numpy(),
                )
            )
            replicate_rows.append(row)  # 保存该重复的统计行。

    overall_frame = pd.DataFrame(overall_rows)  # 将池化记录装配为表。
    replicate_frame = pd.DataFrame(replicate_rows)  # 将重复层记录装配为表。
    # 对各参数组合计算重复间离散程度。
    dispersion = (
        replicate_frame.groupby(keys, as_index=False)
        .agg(
            replicate_return_std_sd=("return_std", "std"),  # 用重复层收益标准差的样本标准差衡量波动率估计离散。
            replicate_kurtosis_sd=("return_excess_kurtosis", "std"),  # 计算重复层超额峰度的样本标准差。
            replicate_p_abs_z_gt_3_sd=("p_abs_z_gt_3", "std"),  # 计算双侧 3σ 极端事件概率的重复间标准差。
            replicate_p_abs_z_gt_5_sd=("p_abs_z_gt_5", "std"),  # 计算双侧 5σ 极端事件概率的重复间标准差。
            replicate_p_abs_z_gt_8_sd=("p_abs_z_gt_8", "std"),  # 计算双侧 8σ 极端事件概率的重复间标准差。
            replicate_negative_crash_probability_sd=(  # 汇总单侧崩盘概率的重复间标准差。
                "negative_crash_probability",
                "std",
            ),
        )
    )
    overall_frame = overall_frame.merge(dispersion, on=keys, validate="one_to_one")  # 按参数组合把重复间离散度接回池化表。
    return data, overall_frame, replicate_frame
