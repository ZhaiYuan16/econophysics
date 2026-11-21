"""Statistical observables for percolation and market simulations."""
# 从团簇配置和交易收益计算统计量。
# finite_cluster_sizes() 和 cluster_histogram() 筛选、汇总团簇规模。
# standardize_returns() 和 nonzero_absolute_standardized_returns() 准备标准化收益与尾部样本，return_statistics() 和 return_distribution_table() 输出描述统计和分布。
#
# 供渗流临界性质实验、市场收益率的影响因素实验和网络拓扑压力测试汇总模拟结果。

from __future__ import annotations  # 延后求值类型注解。

from typing import Literal  # 标准库类型标注，用字面值限定团簇排除规则。

import numpy as np  # NumPy 用于筛选、计数、标准化和分布网格。
from scipy import stats  # SciPy 统计模块用于估计收益率偏度和超额峰度。

from .percolation import ClusterConfiguration  # src/local_market/percolation.py，提供带规模和贯通编号的格点团簇配置。

ClusterExclusion = Literal["largest", "spanning", "none"]


def finite_cluster_sizes(
    configuration: ClusterConfiguration,
    exclusion: ClusterExclusion = "largest",
) -> np.ndarray:
    """Return cluster sizes after applying a documented exclusion rule.

    ``largest`` removes exactly one largest cluster from each realization.
    ``spanning`` removes every spanning/wrapping cluster, while ``none``
    retains all clusters.
    """
    # 按指定规则从配置中筛出用于有限团簇统计的规模。
    # 引用
    #     src/local_market/criticality_revision.py 的 scan() 分别按最大团簇和贯通规则计算矩与相关长度。
    #     src/local_market/experiments.py 的 run_criticality_scan() 对每份格点配置统计规模分布和易感性。
    #     src/local_market/network_analysis.py 的 run_network_criticality_scan() 用网络规模与最大团簇编号计算有限团簇统计。
    #     defense/cb_experiment.py 的 occupancy_block() 内部 compute() 比较两种排除口径。
    # 输入
    #     configuration 为 ClusterConfiguration 或调用方传入的同接口对象，提供规模数组和所选规则需要的标签。
    #     exclusion 默认为 largest，只排除一个最大团簇，spanning 排除所有贯通或绕行团簇，none 保留全部。
    # 输出 一维规模数组，按原团簇顺序排列。空配置返回空数组，非空配置遇到非法排除规则抛出 ValueError。
    if exclusion == "none" or configuration.cluster_count == 0:
        return configuration.sizes.copy()  # 保留原顺序并返回独立副本。
    keep = np.ones(configuration.cluster_count, dtype=bool)  # 每个团簇占掩码的一位，初始全部纳入。
    if exclusion == "largest":
        keep[configuration.largest_label - 1] = False  # 将从 1 开始的团簇标签换成从 0 开始的数组下标。
    elif exclusion == "spanning":
        for label in configuration.spanning_labels:
            keep[label - 1] = False  # 逐个移除标记为贯通或绕行的团簇。
    else:
        raise ValueError("exclusion must be 'largest', 'spanning', or 'none'")
    return configuration.sizes[keep]


def cluster_histogram(sizes: np.ndarray, max_size: int | None = None) -> np.ndarray:
    """Count clusters by integer size; element zero is retained as zero."""
    # 按整数规模给团簇计数。
    # 引用
    #     defense/cb_experiment.py 的 simulate_cb() 将各拓扑汇总后的规模转换为演示用直方图。
    # 输入
    #     sizes 为一维正整数规模序列，可为空。
    #     max_size 为直方图的最小覆盖上界，包含该规模，默认由样本最大规模决定。
    # 输出 一维 int64 数组，索引为团簇规模，索引 0 为零，空样本返回相应长度的全零数组，非正规模抛出 ValueError。
    sizes = np.asarray(sizes, dtype=np.int64)  # 将规模转换为整数以用作频数数组下标。
    if sizes.size == 0:
        length = 1 if max_size is None else max_size + 1
        return np.zeros(length, dtype=np.int64)  # 空样本仍保留所需的规模索引。
    if np.any(sizes <= 0):  # 排除无法作为正规模的计数输入。
        raise ValueError("cluster sizes must be positive")
    minlength = int(sizes.max()) + 1 if max_size is None else max_size + 1  # 使直方图至少包含指定上界，当未指定上界时包含所观测到的最大规模。
    return np.bincount(sizes, minlength=minlength).astype(np.int64)  # 按规模整数计数，超过 max_size 的样本会延长数组。


def standardize_returns(returns: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize returns using the sample standard deviation (ddof=1)."""
    # 按样本均值和样本标准差计算标准化收益 z。
    # 引用
    #     src/local_market/experiments.py 的 summarize_market_returns() 按格点市场参数池化标准化。
    #     src/local_market/network_analysis.py 的 summarize_network_market() 按网络场景标准化。
    #     scripts/analysis/audit_downstream_statistics.py 的 validate_saved_distributions() 重算保存表中的分布。
    #     defense/live.py 的 return_table() 为演示收益计算 z。
    # 输入
    #     returns 为至少两个观测的数值数组，均值和标准差沿所有元素计算。
    # 输出依次为
    #     z，与输入形状相同。
    #     原始收益的算术均值。
    #     样本标准差，分母为观测数减一，观测不足或标准差为零时抛出 ValueError。
    values = np.asarray(returns, dtype=float)  # 转换为浮点以保留分数收益及标准化结果。
    if values.size < 2:  # 样本标准差至少需要两个观测。
        raise ValueError("at least two returns are required")
    mean = float(values.mean())  # 用全体观测的均值居中。
    standard_deviation = float(values.std(ddof=1))  # 用 N−1 分母估计标准差。
    if standard_deviation == 0.0:  # 常量收益序列不存在可用的标准化尺度。
        raise ValueError("returns have zero variance")
    return (values - mean) / standard_deviation, mean, standard_deviation


def nonzero_absolute_standardized_returns(
    returns: np.ndarray, standardized: np.ndarray
) -> np.ndarray:
    """Tail sample: discard exact zero raw returns before taking absolute z."""
    # 按同一观测位置剔除零原始收益和非有限值，然后取标准化收益的绝对值。
    # 引用
    #     return_distribution_table() 用它确定绝对收益尾部阈值的取点范围。
    #     scripts/analysis/analyze_market_returns.py 的 main() 准备尾部拟合样本。
    #     scripts/analysis/audit_market_tails_visual.py 的 main() 按修正口径重拟合格点收益尾部。
    # 输入
    #     returns 为原始收益数组。
    #     standardized 为等形状的标准化收益数组 z。
    # 输出 一维 绝对 z 样本，按筛选前的元素顺序排列，形状不一致时抛出 ValueError。
    raw = np.asarray(returns, dtype=float)  # 以浮点表示原始收益，供有限性和精确零判定。
    z = np.asarray(standardized, dtype=float)  # 以浮点表示同一观测的 z。
    if raw.shape != z.shape:  # 原始收益和 z 必须逐元素对齐。
        raise ValueError("returns and standardized values must have the same shape")
    valid = np.isfinite(raw) & np.isfinite(z) & (raw != 0)  # 保留原始收益非零且两种表示均有限的观测。
    return np.abs(z[valid])  # 在筛选后才对标准化收益取绝对值。


def return_statistics(
    returns: np.ndarray,
    standardized: np.ndarray,
    extreme_thresholds: tuple[float, ...],
    active_cluster_counts: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Descriptive moments and standardized extreme-event frequencies."""
    # 计算收益矩、零收益比例和标准化极端事件频数，可同时汇总活跃团簇数。
    # 引用
    #     src/local_market/experiments.py 的 summarize_market_returns() 和 src/local_market/network_analysis.py 的 summarize_network_market() 分别生成格点和网络的池化及重复层统计。
    #     defense/live.py 的 return_table() 提供演示表中的收益统计。
    # 输入
    #     returns 为至少两个观测的原始收益数组。
    #     standardized 为与 returns 等形状的 z 数组。
    #     extreme_thresholds 为各个双侧阈值 c，统计 绝对 z > c。
    #     active_cluster_counts 为可选的逐笔活跃团簇计数，默认 None 时省略该项均值。
    # 输出 指标字典，含观测数、收益矩、零收益比例、z < −5 的比例、指定阈值的双侧比例和可选活跃团簇均值。
    returns = np.asarray(returns, dtype=float)  # 统一原始收益的计算类型。
    z = np.asarray(standardized, dtype=float)  # 统一用于极端事件阈值比较的 z 类型。
    if returns.shape != z.shape or returns.size < 2:  # 原始收益和 z 要对齐，并至少包含两个观测。
        raise ValueError("returns and standardized values must have equal nontrivial shape")
    result: dict[str, float | int] = {
        "observations": int(returns.size),  # 把整个数组中的元素数作为样本量。
        "return_mean": float(returns.mean()),  # 对全样本求算术均值。
        "return_std": float(returns.std(ddof=1)),  # 以 N−1 为分母计算样本标准差。
        "return_skewness": float(stats.skew(returns, bias=False)),  # 使用经样本偏差修正的偏度。
        "return_excess_kurtosis": float(stats.kurtosis(returns, fisher=True, bias=False)),  # 使用经偏差修正的 Fisher 超额峰度，正态基准为 0。
        "zero_return_fraction": float(np.mean(returns == 0)),  # 以全样本为分母计精确零收益。
        "negative_crash_probability": float(np.mean(z < -5.0)),  # 以全样本为分母计严格跌破 −5 的观测。
    }
    if active_cluster_counts is not None:
        # 仅在调用方提供活跃团簇数时记录其均值。
        result["mean_active_cluster_count"] = float(
            np.asarray(active_cluster_counts).mean()  # 对传入的所有活跃团簇计数求均值。
        )
    for threshold in extreme_thresholds:
        label = f"p_abs_z_gt_{threshold:g}"
        result[label] = float(np.mean(np.abs(z) > threshold))  # 对双侧严格超过阈值的指示变量取全样本均值。
    return result


def return_distribution_table(
    standardized: np.ndarray,
    pdf_limit: float = 12.0,
    pdf_bins: int = 240,
    ccdf_points: int = 120,
    *,
    raw_returns: np.ndarray | None = None,
    include_pdf: bool = True,
) -> list[dict[str, float | int | str]]:
    """Build absolute-return CCDF data and, by default, signed-return PDF data."""
    # 生成有符号 z 的密度直方图数据和 绝对 z 的经验 CCDF 数据。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 和 scripts/analysis/finalize_market_statistics.py 的 main() 生成格点市场分布表。
    #     scripts/simulation/compare_networks.py 的 main() 生成网络收益分布，scripts/analysis/audit_downstream_statistics.py 的 validate_saved_distributions() 重算保存表。
    # 输入
    #     standardized 为收益的标准化数组 z。
    #     pdf_limit 为有符号直方图的对称画图范围，默认 ±12。
    #     pdf_bins 为该范围内的箱数，默认 240。
    #     ccdf_points 为 CCDF 的对数间距阈值数，默认 120。
    #     raw_returns 可选原始收益，用于筛选阈值的候选范围，形状须与 standardized 相同。
    #     include_pdf 默认 True，控制是否生成有符号 PDF 数据。
    # 输出 逐点字典列表，列为 distribution、x、value 和 count，无候选 绝对 z 时不生成 CCDF 行。
    z = np.asarray(standardized, dtype=float)  # 将 z 转为浮点数组供两种分布共用。
    rows: list[dict[str, float | int | str]] = []
    if include_pdf:
        edges = np.linspace(-pdf_limit, pdf_limit, pdf_bins + 1)  # 在 ±pdf_limit 间等间隔划分直方图。
        counts, _ = np.histogram(z, bins=edges)  # 统计落在绘图范围内每个箱的观测数。
        widths = np.diff(edges)  # 用相邻箱边之差取得每一箱的宽度。
        centers = (edges[:-1] + edges[1:]) / 2.0  # 以箱中心作为 PDF 表的横坐标。
        rows.extend(
            {
                "distribution": "signed_pdf",
                "x": float(center),
                "value": float(count / (z.size * width)),  # 除以全样本数和箱宽，保留截去图外概率后的密度。
                "count": int(count),
            }
            for center, width, count in zip(centers, widths, counts, strict=True)  # 配对各箱的中心、宽度和计数。
            if count > 0
        )
    absolute = np.abs(z)  # 保留全样本的 绝对 z，供经验 CCDF 计数。
    # 筛出的样本仅用于决定 CCDF 阈值范围。
    positive = (nonzero_absolute_standardized_returns(raw_returns, z)
                if raw_returns is not None else absolute[absolute > 0])
    if positive.size:
        lower = max(float(positive.min()), 0.05)  # 阈值下限至少为 0.05。
        upper = max(lower * 1.001, float(positive.max()))  # 保证上限严格大于下限，同时覆盖样本最大值。
        thresholds = np.geomspace(lower, upper, ccdf_points)  # 在上下限间按等比间距生成阈值。
        # 给每个阈值写入全样本的超越比例和计数。
        rows.extend(
            {
                "distribution": "absolute_ccdf",
                "x": float(threshold),  # 保存正的绝对标准化收益阈值。
                "value": float(np.mean(absolute >= threshold)),  # 由全样本中 绝对 z ≥ 阈值的占比求 CCDF，分母含零原始收益。
                "count": int(np.count_nonzero(absolute >= threshold)),  # 保存达到或超过该阈值的全样本观测数。
            }
            for threshold in thresholds
        )
    return rows
