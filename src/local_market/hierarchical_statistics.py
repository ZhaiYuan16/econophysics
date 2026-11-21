"""Distribution statistics and tail fits from resampled configuration blocks."""
# 用按配置分块的稀疏频数表做分层重抽样和收益率统计。
# block_histogram() 构造压缩表，resampled_counts() 聚合重抽结果，select_network_configuration_blocks() 负责网络两级抽样。
# weighted_statistics() 汇总矩和尾部拟合，selection_aware_tail_gof() 在拟合优度检验中重选尾部下界。
#
# 市场收益率的影响因素实验和网络拓扑压力测试用本模块从交易系综估计统计不确定性。
from __future__ import annotations  # 延后求值类型注解。

import numpy as np  # NumPy 用于随机抽样、计数、加权计算和分位数定位。
from scipy import sparse  # SciPy 稀疏矩阵用于压缩不同配置中的重复收益。
from typing import Callable  # 标准库 Callable 标注配置块选择回调的签名。


def block_histogram(values: np.ndarray, block_ids: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Compress repeated discrete returns without losing their block membership."""
    # 将同一配置块中的重复收益压缩成稀疏频数表。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 与 scripts/simulation/compare_networks.py 的 main() 分别压缩格点和网络样本。
    #     scripts/analysis/audit_downstream_statistics.py 的 audit_one_family() 和 paired_market_activity_comparisons() 为单组审计和活跃率配对比较建立块表。
    #     scripts/analysis/audit_market_tails_visual.py 的 corrected_uncertainty() 为修正口径的尾部重抽样建立块表。
    # 输入
    #     values 为非空的一维逐笔收益。
    #     block_ids 为与 values 等长的非负配置块编号，可有缺号。
    # 输出依次为
    #     一维升序收益支持点。
    #     CSR 频数矩阵，行索引为块编号，列索引为收益支持点索引。
    support, columns = np.unique(np.asarray(values, dtype=float), return_inverse=True)  # 提取不同收益值，并把每笔收益映射到对应的列。
    # 每笔交易在对应块与收益的交叉位置贡献一票，转 CSR 时合并重复位置。
    matrix = sparse.coo_matrix(
        (np.ones(len(columns), dtype=np.int32), (block_ids, columns)),  # 用块编号做行索引，用收益映射做列索引。
        shape=(int(np.max(block_ids)) + 1, len(support)),  # 行数覆盖最大块编号，列数覆盖所有支持点。
    ).tocsr()
    return support, matrix


def resampled_counts(matrix: sparse.csr_matrix, selected_blocks: np.ndarray) -> np.ndarray:
    # 把一次有放回重抽所选的配置块合并成收益频数。
    # 引用
    #     selection_aware_tail_gof() 恢复本轮分块抽样的频数。
    #     scripts/analysis/analyze_market_returns.py 的 main() 和 scripts/simulation/compare_networks.py 的 main() 为分层 bootstrap 聚合样本。
    #     scripts/analysis/audit_downstream_statistics.py 的 audit_one_family()、paired_market_activity_comparisons() 以及 scripts/analysis/audit_market_tails_visual.py 的 corrected_uncertainty() 为审计和配对比较重建频数。
    # 输入
    #     matrix 为块×收益支持点的 CSR 频数矩阵。
    #     selected_blocks 为本轮抽到的矩阵行编号，可重复。
    # 输出 与矩阵列对齐的一维 int64 频数数组。
    multiplicities = np.bincount(selected_blocks, minlength=matrix.shape[0])  # 按行编号统计各配置块被抽中的次数。
    return np.asarray(multiplicities @ matrix).ravel().astype(np.int64)  # 用抽中次数加权求和各块，再转成一维整数频数。


def select_network_configuration_blocks(
    rng: np.random.Generator, networks: int, configurations: int
) -> np.ndarray:
    """Draw networks, then configurations within each selected network."""
    # 为网络拓扑压力测试先重抽网络、再重抽每个所选网络内的配置块。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 与 scripts/analysis/audit_downstream_statistics.py 的 audit_one_family() 在各自的 selector 回调中使用，供 bootstrap 和 selection_aware_tail_gof() 抽样。
    # 输入
    #     rng 为负责两级抽样的 NumPy 随机数生成器。
    #     networks 为网络实现数，必须为正。
    #     configurations 为每份网络的配置数，所有网络相同且为正。
    # 输出 长度为 networks×configurations 的一维块编号数组，允许重复。
    selected_networks = rng.integers(0, networks, size=networks)  # 从 0 到 networks−1 有放回抽取与原样本数量相同的网络。
    # 对每次抽中的网络独立有放回抽配置，再按网络优先的块编号拼接。
    return np.concatenate([
        network * configurations + rng.integers(0, configurations, size=configurations)  # 用网络编号乘每网配置数加上网内配置编号。
        for network in selected_networks
    ])


def weighted_quantile(values: np.ndarray, counts: np.ndarray, probability: float) -> float:
    """Match NumPy's linear sample quantile without expanding observations."""
    # 从有序支持和频数计算与展开样本相同的线性分位数。
    # 引用
    #     weighted_pareto_fit() 用它从重抽样本确定候选 xmin。
    # 输入
    #     values 为一维升序支持点。
    #     counts 为与 values 对齐的非负整数频数，总数须为正。
    #     probability 为 [0, 1] 内的分位概率。
    # 输出 插值得到的浮点样本分位数。
    cumulative = np.cumsum(counts)  # 累计频数把展开后的样本秩映射回支持点。
    n = int(cumulative[-1])  # 最后一个累计值就是展开后的样本数。
    position = (n - 1) * probability  # 按 NumPy 线性规则在从 0 开始的样本秩上定位。
    lower, upper = int(np.floor(position)), int(np.ceil(position))  # 找出插值跨度两端的整数秩。
    lo = values[np.searchsorted(cumulative, lower + 1)]  # 定位下端秩所属的支持点，转成从 1 开始的频数位置。
    hi = values[np.searchsorted(cumulative, upper + 1)]  # 定位上端秩所属的支持点。
    return float(lo + (position - lower) * (hi - lo))  # 在下端和上端的收益值之间按小数秩线性插值。


def weighted_pareto_fit(
    absolute_values: np.ndarray,
    counts: np.ndarray,
    quantiles: tuple[float, ...],
    minimum_count: int,
) -> dict[str, float | int]:
    """Reselect xmin by the original minimum-KS rule after each bootstrap draw."""
    # 对计数压缩的绝对标准化收益拟合连续 Pareto 尾，并按最小 KS 距离重选下界。
    # 引用
    #     weighted_statistics() 汇总重抽样本的尾部，selection_aware_tail_gof() 对合成样本重选下界。
    #     scripts/analysis/audit_market_tails_visual.py 的 corrected_uncertainty() 按配置重抽后重拟合尾部。
    # 输入
    #     absolute_values 为一维绝对标准化收益支持点，可未排序。
    #     counts 为与 absolute_values 对齐的非负整数频数。
    #     quantiles 为候选下界的分位概率。
    #     minimum_count 为尾部检验所需的最小展开观测数。
    # 输出 最优候选的分位数、xmin、PDF 指数、KS 距离和尾样本数，无有效候选时抛出 ValueError。
    order = np.argsort(absolute_values)  # 取得收益支持点的升序索引，使分位数计算可按秩定位。
    x = absolute_values[order]
    weights = counts[order]
    keep = (x > 0) & (weights > 0)  # 只保留正幅度且在本轮样本中出现的支持点。
    x, weights = x[keep], weights[keep]
    candidates = []
    for q in quantiles:
        xmin = weighted_quantile(x, weights, q)  # 按本轮频数计算候选阈值。
        tail = x >= xmin
        n = int(weights[tail].sum())  # 用尾部的展开频数计算样本量。
        if n < minimum_count:
            continue
        log_sum = float(np.dot(weights[tail], np.log(x[tail] / xmin)))  # 用频数加权累计各尾值相对下界的对数比。
        if log_sum <= 0:
            continue
        alpha = 1 + n / log_sum  # 由加权对数比得到连续 Pareto 的 PDF 指数。
        cumulative = np.cumsum(weights[tail])  # 累计尾频数，以定位各组重复值的首尾样本秩。
        empirical_last = cumulative / n  # 同值组最后一笔对应的经验累计比例。
        empirical_first = (cumulative - weights[tail] + 1) / n  # 同值组第一笔对应的经验累计比例。
        model = 1 - (x[tail] / xmin) ** (1 - alpha)  # 在各尾支持点计算模型的条件 CDF。
        # 同时比较同值组首笔和末笔，取两组最大偏差中的较大值。
        ks = float(max(np.max(np.abs(empirical_first - model)),
                       np.max(np.abs(empirical_last - model))))
        candidates.append({"xmin_quantile": q, "xmin": xmin, "alpha_pdf": alpha,  # 保存每个合格候选的参数和比较指标。
                           "ks_distance": ks, "tail_count": n})
    if not candidates:  # 所有候选都无足够尾观测或无法得到有效指数时失败。
        raise ValueError("no valid Pareto tail candidate")
    return min(candidates, key=lambda candidate: candidate["ks_distance"])  # 按最低 KS 距离返回候选。


def weighted_statistics(
    support: np.ndarray, counts: np.ndarray, quantiles: tuple[float, ...], minimum_count: int,
    exclude_zero_returns: bool = False,
) -> dict[str, float | int]:
    # 用收益频数计算均值、样本标准差、经修正的超额峰度、极端概率和尾参数。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 和 scripts/simulation/compare_networks.py 的 main() 生成分层 bootstrap 指标。
    #     scripts/analysis/audit_downstream_statistics.py 的 audit_one_family() 生成点估计及重抽分布，paired_market_activity_comparisons() 在活跃率间配对比较波动率。
    # 输入
    #     support 为一维原始收益支持点。
    #     counts 为同长度的非负频数，计算校正峰度需至少 4 笔且收益有方差。
    #     quantiles 为选尾下界的候选分位概率。
    #     minimum_count 为最少尾部展开观测数。
    #     exclude_zero_returns 默认 False，开启时只在尾部拟合中剔除原始零收益。
    # 输出 收益均值、样本标准差、超额峰度、三种事件比例和最优尾部拟合字段组成的字典。
    n = int(counts.sum())  # 用展开后的交易笔数作总体计数。
    mean = float(np.dot(support, counts) / n)  # 按频数加权求原始收益均值。
    variance = float(np.dot((support - mean) ** 2, counts) / (n - 1))  # 样本方差的分母为 N−1。
    std = np.sqrt(variance)  # 将样本方差开方得到 z 的标准化尺度。
    z = (support - mean) / std
    centered = support - mean
    m2 = float(np.dot(centered**2, counts) / n)  # 按 N 归一化求二阶中心矩。
    m4 = float(np.dot(centered**4, counts) / n)  # 按 N 归一化求四阶中心矩。
    g2 = m4 / m2**2 - 3  # 四阶矩除以二阶矩平方，再减正态基准 3。
    kurtosis = (n - 1) * ((n + 1) * g2 + 6) / ((n - 2) * (n - 3))  # 用样本量修正 Fisher 超额峰度。
    eligible = support != 0 if exclude_zero_returns else np.ones(len(support), dtype=bool)  # 只对尾部应用可选的原始零收益筛选。
    tail = weighted_pareto_fit(np.abs(z[eligible]), counts[eligible], quantiles, minimum_count)  # 用筛选后的绝对 z 和频数重选并估计 Pareto 尾。
    return {"return_mean": mean, "return_std": std, "return_excess_kurtosis": kurtosis,
            "zero_return_fraction": float(counts[support == 0].sum() / n),  # 以全部交易为分母求原始零收益比例。
            "p_abs_z_gt_5": float(counts[np.abs(z) > 5].sum() / n),  # 以全部交易为分母求绝对 z 严格超过 5 的比例。
            "negative_crash_probability": float(counts[z < -5].sum() / n),  # 以全部交易为分母求 z 严格小于 −5 的比例。
            **tail}


def selection_aware_tail_gof(
    support: np.ndarray, matrix: sparse.csr_matrix, fit: dict,
    quantiles: tuple[float, ...], minimum: int, rng: np.random.Generator,
    draws: int, selector: Callable[[np.random.Generator], np.ndarray],
    exclude_zero_returns: bool = False,
) -> float:
    """Semiparametric common-Pareto null with resampled block counts and xmin selection."""
    # 构造分块重抽的半参数 Pareto 零假设，在每轮重选 xmin 后比较 KS 距离。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main()、scripts/simulation/compare_networks.py 的 main() 和 scripts/analysis/audit_downstream_statistics.py 的 audit_one_family() 传入各自的块选择器，检验格点或网络尾部。
    # 输入
    #     support 为原始收益支持点。
    #     matrix 为块×收益支持点的频数矩阵。
    #     fit 为原样本的拟合，含 xmin、alpha_pdf 和 ks_distance。
    #     quantiles 为重选下界的候选分位概率。
    #     minimum 为尾部最少观测数。
    #     rng 为传给 selector 和合成尾部的随机数生成器。
    #     draws 为 Monte Carlo 重复数。
    #     selector 为接收 rng 并返回一维块行编号的回调，本函数每轮调用一次。
    #     exclude_zero_returns 默认 False，开启时不让原始零收益参与尾部和经验主体。
    # 输出 加一修正的 KS 拟合优度 p 值，计算为 (超越次数+1)/(draws+1)。
    original = np.asarray(matrix.sum(axis=0)).ravel()  # 沿块轴合并原样本频数，为统一标准化准备权重。
    mean = np.dot(support, original) / original.sum()  # 计算整份原样本的加权均值。
    std = np.sqrt(np.dot((support - mean) ** 2, original) / (original.sum() - 1))  # 以 N−1 为分母计算原样本的标准差。
    absolute = np.abs((support - mean) / std)  # 用固定的原样本均值和标准差得到绝对 z。
    eligible = support != 0 if exclude_zero_returns else np.ones(len(support), dtype=bool)  # 根据原始收益的精确零值确定主体和尾部资格。
    xmin, alpha = float(fit["xmin"]), float(fit["alpha_pdf"])  # 使用原拟合下界划分主体与尾部，并保留其指数作为零假设。
    exceed = 0
    for _ in range(draws):  # 每轮重新抽块、生成尾部并选择下界。
        counts = resampled_counts(matrix, selector(rng))  # 让调用方选择本轮配置块，并合计相应的重抽频数。
        body_mask = eligible & (absolute > 0) & (absolute < xmin)  # 保留低于拟合下界的正幅度作为经验主体。
        n_tail = int(counts[eligible & (absolute >= xmin)].sum())  # 从本轮所选块统计需要替换的尾部样本数。
        synthetic_tail = xmin * (1 - rng.random(n_tail)) ** (-1 / (alpha - 1))  # 以逆 CDF 从原拟合的连续 Pareto 尾部生成同量合成幅度。
        # 用经验主体和合成尾部重选阈值并拟合零假设样本。
        synthetic = weighted_pareto_fit(
            np.concatenate([absolute[body_mask], synthetic_tail]),  # 把保留的主体支持与合成尾值拼在一起。
            np.concatenate([counts[body_mask], np.ones(n_tail, dtype=np.int64)]),  # 主体沿用重抽频数，每个合成观测贡献一次计数。
            quantiles, minimum,
        )
        exceed += synthetic["ks_distance"] >= fit["ks_distance"]  # 平局也计入达到原样本 KS 的次数。
    return (exceed + 1) / (draws + 1)
