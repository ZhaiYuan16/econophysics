"""Tail estimators and distribution comparisons for absolute returns."""
# 估计绝对标准化收益的连续尾部。
# pareto_mle() 使用给定下界，select_pareto_tail() 在候选下界间按 KS 距离选择。
# pareto_fixed_xmin_gof_pvalue() 在固定下界下检验拟合优度，compare_tail_distributions() 比较幂律、指数和截断幂律。
#
# 市场收益率的影响因素实验和网络拓扑压力测试把收益样本交给本模块得到尾部参数、敏感性和模型比较表。

from __future__ import annotations  # 延后求值类型注解。

import warnings  # 标准库警告模块，将第三方拟合的警告过滤限制在比较函数内。
from dataclasses import dataclass  # 标准库数据类装饰器，为拟合结果容器生成构造方法等。

import numpy as np  # NumPy 用于样本筛选、分位数、对数和 Monte Carlo 抽样。
import powerlaw  # powerlaw 库负责连续尾部的备选模型和似然比比较。


@dataclass(frozen=True)  # 冻结拟合结果字段的绑定。
class TailFit:
    xmin: float  # 拟合的绝对标准化收益下界。
    alpha_pdf: float  # 连续 Pareto 的概率密度指数 α。
    ks_distance: float  # 模型 CDF 与经验 CDF 之间的 KS 距离。
    tail_count: int  # 达到下界的尾部观测数。
    tail_fraction: float  # 尾部观测数除以筛选后的有限正样本数。


def positive_finite(values: np.ndarray) -> np.ndarray:
    # 把收益幅度转为浮点样本并筛出有限正值。
    # 引用
    #     pareto_mle() 清洗固定下界的输入，select_pareto_tail() 清洗分位数样本，compare_tail_distributions() 清洗第三方拟合的输入。
    # 输入
    #     values 为可转换为浮点数组的观测。
    # 输出 一维有限正值数组，保持原元素遍历顺序，可为空。
    values = np.asarray(values, dtype=float)  # 确保后续有限性检测和估计使用浮点。
    return values[np.isfinite(values) & (values > 0)]  # 同时筛掉无穷、NaN、负值和零值。


def pareto_mle(values: np.ndarray, xmin: float) -> tuple[float, float, int]:
    """Continuous Pareto PDF exponent and KS distance above fixed xmin."""
    # 计算固定下界的连续 Pareto PDF 指数和 KS 距离。
    # 引用
    #     select_pareto_tail() 评估阈值候选，pareto_fixed_xmin_gof_pvalue() 在每个合成样本上重拟合。
    #     scripts/analysis/analyze_market_returns.py 的 main() 与 scripts/simulation/compare_networks.py 的 main() 在样本长度敏感性检查中使用固定阈值。
    # 输入
    #     values 为待筛选的幅度样本。
    #     xmin 为正尾下界，含端点。
    # 输出依次为
    #     PDF 指数 α。
    #     经验与模型 CDF 的最大绝对差。
    #     达到下界的样本数。尾样本少于两个时前两项为 NaN，对数比总和非正时为无穷。
    positive = positive_finite(values)  # 剔除非有限值和非正值。
    tail = np.sort(positive[positive >= xmin])  # 只取含边界的尾样本并升序排列。
    count = int(tail.size)  # 用尾样本数作为经验 CDF 和估计式的分母。
    if count < 2:  # 不足两项时无法估计。
        return float("nan"), float("nan"), count
    log_sum = float(np.log(tail / xmin).sum())  # 累计样本对 xmin 的对数比，供极大似然指数使用。
    if log_sum <= 0:  # 下界以上没有对数跨度时有限指数无法估计。
        return float("inf"), float("inf"), count
    alpha = 1.0 + count / log_sum
    empirical_cdf = np.arange(1, count + 1, dtype=float) / count  # 用从 1 开始的排序秩除以尾样本数得到经验 CDF。
    model_cdf = 1.0 - (tail / xmin) ** (1.0 - alpha)
    ks = float(np.max(np.abs(empirical_cdf - model_cdf)))  # 取在样本点上评估的最大累计概率差。
    return float(alpha), ks, count  # 返回匹配该下界的指数、最大累计差和观测数。


def select_pareto_tail(
    values: np.ndarray,
    quantiles: tuple[float, ...],
    minimum_count: int,
) -> tuple[TailFit, list[dict[str, float | int]]]:
    """Choose xmin from prespecified quantiles by minimum KS distance."""
    # 从指定样本分位数中选择连续 Pareto 尾部下界。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 做主拟合和长度子样本的重选。
    #     scripts/simulation/compare_networks.py 的 main() 为各网络场景选择下界。
    #     scripts/analysis/audit_market_tails_visual.py 的 main() 按修正口径重新选下界。
    # 输入
    #     values 为待拟合的幅度样本。
    #     quantiles 为 [0, 1] 内的候选分位概率。
    #     minimum_count 为可纳入候选的最少尾部观测数。
    # 输出依次为
    #     最小 KS 距离候选的 TailFit。
    #     保留的候选字典列表。样本不足或没有合格候选时抛出 ValueError。
    positive = positive_finite(values)  # 取分位数前仅保留有限的正幅度。
    if positive.size < minimum_count:  # 正幅度总数不足时没有符合要求的阈值。
        raise ValueError("not enough positive observations for tail fitting")
    rows: list[dict[str, float | int]] = []
    seen: set[float] = set()  # 记录已处理过的尾下界，避免重复拟合同一批样本。
    for quantile in quantiles:
        xmin = float(np.quantile(positive, quantile))  # 把分位概率换算成幅度下界。
        if xmin <= 0 or xmin in seen:
            continue
        seen.add(xmin)  # 标记已评估阈值，避免重复分位数浪费拟合。
        alpha, ks, count = pareto_mle(positive, xmin)  # 在当前下界计算指数、KS 距离和尾样本数。
        if count < minimum_count or not np.isfinite(alpha):  # 要求尾观测数足够且指数为有限值。
            continue
        rows.append(  # 保存满足样本数和有限指数要求的候选。
            {
                "xmin_quantile": quantile,
                "xmin": xmin,
                "alpha_pdf": alpha,
                "ks_distance": ks,
                "tail_count": count,
                "tail_fraction": count / positive.size,  # 用全部有限正幅度为分母计算尾样本占比。
            }
        )
    if not rows:  # 筛选后至少需要一个有效候选。
        raise ValueError("no xmin candidate retained enough tail observations")
    best = min(rows, key=lambda row: float(row["ks_distance"]))  # 从有效候选中选出经验与模型 CDF 最接近的一个。
    return (
        # 把获选候选转换为具名拟合结果。
        TailFit(
            xmin=float(best["xmin"]),
            alpha_pdf=float(best["alpha_pdf"]),
            ks_distance=float(best["ks_distance"]),
            tail_count=int(best["tail_count"]),
            tail_fraction=float(best["tail_fraction"]),
        ),
        rows,
    )


def pareto_fixed_xmin_gof_pvalue(
    fit: TailFit,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    """Conditional parametric-bootstrap KS p-value at a fixed xmin."""
    # 在固定 xmin 下对连续 Pareto 尾部做条件参数 bootstrap KS 检验。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 与 scripts/simulation/compare_networks.py 的 main() 输出格点和网络尾部的固定阈值检验结果。
    # 输入
    #     fit 为原样本 TailFit，须有正下界、指数 α > 1 和可用的尾样本数。
    #     replicates 为合成样本数。
    #     rng 为合成 Pareto 样本使用的 NumPy 随机数生成器。
    # 输出 经加一修正的 Monte Carlo p 值。
    synthetic_distances = np.empty(replicates, dtype=float)  # 为每轮合成样本的 KS 距离预分配空间。
    for replicate in range(replicates):  # 独立抽取每次参数 bootstrap 的尾部样本。
        uniform = rng.random(fit.tail_count)  # 按原尾样本数抽取独立均匀随机数。
        synthetic = fit.xmin * (1.0 - uniform) ** (-1.0 / (fit.alpha_pdf - 1.0))  # 由 Pareto 的逆 CDF 生成合成尾部样本。
        _, distance, _ = pareto_mle(synthetic, fit.xmin)  # 保持 xmin 固定并重新拟合指数，再取得 KS。
        synthetic_distances[replicate] = distance
    exceedances = int(np.count_nonzero(synthetic_distances >= fit.ks_distance))  # 统计合成 KS 大于或等于观测 KS 的次数。
    # 分子和分母各加 1，避免有限次模拟中恰无超越就报告 p=0。
    return float((exceedances + 1) / (replicates + 1))


def compare_tail_distributions(
    values: np.ndarray, xmin: float
) -> dict[str, float]:
    """Compare a continuous power law with exponential and truncated alternatives."""
    # 在共同固定下界比较连续幂律、指数和截断幂律。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 和 scripts/simulation/compare_networks.py 的 main() 将比较结果与格点和网络的主尾部拟合一起保存。
    # 输入
    #     values 为供第三方库拟合的幅度样本，先剔除非有限值和非正值。
    #     xmin 为共同的正尾下界。
    # 输出 幂律 PDF 指数，以及对指数和截断幂律的两组似然比和对应 p 值。
    positive = positive_finite(values)  # 按连续正尾模型的口径清洗输入。
    with warnings.catch_warnings():  # 把警告设置的作用域限制在第三方拟合过程。
        warnings.simplefilter("ignore")  # 抑制当前拟合与比较产生的警告。
        # 在给定下界之上拟合连续分布。
        fit = powerlaw.Fit(
            positive,
            xmin=xmin,
            discrete=False,
            verbose=False,
            parameter_ranges={"alpha": (1.000001, None)},  # 约束幂律指数下限高于 1，确保连续幂律可归一。
        )
        # 用归一化似然比比较纯幂律与指数尾。
        ratio_exponential, p_exponential = fit.distribution_compare(
            "power_law", "exponential", normalized_ratio=True
        )
        # 用库的比较方法计算纯幂律与截断幂律的似然比及 p 值。
        ratio_truncated, p_truncated = fit.distribution_compare(
            "power_law", "truncated_power_law"
        )
    return {
        "powerlaw_package_alpha_pdf": float(fit.power_law.alpha),  # 单独保存第三方库估计的纯幂律指数。
        "loglikelihood_ratio_power_vs_exponential": float(ratio_exponential),  # 保存纯幂律对指数模型的归一化似然比。
        "p_power_vs_exponential": float(p_exponential),  # 保存纯幂律对指数模型比较的 p 值。
        "loglikelihood_ratio_power_vs_truncated": float(ratio_truncated),  # 保存纯幂律对截断幂律的似然比。
        "p_power_vs_truncated": float(p_truncated),  # 保存嵌套的截断模型比较所给的 p 值。
    }
