#!/usr/bin/env python3
"""Reassess market and network return statistics at their actual sampling levels."""
# 按市场构型和网络拓扑的抽样层级审计收益统计，复核已保存的经验分布。
# audit_one_family() 重算统计和区间，validate_saved_distributions() 核对分布表，paired_market_activity_comparisons() 比较共用构型的活动率，绘图函数展示区间。

# main() 读取市场收益率的影响因素实验与网络拓扑压力测试的归档，生成可并排检查的市场和网络审计结果。
from __future__ import annotations  # 延后求值类型注解。

import argparse  # 解析输出路径及两种重采样次数。
import hashlib  # 计算输入归档的 SHA-256。
import json  # 写出审计方法和来源摘要。
from pathlib import Path  # 拼接归档路径及审计输出目录。

import matplotlib  # 配置统计图的批处理后端。
matplotlib.use("Agg")  # 在无窗口环境生成 PNG。
import matplotlib.pyplot as plt  # 绘制带重采样区间的统计和尾指数图。
import numpy as np  # 构造块索引、重抽并比较收益统计。
import pandas as pd  # 读写归档、分组收益并汇总区间。

from local_market.config import MarketAnalysisConfig, NetworkAnalysisConfig  # src/local_market/config.py 的 MarketAnalysisConfig 和 NetworkAnalysisConfig 分别加载格点市场和网络实验配置。
from local_market.statistics import return_distribution_table, standardize_returns  # src/local_market/statistics.py 的 return_distribution_table 重建经验分布，standardize_returns 重算每条件 z。
from local_market.hierarchical_statistics import (  # 从 src/local_market/hierarchical_statistics.py 导入分层重采样方法。
    block_histogram, resampled_counts, select_network_configuration_blocks,  # block_histogram 建立块计数，resampled_counts 聚合频数，select_network_configuration_blocks 重抽网络与其构型。
    selection_aware_tail_gof, weighted_statistics,  # selection_aware_tail_gof 检验重选阈值后的拟合，weighted_statistics 从频数计算矩、极端概率和尾参数。
)


def audit_one_family(
    frame: pd.DataFrame, old_summary: pd.DataFrame, old_tails: pd.DataFrame,
    keys: list[str], quantiles: tuple[float, ...], minimum: int,
    configurations: int, networks: int | None, seed: int, bootstraps: int, gof_draws: int,
    exclude_zero_returns: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 功能：对指定的市场或网络数据重算点估计、构型层级区间及尾部拟合优度。
    # 引用：main() 分别为市场与网络调用本函数，将两套审计及逐轮结果存档。
    # 输入
    # frame：逐笔原始收益及条件键，市场须含 replicate 和 configuration，网络须含 network_replicate 和 step。
    # old_summary：按相同条件键保存的原收益统计。
    # old_tails：按相同条件键保存的原尾拟合与区间。
    # keys：条件分组列，顺序与分组键值一一对应。
    # quantiles：候选尾阈值的分位数。
    # minimum：候选尾阈值所需的最少尾样本。
    # configurations：每个市场重复或每个独立网络的占据构型数量。
    # networks：独立网络数量。传 None 时按市场构型抽样。
    # seed：派生各条件随机流的主种子。
    # bootstraps：用于经验区间的正整数重采样次数。
    # gof_draws：半参数零假设的模拟次数。
    # exclude_zero_returns：默认 False。为真时在尾部拟合前剔除原始零收益。
    # 输出依次为
    # audit：逐条件的原拟合、重算点估计、95% 区间、阈值选择频率与检验值。
    # draws：每个条件每轮 bootstrap 一行的统计表，可用于组间差值。
    rows, draws_rows = [], []
    # 对每个条件独立分析，编号用于派生随机流。
    for group_index, (key, group) in enumerate(frame.groupby(keys, sort=True)):
        key = key if isinstance(key, tuple) else (key,)  # 将单列键统一成元组。
        parameters = dict(zip(keys, key, strict=True))  # 为当前分组生成条件字段。
        if networks is None:
            ids = group.replicate.to_numpy(dtype=int) * configurations + group.configuration.to_numpy(dtype=int)  # 市场块 ID 由重复号和重复内构型号确定。
            count = int(group.replicate.nunique()) * configurations  # 计算预期的市场独立构型总数。
            assert np.array_equal(np.bincount(ids), np.full(count, len(group) // count))  # 校验每个市场构型的交易笔数相同。
            selector = lambda rng: rng.integers(0, count, size=count)  # 市场选择器由本函数和 selection_aware_tail_gof() 调用，用传入随机流在全部构型块上有放回抽样。
        else:
            # Each saved network step spans exactly one configuration block.
            ids = group.network_replicate.to_numpy(dtype=int) * configurations + group.step.to_numpy(dtype=int) // (len(group) // (networks * configurations))  # 根据每构型交易数把网络内交易步映射到构型号，再加网络重复偏移。
            assert len(np.unique(ids)) == networks * configurations  # 核对实际块数为网络数乘每网络构型数。
            # 核对网络构型块都含同样数量的交易。
            assert np.array_equal(np.bincount(ids),
                                  np.full(networks * configurations, len(group) // (networks * configurations)))
            selector = lambda rng: select_network_configuration_blocks(rng, networks, configurations)  # 网络选择器由本函数和 selection_aware_tail_gof() 调用，用传入随机流先抽网络再抽其内构型。
        # 在共同收益支持上建立构型计数矩阵。
        support, matrix = block_histogram(group["return"].to_numpy(dtype=float), ids)
        original_counts = np.asarray(matrix.sum(axis=0)).ravel()  # 沿构型轴累加，得到原样本频数。
        # 使用原频数重算收益矩、极端概率和尾拟合。
        point = weighted_statistics(support, original_counts, quantiles, minimum,
                                    exclude_zero_returns=exclude_zero_returns)  # 尾样本按这一数据族的零收益口径筛选。
        old_stat = old_summary
        # 逐列匹配归档的同条件收益统计。
        for column, value in parameters.items():
            old_stat = old_stat[old_stat[column] == value]
        old_stat = old_stat.iloc[0]
        old_tail = old_tails
        # 逐列匹配归档的同条件尾拟合。
        for column, value in parameters.items():
            old_tail = old_tail[old_tail[column] == value]
        old_tail = old_tail.iloc[0]
        # 对峰度、零比例及两个极端概率逐项验证。
        for column in ("return_excess_kurtosis", "zero_return_fraction", "p_abs_z_gt_5", "negative_crash_probability"):
            assert np.isclose(point[column], old_stat[column], atol=1e-10), (parameters, column)  # 核对重算值在要求的数值容差内。
        # 检查旧阈值、指数及 KS，差异写到终端。
        for column in ("xmin", "alpha_pdf", "ks_distance"):
            if not np.isclose(point[column], old_tail[column], atol=1e-10):  # 尾拟合有差异时报告首个不同字段。
                print("recomputed tail differs from saved legacy fit", parameters, column,
                      flush=True)
                break
        rng = np.random.default_rng(np.random.SeedSequence([seed, 810, group_index]))  # 为该条件的经验 bootstrap 派生随机流。
        samples = []
        for replicate in range(bootstraps):
            counts = resampled_counts(matrix, selector(rng))  # 选择整构型块并恢复抽样频数。
            # 按本轮的频数重算收益统计和尾拟合。
            result = weighted_statistics(support, counts, quantiles, minimum,
                                         exclude_zero_returns=exclude_zero_returns)
            samples.append(result)  # 留作当前条件的经验区间计算。
            draws_rows.append(parameters | {"bootstrap_replicate": replicate} | result)  # 附上条件和轮编号以保留抽样分布。
        bootstrap = pd.DataFrame(samples)  # 将逐轮结果整理成区间计算表。
        gof_rng = np.random.default_rng(np.random.SeedSequence([seed, 811, group_index]))  # 为尾部零假设模拟使用单独随机流。
        # 在共同 Pareto 尾零假设下按同一块选择器模拟，每轮重选阈值并比较 KS。
        adjusted_gof = selection_aware_tail_gof(support, matrix, point, quantiles, minimum,
                                gof_rng, gof_draws, selector, exclude_zero_returns=exclude_zero_returns)
        # 将实际样本结构、原估计和修正后的拟合优度放在同一行。
        row = parameters | {
            "observations": len(group), "independent_networks": networks if networks is not None else np.nan,
            "configuration_blocks": matrix.shape[0], "actions_per_configuration": len(group)//matrix.shape[0],  # 块数来自矩阵行数，交易数按构型块平均。
            "old_xmin": float(old_tail.xmin), "old_alpha_pdf": float(old_tail.alpha_pdf),
            "old_alpha_ci_low": float(old_tail.alpha_ci_low),
            "old_alpha_ci_high": float(old_tail.alpha_ci_high),
            "old_fixed_xmin_gof_p": float(old_tail.pareto_fixed_xmin_gof_p),
            "selection_aware_hierarchical_null_gof_p": adjusted_gof,
        }
        # 为每个重算统计保存点估计和双侧 95% 区间。
        for column in ("return_excess_kurtosis", "zero_return_fraction", "p_abs_z_gt_5",
                       "negative_crash_probability", "xmin", "xmin_quantile", "alpha_pdf", "ks_distance"):
            row[column] = point[column]
            row[column + "_ci_low"] = float(bootstrap[column].quantile(.025))  # 取经验 bootstrap 的 2.5% 分位。
            row[column + "_ci_high"] = float(bootstrap[column].quantile(.975))  # 取经验 bootstrap 的 97.5% 分位。
        # 统计重采样时选回原阈值分位数的频率。
        row["original_xmin_quantile_selection_frequency"] = float(
            np.mean(bootstrap.xmin_quantile == point["xmin_quantile"])  # 比较分位数标签，同分位数对应的数值阈值可变化。
        )
        rows.append(row)  # 保存当前条件的审计汇总行。
        # 输出当前条件的指数、区间及拟合优度。
        print("audited", parameters, "alpha", round(point["alpha_pdf"], 4),
              "CI", (round(row["alpha_pdf_ci_low"], 4), round(row["alpha_pdf_ci_high"], 4)),
              "GOF", adjusted_gof, flush=True)
    # 按逐条件审计、逐轮抽样的顺序返回两张表。
    return pd.DataFrame(rows), pd.DataFrame(draws_rows)


def plot_revised_tails(audit: pd.DataFrame, key: str, path: Path) -> None:
    # 功能：绘制市场或网络的 Pareto PDF 指数及分层 bootstrap 区间。
    # 引用：main() 分别用市场活动率和网络拓扑分组生成两张尾指数图。
    # 输入
    # audit：逐条件审计表，含据以绘线的概率比、指数及区间列。
    # key：曲线分组列。activity 用 p_ratio 作横轴，topology 用 relative_p。
    # path：PNG 输出路径。
    # 输出：返回 None，保存图后关闭画布。
    fig, ax = plt.subplots(figsize=(9, 5))  # 建立单面板尾指数图。
    for label, group in audit.groupby(key, sort=True):  # 每个活动率或网络拓扑对应一条曲线。
        x = group["p_ratio" if key == "activity" else "relative_p"].to_numpy()  # 选择实验使用的占据概率比。
        line, = ax.plot(x, group.alpha_pdf, "o-", label=f"{key}={label}")  # 画各条件的 PDF 指数点估计。
        # 遍历同一曲线的条件区间。
        for _, row in group.iterrows():
            ax.vlines(row["p_ratio" if key == "activity" else "relative_p"],  # 在对应概率比上画垂直区间。
                      row.alpha_pdf_ci_low, row.alpha_pdf_ci_high,
                      color=line.get_color(), alpha=.6)  # 区间颜色与点估计曲线一致。
    ax.set_yscale("log")  # 指数使用对数纵轴。
    ax.set_xlabel("p/pc" if key == "activity" else "p/pc effective")  # 横轴标识格点或网络的临界概率口径。
    ax.set_ylabel("Pareto PDF exponent; hierarchical 95% interval")  # 纵轴注明指数类型和区间水平。
    ax.legend(fontsize=8)  # 显示活动率或网络拓扑图例。
    ax.grid(alpha=.2)  # 加读数网格。
    fig.tight_layout()  # 调整尾指数图排版。
    fig.savefig(path, dpi=180)  # 保存尾指数图。
    plt.close(fig)  # 关闭尾指数图。


def plot_revised_moments(audit: pd.DataFrame, group_key: str, x_key: str, path: Path) -> None:
    # 功能：绘制市场或网络的超额峰度、零收益比例及两个极端事件概率。
    # 引用：main() 用本函数分别生成活动率与网络极端事件统计图。
    # 输入
    # audit：含四项统计及对应区间列的逐条件审计表。
    # group_key：曲线分组列，市场用 p_ratio，网络用 topology。
    # x_key：横轴字段，市场用 activity，网络用 relative_p。
    # path：PNG 输出路径。
    # 输出：返回 None，保存并关闭画布。
    # 定义四个绘图统计，极端事件分别用 |z|>5 和 z<−5。
    columns = (("return_excess_kurtosis", "Excess kurtosis"),
               ("zero_return_fraction", "Zero-return fraction"),
               ("p_abs_z_gt_5", "P(|z|>5)"),
               ("negative_crash_probability", "P(z<-5)"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))  # 建立四个统计面板。
    for label, group in audit.groupby(group_key, sort=True):  # 按占据概率比或网络拓扑分组绘制曲线。
        group = group.sort_values(x_key)  # 在组内按横轴坐标排序。
        # 遍历各统计量及对应子图。
        for ax, (column, title) in zip(axes.flat, columns):
            x = group[x_key].to_numpy()  # 取活动率或网络占据概率比。
            y = group[column].to_numpy()  # 取同条件的统计点估计。
            line, = ax.plot(x, y, "o-", label=f"{group_key}={label}")  # 连接点估计并添加分组标签。
            ax.vlines(x, group[column + "_ci_low"], group[column + "_ci_high"],  # 用逐条件分位数画垂直区间。
                      color=line.get_color(), alpha=.6)  # 区间颜色与点估计相同。
            ax.set_title(title)  # 标识面板对应的统计量。
            ax.grid(alpha=.2)  # 加网格便于比较。
            if x_key == "activity":
                ax.set_xscale("log")  # 活动率横轴用对数刻度。
            ax.set_xlabel("Activity a" if x_key == "activity" else "p/pc effective")  # 标明当前实验的横轴变量。
    axes[0, 0].legend(fontsize=7)  # 在第一幅子图显示分组图例。
    fig.suptitle("Original point estimates with hierarchical 95% bootstrap intervals")  # 标注点估计来源与 95% 区间。
    fig.tight_layout()  # 调整四个统计面板。
    fig.savefig(path, dpi=180)  # 保存统计图。
    plt.close(fig)  # 关闭统计图。


def validate_saved_distributions(
    returns: pd.DataFrame, saved: pd.DataFrame, keys: list[str], absolute_only: bool
) -> pd.DataFrame:
    # 功能：按条件重建市场或网络经验分布并核对归档表。
    # 引用：main() 分别核对市场 PDF 与 CCDF，以及网络 CCDF，保存最大差异。
    # 输入
    # returns：逐笔原始收益及分组列。
    # saved：已保存的分布表，须含分组列、distribution、x、value。
    # keys：市场用概率比和活动率，网络用拓扑和相对占据概率。
    # absolute_only：真时仅重建绝对收益 CCDF，假时也重建有符号 PDF。
    # 输出：逐条件的分布行数和最大绝对值差。形状、网格或数值不符时触发断言。
    rows = []
    # 对每个条件独立核对。
    for key, group in returns.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        parameters = dict(zip(keys, key, strict=True))
        z, _, _ = standardize_returns(group["return"].to_numpy())  # 按组重算标准化收益。
        rebuilt = pd.DataFrame(return_distribution_table(z, include_pdf=not absolute_only))  # 用默认全样本口径重建所需分布，CCDF 阈值取自正的 |z|。
        old = saved
        # 从归档表挑出同条件行。
        for column, value in parameters.items():
            old = old[old[column] == value]
        rebuilt = rebuilt.sort_values(["distribution", "x"]).reset_index(drop=True)  # 重建表按分布类别和横坐标排序。
        old = old.sort_values(["distribution", "x"]).reset_index(drop=True)  # 归档表用相同的排序和从零起的索引。
        # 核对分布点数。
        assert len(rebuilt) == len(old)
        # 核对逐行分布类别。
        assert rebuilt.distribution.tolist() == old.distribution.tolist()
        np.testing.assert_allclose(rebuilt.x, old.x, rtol=1e-12, atol=1e-12)  # 按严格容差验证两套横坐标一致。
        maximum = float(np.max(np.abs(rebuilt.value.to_numpy() - old.value.to_numpy())))  # 取所有分布值配对差的最大绝对值。
        assert maximum < 1e-12
        rows.append(parameters | {"distribution_rows": len(old), "maximum_absolute_value_difference": maximum})  # 保留该条件的分布点数和核对误差。
    return pd.DataFrame(rows)  # 汇总所有条件的验证结果。


def paired_market_activity_comparisons(
    frame: pd.DataFrame, config: MarketAnalysisConfig, draws: int
) -> pd.DataFrame:
    """Preserve the shared occupied configurations when comparing activities."""
    # 功能：在市场收益率的影响因素实验中，配对比较共享占据构型的相邻活动率。
    # 引用：main() 将返回表保存为市场活动率配对比较。
    # 输入
    # frame：逐笔收益表，含 p_ratio、activity、replicate、configuration 和 return。
    # config：市场配置，提供构型数、重复数、种子及尾拟合设置。
    # draws：每个占据概率比的正整数配对 bootstrap 次数。
    # 输出：逐概率比、相邻活动率和统计量的差值 95% 区间及较高活动率统计更大的频率。
    rows = []
    # 每个概率比独立重采样。
    for ratio, ratio_group in frame.groupby("p_ratio", sort=True):
        matrices = {}
        # 按活动率建立并对齐构型块。
        for activity, group in ratio_group.groupby("activity", sort=True):
            block_ids = group.replicate.to_numpy(dtype=int) * config.configurations_per_replicate + group.configuration.to_numpy(dtype=int)  # 用共同的重复号和构型号编码各活动率块 ID。
            matrices[activity] = block_histogram(group["return"].to_numpy(), block_ids)  # 为各活动率分别存储支持和块频数，按共用构型编号配对。
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 813, round(ratio * 10)]))  # 对该占据条件派生确定的随机流。
        estimates = {activity: [] for activity in matrices}
        block_count = config.replicates * config.configurations_per_replicate  # 算出各活动率共用的构型块总数。
        for _ in range(draws):
            selected = rng.integers(0, block_count, size=block_count)  # 每轮有放回抽取一套共用的构型编号。
            for activity, (support, matrix) in matrices.items():
                # 用同一套抽样构型重算各活动率的统计。
                estimates[activity].append(weighted_statistics(
                    support, resampled_counts(matrix, selected),  # 在当前活动率矩阵中汇总同一批块。
                    config.tail_xmin_quantiles, config.tail_minimum_count))  # 使用配置的候选阈值和最少尾样本数。
        activities = sorted(matrices)  # 将活动率从小到大排列。
        for left, right in zip(activities[1:], activities[:-1]):  # 组成相邻配对，left 为较高活动率。
            for statistic in ("return_excess_kurtosis", "p_abs_z_gt_5", "alpha_pdf"):
                # 逐轮作较高活动率减较低活动率的统计差。
                difference = np.array([x[statistic] - y[statistic]
                                       for x, y in zip(estimates[left], estimates[right])])
                # 按概率比、活动率配对和统计量保存差值分布摘要。
                rows.append({"p_ratio": ratio, "activity_left": left, "activity_right": right,
                             "statistic": statistic, "difference_ci_low": np.quantile(difference, .025),  # 报告差值的 2.5% 分位。
                             "difference_ci_high": np.quantile(difference, .975),  # 报告差值的 97.5% 分位。
                             "probability_left_greater": np.mean(difference > 0)})  # 报告差值严格为正的抽样比例。
    # 汇总全部相邻活动率对。
    return pd.DataFrame(rows)


def main() -> None:
    # 功能：审计市场收益率的影响因素实验与网络拓扑压力测试的统计、分布和不确定性。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。读取两类实验的配置及 results/tables 下的归档，命令行可更改输出目录与重采样次数。
    # 输出：返回 None，保存逐条件诊断、逐轮抽样、配对比较、分布验证、图像与来源 JSON。
    # 用模块说明构造审计命令行帮助。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/downstream_audit_20261114"))  # 指定审计产物根目录。
    parser.add_argument("--bootstraps", type=int, default=300)  # 默认使用 300 轮经验 bootstrap。
    parser.add_argument("--gof-draws", type=int, default=100)  # 默认使用 100 轮拟合优度零假设模拟。
    args = parser.parse_args()  # 读取实际选项。
    output = args.output_root
    (output / "tables").mkdir(parents=True, exist_ok=True)  # 建立审计表目录。
    (output / "figures").mkdir(parents=True, exist_ok=True)  # 建立审计图目录。
    base = Path("results/tables")
    market_config = MarketAnalysisConfig.from_yaml("configs/market_analysis.yaml")  # 读取市场系综抽样与尾拟合参数。
    # 读取市场分组键、构型标识和原始收益。
    market = pd.read_csv(base / "market_returns.csv.gz",
                         usecols=["p_ratio", "activity", "replicate", "configuration", "return"])
    market_summary = pd.read_csv(base / "market_summary.csv")  # 读取市场收益统计基线。
    market_tails = pd.read_csv(base / "market_tail_fits.csv")  # 读取市场原尾拟合和区间。
    # 对市场条件重算分层统计。
    market_audit, market_draws = audit_one_family(
        market, market_summary, market_tails, ["p_ratio", "activity"],
        market_config.tail_xmin_quantiles, market_config.tail_minimum_count,
        market_config.configurations_per_replicate, None,  # 市场分析在全部独立构型块上抽样。
        market_config.seed, args.bootstraps, args.gof_draws, exclude_zero_returns=True)  # 市场尾样本剔除原始零收益。
    market_audit.to_csv(output / "tables/market_revised_statistics.csv", index=False)  # 保存市场审计汇总。
    market_draws.to_csv(output / "tables/market_bootstrap_draws.csv.gz", index=False, compression="gzip")  # 压缩保存市场逐轮结果。
    # 用共用构型配对计算相邻活动率的差，并保存比较表。
    paired_market_activity_comparisons(market, market_config, args.bootstraps).to_csv(
        output / "tables/market_paired_activity_comparisons.csv", index=False)
    # 重建市场全样本 PDF 和 CCDF 并保存核对结果。
    validate_saved_distributions(
        market, pd.read_csv(base / "market_distributions.csv"),  # 读取市场归档的经验分布。
        ["p_ratio", "activity"], False,
    ).to_csv(output / "tables/market_distribution_validation.csv", index=False)
    # 绘制市场尾指数和各条件的重抽区间。
    plot_revised_tails(market_audit, "activity", output / "figures/market_tail_exponents_revised.png")
    # 绘制市场收益统计随活动率变化的区间图。
    plot_revised_moments(market_audit, "p_ratio", "activity",
                         output / "figures/market_activity_statistics_revised.png")

    # 读取网络系综的抽样和尾拟合参数。
    network_config = NetworkAnalysisConfig.from_yaml("configs/network_analysis.yaml")
    # 读取网络拓扑、占据条件、网络编号、交易步和原始收益。
    network = pd.read_csv(base / "network_market_returns.csv.gz",
                          usecols=["topology", "relative_p", "network_replicate", "step", "return"])
    network_summary = pd.read_csv(base / "network_market_summary.csv")  # 读取网络收益统计基线。
    network_tails = pd.read_csv(base / "network_tail_fits.csv")  # 读取网络原尾拟合和区间。
    # 对网络条件按网络及构型两级抽样审计。
    network_audit, network_draws = audit_one_family(
        network, network_summary, network_tails, ["topology", "relative_p"],
        network_config.tail_xmin_quantiles, network_config.tail_minimum_count,
        network_config.market_configurations_per_network, network_config.network_replicates,  # 传入网络内构型数和独立网络数，启用两级抽样。
        network_config.seed, args.bootstraps, args.gof_draws)  # 沿用默认的网络尾部零收益设置。
    network_audit.to_csv(output / "tables/network_revised_statistics.csv", index=False)  # 保存网络审计汇总。
    network_draws.to_csv(output / "tables/network_bootstrap_draws.csv.gz", index=False, compression="gzip")  # 压缩保存网络逐轮结果。
    # 重建网络绝对收益 CCDF 并保存核对结果。
    validate_saved_distributions(
        network, pd.read_csv(base / "network_market_distributions.csv"),  # 读取网络归档 CCDF。
        ["topology", "relative_p"], True,
    ).to_csv(output / "tables/network_distribution_validation.csv", index=False)
    # 绘制各网络拓扑的尾指数和区间。
    plot_revised_tails(network_audit, "topology", output / "figures/network_tail_exponents_revised.png")
    # 绘制各网络拓扑的极端事件等统计。
    plot_revised_moments(network_audit, "topology", "relative_p",
                         output / "figures/network_extreme_events_revised.png")
    comparisons = []
    # 在相同相对占据概率下比较拓扑。
    for ratio in network_config.relative_p_values:
        selected = network_draws[np.isclose(network_draws.relative_p, ratio)]  # 只选当前相对占据概率的重抽结果。
        # 遍历报告指定的网络拓扑对。
        for left, right in (("ba", "ws"), ("ba", "er"), ("ba", "lattice"),
                            ("ws", "er"), ("er", "lattice")):
            left_draws = selected[selected.topology == left].sort_values("bootstrap_replicate")  # 提取左侧拓扑的重复统计并按轮编号排序。
            right_draws = selected[selected.topology == right].sort_values("bootstrap_replicate")  # 对右侧拓扑应用相同的轮编号排序。
            for column in ("alpha_pdf", "p_abs_z_gt_5", "return_excess_kurtosis"):
                difference = left_draws[column].to_numpy() - right_draws[column].to_numpy()  # 计算两个拓扑独立重抽结果的左减右差值。
                # 记录差值 95% 区间及左侧统计量更大的频率。
                comparisons.append({"relative_p": ratio, "left": left, "right": right,
                                    "statistic": column, "difference_ci_low": np.quantile(difference, .025),
                                    "difference_ci_high": np.quantile(difference, .975),
                                    "probability_left_greater": np.mean(difference > 0)})
    # 写出网络拓扑逐对比较表。
    pd.DataFrame(comparisons).to_csv(output / "tables/network_pairwise_comparisons.csv", index=False)
    # 确定需保存哈希的归档输入文件。
    sources = [base / name for name in (
        "market_returns.csv.gz", "market_summary.csv", "market_tail_fits.csv",
        "market_distributions.csv", "network_market_returns.csv.gz",
        "network_market_summary.csv", "network_tail_fits.csv",
        "network_market_distributions.csv")]
    # 记录样本层级、重抽设置、标准化和检验口径。
    metadata = {
        "bootstrap_draws": args.bootstraps,
        "gof_null_draws": args.gof_draws,
        "market_seed": market_config.seed,
        "network_seed": network_config.seed,
        "market_resampling": "400 occupied configurations, whole 500-action blocks; paired across activities for differences",
        "network_resampling": "4 networks, then 60 occupied configurations within each selected network, whole 400-action blocks",
        "standardization": "recomputed on every bootstrap ensemble; tail point estimates follow the selected raw-zero rule",
        "market_tail_zero_rule": "exclude raw return==0 before |z| tail selection; other moments use all returns",
        "xmin": "minimum KS among configured positive-|z| quantiles, reselected in every bootstrap draw",
        "gof": "semiparametric common-Pareto tail null preserving block body and tail counts; xmin reselected",
        "source_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},  # 对各归档的原始字节计算 SHA-256。
    }
    # 保存审计方法和输入哈希。
    (output / "tables/audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    # 启动市场与网络统计审计。
    main()
