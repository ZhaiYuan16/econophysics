#!/usr/bin/env python3
"""Audit saved market ensembles and generate separate CB trajectory artifacts."""
# 审计市场收益率的影响因素实验中归档系综的构型分块，并生成独立的固定构型对照轨迹。
# acf() 和 ljung_box() 检查收益序列的时间相关，main() 重建构型、核对条件方差并输出对照诊断。

# 本脚本使用 analyze_market_returns.py 生成的市场归档。finalize_market_statistics.py 读取本脚本保存的固定构型轨迹。
from __future__ import annotations  # 延后求值类型注解。

import argparse  # 解析归档路径、配置路径和审计目录。
import json  # 保存理论口径及诊断摘要。
from pathlib import Path  # 表示归档文件和图表路径。

import matplotlib  # 配置用于批量保存审计图的绘图后端。
matplotlib.use("Agg")  # 在无窗口环境里生成图像。
import matplotlib.pyplot as plt  # 绘制条件方差、顺序对照及自相关。
import numpy as np  # 重建随机流、抽样并计算收益统计。
import pandas as pd  # 读取归档并对交易、构型结果制表。
from scipy.signal import fftconvolve  # 用 FFT 卷积计算多个滞后的未归一化相关。
from scipy.stats import chi2, pearsonr  # chi2 给出 Ljung–Box 的右尾 p 值，pearsonr 计算活动率间的配对相关。

from local_market.cb_model import simulate_quenched  # src/local_market/cb_model.py 的 simulate_quenched 在单个构型上生成独立行动轨迹。
from local_market.config import MarketAnalysisConfig  # src/local_market/config.py 的 MarketAnalysisConfig 读取归档系综和模型参数。
from local_market.experiments import run_quenched_market_trajectories  # src/local_market/experiments.py 的 run_quenched_market_trajectories 生成共同构型上的对照轨迹。
from local_market.networks import sample_lattice_occupancy  # src/local_market/networks.py 的 sample_lattice_occupancy 重建占据状态。
from local_market.percolation import identify_lattice_clusters  # src/local_market/percolation.py 的 identify_lattice_clusters 识别格点团簇。
from local_market.plotting import plot_market_ensemble_samples, plot_market_time_series  # src/local_market/plotting.py 的 plot_market_ensemble_samples 展示系综样本，plot_market_time_series 绘制固定构型时序。


def acf(values: np.ndarray, max_lag: int = 50) -> np.ndarray:
    # 功能：为市场收益率的影响因素实验计算正滞后的样本自相关。
    # 引用：main() 对归档和对照序列调用本函数，再用系数绘图和执行 Ljung–Box 检验。
    # 输入
    # values：非空的一维数值时间序列，通常要求非恒定。
    # max_lag：最大正滞后，默认 50。
    # 输出：按滞后 1 到 min(max_lag, n−1) 排列的数组。恒定序列的零滞后分母为零。
    x = np.asarray(values, dtype=float)  # 将输入转成浮点数组。
    x = x - x.mean()  # 减去整条序列的均值。
    # 以序列与其反转的完整卷积求各滞后的离差乘积和，从零滞后开始切片。
    covariance = fftconvolve(x, x[::-1], mode="full")[len(x)-1:len(x)+max_lag]
    return covariance[1:] / covariance[0]  # 用零滞后的平方和作共同分母，只返回正滞后。


def ljung_box(coefficients: np.ndarray, n: int) -> float:
    # 功能：为市场收益率的影响因素实验计算所有给定滞后自相关的 Ljung–Box 联合检验 p 值。
    # 引用：main() 对原始、打乱、固定和独立种子对照序列使用同一检验。
    # 输入
    # coefficients：一维自相关数组，首项对应滞后 1，长度 m 应满足 0 < m < n。
    # n：用于计算这些自相关的原始序列长度。
    # 输出：按自由度 m 的卡方分布计算的右尾 p 值。
    max_lag = len(coefficients)  # 由系数个数确定最大滞后和检验自由度。
    statistic = n * (n + 2) * np.sum(coefficients**2 / (n - np.arange(1, max_lag + 1)))  # 每个平方系数除以该滞后的重叠样本数，再乘 n(n+2)。
    return float(chi2.sf(statistic, max_lag))  # 把检验统计量换成右尾概率。


def main() -> None:
    # 功能：复核市场收益率的影响因素实验的构型条件方差、系综顺序及固定构型的时间相关。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。通过命令行指定市场归档、模拟配置和审计目录。
    # 输出：返回 None，保存构型方差、对照轨迹、自相关、种子重复和 bootstrap 图表及元数据。
    # 建立审计命令行解析器。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/tables/market_returns.csv.gz"))  # 指定逐笔市场收益率文件。
    parser.add_argument("--config", type=Path, default=Path("configs/market_analysis.yaml"))  # 指定用于重建同一随机流的市场配置。
    parser.add_argument("--output-root", type=Path, default=Path("results/market_time_audit_20261113"))  # 指定审计输出根目录。
    args = parser.parse_args()  # 解析输入和输出路径。
    config = MarketAnalysisConfig.from_yaml(args.config)  # 读取构型数量、交易次数和种子。
    root = args.output_root
    (root / "figures").mkdir(parents=True, exist_ok=True)  # 建立图形目录。
    (root / "tables").mkdir(parents=True, exist_ok=True)  # 建立表格目录。
    columns = ["p_ratio", "activity", "replicate", "step", "configuration", "return", "z"]
    frame = pd.read_csv(args.source, usecols=columns)  # 仅读取重建顺序和计算收益统计所需的列。
    critical = frame[np.isclose(frame.p_ratio, 1.0)].copy()  # 选取理论临界占据条件。
    plot_market_ensemble_samples(critical, root / "figures/market_ensemble_samples.png")  # 绘制临界条件下顺序拼接的系综样本。

    # Recreate precisely the stored occupancy streams; compare saved structure IDs below.
    structures = pd.read_csv(args.source.parent / "market_structures.csv")  # 读取归档的构型统计以验证重建。
    rows = []
    selected = critical[critical.replicate == 0]  # 用首个重复的交易序列作顺序对照。
    for replicate in range(config.replicates):
        occupancy_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 10, 1, replicate]))  # 使用市场系综在临界占据条件下的种子键。
        for configuration in range(config.configurations_per_replicate):
            occupied = sample_lattice_occupancy(config.length, config.p_c, occupancy_rng)  # 在 p_c 下抽样当前方格占据。
            clusters = identify_lattice_clusters(occupied, config.boundary)  # 用归档配置中的边界条件识别团簇。
            # 按概率比、重复号和构型号匹配归档记录。
            old = structures[(np.isclose(structures.p_ratio, 1.0)) &
                             (structures.replicate == replicate) &
                             (structures.configuration == configuration)].iloc[0]
            assert clusters.cluster_count == old.cluster_count  # 核对重建团簇数。
            assert clusters.largest_size == old.largest_size  # 核对重建最大团簇规模。
            sum_squares = int(np.dot(clusters.sizes, clusters.sizes))  # 用团簇规模向量内积求 ∑s²。
            for activity in config.activities:
                # 从临界交易池筛出同一重复、构型与活动率。
                group = critical[(critical.replicate == replicate) &
                                 (np.isclose(critical.activity, activity)) &
                                 (critical.configuration == configuration)]
                # 核对该构型的逐笔抽样数。
                assert len(group) == config.trades_per_configuration
                # 计算该重复在全条件拼接序列中的起始偏移。
                offset = replicate * config.configurations_per_replicate * config.trades_per_configuration
                # 保存此配置在拼接序列和重复内部的首末索引。
                rows.append({"replicate": replicate, "configuration": configuration,
                             "activity": activity, "start_index": offset + int(group.step.min()),  # 全条件索引加上在重复中的偏移。
                             "end_index_inclusive": offset + int(group.step.max()),
                             "stored_step_start": int(group.step.min()),
                             "stored_step_end_inclusive": int(group.step.max()),
                             "cluster_count": clusters.cluster_count,
                             "largest_size": clusters.largest_size,
                             "sum_s_squared": sum_squares,
                             "theoretical_variance": activity * sum_squares / config.market_depth**2,  # 理论方差由独立行动的方差 a 加权每个团簇的 s²，再除以市场深度平方。
                             "empirical_variance": float(group["return"].var(ddof=1))})  # 经验条件方差用同一构型内的收益，分母为交易笔数减一。
    variances = pd.DataFrame(rows)  # 将构型及活动率的记录组装为表。
    variances.to_csv(root / "tables/configuration_variances.csv", index=False)  # 保存逐构型条件方差和交易块边界。
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)  # 为四个活动率创建共享构型轴的面板。
    for ax, activity in zip(axes.flat, config.activities):
        g = variances[(variances.replicate == 0) & np.isclose(variances.activity, activity)]  # 仅展示重复 0 的条件方差。
        ax.plot(g.configuration, g.theoretical_variance, label="theory", linewidth=1)  # 画同一批构型的理论方差。
        ax.plot(g.configuration, g.empirical_variance, label="500 draws", linewidth=.8, alpha=.75)  # 叠加构型内交易的经验方差。
        ax.set_title(f"a={activity:g}")  # 标识面板活动率。
        ax.set_ylabel("Conditional return variance")  # 标明纵轴为条件收益率方差。
        ax.grid(alpha=.2)  # 给各面板添加读数网格。
    axes[0, 0].legend()  # 在首幅子图标识理论和经验曲线。
    # 同一底行的两个横轴均为构型编号。
    axes[-1, 0].set_xlabel("Configuration ID")
    axes[-1, 1].set_xlabel("Configuration ID")
    fig.tight_layout()  # 调整方差图子图间距。
    fig.savefig(root / "figures/configuration_variances.png", dpi=180)  # 将条件方差图写出。
    plt.close(fig)  # 释放方差图。

    # 生成共用一个固定临界构型的各活动率对照轨迹。
    fixed, trajectories = run_quenched_market_trajectories(config)
    tests = []
    acf_rows = []
    # 用活动率及其顺序编号选择轨迹和随机流。
    for index, activity in enumerate(config.activities):
        trajectory = trajectories[activity]
        # 保存从第 1 步开始的收益与去掉 t=0 初值的对数价格。
        pd.DataFrame({"time": np.arange(1, 3001), "return": trajectory.returns,
                      "log_price": trajectory.log_prices[1:], "activity": activity}).to_csv(
                          root / f"tables/quenched_a_{activity:g}.csv", index=False)
        ordered = selected[np.isclose(selected.activity, activity)].sort_values("step")  # 按归档交易步恢复首个重复的拼接次序。
        original = ordered["return"].to_numpy()  # 取有符号的原始收益。
        shuffled = np.random.default_rng(np.random.SeedSequence([config.seed, 60, index])).permutation(original)  # 打乱单笔交易，破坏原有相邻关系。
        # The block plot uses shuffled configuration order while retaining within-block draws.
        blocks = original.reshape(config.configurations_per_replicate, config.trades_per_configuration)  # 每行恢复一个构型内的完整交易块。
        reordered = blocks[np.random.default_rng(np.random.SeedSequence([config.seed, 61, index])).permutation(len(blocks))].ravel()  # 只打乱构型块的顺序，展平时保留块内顺序。
        # 对照原序列、逐笔打乱和单构型轨迹的原始或绝对收益。
        for label, data in (("ordered_return", original), ("ordered_absolute", np.abs(original)),
                            ("shuffled_absolute", np.abs(shuffled)),
                            ("fixed_return", trajectory.returns),
                            ("fixed_absolute", np.abs(trajectory.returns))):
            coefficients = acf(data)  # 计算各序列的 1 到 50 阶自相关。
            pvalue = ljung_box(coefficients, len(data))  # 用已有系数求联合检验 p 值。
            # 保存联合检验、点态参考带与最大绝对自相关。
            tests.append({"activity": activity, "series": label, "n": len(data),
                          "ljung_box_lag50_p": pvalue, "ci95_halfwidth": 1.96 / np.sqrt(len(data)),
                          "max_abs_acf_lag1_50": float(np.max(np.abs(coefficients)))})
            # 将每个正滞后的系数和对应序列长度决定的参考界逐行保存。
            acf_rows.extend({"activity": activity, "series": label, "lag": lag,
                             "acf": float(value), "ci95_low": -1.96 / np.sqrt(len(data)),
                             "ci95_high": 1.96 / np.sqrt(len(data))}
                            for lag, value in enumerate(coefficients, 1))
        if activity in (0.1, 0.5):
            fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)  # 为选中活动率建立三个顺序对照面板。
            # 用原序列与乱序块序列的前 3000 笔对照独立固定构型轨迹。
            for ax, data, title in zip(axes, (original[:3000], reordered[:3000], trajectory.returns),
                                       ("Original configuration order", "Shuffled configuration order", "One fixed configuration, independent actions")):
                ax.plot(np.arange(len(data)), data, linewidth=.55)  # 依各序列的索引绘制原始收益。
                ax.set_title(title)  # 按面板标识顺序处理方式。
                ax.set_ylabel("Return")  # 三个面板的纵轴均为原始收益率。
            axes[-1].set_xlabel("Sample index (top two); simulation time (bottom)")  # 区分上方样本索引和底部模拟时间。
            fig.tight_layout()  # 调整顺序对照图间距。
            fig.savefig(root / f"figures/order_comparison_a_{activity:g}.png", dpi=180)  # 按活动率保存顺序对照图。
            plt.close(fig)  # 关闭顺序对照图。
    # 统一绘制各活动率的固定构型轨迹。
    plot_market_time_series(trajectories, root / "figures/market_time_series.png")
    pd.DataFrame(acf_rows).to_csv(root / "tables/acf_lags_1_50.csv", index=False)  # 写出逐滞后自相关。
    pd.DataFrame(tests).to_csv(root / "tables/acf_tests.csv", index=False)  # 写出逐序列检验及摘要。
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)  # 为活动率分配自相关面板。
    acf_table = pd.DataFrame(acf_rows)  # 用表格筛选要叠加的曲线。
    for ax, activity in zip(axes.flat, config.activities):
        for label in ("ordered_absolute", "shuffled_absolute", "fixed_absolute", "ordered_return"):
            g = acf_table[(np.isclose(acf_table.activity, activity)) & (acf_table.series == label)]  # 取当前活动率和序列标签的全部滞后记录。
            ax.plot(g.lag, g.acf, label=label, linewidth=1)  # 画正滞后与自相关系数。
        # 画以 n=50,000 为参照的点态参考带。
        ax.axhspan(-1.96 / np.sqrt(50000), 1.96 / np.sqrt(50000), color="gray", alpha=.15,
                   label="95% pointwise CI (n=50,000)")
        ax.set_title(f"a={activity:g}")  # 标识活动率。
        ax.grid(alpha=.2)  # 添加自相关读数网格。
    axes[0, 0].legend(fontsize=7)  # 在首图解释曲线及参考带。
    # 底行两个面板都使用滞后阶数作横轴。
    axes[-1, 0].set_xlabel("Lag")
    axes[-1, 1].set_xlabel("Lag")
    fig.tight_layout()  # 调整诊断图排版。
    fig.savefig(root / "figures/acf_diagnostics.png", dpi=180)  # 保存自相关诊断图。
    plt.close(fig)  # 关闭自相关诊断图。

    # Independent seed repeats test the conditional model rather than one favorable path.
    repeat_rows = []
    for seed in (701, 702, 703, 704, 705):
        rng = np.random.default_rng(seed)  # 每个重复种子建立独立随机流。
        config_seed = identify_lattice_clusters(sample_lattice_occupancy(config.length, config.p_c, rng), config.boundary)  # 抽取该种子的临界构型，并用配置边界识别团簇。
        for activity in (.01, .1, .5):
            sim = simulate_quenched(config_seed, 10000, activity, config.market_depth, rng)  # 每个活动率在同一构型上抽 10000 步的交易行动。
            for label, values in (("return", sim.returns), ("absolute", np.abs(sim.returns))):  # 分别检查有符号和绝对收益。
                coefficients = acf(values)  # 计算独立种子轨迹的自相关。
                repeat_rows.append({"seed": seed, "activity": activity, "series": label,
                                    "lag50_ljung_box_p": ljung_box(coefficients, len(values)),  # 基于已算系数和样本长度执行联合检验。
                                    "lag1_acf": coefficients[0]})
    # 保存独立种子的联合检验和一阶系数。
    pd.DataFrame(repeat_rows).to_csv(root / "tables/independent_seed_repeats.csv", index=False)

    # Cluster bootstrap keeps all 500 observations of each configuration together.
    bootstrap_rows = []
    tail_table = pd.read_csv(args.source.parent / "market_tail_fits.csv")  # 读取主分析已经选择的尾部阈值。
    for (ratio, activity), group in frame.groupby(["p_ratio", "activity"]):  # 各占据条件和活动率独立进行构型 bootstrap。
        blocks = group["return"].to_numpy().reshape(-1, config.trades_per_configuration)  # 按归档顺序将原始收益还原为每行一个构型块。
        zblocks = group["z"].to_numpy().reshape(blocks.shape)  # 按相同布局组织标准化收益。
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 70, round(ratio*10), round(activity*1000)]))  # 用实验种子、占据条件和活动率派生随机流。
        indices = rng.integers(0, len(blocks), size=(300, len(blocks)))  # 每轮有放回抽取与原来等量的构型，共 300 轮。
        means = blocks.mean(axis=1)  # 预计算各构型的原始收益均值。
        second = (blocks**2).mean(axis=1)  # 预计算各构型的原始收益二阶原点矩。
        variance_boot = second[indices].mean(axis=1) - means[indices].mean(axis=1)**2  # 由抽样块的二阶原点矩减去均值平方计算总体口径方差。
        extreme_boot = (np.abs(zblocks) > 5).mean(axis=1)[indices].mean(axis=1)  # 以抽样块所有交易为分母估计 P(|z|>5)，沿用归档 z。
        # 查找同一占据条件和活动率的主拟合行。
        tail_row = tail_table[(np.isclose(tail_table.p_ratio, ratio)) &
                              (np.isclose(tail_table.activity, activity))].iloc[0]
        xmin = float(tail_row.xmin)  # 将该条件的 xmin 固定用于全部重复。
        tail_mask = np.abs(zblocks) >= xmin  # 用绝对标准化收益筛选阈值以上样本。
        tail_counts = tail_mask.sum(axis=1)  # 按块计数尾部交易。
        tail_logs = np.where(tail_mask, np.log(np.maximum(np.abs(zblocks), xmin) / xmin), 0).sum(axis=1)  # 按块累加尾部 log(|z|/xmin)，其余位置贡献零。
        alpha_boot = 1 + tail_counts[indices].sum(axis=1) / tail_logs[indices].sum(axis=1)  # 用抽中块合计的尾样本数和对数和估计连续 Pareto PDF 指数。
        # 保存三种统计的双侧 95% 百分位区间及原拟合指数。
        bootstrap_rows.append({"p_ratio": ratio, "activity": activity,
                               "return_variance_ci_low": np.quantile(variance_boot, .025),
                               "return_variance_ci_high": np.quantile(variance_boot, .975),
                               "extreme_probability_ci_low": np.quantile(extreme_boot, .025),
                               "extreme_probability_ci_high": np.quantile(extreme_boot, .975),
                               "tail_alpha_fixed_xmin": float(tail_row.alpha_pdf),
                               "tail_alpha_ci_low": np.quantile(alpha_boot, .025),
                               "tail_alpha_ci_high": np.quantile(alpha_boot, .975)})
    # 将构型 bootstrap 区间写表。
    pd.DataFrame(bootstrap_rows).to_csv(root / "tables/configuration_bootstrap_ci.csv", index=False)
    # 以构型编号对齐各活动率的理论条件方差。
    pivot = variances[variances.replicate == 0].pivot(index="configuration", columns="activity", values="theoretical_variance")
    # 求活动率 0.1 和 0.5 之间的配对 Pearson 相关。
    correlation = pearsonr(pivot[0.1], pivot[0.5])
    # 记录配对相关、固定格点构型属性、归档来源和条件方差公式。
    metrics = {"shared_configuration_variance_correlation_a_0.1_0.5": float(correlation.statistic),
               "shared_configuration_correlation_p": float(correlation.pvalue),
               "fixed_cluster_count": fixed.cluster_count,
               "fixed_largest_size": fixed.largest_size,
               "data_source": str(args.source), "theory": "Var(r|s)=a sum(s^2)/lambda^2; zero nonzero-lag conditional covariance"}
    (root / "tables/audit_metadata.json").write_text(json.dumps(metrics, indent=2) + "\n")  # 写外部可查阅的审计元数据。
    print(json.dumps(metrics, indent=2))  # 在终端打印审计摘要。
    print(pd.DataFrame(tests).to_string(index=False))  # 在终端打印逐序列自相关检验表。


if __name__ == "__main__":
    # 启动市场时序审计。
    main()
