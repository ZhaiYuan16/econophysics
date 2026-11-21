#!/usr/bin/env python3
"""Run stage-3 CB return, activity, extreme-event, and tail analysis."""
# 运行市场收益率的影响因素实验，分析活动率、收益分布、极端事件和收益尾部。
# parse_args() 接收实验配置和输出选项。main() 生成系综、拟合尾部、计算活动率标度，并可单独绘制固定构型轨迹。

# 本脚本读取 scripts/simulation/scan_criticality.py 的临界格点统计，归档逐笔收益、构型和拟合供市场审计、核心图汇编及 loglog_ols.py 使用。

from __future__ import annotations  # 延后求值类型注解。

import argparse  # 解析命令行的实验配置和运行选项。
import json  # 将实验方法和来源写成 JSON。
import platform  # 记录生成结果的 Python 版本。
from dataclasses import asdict  # 将实验配置数据类转换为可归档字典。
from pathlib import Path  # 表示配置文件和图表目录。

import matplotlib  # 记录绘图库版本。
import numpy as np  # 处理收益数组、构型重采样及参数区间。
import pandas as pd  # 分组系综收益并读写统计表。
import scipy  # 记录统计计算所用 SciPy 版本。

from local_market.config import MarketAnalysisConfig  # src/local_market/config.py 的 MarketAnalysisConfig 加载市场模拟和尾拟合参数。
from local_market.experiments import (  # 从 src/local_market/experiments.py 导入市场模拟和汇总入口。
    run_market_ensemble,  # 生成逐笔收益和对应构型统计。
    run_quenched_market_trajectories,  # 生成独立固定构型时间轨迹。
    summarize_market_returns,  # 按条件标准化收益并生成系综及逐重复统计。
)
from local_market.plotting import (  # 从 src/local_market/plotting.py 导入市场实验绘图函数。
    plot_active_clusters_by_size,  # 展示期望活跃团簇数随格长的变化。
    plot_activity_statistics,  # 展示各市场条件下的收益统计。
    plot_market_time_series,  # 绘制单构型模拟时间轨迹。
    plot_market_ensemble_samples,  # 展示按构型块串接的系综收益样本。
    plot_return_distributions,  # 绘制收益 PDF 和绝对收益 CCDF。
    plot_tail_exponents,  # 绘制尾指数及拟合诊断。
)
from local_market.statistics import return_distribution_table, nonzero_absolute_standardized_returns  # src/local_market/statistics.py 的 return_distribution_table 计算分布，nonzero_absolute_standardized_returns 筛选尾拟合样本。
from local_market.hierarchical_statistics import (  # 从 src/local_market/hierarchical_statistics.py 导入构型重采样统计。
    block_histogram, resampled_counts, selection_aware_tail_gof, weighted_statistics,  # block_histogram 压缩块频数，resampled_counts 聚合抽中块，selection_aware_tail_gof 重选阈值做检验，weighted_statistics 重算加权统计。
)
from local_market.tail_fitting import (  # 从 src/local_market/tail_fitting.py 导入 Pareto 拟合和检验。
    compare_tail_distributions,  # 比较共同尾阈值上的备选分布。
    pareto_mle,  # 在给定 xmin 下求指数、KS 距离及计数。
    pareto_fixed_xmin_gof_pvalue,  # 计算固定阈值的参数检验 p 值。
    select_pareto_tail,  # 按候选分位数选择 Pareto 尾部起点。
)


def parse_args() -> argparse.Namespace:
    # 功能：解析市场收益率的影响因素实验的命令行选项。
    # 引用：main() 用返回配置路径、输出路径和两个开关运行实验。
    # 输入：无函数参数。读取进程的命令行参数。
    # 输出：argparse.Namespace，含 config、output_root、quiet 和 skip_quenched_trajectories。
    # 使用模块说明作为帮助简介。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)  # 要求显式提供实验 YAML。
    parser.add_argument("--output-root", type=Path, default=Path("results"))  # 默认将表格和图写入 results。
    parser.add_argument("--quiet", action="store_true")  # 可关闭系综模拟的进度显示。
    # 可将固定构型对照轨迹留给 audit_market_time_series.py 生成。
    parser.add_argument(
        "--skip-quenched-trajectories",
        action="store_true",
        help="Leave the dedicated time paths to audit_market_time_series.py.",
    )
    # 解析选项，布尔开关默认关闭。
    return parser.parse_args()


def main() -> None:
    # 功能：生成市场收益率的影响因素实验的市场系综、统计、尾部诊断和配套图表。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。通过命令行提供市场配置及输出目录，活跃团簇标度另需输出目录 tables 下的 criticality_summary.csv。
    # 输出：返回 None，写出实验归档表、图和元数据。缺少前置临界性汇总表时抛出 FileNotFoundError。
    # 读取本轮的运行选项。
    args = parse_args()
    # 加载格长、概率比、各活动率及抽样参数。
    config = MarketAnalysisConfig.from_yaml(args.config)
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    # 建立实验表目录。
    tables.mkdir(parents=True, exist_ok=True)
    # 建立实验图目录。
    figures.mkdir(parents=True, exist_ok=True)

    # 运行固定构型重复抽样的市场系综，返回逐笔收益和构型统计。
    raw_returns, structures = run_market_ensemble(
        config, show_progress=not args.quiet  # quiet 只控制进度显示。
    )
    # 对各市场条件标准化收益，同时生成池化与逐重复统计。
    returns, summary, replicate_summary = summarize_market_returns(raw_returns, config)

    distribution_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    xmin_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
    group_keys = ["length", "p_ratio", "p", "activity"]
    # 逐格长、概率比、绝对占据概率和活动率独立分析。
    for group_index, (key, group) in enumerate(returns.groupby(group_keys, sort=True)):
        parameters = dict(zip(group_keys, key, strict=True))  # 让各诊断行共享同一组条件键。
        z = group["z"].to_numpy(dtype=float)  # z 是当前市场条件池化标准化后的逐笔收益。
        # A zero raw return is still nonzero after mean centering; remove it
        # before selecting candidate quantiles for the conditional tail.
        absolute_z = nonzero_absolute_standardized_returns(  # 对原始非零交易提取绝对标准化收益作条件尾拟合。
            group["return"].to_numpy(dtype=float), z
        )

        # 对所有交易计算收益分布，给各分布点附上市场条件。
        for row in return_distribution_table(
            z, raw_returns=group["return"].to_numpy(dtype=float)  # 原始非零交易用于限定 CCDF 阈值范围。分布计数和分母使用全部交易。
        ):
            distribution_rows.append(parameters | row)  # 为分布点附上当前市场条件。

        # 在候选分位数上选择 KS 距离最小的 Pareto 尾阈值。
        best, candidates = select_pareto_tail(
            absolute_z,
            config.tail_xmin_quantiles,  # 由配置提供所搜索的阈值分位数。
            config.tail_minimum_count,  # 仅考虑尾样本数达到配置下限的候选。
        )
        # 为当前条件生成可复现的构型 bootstrap 随机流。
        bootstrap_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 30, group_index])
        )
        # 将重复号和重复内构型号编码为全条件的块 ID。
        block_ids = (group["replicate"].to_numpy(dtype=int) *
                     config.configurations_per_replicate + group["configuration"].to_numpy(dtype=int))
        # 将原始收益压缩成共同取值支持和逐构型计数。
        support, block_matrix = block_histogram(group["return"].to_numpy(), block_ids)
        # selector 接收随机数生成器，读取 config，返回一维构型块 ID。主循环和 selection_aware_tail_gof() 均调用它。
        selector = lambda rng: np.concatenate([
            replicate * config.configurations_per_replicate
            + rng.integers(0, config.configurations_per_replicate,  # 每个重复内有放回抽取原构型数，保持各重复的权重一致。
                           size=config.configurations_per_replicate)
            for replicate in range(config.replicates)
        ])
        # 每轮重抽整构型，重算标准化收益及尾部统计。
        estimates = [weighted_statistics(
            support, resampled_counts(block_matrix, selector(bootstrap_rng)),  # 用 selector 返回的块索引汇总原始收益的频数。
            config.tail_xmin_quantiles, config.tail_minimum_count, True,  # 按原阈值设置重选 xmin，并排除原始零收益。
        ) for _ in range(config.bootstrap_replicates)]
        alpha_values = np.array([estimate["alpha_pdf"] for estimate in estimates])  # 收集每轮的连续 Pareto PDF 指数。
        xmin_values = np.array([estimate["xmin"] for estimate in estimates])  # 收集每轮重新选择的尾部阈值。
        alpha_se = float(alpha_values.std(ddof=1))  # 用重抽指数的样本标准差作为标准误。
        alpha_low, alpha_high = np.quantile(alpha_values, [.025, .975])  # 取指数的双侧 95% 百分位区间。
        # 对原拟合执行固定阈值的参数式拟合优度检验。
        gof_pvalue = pareto_fixed_xmin_gof_pvalue(
            best, config.bootstrap_replicates, bootstrap_rng
        )
        # 用保持块体尾频数的半参数零假设评估重新选择阈值后的拟合优度。
        selection_gof = selection_aware_tail_gof(
            support, block_matrix, {"xmin": best.xmin, "alpha_pdf": best.alpha_pdf,  # 原样本的 xmin、指数和 KS 距离定义尾部零假设及比较基准。
                                    "ks_distance": best.ks_distance},
            config.tail_xmin_quantiles, config.tail_minimum_count,
            bootstrap_rng, config.bootstrap_replicates, selector, True,  # 使用同一分层块选择器，并在拟合前排除原始零收益。
        )
        # 在同一尾阈值下比较纯幂律与备选分布。
        comparison = compare_tail_distributions(absolute_z, best.xmin)
        # 将尾拟合、区间、模型比较及可靠性判据写成一条市场条件记录。
        tail_rows.append(
            parameters
            | {
                "tail_definition": "absolute standardized return |z|",
                "xmin": best.xmin,
                "alpha_pdf": best.alpha_pdf,
                "alpha_ccdf": best.alpha_pdf - 1.0,  # CCDF 指数等于 PDF 指数减一。
                "ks_distance": best.ks_distance,
                "tail_count": best.tail_count,
                "nonzero_return_count": int(np.count_nonzero(group["return"].to_numpy())),  # 用原始收益判定哪些交易为非零。
                "unique_tail": int(np.unique(absolute_z[absolute_z >= best.xmin]).size),  # 对达到阈值的不同绝对幅度计数。
                "tail_dynamic_range": float(absolute_z.max() / best.xmin),  # 用最大幅度与阈值之比衡量尾部动态范围。
                # 同时检查拟合优度、尾样本量、幅度跨度、不同值数量和似然比较。
                "power_law_reliable": bool(
                    selection_gof >= 0.1 and best.tail_count >= 1000
                    and absolute_z.max() / best.xmin >= 10
                    and np.unique(absolute_z[absolute_z >= best.xmin]).size >= 30
                    and comparison["loglikelihood_ratio_power_vs_truncated"] >= -2
                    and comparison["loglikelihood_ratio_power_vs_exponential"] >= 0
                ),
                "tail_fraction_of_nonzero": best.tail_fraction,  # 尾部比例以原始非零交易为样本池。
                "alpha_bootstrap_se": alpha_se,
                "alpha_ci_low": alpha_low,
                "alpha_ci_high": alpha_high,
                "xmin_ci_low": float(np.quantile(xmin_values, .025)),  # 用重选的 xmin 分布估计阈值区间下界。
                "xmin_ci_high": float(np.quantile(xmin_values, .975)),  # 用重选的 xmin 分布估计阈值区间上界。
                "selection_aware_hierarchical_null_gof_p": selection_gof,
                "pareto_fixed_xmin_gof_p": gof_pvalue,
            }
            | comparison
        )
        # 保存各候选阈值，供选择敏感性分析。
        xmin_rows.extend(parameters | candidate for candidate in candidates)

        # 在每个独立重复上用共同阈值比较拟合稳定性。
        for replicate, replicate_group in group.groupby("replicate", sort=True):
            nonzero = replicate_group["return"].to_numpy(dtype=float) != 0  # 在标准化前的收益上判定零交易。
            # 使用池化拟合的固定 xmin 求当前重复的指数、KS 和尾部计数。
            alpha, ks, count = pareto_mle(
                np.abs(replicate_group["z"].to_numpy(dtype=float)[nonzero]), best.xmin
            )
            # 记录重复编号、共同阈值和拟合诊断。
            seed_rows.append(
                parameters
                | {
                    "replicate": int(replicate),
                    "common_xmin": best.xmin,
                    "alpha_pdf": alpha,
                    "ks_distance": ks,
                    "tail_count": count,
                }
            )

        # 为当前条件的样本量敏感性分析派生独立随机流。
        nested_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 40, group_index])
        )
        order = nested_rng.permutation(absolute_z.size)  # 仅打乱一次，以便不同样本比例使用相互嵌套的前缀。
        for fraction in config.length_sensitivity_fractions:
            sample = absolute_z[order[: max(2, int(fraction * absolute_z.size))]]  # 取打乱序列的前缀，申请至少两个样本且不超过可用样本。
            # 对当前子样本重新选择 Pareto 阈值。
            length_fit, _ = select_pareto_tail(
                sample,
                config.tail_xmin_quantiles,
                min(config.tail_minimum_count, max(50, sample.size // 20)),  # 小样本使用较低的尾部计数门槛，仍受原配置上限约束。
            )
            # 保存实际样本数及重新拟合的尾参数。
            length_rows.append(
                parameters
                | {
                    "sample_fraction": fraction,
                    "sample_count": sample.size,
                    "xmin": length_fit.xmin,
                    "alpha_pdf": length_fit.alpha_pdf,
                    "ks_distance": length_fit.ks_distance,
                    "tail_count": length_fit.tail_count,
                }
            )

    # 将逐条件 PDF 和 CCDF 行合成分布表。
    distributions = pd.DataFrame(distribution_rows)
    # 将主尾拟合和模型比较合成拟合表。
    tail_fits = pd.DataFrame(tail_rows)
    # 组成候选阈值敏感性表。
    xmin_sensitivity = pd.DataFrame(xmin_rows)
    # 组成独立重复敏感性表。
    seed_sensitivity = pd.DataFrame(seed_rows)
    # 组成子样本量敏感性表。
    length_sensitivity = pd.DataFrame(length_rows)

    criticality_path = tables / "criticality_summary.csv"
    # 在绘制活动团簇的格长依赖前，要求已有第二阶段的汇总。
    if not criticality_path.exists():
        raise FileNotFoundError(
            "stage-2 criticality_summary.csv is required for activity-size scaling"
        )
    # 读取渗流临界性质实验的格点扫描汇总。
    criticality = pd.read_csv(criticality_path)
    # 选取 p/p_c 近似为 1 的格长、构型数和团簇计数统计。
    critical_sizes = criticality[np.isclose(criticality["p_ratio"], 1.0)][
        ["length", "p", "samples", "mean_cluster_count", "cluster_count_sem"]
    ]
    # 为每个活动率复制一组临界格长统计。
    activity_scaling = pd.concat(
        [critical_sizes.assign(activity=activity) for activity in config.activities],  # 将当前活动率附在该组的每个格长上。
        ignore_index=True,  # 按新顺序生成行索引。
    )
    # 由每个团簇独立以概率 a 活跃，得到 E[K_active]=a E[K]。
    activity_scaling["expected_active_cluster_count"] = (
        activity_scaling["activity"] * activity_scaling["mean_cluster_count"]
    )
    # 将团簇总数均值的标准误同乘 a。
    activity_scaling["expected_active_cluster_count_sem"] = (
        activity_scaling["activity"] * activity_scaling["cluster_count_sem"]
    )

    paths = {
        "returns": tables / "market_returns.csv.gz",
        "structures": tables / "market_structures.csv",
        "summary": tables / "market_summary.csv",
        "replicates": tables / "market_replicates.csv",
        "distributions": tables / "market_distributions.csv",
        "tail_fits": tables / "market_tail_fits.csv",
        "tail_xmin_sensitivity": tables / "market_tail_xmin_sensitivity.csv",
        "tail_seed_sensitivity": tables / "market_tail_seed_sensitivity.csv",
        "tail_length_sensitivity": tables / "market_tail_length_sensitivity.csv",
        "active_clusters_by_size": tables / "market_active_clusters_by_size.csv",
    }
    # 按 gzip 压缩保存逐笔原始与标准化收益。
    returns.to_csv(paths["returns"], index=False, compression="gzip")
    # 保存构型统计供时序审计复核。
    structures.to_csv(paths["structures"], index=False)
    # 保存条件池化统计。
    summary.to_csv(paths["summary"], index=False)
    # 保存各独立重复的统计。
    replicate_summary.to_csv(paths["replicates"], index=False)
    # 保存 PDF 与 CCDF 数据。
    distributions.to_csv(paths["distributions"], index=False)
    # 保存主尾拟合和分布比较。
    tail_fits.to_csv(paths["tail_fits"], index=False)
    # 保存阈值扫描表。
    xmin_sensitivity.to_csv(paths["tail_xmin_sensitivity"], index=False)
    # 保存逐重复的尾拟合。
    seed_sensitivity.to_csv(paths["tail_seed_sensitivity"], index=False)
    # 保存子样本量敏感性。
    length_sensitivity.to_csv(paths["tail_length_sensitivity"], index=False)
    # 保存预期活跃团簇数及标准误。
    activity_scaling.to_csv(paths["active_clusters_by_size"], index=False)

    # 绘制拼接的系综收益样本。
    plot_market_ensemble_samples(returns, figures / "market_ensemble_samples.png")
    # 按开关决定由此处还是时序审计入口生成专用轨迹。
    if args.skip_quenched_trajectories:
        time_series_description = (
            "Dedicated quenched trajectories are generated by audit_market_time_series.py. "
            "market_ensemble_samples.png has no price path."
        )
    else:
        _, trajectories = run_quenched_market_trajectories(config)  # 在一个临界固定构型上生成不同活动率的独立行动轨迹。
        plot_market_time_series(trajectories, figures / "market_time_series.png")  # 绘制模拟时间上的收益、绝对收益和对数价格。
        time_series_description = (
            "market_time_series.png is a separate 3000-step quenched trajectory at p=pc; "
            "its fixed occupancy seed is [config.seed,50], and each activity has an "
            "independent action stream [config.seed,51,activity_index]. "
            "market_ensemble_samples.png has no price path."
        )
    # 绘制标准化收益 PDF 及绝对收益 CCDF。
    plot_return_distributions(distributions, figures / "market_pdf_ccdf.png")
    # 绘制条件汇总随活动率的变化。
    plot_activity_statistics(summary, figures / "market_activity_statistics.png")
    # 绘制各市场条件的尾指数及可靠性诊断。
    plot_tail_exponents(tail_fits, figures / "market_tail_exponents.png")
    # 绘制临界点的预期活跃团簇数随格长的变化。
    plot_active_clusters_by_size(
        activity_scaling, figures / "market_active_clusters_by_size.png"
    )

    # 归档实验配置、抽样设计、统计口径及软件环境。
    metadata = {
        "config": asdict(config),  # 将配置数据类转成普通字典。
        "sampling": (
            "Independent occupied configurations are generated per p and replicate; "
            "each is reused for multiple independent cluster-action draws and across "
            "activities. The concatenation is an ensemble ordering, not evidence of "
            "temporal volatility clustering."
        ),
        "time_series": time_series_description,
        "activity_size_scaling": (
            "Expected active cluster counts use E[K_active]=a E[K] with the "
            "p/pc=1 cluster counts produced in stage 2."
        ),
        "standardization": "per (p/p_c, activity), sample standard deviation ddof=1",
        "tail_method": {
            "variable": "absolute standardized return |z|, excluding zeros",
            "exponent": "continuous Pareto PDF exponent alpha",
            "xmin_selection": "minimum KS distance over configured quantile candidates",
            "confidence_interval": "configuration-block bootstrap with xmin reselected in every draw",
            "goodness_of_fit": (
                "legacy conditional parametric KS p-value at fixed xmin is retained; "
                "selection-aware semiparametric null resamples configuration blocks, "
                "preserves their body and tail counts, generates common Pareto tail "
                "magnitudes, and reselects xmin in every draw"
            ),
            "distribution_comparison": (
                "powerlaw package likelihood ratios; positive ratio favors pure power law"
            ),
        },
        "software": {
            "python": platform.python_version(),  # 记录当前 Python 版本。
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {name: str(path) for name, path in paths.items()},  # 将输出文件路径转换为 JSON 可序列化的字符串。
    }
    # 保存可用于复现实验的 UTF-8 元数据。
    (tables / "market_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 打印市场系综的汇总统计。
    print(summary.to_string(index=False))
    # 打印尾拟合及诊断表。
    print(tail_fits.to_string(index=False))
    # 报告第三阶段产物的根目录。
    print(f"saved stage-3 outputs under {args.output_root}")


if __name__ == "__main__":
    # 启动市场收益率分析。
    main()
