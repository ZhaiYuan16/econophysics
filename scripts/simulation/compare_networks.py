#!/usr/bin/env python3
"""Run stage-4 network-topology criticality and market stress tests."""
# 运行网络拓扑压力测试。parse_args() 读取配置与输出选项，main() 是命令行入口，组织网络系综、
# 拓扑各自的渗流阈值估计、匹配相对临界位置的交易、收益分布以及 Pareto 尾部诊断。
#
# 本脚本从网络生成和网络节点占据参数出发，产出结构、渗流与市场结果。网络收益表供下游统计审计
# 复算分层区间与极端收益概率，也供 log-log OLS 分析拟合尾部。

from __future__ import annotations  # 延后求值函数的类型注解。

import argparse  # 标准库，用于声明并解析命令行选项。
import json  # 标准库，用于将网络模拟的元数据写为 JSON。
import platform  # 标准库，用于记录 Python 版本。
from dataclasses import asdict  # 标准库，将网络分析配置数据类转换为元数据字典。
from pathlib import Path  # 标准库，用路径对象定位配置、表格和输出图。

import matplotlib  # 第三方 Matplotlib 绘图库，这里记录绘图环境版本。
import networkx as nx  # 第三方 NetworkX 网络分析库，这里记录建图环境版本。
import numpy as np  # 第三方 NumPy，用于收益幅度、分层抽样随机流和区间估计。
import pandas as pd  # 第三方 pandas，用于按场景分组、合并并保存结构与交易表。
import scipy  # 第三方 SciPy 科学计算库，这里记录统计计算环境版本。

from local_market.config import NetworkAnalysisConfig  # 本地 src/local_market/config.py，读取与校验拓扑、渗流和交易设置。
from local_market.network_analysis import (  # 本地 src/local_market/network_analysis.py，网络系综和交易统计入口。
    build_network_ensemble,  # 生成各拓扑的网络实现、逐图结构指标和度频数。
    run_network_criticality_scan,  # 扫描网络上的节点渗流，用易感性峰定位各拓扑有效临界概率。
    run_network_market_comparison,  # 在相对临界位置匹配的情况下生成逐笔交易和占据配置记录。
    summarize_network_market,  # 对场景收益统一标准化，汇总池化及逐网络交易指标。
)
from local_market.plotting import (  # 本地 src/local_market/plotting.py，网络拓扑压力测试的绘图函数。
    plot_network_criticality,  # 绘制网络渗流曲线和有效临界阈值。
    plot_network_degree_distributions,  # 绘制各拓扑网络节点的度分布。
    plot_network_market_comparison,  # 绘制市场极端收益等场景统计。
    plot_network_return_ccdfs,  # 绘制绝对标准化收益的经验 CCDF。
    plot_network_structure_metrics,  # 绘制逐图结构指标。
    plot_network_tail_exponents,  # 绘制收录分层区间的网络尾指数。
)
from local_market.statistics import return_distribution_table  # 本地 src/local_market/statistics.py，生成经验分布逐点表，此处仅取 CCDF。
from local_market.hierarchical_statistics import (  # 本地 src/local_market/hierarchical_statistics.py，按网络和占据配置两级做统计推断。
    block_histogram, resampled_counts, select_network_configuration_blocks,  # block_histogram 压缩逐块收益，resampled_counts 合并重抽频数，select_network_configuration_blocks 执行两级抽样。
    selection_aware_tail_gof, weighted_statistics,  # selection_aware_tail_gof 检验重选下界后的尾拟合，weighted_statistics 从重抽频数计算交易与尾部统计。
)
from local_market.tail_fitting import (  # 本地 src/local_market/tail_fitting.py，连续 Pareto 拟合、检验和模型比较。
    compare_tail_distributions,  # 在同一尾样本上比较备选分布。
    pareto_fixed_xmin_gof_pvalue,  # 在固定尾部下界时计算拟合优度 p 值。
    pareto_mle,  # 在指定阈值下估计 Pareto PDF 指数、KS 距离和尾样本数。
    select_pareto_tail,  # 用最小 KS 距离从候选下界中选择 Pareto 尾。
)


def parse_args() -> argparse.Namespace:
    # 解析网络拓扑压力测试的命令行选项。
    # 由本文件的 main() 调用，用于定位配置和结果，并控制模拟进度显示。
    # 输入无函数参数，从进程命令行读取选项。
    # 输出 argparse.Namespace，含必填的配置路径 config、默认 results 的 output_root 和默认关闭的 quiet 开关。
    parser = argparse.ArgumentParser(description=__doc__)  # 用原模块说明作为命令行帮助的简介。
    parser.add_argument("--config", type=Path, required=True)  # 要求提供网络拓扑压力测试的 YAML 配置路径。
    parser.add_argument("--output-root", type=Path, default=Path("results"))  # 指定表格和图的共同父目录。
    parser.add_argument("--quiet", action="store_true")  # 请求关闭网络渗流和交易进度显示。
    return parser.parse_args()  # 返回解析结果，参数不合法时由 argparse 退出。


def main() -> None:
    # 运行网络拓扑压力测试，比较结构、渗流阈值、交易分布和尾部统计。
    # 由本文件的 __main__ 入口调用。scripts/analysis/audit_downstream_statistics.py 的 main() 读取收益、汇总和尾拟合等表进行审计，
    # scripts/analysis/loglog_ols.py 的 main() 从逐笔收益表拟合尾部。scripts/figures/plot_network_degree.py 的模块顶层可用度频数表重绘。
    # 输入无函数参数，命令行提供配置路径、输出根目录和进度开关。
    # 输出 None。写出网络结构、渗流、逐笔交易与分布等表格、图和配置元数据，并打印主要统计结果。
    args = parse_args()  # 读取运行选项，确定配置与结果位置。
    config = NetworkAnalysisConfig.from_yaml(args.config)  # 加载网络规模、拓扑参数、扫描网格、交易和尾部分析设置。
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)  # 创建表格和元数据目录，允许复用。
    figures.mkdir(parents=True, exist_ok=True)  # 创建输出图目录，允许复用。

    ensemble, network_metrics, degree_distributions = build_network_ensemble(config)  # 构建拓扑系综，并分别记录逐图结构指标与度频数。
    # 在同一批网络上扫描节点渗流，得到逐配置样本、扫描汇总和拓扑各自的有效临界点。
    critical_samples, critical_summary, pc_estimates = run_network_criticality_scan(
        config, ensemble, show_progress=not args.quiet  # 复用网络系综，quiet 未开启时显示进度。
    )
    # 用各拓扑自身的有效临界点换算占据概率，运行可比的固定团簇交易。
    raw_returns, market_structures = run_network_market_comparison(
        config, ensemble, pc_estimates, show_progress=not args.quiet  # 对同一批网络模拟，按配置的相对概率选择各拓扑的占据概率。
    )
    # 依次得到含标准化收益 z 的逐笔表、场景池化汇总和逐网络汇总。
    returns, market_summary, market_replicates = summarize_network_market(
        raw_returns, config  # 同一场景的交易共用池化均值和样本标准差。
    )

    distribution_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    tail_seed_rows: list[dict[str, object]] = []
    keys = ["topology", "relative_p", "p", "activity"]  # 场景键依次为拓扑、相对临界概率比、实际占据概率和团簇活跃率。
    for group_index, (key, group) in enumerate(returns.groupby(keys, sort=False)):  # 按场景分别构造经验分布和尾拟合，沿用场景首次出现的顺序。
        parameters = dict(zip(keys, key, strict=True))  # 保存场景参数，附到 CCDF 和尾拟合等输出行。
        absolute_z = np.abs(group["z"].to_numpy(dtype=float))  # 每笔交易取绝对标准化收益 |z| 作为尾拟合的幅度。
        for distribution_row in return_distribution_table(group["z"].to_numpy(), include_pdf=False):  # 仅生成 |z| 的经验 CCDF，达到或超过阈值的比例以场景全部交易为分母。
            distribution_rows.append(parameters | distribution_row)  # 为当前 CCDF 点附上场景参数。
        # 从场景的有限正幅度分位数中选候选下界，并保留 KS 距离最小的有效连续 Pareto 尾。
        fit, _ = select_pareto_tail(
            absolute_z,
            config.tail_xmin_quantiles,  # 将给定分位数处的幅度作为候选下界。
            config.tail_minimum_count,  # 尾样本数须达到配置要求，且指数须为有限值。
        )
        # 为当前场景建立可复现的随机数流，供重抽样和拟合检验。
        rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 200, group_index])  # 用主种子、固定标识 200 和场景编号派生随机种子。
        )
        # 用网络编号乘每网配置数，再加网内配置编号，得到跨网络唯一的配置块索引。
        block_ids = (group["network_replicate"].to_numpy(dtype=int) *
                     config.market_configurations_per_network + group["configuration"].to_numpy(dtype=int))
        support, block_matrix = block_histogram(group["return"].to_numpy(), block_ids)  # 得到原始收益支持点和稀疏频数矩阵，矩阵行对应配置块，列对应支持点。
        # main() 的 selector 回调对网络与网内配置两级有放回重抽。
        # main() 在区间估计时直接调用它，随后传给 selection_aware_tail_gof() 供检验逐轮调用。
        # 输入 draw_rng 为 NumPy 随机数生成器，回调还读取外层 config 中的网络数和每网配置数。
        # 输出一维配置块编号，可重复，长度等于网络数乘每网配置数。
        selector = lambda draw_rng: select_network_configuration_blocks(
            draw_rng, config.network_replicates, config.market_configurations_per_network)  # 保持每级的抽样规模与原网络系综相同。
        # 每轮合并整块重抽的收益频数，重新标准化并重选尾部下界。
        estimates = [weighted_statistics(
            support, resampled_counts(block_matrix, selector(rng)),  # 先抽块编号再合并频数，保留块内所有交易。
            config.tail_xmin_quantiles, config.tail_minimum_count,  # 用与原拟合相同的分位数规则和最少尾观测数重选下界。
        ) for _ in range(config.bootstrap_replicates)]
        alpha_values = np.array([estimate["alpha_pdf"] for estimate in estimates])  # 收集各次重抽的 Pareto PDF 指数。
        xmin_values = np.array([estimate["xmin"] for estimate in estimates])  # 收集各次重抽重新选出的下界。
        alpha_se = float(alpha_values.std(ddof=1))  # 用重抽指数的样本标准差估计指数标准误。
        alpha_low, alpha_high = np.quantile(alpha_values, [.025, .975])  # 取重抽指数的 2.5% 和 97.5% 分位数作为区间端点。
        # 固定原拟合的 xmin，逐轮生成 Pareto 尾样本、重估指数并比较 KS 距离以计算 p 值。
        gof = pareto_fixed_xmin_gof_pvalue(
            fit, config.bootstrap_replicates, rng
        )
        # 用网络和配置两级重抽构造半参数零假设，保留经验主体与重抽尾样本数，并对合成样本重选 xmin。
        selection_gof = selection_aware_tail_gof(
            support, block_matrix, {"xmin": fit.xmin, "alpha_pdf": fit.alpha_pdf,  # 使用场景原拟合的下界、指数与 KS 距离作为检验基准。
                                    "ks_distance": fit.ks_distance},
            config.tail_xmin_quantiles, config.tail_minimum_count,  # 候选下界和最少尾观测数与点估计规则相同。
            rng, config.bootstrap_replicates, selector,  # 由 selector 逐轮抽取配置块，并采用配置指定的检验轮数。
        )
        comparison = compare_tail_distributions(absolute_z, fit.xmin)  # 在原拟合阈值处对同一场景尾数据比较候选模型。
        # 保存当前场景的尾定义、拟合结果、重抽区间和两种拟合优度。
        tail_rows.append(
            parameters
            | {
                "tail_definition": "absolute standardized return |z|",
                "xmin": fit.xmin,
                "alpha_pdf": fit.alpha_pdf,
                "alpha_ccdf": fit.alpha_pdf - 1.0,  # 连续 Pareto 的 CCDF 指数等于 PDF 指数减一。
                "ks_distance": fit.ks_distance,
                "tail_count": fit.tail_count,
                "tail_fraction_of_nonzero": fit.tail_fraction,  # 分母为具有有限正 |z| 的交易数。
                "alpha_bootstrap_se": alpha_se,
                "alpha_ci_low": alpha_low,
                "alpha_ci_high": alpha_high,
                "xmin_ci_low": float(np.quantile(xmin_values, .025)),  # 下界区间的下端是重抽 xmin 的 2.5% 分位数。
                "xmin_ci_high": float(np.quantile(xmin_values, .975)),  # 下界区间的上端是重抽 xmin 的 97.5% 分位数。
                "selection_aware_hierarchical_null_gof_p": selection_gof,
                "pareto_fixed_xmin_gof_p": gof,
            }
            | comparison  # 将同一尾样本上的备选模型比较结果并入该行。
        )
        for replicate, replicate_group in group.groupby("network_replicate"):  # 按网络实现拆分场景，检查尾拟合对网络种子的敏感性。
            # 在单个网络中使用场景共用的标准化收益和尾部下界，计算该网络的指数、KS 距离与计数。
            alpha, ks, count = pareto_mle(
                np.abs(replicate_group["z"].to_numpy()), fit.xmin
            )
            # 保存当前网络实现的尾部诊断及共用阈值。
            tail_seed_rows.append(
                parameters
                | {
                    "network_replicate": int(replicate),
                    "common_xmin": fit.xmin,
                    "alpha_pdf": alpha,
                    "ks_distance": ks,
                    "tail_count": count,
                }
            )

    distributions = pd.DataFrame(distribution_rows)  # 将各场景的经验 CCDF 点整理成长表。
    tail_fits = pd.DataFrame(tail_rows)  # 将各场景的池化尾拟合整理成表。
    tail_seed_sensitivity = pd.DataFrame(tail_seed_rows)  # 将共用阈值下的逐网络尾拟合整理成敏感性表。
    # 跨网络实现汇总各拓扑的结构指标，并与渗流阈值对齐。
    network_metric_summary = (
        network_metrics.groupby("topology", as_index=False)
        .agg(
            mean_degree=("mean_degree", "mean"),
            mean_degree_sd=("mean_degree", "std"),  # 对各网络实现的平均度取样本标准差。
            degree_std=("degree_std", "mean"),  # 先在各网络上计算度标准差，再对同一拓扑的网络取均值。
            maximum_degree=("maximum_degree", "mean"),
            clustering_coefficient=("clustering_coefficient", "mean"),
            clustering_coefficient_sd=("clustering_coefficient", "std"),  # 计算网络间聚类系数的样本标准差。
            largest_connected_component_fraction=(
                "largest_connected_component_fraction",
                "mean",  # 先算各网络最大连通分量占全部节点的比例，再对网络取均值。
            ),
            average_shortest_path_length_approx=(
                "average_shortest_path_length_approx",
                "mean",  # 对各网络最大连通分量内估计的平均最短路径长度取均值。
            ),
            average_shortest_path_length_approx_sd=(
                "average_shortest_path_length_approx",
                "std",  # 计算该路径长度估计值在网络实现之间的样本标准差。
            ),
        )
        .merge(pc_estimates, on="topology", validate="one_to_one")  # 按拓扑并入有效临界点等估计，要求两边各拓扑均至多一行。
    )

    # 统一记录各结果表的文件位置，逐笔交易表单独压缩。
    paths = {
        "network_metrics": tables / "network_metrics.csv",
        "network_metric_summary": tables / "network_metric_summary.csv",
        "degree_distributions": tables / "network_degree_distributions.csv",
        "criticality_samples": tables / "network_criticality_samples.csv",
        "criticality_summary": tables / "network_criticality_summary.csv",
        "effective_pc": tables / "network_effective_pc.csv",
        "market_structures": tables / "network_market_structures.csv",
        "market_returns": tables / "network_market_returns.csv.gz",
        "market_summary": tables / "network_market_summary.csv",
        "market_replicates": tables / "network_market_replicates.csv",
        "market_distributions": tables / "network_market_distributions.csv",
        "tail_fits": tables / "network_tail_fits.csv",
        "tail_seed_sensitivity": tables / "network_tail_seed_sensitivity.csv",
    }
    # 将未压缩的表与输出路径按同名键对应。
    frames = {
        "network_metrics": network_metrics,
        "network_metric_summary": network_metric_summary,
        "degree_distributions": degree_distributions,
        "criticality_samples": critical_samples,
        "criticality_summary": critical_summary,
        "effective_pc": pc_estimates,
        "market_structures": market_structures,
        "market_summary": market_summary,
        "market_replicates": market_replicates,
        "market_distributions": distributions,
        "tail_fits": tail_fits,
        "tail_seed_sensitivity": tail_seed_sensitivity,
    }
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False)  # 写出各汇总和诊断 CSV，省略行索引。
    returns.to_csv(paths["market_returns"], index=False, compression="gzip")  # 将含原始与标准化收益的逐笔表写为 gzip 压缩 CSV。

    plot_network_structure_metrics(network_metrics, figures / "network_metrics.png")  # 按拓扑绘制逐图结构指标。
    # 从逐图度频数绘制拓扑间度分布。
    plot_network_degree_distributions(
        degree_distributions, figures / "network_degree_distributions.png"
    )
    # 对照渗流曲线及各拓扑的有效临界点。
    plot_network_criticality(
        critical_summary, pc_estimates, figures / "network_criticality.png"
    )
    # 用市场池化统计绘制极端交易事件对照。
    plot_network_market_comparison(
        market_summary, figures / "network_extreme_events.png"
    )
    # 绘制绝对标准化收益的经验 CCDF。
    plot_network_return_ccdfs(
        distributions, figures / "network_return_ccdfs.png"
    )
    # 绘制各场景 Pareto 尾指数和对应的分层重抽区间。
    plot_network_tail_exponents(
        tail_fits, figures / "network_tail_exponents.png"
    )

    # 记录配置、拓扑匹配与统计口径、软件版本和输出表位置。
    metadata = {
        "config": asdict(config),  # 将网络配置数据类展开为可序列化字典。
        "comparison": (
            "Networks are matched by node count and target mean degree. Market "
            "comparisons use the same activity and positions relative to each "
            "topology's susceptibility-peak effective threshold."
        ),
        "path_length": (
            "Average shortest path is estimated within the underlying graph's "
            "largest connected component using configured sampled sources."
        ),
        "tail": (
            "Continuous Pareto PDF fit to nonzero |z|; xmin chosen from fixed "
            "quantile candidates by KS distance. Confidence intervals resample "
            "four networks, then configurations within each network, with xmin "
            "reselected. The selection-aware semiparametric GOF retains block "
            "tail counts and refits xmin under a common-Pareto tail null."
        ),
        "software": {
            "python": platform.python_version(),  # 记录运行本实验的 Python 版本。
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "networkx": nx.__version__,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    # 将网络拓扑压力测试的元数据写为 UTF-8 JSON。
    (tables / "network_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(network_metric_summary.to_string(index=False))  # 打印聚合结构指标及有效临界点。
    print(market_summary.to_string(index=False))  # 打印场景市场统计。
    print(tail_fits.to_string(index=False))  # 打印池化尾拟合及诊断结果。
    print(f"saved stage-4 outputs under {args.output_root}")  # 报告网络拓扑压力测试的结果目录。


if __name__ == "__main__":
    main()  # 直接运行脚本时执行网络拓扑压力测试。
