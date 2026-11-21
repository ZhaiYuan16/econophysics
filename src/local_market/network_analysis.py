"""Network-topology stress tests for the Cont-Bouchaud model."""
# 在匹配规模的 lattice、ER、WS 和 BA 网络上执行网络拓扑压力测试。
# build_network_ensemble() 构图并统计结构，run_network_criticality_scan() 估计各拓扑的有效临界点。
# run_network_market_comparison() 按相对有效临界概率生成交易，summarize_network_market() 标准化收益并汇总。
#
# 网络比较脚本把图实例和临界点依次交给扫描与交易，用各拓扑的收益分布、极端概率和尾部比较模型的拓扑敏感性。

from __future__ import annotations  # 延后求值类型注解。

import numpy as np  # NumPy 用于派生随机流、抽样占据状态和汇总规模矩。
import pandas as pd  # Pandas 用于组装网络结构、渗流和逐笔交易数据。
from tqdm import tqdm  # tqdm 用于显示渗流和交易系综进度。

from .cb_model import sample_demands_and_active_counts  # src/local_market/cb_model.py，在固定网络占据配置上批量抽样需求和活跃团簇数。
from .config import NetworkAnalysisConfig  # src/local_market/config.py，定义网络压力测试的拓扑、样本量和交易参数。
from .networks import (  # src/local_market/networks.py，提供匹配网络的生成与测量。
    GraphRealization,  # 承载一份原图、稀疏邻接矩阵和种子。
    generate_matched_graph,  # 按拓扑生成匹配节点数和目标平均度的图。
    graph_structure_metrics,  # 提取整图结构并近似最短路。
    identify_graph_clusters,  # 给占据节点划分团簇。
)
from .statistics import finite_cluster_sizes, return_statistics, standardize_returns  # src/local_market/statistics.py，筛选有限团簇、汇总收益统计并按样本标准化。


def build_network_ensemble(
    config: NetworkAnalysisConfig,
) -> tuple[dict[str, list[GraphRealization]], pd.DataFrame, pd.DataFrame]:
    # 为网络拓扑压力测试生成各拓扑的网络系综，并测量整图结构和度分布。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 将图样本用于渗流和交易比较，保存结构和度分布表。
    # 输入
    #     config 为 NetworkAnalysisConfig，使用拓扑、规模、目标度、重复数、WS 重连率、路径起点数和种子。
    # 输出依次为
    #     拓扑名称到 GraphRealization 列表的映射，列表索引为网络重复编号。
    #     逐网络的结构指标表。
    #     逐网络按度统计的节点数和比例长表，含出现过的零度节点。
    ensemble: dict[str, list[GraphRealization]] = {}
    metric_rows: list[dict[str, object]] = []
    degree_rows: list[dict[str, object]] = []
    for topology_index, topology in enumerate(config.topologies):  # 用拓扑位置参与派生随机子流。
        realizations = []
        for replicate in range(config.network_replicates):
            # 用实验种子、拓扑和重复编号派生单份图种子。
            seed = int(
                np.random.SeedSequence(
                    [config.seed, 100, topology_index, replicate]
                ).generate_state(1)[0]  # 取种子序列生成的第一个整数交给图生成器。
            )
            # 为当前拓扑生成与实验规模参数匹配的网络。
            realization = generate_matched_graph(
                topology,
                config.node_count,
                config.target_mean_degree,
                seed,
                config.ws_rewiring_probability,
            )
            realizations.append(realization)  # 按照重复编号保存图实例。
            # 为最短路起点抽样派生区别于构图的随机流。
            metric_rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, 101, topology_index, replicate])
            )
            # 统计这份图的结构指标并近似分量内最短路。
            metrics = graph_structure_metrics(
                realization, config.path_sources, metric_rng
            )
            # 将整图指标与拓扑、重复编号和种子一起记录。
            metric_rows.append(
                {
                    "topology": topology,
                    "network_replicate": replicate,
                    "network_seed": seed,
                }
                | metrics
            )
            # 按原图节点迭代提取整数度数。
            degrees = np.fromiter(
                (degree for _, degree in realization.graph.degree()), dtype=np.int64  # 从同一个原图读取每个节点的度，并存入整数数组。
            )
            counts = np.bincount(degrees)  # 计算每个度对应的节点数，索引 0 表示孤立节点。
            # 收集非空度数箱的计数和全图占比。
            degree_rows.extend(
                {
                    "topology": topology,
                    "network_replicate": replicate,
                    "degree": degree,
                    "node_count": int(count),
                    "node_fraction": float(count / config.node_count),  # 节点占比以原图总节点数为分母。
                }
                for degree, count in enumerate(counts)  # 将频数数组索引解释成整数节点度。
                if count > 0
            )
        ensemble[topology] = realizations
    return ensemble, pd.DataFrame(metric_rows), pd.DataFrame(degree_rows)  # 返回可复用的图实例及两张结构汇总表。


def run_network_criticality_scan(
    config: NetworkAnalysisConfig,
    ensemble: dict[str, list[GraphRealization]],
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 为网络拓扑压力测试扫描各拓扑的占据概率，并以池化易感性峰估计有效临界点。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 保存扫描结果，并把有效临界点传给 run_network_market_comparison() 选择交易占据概率。
    # 输入
    #     config 为 NetworkAnalysisConfig，提供各拓扑的 p 网格、样本数、节点数和种子，斜率估计每个网格至少需两个点。
    #     ensemble 为按拓扑和网络重复组织的 GraphRealization 系综。
    #     show_progress 默认 True，显示逐占据配置进度。
    # 输出依次为
    #     每份图的逐占据样本表。
    #     按拓扑和 p 池化的序参量与易感性表。
    #     每个拓扑的有效临界点、峰值、斜率估计和边界标记。
    rows: list[dict[str, object]] = []
    total = sum(len(config.p_grids[name]) for name in config.topologies)  # 汇总各拓扑的占据概率点数。
    total *= config.network_replicates * config.occupancy_samples_per_p
    progress = tqdm(total=total, desc="network criticality", unit="sample", disable=not show_progress)  # 用逐图逐占据样本数初始化进度。
    for topology_index, topology in enumerate(config.topologies):  # 按拓扑扫描，索引用于区分占据随机流。
        for p_index, p in enumerate(config.p_grids[topology]):  # 使用当前拓扑自身的绝对占据概率网格。
            for network_replicate, realization in enumerate(ensemble[topology]):  # 在同一概率点复用每个已有网络实例。
                # 为拓扑、概率点和网络重复组合建立独立的占据随机流。
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 110, topology_index, p_index, network_replicate]
                    )
                )
                for occupancy_sample in range(config.occupancy_samples_per_p):  # 固定图和 p，逐个抽取独立的节点占据配置。
                    occupied = rng.random(config.node_count) < p  # 在固定原图上以 p 独立占据各节点。
                    clusters = identify_graph_clusters(realization.adjacency, occupied)  # 从占据诱导子图识别无向团簇。
                    finite = finite_cluster_sizes(clusters, "largest")  # 剔除一个最大团簇，余下规模用于有限团簇矩。
                    # 保存各占据样本的结构和有限团簇矩。
                    rows.append(
                        {
                            "topology": topology,
                            "p": p,
                            "network_replicate": network_replicate,
                            "occupancy_sample": occupancy_sample,
                            "occupied_count": clusters.occupied_count,
                            "cluster_count": clusters.cluster_count,  # 记录占据团簇个数。
                            "largest_size": clusters.largest_size,
                            "largest_occupied_fraction": clusters.largest_occupied_fraction,  # 最大团簇占已占据节点的比例。
                            "p_infinity_sample": clusters.p_infinity,  # 序参量以所有原图节点为分母。
                            "finite_first_moment": int(finite.sum()),  # 有限团簇规模之和作为池化分母的贡献。
                            "finite_second_moment": int(np.dot(finite, finite)),  # 有限团簇规模平方和作为池化分子的贡献。
                        }
                    )
                    progress.update(1)  # 标记当前占据配置完成。
    progress.close()  # 结束扫描进度显示。
    samples = pd.DataFrame(rows)  # 将逐配置记录转换为表。
    # 在同一拓扑和概率下跨图、跨配置池化。
    summary = (
        samples.groupby(["topology", "p"], as_index=False)
        .agg(
            samples=("occupancy_sample", "count"),  # 计入该组所有图上的占据样本。
            mean_occupied=("occupied_count", "mean"),
            mean_cluster_count=("cluster_count", "mean"),
            mean_largest_size=("largest_size", "mean"),
            p_infinity=("p_infinity_sample", "mean"),  # 对每个占据样本的 P∞ 取平均。
            p_infinity_sem=(
                "p_infinity_sample",
                lambda x: x.std(ddof=1) / np.sqrt(x.count()),  # 序参量标准误为样本标准差除以有效样本数的平方根，由 Pandas 调用该回调。
            ),
            finite_first_moment=("finite_first_moment", "sum"),  # 累加有限团簇规模和。
            finite_second_moment=("finite_second_moment", "sum"),  # 累加有限团簇规模平方和。
        )
    )
    summary["susceptibility_pooled"] = (
        summary["finite_second_moment"] / summary["finite_first_moment"]  # 将池化规模平方和除以池化规模和，空分母产生 NaN。
    )
    estimate_rows = []
    for topology, group in summary.groupby("topology", sort=False):  # 每个拓扑独立确定有效临界点。
        ordered = group.sort_values("p").reset_index(drop=True)  # 按实际概率排序并使网格下标从零开始。
        chi_index = int(ordered["susceptibility_pooled"].idxmax())  # 取池化易感性最大的网格位置。
        slopes = np.gradient(ordered["p_infinity"], ordered["p"])  # 按实际 p 间距近似序参量对 p 的导数。
        slope_index = int(np.argmax(slopes))  # 找出序参量斜率最大的网格位置。
        # 记录每种拓扑的两种网格估计和易感性峰是否在边界。
        estimate_rows.append(
            {
                "topology": topology,
                "effective_pc": float(ordered.loc[chi_index, "p"]),  # 市场阶段以池化易感性峰位作为有效临界概率。
                "pc_criterion": "pooled susceptibility peak, largest cluster removed",
                "chi_peak": float(ordered.loc[chi_index, "susceptibility_pooled"]),
                "pc_max_order_parameter_slope": float(
                    ordered.loc[slope_index, "p"]
                ),
                "max_order_parameter_slope": float(slopes[slope_index]),
                "chi_peak_at_grid_boundary": bool(
                    chi_index in (0, len(ordered) - 1)
                ),
            }
        )
    return samples, summary, pd.DataFrame(estimate_rows)  # 把每个拓扑的临界估计汇为第三张结果表。


def run_network_market_comparison(
    config: NetworkAnalysisConfig,
    ensemble: dict[str, list[GraphRealization]],
    estimates: pd.DataFrame,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 为网络拓扑压力测试按各拓扑相对有效临界点生成固定配置的逐笔交易。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 保存配置结构，并将逐笔收益传给 summarize_network_market() 和尾部分析。
    # 输入
    #     config 为 NetworkAnalysisConfig，提供拓扑、相对概率、交易设置、样本量和种子。
    #     ensemble 为按拓扑和网络重复组织的图实现。
    #     estimates 为 run_network_criticality_scan() 返回的临界估计表，需要 topology 和 effective_pc。
    #     show_progress 默认 True，按场景中的网络重复显示进度。
    # 输出依次为
    #     逐笔需求、收益、活跃团簇和所属配置的交易表。
    #     逐占据配置的网络团簇结构表。
    return_frames: list[pd.DataFrame] = []
    structure_rows: list[dict[str, object]] = []
    total = len(config.topologies) * len(config.relative_p_values) * config.network_replicates  # 进度总数是各拓扑、相对概率和网络重复的组合数。
    progress = tqdm(total=total, desc="network markets", unit="replicate", disable=not show_progress)  # 每完成一份网络上的所有交易配置推进一次。
    pc_lookup = estimates.set_index("topology")["effective_pc"].to_dict()  # 按拓扑索引易感性峰定义的有效临界点。
    for topology_index, topology in enumerate(config.topologies):  # 依次生成不同拓扑的压力测试市场数据。
        effective_pc = float(pc_lookup[topology])  # 取得当前拓扑用于交易扫描的参照概率。
        for ratio_index, relative_p in enumerate(config.relative_p_values):  # 对每个拓扑扫描相对自身有效临界点的概率比。
            p = min(1.0, effective_pc * relative_p)  # 把相对概率换成实际占据概率，上限为 1。
            for network_replicate, realization in enumerate(ensemble[topology]):  # 复用扫描用的各网络，分别派生随机数流。
                # 为当前网络和市场参数组合派生占据随机流。
                occupancy_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 120, topology_index, ratio_index, network_replicate]
                    )
                )
                # 为同一组合派生独立的团簇交易随机流。
                action_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 121, topology_index, ratio_index, network_replicate]
                    )
                )
                demand_blocks = []
                active_blocks = []
                for sample in range(config.market_configurations_per_network):  # 每份网络在该市场参数下抽取多套占据状态。
                    occupied = occupancy_rng.random(config.node_count) < p  # 初始网络固定，各配置独立抽取节点占据状态。
                    clusters = identify_graph_clusters(realization.adjacency, occupied)  # 确定当前配置中的占据团簇及其规模。
                    # 记录每份配置的团簇结构，供交易样本追溯。
                    structure_rows.append(
                        {
                            "topology": topology,
                            "relative_p": relative_p,
                            "p": p,
                            "network_replicate": network_replicate,
                            "configuration": sample,
                            "occupied_count": clusters.occupied_count,
                            "cluster_count": clusters.cluster_count,
                            "largest_size": clusters.largest_size,
                            "largest_occupied_fraction": clusters.largest_occupied_fraction,
                        }
                    )
                    # 按当前团簇规模批量抽取需求和活跃团簇数。
                    demands, active = sample_demands_and_active_counts(
                        clusters.sizes,
                        config.trades_per_configuration,
                        config.activity,
                        action_rng,
                    )
                    demand_blocks.append(demands)  # 按配置顺序保留需求区块。
                    active_blocks.append(active)  # 按相同顺序保留活跃团簇数。
                demands = np.concatenate(demand_blocks)  # 把该网络重复中的需求区块拼成一维序列。
                # 建立该网络重复的逐笔交易表。
                return_frames.append(
                    pd.DataFrame(
                        {
                            "topology": topology,
                            "relative_p": relative_p,
                            "p": p,
                            "activity": config.activity,
                            "network_replicate": network_replicate,
                            "step": np.arange(demands.size, dtype=np.int64),  # 交易步在本网络重复内从 0 连续编号。
                            "configuration": np.repeat(  # 将每个配置编号重复到其建模交易笔数。
                                np.arange(config.market_configurations_per_network, dtype=np.int64),  # 生成从零开始的配置索引。
                                config.trades_per_configuration,
                            ),
                            "demand": demands,
                            "active_cluster_count": np.concatenate(active_blocks),  # 按配置顺序对齐并拼接逐笔活跃团簇数。
                            "return": demands.astype(float) / config.market_depth,  # 逐笔以市场深度归一超额需求得到对数收益。
                        }
                    )
                )
                progress.update(1)  # 本网络重复的所有交易已生成。
    progress.close()  # 关闭网络交易进度。
    return pd.concat(return_frames, ignore_index=True), pd.DataFrame(structure_rows)  # 合并所有场景的交易表，并将配置列表转换成独立结构表。


def summarize_network_market(
    returns: pd.DataFrame, config: NetworkAnalysisConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 为网络拓扑压力测试按场景池化标准化收益，汇总池化和网络重复统计。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 将标准化收益、池化和重复统计用于尾拟合与拓扑比较。
    # 输入
    #     returns 为 run_network_market_comparison() 的逐笔交易表，含参数、网络重复、收益和活跃团簇数。
    #     config 为 NetworkAnalysisConfig，使用极端事件阈值。
    # 输出依次为
    #     新增 z 的交易表副本。
    #     按场景池化的统计量和网络重复之间的样本标准差。
    #     按场景与网络重复拆分的统计量。
    data = returns.copy()  # 复制交易表以保留调用方的原始输入。
    data["z"] = np.nan
    overall_rows = []
    replicate_rows = []
    keys = ["topology", "relative_p", "p", "activity"]
    for key, group in data.groupby(keys, sort=False):  # 按拓扑、实际及相对概率、活跃率合并全部网络重复。
        z, mean, standard_deviation = standardize_returns(group["return"].to_numpy())  # 用组内池化均值与样本标准差统一计算 z。
        data.loc[group.index, "z"] = z
        parameters = dict(zip(keys, key, strict=True))  # 将分组键转换为输出记录的参数列。
        # 计算池化收益矩、极端概率和活跃团簇均值。
        row = parameters | return_statistics(
            group["return"].to_numpy(),
            z,
            config.extreme_thresholds,
            group["active_cluster_count"].to_numpy(),  # 取逐笔活跃团簇数，用于统计均值。
        )
        row["standardization_mean"] = mean
        row["standardization_std"] = standard_deviation
        overall_rows.append(row)  # 保存当前场景的池化统计。
        for replicate, replicate_group in group.groupby("network_replicate"):  # 再按网络实现拆分当前场景。
            # 按网络重复计算收益矩，同时沿用场景共同的 z。
            replicate_rows.append(
                parameters
                | {"network_replicate": int(replicate)}
                | return_statistics(
                    replicate_group["return"].to_numpy(),
                    data.loc[replicate_group.index, "z"].to_numpy(),  # 从原表取得该网络交易对应的共同标准化值。
                    config.extreme_thresholds,
                    replicate_group["active_cluster_count"].to_numpy(),
                )
            )
    overall = pd.DataFrame(overall_rows)  # 将各场景池化结果整理为表。
    replicates = pd.DataFrame(replicate_rows)  # 将各场景内的网络重复结果整理为表。
    # 汇总场景内网络重复之间的峰度和极端概率的样本标准差。
    dispersion = (
        replicates.groupby(keys, as_index=False)
        .agg(
            replicate_kurtosis_sd=("return_excess_kurtosis", "std"),  # 计算重复层超额峰度的样本标准差。
            replicate_p_abs_z_gt_5_sd=("p_abs_z_gt_5", "std"),  # 计算绝对 z 大于 5 的概率在网络重复之间的标准差。
            replicate_negative_crash_probability_sd=(  # 计算 z 小于 −5 的概率在网络重复之间的标准差。
                "negative_crash_probability",
                "std",
            ),
        )
    )
    overall = overall.merge(dispersion, on=keys, validate="one_to_one")  # 将重复间离散度按参数组合一对一合并到池化表。
    return data, overall, replicates
