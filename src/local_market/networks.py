"""Network and occupancy generators used by the market models."""
# 生成格点占据样本和匹配规模的网络，识别占据团簇，并测量占据前的整图结构。
# sample_lattice_occupancy() 抽取正方格点占据状态。generate_matched_graph()、identify_graph_clusters() 和 graph_structure_metrics() 分别构图、划分团簇和统计整图。
#
# 格点样本交给基线交易、渗流临界性质实验和市场收益率的影响因素实验。后三个函数为网络拓扑压力测试提供图结构和占据团簇。

from __future__ import annotations  # 延后求值类型注解。

from dataclasses import dataclass  # 标准库数据类装饰器，生成图和团簇结果的构造方法等。
from typing import Literal  # 标准库类型标注，限定支持的拓扑名称。

import networkx as nx  # NetworkX 用于生成图并计算原图的连通性、聚类和最短路。
import numpy as np  # NumPy 用于抽样、标记节点和汇总团簇规模。
from scipy import sparse  # SciPy 稀疏矩阵模块，用于按节点索引保存和切片邻接矩阵。
from scipy.sparse import csgraph  # SciPy 稀疏图算法，用于识别占据诱导子图的连通分量。

Topology = Literal["lattice", "er", "ws", "ba"]


@dataclass(frozen=True)  # 冻结图结果的字段绑定。
class GraphRealization:
    topology: Topology  # 拓扑类型，取 lattice、er、ws 或 ba。
    graph: nx.Graph  # 节点编号为 0 到 N−1 的 NetworkX 无向图。
    adjacency: sparse.csr_matrix  # 按该节点顺序排列的 N×N CSR 邻接矩阵。
    seed: int  # 生成该图时传入的种子。


@dataclass(frozen=True)  # 冻结占据团簇结果的字段绑定。
class GraphClusters:
    labels: np.ndarray  # 长度 N，沿原图编号排列，0 表示未占据，正整数为团簇编号。
    sizes: np.ndarray  # 长度 K，索引 label−1 对应正编号 label 的团簇规模。
    occupied_count: int  # 占据节点总数。
    largest_label: int  # 最大团簇编号，规模并列时取最先编号者，空占据时为 0。
    largest_size: int  # 最大团簇的节点数，空占据时为 0。

    @property
    def cluster_count(self) -> int:
        # 读取当前图的占据团簇总数。
        # 引用
        #     src/local_market/network_analysis.py 的 run_network_criticality_scan() 和 run_network_market_comparison() 记录各占据样本的团簇数。
        #     defense/cb_experiment.py 的 occupancy_block() 内部 compute() 记录网络样本的团簇数。
        # 输入
        #     self 为 GraphClusters 实例，读取 sizes。
        # 输出 整数团簇数，空占据时为 0。
        return int(self.sizes.size)

    @property
    def largest_occupied_fraction(self) -> float:
        # 计算最大团簇占已占据节点的比例。
        # 引用
        #     src/local_market/network_analysis.py 的 run_network_criticality_scan() 和 run_network_market_comparison() 记录最大团簇占已占据节点的比例。
        # 输入
        #     self 为 GraphClusters 实例，读取 largest_size 和 occupied_count。
        # 输出 比例，空占据时为 0.0。
        if self.occupied_count == 0:
            return 0.0
        return self.largest_size / self.occupied_count

    @property
    def p_infinity(self) -> float:
        # 计算最大团簇占原图全体节点的比例 P∞。
        # 引用
        #     src/local_market/network_analysis.py 的 run_network_criticality_scan() 用它汇总各占据概率的序参量。
        #     defense/cb_experiment.py 的 occupancy_block() 内部 compute() 记录网络样本的 P∞。
        # 输入
        #     self 为 GraphClusters 实例，读取 largest_size 和 labels 的长度。
        # 输出 比例，原图须至少有一个节点。空占据时返回 0.0。
        return self.largest_size / self.labels.size


def sample_lattice_occupancy(
    length: int, p: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample occupied sites on an ``length x length`` square lattice."""
    # 独立地为正方格点的每个位置抽取占据状态，供后续团簇识别使用。
    # 引用
    #     src/local_market/experiments.py 的 run_baseline() 内部 make_configuration() 为基线交易准备固定配置或逐步更新配置。
    #     src/local_market/experiments.py 的 run_criticality_scan() 为各尺寸和占据概率生成渗流样本。
    #     src/local_market/experiments.py 的 run_market_ensemble() 为不同活跃率生成共用的格点样本。
    #     src/local_market/experiments.py 的 run_quenched_market_trajectories() 为临界格点上的固定配置交易生成一份占据状态。
    #     src/local_market/criticality_revision.py 的 scan() 为开放边界渗流临界性质实验生成样本。
    #     scripts/simulation/run_small_validation.py 的 main() 为三点校验生成格点，随后统计团簇并模拟交易。
    #     scripts/analysis/audit_market_time_series.py 的 main() 重建市场系综的占据状态，并为固定配置的对照交易轨迹抽样。
    #     scripts/figures/plot_correlation_length_preview.py 的模块顶层循环为相关长度预览抽取开放格点样本。
    #     defense/cb_experiment.py 的 occupancy_block() 内部 compute() 为格点答辩演示抽样。
    #     src/local_market/__init__.py 将本函数重新导出为 local_market.sample_lattice_occupancy。
    # 输入
    #     length 为正整数，表示格点的行数和列数。
    #     p 为单个格点的占据概率，范围为 [0, 1]。
    #     rng 为 NumPy 随机数生成器，抽样会推进其状态。
    # 输出形状为 (length, length) 的布尔数组，True 表示占据。边长非正或 p 越界时抛出 ValueError。
    if length <= 0:  # 拒绝无法定义非空方格的边长。
        raise ValueError("length must be positive")
    if not 0.0 <= p <= 1.0:  # 占据概率的两个边界值均可用于抽样。
        raise ValueError("p must lie in [0, 1]")
    # 各位置独立从 [0, 1) 抽取均匀随机数，小于 p 时标记为已占据。
    return rng.random((length, length)) < p


def generate_matched_graph(
    topology: Topology,
    node_count: int,
    mean_degree: int,
    seed: int,
    ws_rewiring_probability: float = 0.1,
) -> GraphRealization:
    """Generate a graph with approximately matched size and mean degree."""
    # 为网络拓扑压力测试生成同节点数、近似匹配平均度的无向图。
    # 引用
    #     src/local_market/network_analysis.py 的 build_network_ensemble() 为各拓扑建立多份图样本。
    #     defense/cb_experiment.py 的 network_adjacency() 内部 generate() 构造用于答辩演示的邻接矩阵。
    # 输入
    #     topology 为格点、ER、WS 或 BA 拓扑。
    #     node_count 为正节点数 N，格点要求 N 为完全平方数，ER 公式要求 N 大于 1。
    #     mean_degree 为正的目标平均度。周期方格的邻接关系由格点本身决定，WS 和 BA 要求偶数，随机图生成器还有各自的规模约束。
    #     seed 为随机网络生成种子。
    #     ws_rewiring_probability 为 WS 边重连概率，默认 0.1。
    # 输出 GraphRealization，含拓扑、原图、节点对齐的邻接矩阵和种子。
    if node_count <= 0 or mean_degree <= 0:
        raise ValueError("node_count and mean_degree must be positive")
    if topology == "lattice":
        length = int(round(np.sqrt(node_count)))  # 以节点数平方根恢复正方格边长。
        if length * length != node_count:  # 拒绝无法组成正方格的节点数。
            raise ValueError("lattice node_count must be a perfect square")
        # 构造周期四邻域方格，按坐标排序重新编号以对齐矩阵索引。
        graph = nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(length, length, periodic=True), ordering="sorted"
        )
    elif topology == "er":
        # 在 ER 图中用目标平均度除以 N−1 得到每条候选边的连边概率。
        graph = nx.fast_gnp_random_graph(
            node_count, mean_degree / (node_count - 1), seed=seed
        )
    elif topology == "ws":
        if mean_degree % 2:  # WS 近邻环按两侧各连一半邻居，要求偶数目标度。
            raise ValueError("Watts-Strogatz mean_degree must be even")
        # 先以目标偶数度连近邻环，再按给定概率重连。
        graph = nx.watts_strogatz_graph(
            node_count, mean_degree, ws_rewiring_probability, seed=seed
        )
    elif topology == "ba":
        if mean_degree % 2:  # 本实验用目标均度的一半作为 BA 新节点的连边数。
            raise ValueError("BA target mean_degree must be even")
        # 每个 BA 新节点加 mean_degree/2 条边，使大图平均度接近目标值。
        graph = nx.barabasi_albert_graph(node_count, mean_degree // 2, seed=seed)
    else:
        raise ValueError("unknown topology")  # 拒绝实验支持范围外的拓扑。
    # 显式指定节点顺序，将原图编码为 CSR 邻接矩阵。
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=range(node_count), format="csr", dtype=np.uint8
    )
    return GraphRealization(topology, graph, sparse.csr_matrix(adjacency), seed)  # 统一封装为 csr_matrix，保持下游的行列切片接口。


def identify_graph_clusters(
    adjacency: sparse.csr_matrix, occupied: np.ndarray
) -> GraphClusters:
    """Find components in the occupied-node induced subgraph."""
    # 从无向图的占据诱导子图识别团簇，并将标签映射回原节点。
    # 引用
    #     src/local_market/network_analysis.py 的 run_network_criticality_scan() 记录团簇统计，run_network_market_comparison() 用团簇规模抽样交易。
    #     defense/cb_experiment.py 的 occupancy_block() 内部 compute() 为演示生成网络团簇和规模。
    # 输入
    #     adjacency 为形状 (N, N) 的 CSR 邻接矩阵，行列顺序一致。
    #     occupied 为长度 N 的一维节点占据掩码，可转换成布尔值。
    # 输出 GraphClusters，包含标签、规模、占据数和最大团簇。掩码形状不符时抛出 ValueError。
    occupied = np.asarray(occupied, dtype=bool)  # 统一转为布尔占据掩码，便于提取节点编号。
    if occupied.ndim != 1 or occupied.size != adjacency.shape[0]:  # 保证掩码按原图的一维节点编号对齐。
        raise ValueError("occupied must match the adjacency node dimension")
    occupied_nodes = np.flatnonzero(occupied)  # 提取原图中所有占据节点的编号。
    labels = np.zeros(occupied.size, dtype=np.int32)  # 预留全图标签，未占据节点保留 0。
    if occupied_nodes.size == 0:
        return GraphClusters(labels, np.empty(0, dtype=np.int64), 0, 0, 0)  # 空占据时保留全零标签和空规模数组。
    induced = adjacency[occupied_nodes][:, occupied_nodes]  # 只保留占据节点之间的原图边。
    # 给占据诱导子图按无向连通分量编号。
    count, component_labels = csgraph.connected_components(
        induced, directed=False, return_labels=True
    )
    labels[occupied_nodes] = component_labels.astype(np.int32) + 1  # 将零基分量号加一并写回原图位置。
    sizes = np.bincount(component_labels, minlength=count).astype(np.int64)  # 按零基分量号统计占据节点数。
    largest_label = int(np.argmax(sizes)) + 1  # 将最大规模的数组下标转为正编号。
    # 打包全图标签和占据团簇的汇总统计。
    return GraphClusters(
        labels=labels,
        sizes=sizes,
        occupied_count=int(occupied_nodes.size),
        largest_label=largest_label,
        largest_size=int(sizes[largest_label - 1]),
    )


def graph_structure_metrics(
    realization: GraphRealization,
    path_sources: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Compute graph metrics, approximating path length from sampled sources."""
    # 测量占据前的网络结构，并从最大连通分量抽起点近似平均最短路。
    # 引用
    #     src/local_market/network_analysis.py 的 build_network_ensemble() 保存每份图的结构指标供网络拓扑压力测试比较。
    # 输入
    #     realization 为含至少一个节点的 GraphRealization，使用其原图。
    #     path_sources 为不放回抽取的起点数上限，通常为正整数且至多取分量节点数。
    #     rng 为用于抽取起点的 NumPy 随机数生成器。
    # 输出 一个字典，含点数、边数、均度、度样本标准差和最大度、平均聚类系数、最大分量比例、最短路估计和实际起点数。
    graph = realization.graph
    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=float)  # 沿原图节点逐个取度数，供全图均值、标准差和最大值共用。
    components = list(nx.connected_components(graph))  # 枚举原图连通分量，用节点集合表示。
    largest_nodes = max(components, key=len)  # 按节点数找最大分量。
    largest = graph.subgraph(largest_nodes)  # 限制路径计算只在最大分量中进行。
    # 在最大分量中不放回选择路径起点。
    sources = rng.choice(
        np.fromiter(largest_nodes, dtype=np.int64),
        size=min(path_sources, largest.number_of_nodes()),
        replace=False,
    )
    source_means = []
    for source in sources:  # 对每个抽样起点独立做一次路径计算。
        distances = nx.single_source_shortest_path_length(largest, int(source))  # 求该起点到分量内各节点的最短边数，自身距离为 0。
        source_means.append(sum(distances.values()) / max(1, len(distances) - 1))  # 分母排除起点自身，单节点分量给出 0。
    return {
        "node_count": graph.number_of_nodes(),  # 计数包含最大分量之外的节点。
        "edge_count": graph.number_of_edges(),  # 计数覆盖整张原图。
        "mean_degree": float(degrees.mean()),  # 以原图全部节点为分母。
        "degree_std": float(degrees.std(ddof=1)),  # 样本标准差使用 N−1 分母。
        "maximum_degree": int(degrees.max()),  # 取原图节点的最大度。
        "clustering_coefficient": float(nx.average_clustering(graph)),  # 计算整张图的平均局部聚类系数。
        "largest_connected_component_fraction": (
            largest.number_of_nodes() / graph.number_of_nodes()  # 最大分量的节点数除以原图节点数。
        ),
        "average_shortest_path_length_approx": float(np.mean(source_means)),  # 等权平均已抽起点的平均距离，未抽起点时为 NaN。
        "path_source_count": int(len(sources)),  # 记录实际用于近似的起点数。
    }
