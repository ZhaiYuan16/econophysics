"""Cluster identification for square-lattice site percolation."""
# 将二维格点的占据状态划分为四邻域团簇，统计团簇规模及其是否贯通或绕行。
# 对外入口 identify_lattice_clusters() 按边界类型分派，并返回含团簇标签、规模和统计量的 ClusterConfiguration。
# 内部的 _open_clusters() 判断团簇是否贯通开放边界，_periodic_clusters() 检测周期边界下的绕行闭环。
#
# 在项目中，本模块把格点占据状态转成下游可用的团簇配置，为渗流临界性扫描提供结构统计，
# 为 Cont-Bouchaud 交易和 SOC 模拟提供团簇规模或格点标签。

from __future__ import annotations  # 延后求值类型注解。

from dataclasses import dataclass  # 标准库数据类装饰器，为团簇配置生成构造方法等。
from typing import Literal  # 标准库类型标注，用字符串字面值限定 Boundary 的取值。

import numpy as np  # 第三方数组库，用于存储格点标签、统计规模和维护并查集。
from scipy import ndimage  # SciPy 的多维数组处理模块，用于标记开放边界下的连通团簇。

Boundary = Literal["open", "periodic"]


@dataclass(frozen=True)  # 冻结格点团簇配置字段的绑定。
class ClusterConfiguration:
    """A labelled occupied lattice and its cluster-level observables.

    Positive labels are consecutive cluster identifiers; zero denotes an
    unoccupied site. ``spans`` means open-boundary spanning or periodic
    wrapping, according to ``boundary``.
    """
    # 某一构型生成团簇后的团簇特征记录。
    # 由 _empty_configuration()、_open_clusters() 和 _periodic_clusters() 构造。
    labels: np.ndarray  # 与格点同形，0 表示未占据，正整数是团簇编号。
    sizes: np.ndarray  # 一维规模数组，编号 k 的团簇占据 sizes[k - 1] 个格点。
    occupied_count: int  # 全部团簇的占据格点总数。
    largest_label: int  # 最大团簇的编号，并列时取最小编号，空配置记为 0。
    largest_size: int  # 最大团簇的格点数，空配置记为 0。
    spans: bool  # 是否有团簇贯通开放边界或在周期边界绕行。
    spanning_labels: tuple[int, ...]  # 所有贯通或绕行团簇的编号，按边界类型解释。
    boundary: Boundary  # open 为开放边界，periodic 为周期边界。
    horizontal_spanning_labels: tuple[int, ...] = ()  # 开放边界下同时接触左右边界的团簇编号。
    vertical_spanning_labels: tuple[int, ...] = ()  # 开放边界下同时接触上下边界的团簇编号。

    @property
    def cluster_count(self) -> int:
        # 统计当前配置的团簇总数。
        # 由 src/local_market/cb_model.py 的 sample_demands_and_action_once() 确定抽样次数，
        # 由 src/local_market/cb_model.py 的 simulate_quenched() 和 simulate_annealed() 记录各交易步的团簇数。
        # 由 src/local_market/statistics.py 的 finite_cluster_sizes() 判断空配置并为团簇筛选构造掩码。
        # 由 src/local_market/experiments.py 的 run_criticality_scan() 和 run_market_ensemble() 记录配置结构。
        # 由 scripts/simulation/run_small_validation.py 的 main() 按占据概率 p 汇总团簇数，
        # 由 scripts/analysis/audit_market_time_series.py 的 main() 核对重建配置并记录固定格点的团簇数。
        # 由 defense/cb_experiment.py 的 occupancy_block() 内部 compute() 记录格点样本的团簇数。
        # 输入 self 为当前 ClusterConfiguration，读取其 sizes。
        # 输出整数团簇数。
        return int(self.sizes.size)  # 将规模数组的长度转为 Python 整数。

    @property
    def largest_occupied_fraction(self) -> float:
        # 计算最大团簇占全部已占据格点的比例。
        # 由 src/local_market/experiments.py 的 run_criticality_scan() 和 run_market_ensemble() 记录各配置的比例。
        # 由 scripts/analysis/audit_market_time_series.py 的 main() 记录重建格点配置的最大团簇占据比例。
        # 由 scripts/simulation/run_small_validation.py 的 main() 按占据概率 p 汇总该比例。
        # 输入 self 为当前 ClusterConfiguration，读取其 largest_size 和 occupied_count。
        # 输出比例，空占据配置返回 0.0。
        if self.occupied_count == 0:  # 无占据格点时避免用零作为比例分母。
            return 0.0
        return self.largest_size / self.occupied_count

    @property
    def p_infinity(self) -> float:
        # 计算最大团簇占格点总数的比例 P∞。
        # 由 src/local_market/criticality_revision.py 的 scan() 和 src/local_market/experiments.py 的 run_criticality_scan() 记录序参量样本。
        # 由 scripts/figures/plot_correlation_length_preview.py 的模块顶层循环汇总序参量。
        # 由 defense/cb_experiment.py 的 occupancy_block() 内部 compute() 记录格点样本的序参量。
        # 输入 self 为当前 ClusterConfiguration，读取其 largest_size 和 labels 的格点总数。
        # 输出比例，分母包含已占据和未占据的格点。
        return self.largest_size / self.labels.size  # 包括未占据的格点，以全部格点作为分母。

    @property
    def spans_horizontal(self) -> bool:
        # 判断开放边界下是否有团簇贯通左右两侧。
        # 由 src/local_market/criticality_revision.py 的 scan() 记录水平贯通事件。
        # 由 scripts/figures/plot_correlation_length_preview.py 的模块顶层循环核对并汇总水平贯通结果。
        # 由 defense/cb_experiment.py 的 occupancy_block() 内部 compute() 记录开放格点样本的水平贯通事件。
        # 输入 self 为当前 ClusterConfiguration，读取其 boundary 和 horizontal_spanning_labels。
        # 输出是否贯通的布尔值，周期边界抛出 ValueError。
        if self.boundary != "open":  # 左右贯通只对开放边界定义。
            raise ValueError("horizontal open-boundary spanning is undefined for periodic lattices")
        return bool(self.horizontal_spanning_labels)  # 有水平贯通团簇编号时返回 True。

    @property
    def spans_vertical(self) -> bool:
        # 判断开放边界下是否有团簇贯通上下两侧。
        # 由 src/local_market/criticality_revision.py 的 scan() 记录垂直贯通事件。
        # 输入 self 为当前 ClusterConfiguration，读取其 boundary 和 vertical_spanning_labels。
        # 输出是否贯通的布尔值，周期边界抛出 ValueError。
        if self.boundary != "open":  # 上下贯通只对开放边界定义。
            raise ValueError("vertical open-boundary spanning is undefined for periodic lattices")
        return bool(self.vertical_spanning_labels)  # 有垂直贯通团簇编号时返回 True。


def _empty_configuration(shape: tuple[int, int], boundary: Boundary) -> ClusterConfiguration:
    # 构造完全未占据的格点配置。
    # 由 _open_clusters() 和 _periodic_clusters() 在找不到占据团簇时调用，统一返回空结果。
    # 输入
    #     shape 为格点阵列的行数和列数。
    #     boundary 为结果中保留的边界类型。
    # 输出 ClusterConfiguration，标签全为 0，团簇规模数组为空，各计数为 0。
    # 构造空团簇配置。
    return ClusterConfiguration(
        labels=np.zeros(shape, dtype=np.int32),  # 标签全部置为 0，表示格点均未占据。
        sizes=np.empty(0, dtype=np.int64),  # 规模数组长度为 0，表示没有团簇。
        occupied_count=0,
        largest_label=0,
        largest_size=0,
        spans=False,
        spanning_labels=(),
        boundary=boundary,
    )


def _open_clusters(occupied: np.ndarray) -> ClusterConfiguration:
    # 在开放边界下识别四邻域团簇并判定水平、垂直贯通。
    # 由 identify_lattice_clusters() 在 open 分支调用，得到开放格点的团簇统计。
    # 输入 occupied 为二维布尔占据数组，True 表示该格点已占据。
    # 输出 ClusterConfiguration，包含团簇标签、规模和两个方向的贯通编号。
    
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8) # np.array 构造十字形四邻域。
    labels, count = ndimage.label(occupied, structure=structure)    # ndimage.label 标记连通分量并统计团簇数。
    if count == 0:
        return _empty_configuration(occupied.shape, "open")  # 构造开放边界的空配置。

    # ravel 展平标签，bincount 按编号计数，去掉未占据的 0 号并转换为 int64。
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(np.int64)
    top = set(labels[0, labels[0] > 0].tolist())  # 收集上边界的正团簇编号并去重。
    bottom = set(labels[-1, labels[-1] > 0].tolist())  # 收集下边界的正团簇编号并去重。
    left = set(labels[labels[:, 0] > 0, 0].tolist())  # 收集左边界的正团簇编号并去重。
    right = set(labels[labels[:, -1] > 0, -1].tolist())  # 收集右边界的正团簇编号并去重。
    horizontal = tuple(sorted(left & right))  # 取左右边界编号的交集，按编号顺序记录水平贯通团簇。
    vertical = tuple(sorted(top & bottom))  # 取上下边界编号的交集，按编号顺序记录垂直贯通团簇。
    spanning = tuple(sorted(set(horizontal) | set(vertical)))  # 合并两个方向的贯通团簇，去重后按编号排序。
    largest_label = int(np.argmax(sizes)) + 1  # 找到最大规模的首个索引并转为从 1 开始的编号。
    # ClusterConfiguration 汇总统计量。
    return ClusterConfiguration(
        labels=labels.astype(np.int32, copy=False), # astype 统一标签类型。
        sizes=sizes,
        occupied_count=int(sizes.sum()),    # sum 汇总占据数。
        largest_label=largest_label,
        largest_size=int(sizes[largest_label - 1]),  # 将正标签转换为规模数组下标并读取最大规模。
        spans=bool(spanning),           # bool 判断是否贯通。
        spanning_labels=spanning,
        boundary="open",
        horizontal_spanning_labels=horizontal,
        vertical_spanning_labels=vertical,
    )


def _periodic_clusters(occupied: np.ndarray) -> ClusterConfiguration:
    """Label periodic clusters and detect non-contractible wrapping cycles."""
    # 用带位移的并查集识别周期边界下的四邻域团簇和绕行闭环。
    # 由 identify_lattice_clusters() 在 periodic 分支调用，得到周期格点的团簇统计。
    # 输入 occupied 为二维布尔占据数组，True 表示该格点已占据。
    # 输出 ClusterConfiguration，包含团簇标签、规模和绕行团簇编号。
    rows, cols = occupied.shape
    flat_occupied = occupied.ravel()  # 按行展平占据数组，便于按一维编号查找格点。
    occupied_nodes = np.flatnonzero(flat_occupied)  # 取出值为 True 的格点编号。
    if occupied_nodes.size == 0:
        return _empty_configuration(occupied.shape, "periodic")  # 构造周期边界的空配置。

    total = rows * cols
    parent = np.arange(total, dtype=np.int64)  # 初始时每个格点的父节点都是自己。
    rank = np.zeros(total, dtype=np.int8)  # 将秩置为 0，供后续选择合并方向。
    # displacement from a node to its parent in the unwrapped covering lattice
    dr = np.zeros(total, dtype=np.int64)  # 展开格点上，节点相对父节点的行坐标差，初始为 0。
    dc = np.zeros(total, dtype=np.int64)  # 展开格点上，节点相对父节点的列坐标差，初始为 0。
    wraps = np.zeros(total, dtype=bool)  # 尚未发现绕行闭环，各标记初始为 False。

    def find(node: int) -> tuple[int, int, int]:
        # 查找根节点并压缩路径，同时累计节点相对根的位移。
        # 由 _periodic_clusters() 内部的 union() 比较边两端所属团簇及位移。
        # 由 _periodic_clusters() 的主流程收集团簇根和读取绕行标记，_periodic_clusters() 内部的 find() 也递归查找父节点。
        # 输入 node 为格点的一维编号，读取和更新外层 parent、dr、dc。
        # 输出依次为
        #     root，团簇的根节点编号。
        #     行位移，节点行坐标减去根节点行坐标。
        #     列位移，节点列坐标减去根节点列坐标。
        if parent[node] == node:
            return node, 0, 0
        root, root_dr, root_dc = find(int(parent[node]))  # 转为整数索引，递归取得父节点的根与累计位移。
        dr[node] += root_dr  # 沿递归路径累加行位移，改为相对根节点。
        dc[node] += root_dc  # 同样累加列位移，保持展开格点的二维位置关系。
        parent[node] = root  # 将节点直接连到根，缩短后续查找路径。
        return root, int(dr[node]), int(dc[node])  # 将累计行、列位移转为 Python 整数。

    def union(u: int, v: int, edge_dr: int, edge_dc: int) -> None:
        # 加入一条四邻域边，合并两个团簇或检测同一团簇中的绕行闭环。
        # 由 _periodic_clusters() 遍历占据格点时调用，处理每条向右或向下的占据邻接边。
        # 输入
        #     u 为起点格点的一维编号。
        #     v 为终点格点的一维编号。
        #     edge_dr 为展开格点上 v 相对 u 的行位移。
        #     edge_dc 为展开格点上 v 相对 u 的列位移。
        # 输出 None，通过更新外层 parent、rank、dr、dc 和 wraps 记录连接与绕行状态。
        root_u, u_dr, u_dc = find(u)  # 查找起点所属的根，取得起点相对该根的行、列位移。
        root_v, v_dr, v_dc = find(v)  # 查找终点所属的根，取得终点相对该根的行、列位移。
        if root_u == root_v:
            # 位移与新边不一致时，闭环在展开格点上有非零净位移，说明团簇绕行。
            if (v_dr - u_dr, v_dc - u_dc) != (edge_dr, edge_dc):
                wraps[root_u] = True  # 已有闭环的净位移非零，标记这个连通团簇绕行。
            return

        # 新边给出展开格点上从 root_u 到 root_v 的位移。
        rel_dr = edge_dr + u_dr - v_dr  # 由新边和两端节点相对各自根的位移反推出两根的行坐标差。
        rel_dc = edge_dc + u_dc - v_dc  # 用相同的位移关系得到两根的列坐标差。
        combined_wrap = bool(wraps[root_u] or wraps[root_v])  # 合并后继承任一原团簇的绕行状态。
        # 将较低秩的根接到较高秩的根，按连接方向保存根之间的位移。
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
            dr[root_u] = -rel_dr
            dc[root_u] = -rel_dc
            wraps[root_v] = combined_wrap
        else:
            parent[root_v] = root_u
            dr[root_v] = rel_dr
            dc[root_v] = rel_dc
            wraps[root_u] = combined_wrap
            if rank[root_u] == rank[root_v]:
                rank[root_u] += 1

    # 遍历格点的行和列。
    for row in range(rows):
        for col in range(cols):
            u = row * cols + col  # 把格点行列坐标换成行优先的一维索引。
            if not flat_occupied[u]:
                continue
            right = row * cols + ((col + 1) % cols)  # 取右邻居编号，最右列接回第一列。
            down = ((row + 1) % rows) * cols + col  # 取下邻居编号，最下行接回第一行。
            if flat_occupied[right]:
                union(u, right, 0, 1)  # 加入向右的占据邻接边，合并团簇或检测绕行。
            if flat_occupied[down]:
                union(u, down, 1, 0)  # 加入向下的占据邻接边，合并团簇或检测绕行。

    roots = np.array([find(int(node))[0] for node in occupied_nodes], dtype=np.int64)   # find 取每个占据点的根，np.array 将根编号收集为整数数组。
    
    unique_roots, inverse, sizes = np.unique(roots, return_inverse=True, return_counts=True)    # unique 按根编号分组，返回各团簇的根、每个占据点的分组索引和各组规模。
    labels_flat = np.zeros(total, dtype=np.int32)   # zeros 保留空位的 0 号标签
    labels_flat[occupied_nodes] = inverse.astype(np.int32) + 1  # astype 转换分组索引类型后加 1 作为团簇编号。

    spanning_labels = tuple(    # tuple 保存有绕行的团簇编号。
        index + 1
        for index, root in enumerate(unique_roots)  # 枚举各团簇的根及其排序索引。
        if wraps[find(int(root))[0]]    # 按根读取绕行标记，只保留发生绕行的团簇。
    )
    sizes = sizes.astype(np.int64)  # 将各团簇的规模统一为 int64。
    largest_label = int(np.argmax(sizes)) + 1  # 取最大规模的首个索引并转为从 1 开始的团簇编号。
    # ClusterConfiguration 汇总周期边界结果
    return ClusterConfiguration(
        labels=labels_flat.reshape(occupied.shape), # reshape 恢复格点形状
        sizes=sizes,
        occupied_count=int(occupied_nodes.size),  # 用已占据索引数组的长度记录占据格点数。
        largest_label=largest_label,
        largest_size=int(sizes[largest_label - 1]),  # 从正标签定位规模数组中的最大团簇。
        spans=bool(spanning_labels),        # bool 判断是否有绕行。
        spanning_labels=spanning_labels,
        boundary="periodic",
    )


def identify_lattice_clusters(
    occupied: np.ndarray, boundary: Boundary = "open"
) -> ClusterConfiguration:
    """Identify four-neighbour clusters in a two-dimensional Boolean array."""
    # 将格点占据状态划分为四邻域团簇，并按边界类型统计贯通或绕行。
    # 由 src/local_market/experiments.py 的 run_baseline() 内部 make_configuration() 生成基线交易配置，
    # 由 src/local_market/experiments.py 的 run_criticality_scan() 测量渗流结构，run_market_ensemble() 为各活跃率准备共用配置，
    # run_quenched_market_trajectories() 为临界格点上的固定配置交易准备团簇。
    # 由 src/local_market/criticality_revision.py 的 scan() 计算开放格点的贯通和规模统计。
    # 由 src/local_market/soc_research.py 的 simulate() 在每轮交易前识别当前团簇，供抽样和耗散使用。
    # 由 scripts/simulation/run_small_validation.py 的 main() 为每份配置记录结构并模拟交易，
    # 由 scripts/analysis/audit_market_time_series.py 的 main() 重建市场系综配置并生成对照轨迹。
    # 由 scripts/figures/plot_correlation_length_preview.py 的模块顶层循环重建配置并计算相关长度。
    # 由 defense/cb_experiment.py 的 occupancy_block() 内部 compute() 为格点样本测量渗流统计。
    # 由 src/local_market/__init__.py 在包级重新导出。
    # 输入
    #     occupied 为可转换成二维布尔数组的格点占据状态，行数和列数都须大于 0。
    #     boundary 为边界类型，默认 open，periodic 表示相对两侧相接。
    # 输出 ClusterConfiguration，包含标签、规模和对应边界的贯通或绕行结果，非法输入抛出 ValueError。

    occupied = np.asarray(occupied, dtype=bool) # asarray 转为布尔占据数组
    if occupied.ndim != 2 or min(occupied.shape) == 0:  # min 检查较短的一边是否为空。
        raise ValueError("occupied must be a non-empty two-dimensional array")
    if boundary == "open":
        return _open_clusters(occupied)  # 标记开放边界下的团簇并检查对边贯通。
    if boundary == "periodic":
        return _periodic_clusters(occupied)  # 合并周期邻居并检测团簇绕行。
    raise ValueError("boundary must be 'open' or 'periodic'")  # 拒绝两种已实现边界之外的输入。
