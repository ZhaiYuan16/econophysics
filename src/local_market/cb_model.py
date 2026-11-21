"""Trading dynamics for the Cont-Bouchaud market model."""
# 按团簇规模对买卖行动加权，生成 Cont-Bouchaud 模型的超额需求与交易轨迹。
# 对外的交易模拟函数有 simulate_quenched() 和 simulate_annealed()，分别固定团簇配置和每步重建配置。
# 对外的抽样函数有 sample_demands_and_action_once() 和 sample_demands_and_active_counts()，
# 前者返回单步需求和各团簇行动，后者用固定团簇规模批量生成逐步需求和活跃团簇数。
#
# 在项目中，本模块把格点或网络的团簇接入交易模型，为市场基线、市场系综和网络比较生成交易数据，
# 并为 SOC 模拟判断是否触发耗散、哪些团簇与需求同向提供依据。

from __future__ import annotations  # 延后求值类型注解，用于声明数组、配置和返回对象类型。

from dataclasses import dataclass  # 标准库，自动生成结果数据类的构造方法等。
from typing import Callable  # 标准库类型标注，声明“无参数并返回团簇配置”的回调接口。

import numpy as np  # 第三方 NumPy 库，提供随机抽样、数组、矩阵乘法与累计和运算。

from .percolation import ClusterConfiguration  # 本地 src/local_market/percolation.py，保存格点团簇标签与规模。


@dataclass(frozen=True)  # 冻结交易结果字段的绑定。
class MarketSimulation:
    # 每个字段对应一次交易模拟的一项逐步记录。
    # 由 simulate_quenched() 或 simulate_annealed() 构造并返回给实验入口。
    demands: np.ndarray         # 长度 T，每步超额需求 D。
    returns: np.ndarray         # 长度 T，每步对数收益率 r = D/λ。
    log_prices: np.ndarray      # 长度 T+1，初始对数价格及每步交易后的对数价格。
    occupied_counts: np.ndarray # 长度 T，每步使用配置的占据节点数。
    cluster_counts: np.ndarray  # 长度 T，每步使用配置的团簇数。

def sample_demands_and_action_once(
    configuration: ClusterConfiguration, # 提供团簇数 K 和团簇大小 s。
    activity: float,                     # 团簇参与交易的概率 a。
    rng: np.random.Generator             # 外部随机数生成器。
) -> tuple[int, np.ndarray]:             # D = Σsαφα: 超额需求。actions: 每个团簇的行动数组，-1 表示卖出，0 表示不交易，+1 表示买入。
    """Sample excess demand and the number of active clusters per step."""
    # 抽取各团簇在一次交易中的行动，并按团簇规模加权计算超额需求。
    # 由 simulate_annealed() 每步调用，用于计算当前交易步的需求。
    # 由 src/local_market/soc_research.py 的 simulate() 逐轮调用，用需求判断是否触发耗散，用行动识别与需求同向的团簇。
    
    # 输入
    #     configuration 为团簇配置，读取其中按标签顺序排列的一维规模。
    #     activity 为团簇参与交易的概率 a，范围为 [0, 1]。
    #     rng 为抽取团簇行动的 NumPy 随机数生成器。
    # 输出依次为
    #     整数超额需求 D，空配置为 0。
    #     长度为团簇数的行动数组，与 configuration.sizes 对齐，依次编码卖出 −1、闲置 0、买入 +1。
    if not 0.0 <= activity <= 1.0:  # 活跃率须在合法概率范围内。
        raise ValueError("activity must lie in [0, 1]")
    cluster_count = configuration.cluster_count  # 读取团簇数，确定当步抽样次数。
    draws = rng.random(cluster_count)  # 为每个团簇抽取一个均匀随机数。
    actions = np.zeros(cluster_count, dtype=np.int8)  # 将每个团簇的行动初始化为不交易。
    actions[draws < activity / 2.0] = -1  # 随机数落在下端 a/2 区间的团簇卖出。
    actions[draws >= 1.0 - activity / 2.0] = 1  # 随机数落在上端 a/2 区间的团簇买入。

    if actions.shape != configuration.sizes.shape:  # 逐团簇的行动必须与团簇规模数组一一对应。
        raise ValueError("one action is required for each cluster")
    D = int(np.dot(configuration.sizes, actions))  # 按规模加权求超额需求，并转为 Python 整数。
    return D, actions


def sample_demands_and_active_counts(
    cluster_sizes: np.ndarray,      # 为一维团簇规模数组，每个元素对应一个团簇。
    steps: int,                     # 交易步数 T。
    activity: float,                # 团簇参与交易的概率 a。
    rng: np.random.Generator,       # 外部随机数生成器。
) -> tuple[np.ndarray, np.ndarray]: # demand，超额需求，数组长度T。active_counts: 当步非零行动的团簇数量，数组长度T。
    """Sample excess demand and the number of active clusters per step."""
    # 固定团簇规模，用矩阵运算同时抽取多个交易步的需求与活跃团簇数。
    # 由 simulate_quenched() 调用，为固定配置交易提供需求序列。
    # 由 src/local_market/experiments.py 的 run_market_ensemble() 为格点市场系综抽样，
    # 由 src/local_market/network_analysis.py 的 run_network_market_comparison() 为网络市场系综抽样。
    # 由 defense/cb_experiment.py 的 action_block() 内部 compute() 分批抽取交易需求和活跃团簇数。
    
    # 输入
    #     cluster_sizes 为一维正团簇规模数组，可为空。
    #     steps 为非负的交易步数 T。
    #     activity 为每步每个团簇参与交易的概率 a，范围为 [0, 1]。
    #     rng 为批量抽取各步团簇行动的 NumPy 随机数生成器。
    # 输出依次为
    #     长度为 steps 的超额需求数组，按团簇规模加权。
    #     长度为 steps 的活跃团簇数数组，逐步数非零行动。空规模时两个数组均为零。
    if steps < 0:  # 允许空交易序列，拒绝负步数。
        raise ValueError("steps must be non-negative")
    if not 0.0 <= activity <= 1.0:  # 活跃率须允许将概率均分到买入和卖出。
        raise ValueError("activity must lie in [0, 1]")
    sizes = np.asarray(cluster_sizes, dtype=np.int64)  # 将团簇规模转为整数数组，供校验和加权。
    if sizes.ndim != 1 or np.any(sizes <= 0):   # any 检查规模数组中是否有非正值。
        raise ValueError("cluster_sizes must be a one-dimensional positive array")
    
    draws = rng.random((steps, sizes.size))     # random 按每步每团簇独立抽样。
    actions = (draws >= 1.0 - activity / 2.0).astype(np.int8)   # 将买入位置记为 +1，其余位置暂记为 0。
    actions -= (draws < activity / 2.0).astype(np.int8)  # 将卖出位置减 1，记为 -1。
    demands = actions @ sizes       # 矩阵乘规模向量，逐步计算 Σsαφα。
    active_counts = np.count_nonzero(actions, axis=1).astype(np.int64)  # 逐行统计买卖团簇数并转为 int64。
    return demands, active_counts


def _validate_market_inputs(steps: int, market_depth: float) -> None:
    # 校验交易步数非负、市场深度为正。
    # 由 simulate_quenched() 和 simulate_annealed() 在抽样前调用，使交易参数合法后进入模拟。
    # 输入
    #     steps 为交易步数 T，须为非负。
    #     market_depth 为收益率的分母 λ，须为正。
    # 输出为 None，参数非法时触发 ValueError。
    
    if steps < 0:  # 拒绝负步数，允许只保留初始价格。
        raise ValueError("steps must be non-negative")
    if market_depth <= 0.0:  # 收益率的分母 λ 必须为正。
        raise ValueError("market_depth must be positive")


def simulate_quenched(
    configuration: ClusterConfiguration,    # 本次交易期间保持固定的 ClusterConfiguration。
    steps: int,                             # 交易步数 T。
    activity: float,                        # 团簇每步参与交易的概率 a。
    market_depth: float,                    # 市场深度 λ，收益率按 r=D/λ 计算。
    rng: np.random.Generator,               # 外部随机数生成器。
    initial_log_price: float = 0.0,         # 初始对数价格，默认 0，对应初始价格 1。
) -> MarketSimulation:                      # 输出：MarketSimulation 对象。
    """Trade repeatedly on one fixed cluster configuration."""
    # 固定一份占据与团簇配置，模拟 steps 步交易，得到需求、收益率和对数价格。
    # 由 src/local_market/experiments.py 的 run_baseline() 在 quenched 分支调用。
    # 由 src/local_market/experiments.py 的 run_quenched_market_trajectories() 在同一临界格点上生成不同活跃率的轨迹。
    # 由 scripts/simulation/run_small_validation.py 的 main() 对每份抽取的配置调用，随后保存需求与收益率并按 p 汇总。
    # 由 scripts/analysis/audit_market_time_series.py 的 main() 生成不同随机种子下的固定配置轨迹，以检查自相关。
    # 由 src/local_market/__init__.py 在包级重新导出。

    # 输入按上述签名顺序给出，steps 可为 0，activity 取 [0, 1]，market_depth 须为正。
    # 输出 MarketSimulation，含需求、收益、对数价格及固定配置计数。对数价格含初始值，长度为 steps+1。
    _validate_market_inputs(steps, market_depth)        # 检查步数、市场深度。
    demands,_ = sample_demands_and_active_counts(configuration.sizes, steps, activity, rng) # 固定规模，批量抽取需求。
    returns = demands.astype(float) / market_depth  # 将需求转换为浮点数后计算收益率 r = D/λ。
    log_prices = np.concatenate(    # 拼接初始值与逐步对数价格。
        ([float(initial_log_price)], float(initial_log_price) + np.cumsum(returns))  # 用初始值加累计收益率，得到各步对数价格。
    )
    # 构造结果对象。
    return MarketSimulation(
        demands=demands,
        returns=returns,
        log_prices=log_prices,
        occupied_counts=np.full(steps, configuration.occupied_count, dtype=np.int64),   # 将固定配置的占据节点数复制到每个交易步。
        cluster_counts=np.full(steps, configuration.cluster_count, dtype=np.int64),  # 将固定配置的团簇数复制到每个交易步。
    )


def simulate_annealed(
    configuration_factory: Callable[[], ClusterConfiguration],      # 生成团簇配置的回调函数。
    steps: int,                             # 交易步数 T。
    activity: float,                        # 团簇每步参与交易的概率 a。
    market_depth: float,                    # 市场深度 λ，收益率按 r=D/λ 计算。
    rng: np.random.Generator,               # 外部随机数生成器。
    initial_log_price: float = 0.0,         # 初始对数价格，默认 0，对应初始价格 1。
) -> MarketSimulation:                      # 输出：MarketSimulation 对象，需求和收益率长度为 T，对数价格长度为 T+1。
    """Regenerate the occupied configuration before every trading step."""
    # 以 annealed 模式交易，每步重新生成占据状态和团簇，再计算当步需求、收益率和价格。
    # 由 src/local_market/experiments.py 的 run_baseline() 在 annealed 分支调用，接收其内部的 make_configuration() 回调以逐步重抽配置。
    # 由 src/local_market/__init__.py 在包级重新导出。

    # 输入按上述签名顺序给出，steps 可为 0，market_depth 须为正，有交易步时检查 activity 是否在 [0, 1]。
    # 输出 MarketSimulation，含逐步需求、收益、对数价格和每步重新抽取的配置计数。
    _validate_market_inputs(steps, market_depth)         # 检查步数和市场深度。
    demands = np.empty(steps, dtype=np.int64)  # 为各步超额需求预分配数组。
    occupied_counts = np.empty(steps, dtype=np.int64)  # 为各步占据节点数预分配数组。
    cluster_counts = np.empty(steps, dtype=np.int64)  # 为各步团簇数预分配数组。
    for step in range(steps):  # 按交易步索引写入各项记录。
        configuration = configuration_factory()         # 生成当步团簇配置，基线的回调来自 src/local_market/experiments.py 的 run_baseline() 内部 make_configuration()。
        demands[step],_ = sample_demands_and_action_once(configuration, activity, rng)  # 对当前全部团簇抽行动，并加权为当步需求。
        occupied_counts[step] = configuration.occupied_count  # 记录当步配置的已占据节点数。
        cluster_counts[step] = configuration.cluster_count  # 读取当前配置的团簇数。
    returns = demands.astype(float) / market_depth      # 将需求转换为浮点数后计算收益率 r = D/λ。
    log_prices = np.concatenate(    # 拼接初始值与逐步对数价格。
        ([float(initial_log_price)], float(initial_log_price) + np.cumsum(returns)) # 用初始值加累计收益率，得到各步对数价格。
    )
    # 将各步结果打包返回给实验入口，再交给脚本保存。
    return MarketSimulation(        
        demands=demands,
        returns=returns,
        log_prices=log_prices,
        occupied_counts=occupied_counts,
        cluster_counts=cluster_counts,
    )
