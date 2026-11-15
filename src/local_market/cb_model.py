"""Trading dynamics for the Cont-Bouchaud market model."""

from __future__ import annotations  # 延后求值类型注解，用于声明数组、配置和返回对象类型。

from dataclasses import dataclass  # 标准库，自动生成结果数据类的构造方法等。
from typing import Callable  # 标准库类型标注，声明“无参数并返回团簇配置”的回调接口。

import numpy as np  # 第三方 NumPy 库，提供随机抽样、数组、矩阵乘法与累计和运算。

from .percolation import ClusterConfiguration  # 本地 src/local_market/percolation.py，保存格点团簇标签与规模。

@dataclass(frozen=True)
class MarketSimulation:
    # 每个字段对应一次交易模拟的一项逐步记录。
    # 由 simulate_quenched() 或 simulate_annealed() 构造并返回给实验入口。
    demands: np.ndarray         # 长度 T，每步超额需求 D。
    returns: np.ndarray         # 长度 T，每步对数收益率 r = D/λ。
    log_prices: np.ndarray      # 长度 T+1，初始对数价格及每步交易后的对数价格。
    occupied_counts: np.ndarray # 长度 T，每步使用配置的占据节点数。
    cluster_counts: np.ndarray  # 长度 T，每步使用配置的团簇数。

def sample_demands_and_active_once(
    configuration: ClusterConfiguration, activity: float, rng: np.random.Generator
) -> tuple[int, np.ndarray]:
    # 按团簇规模汇总一次交易的超额需求。
    # 由 simulate_annealed() 每步调用，用于计算当前交易步的需求。
    # 输入：
    #       configuration，提供团簇数 K 和团簇大小 s ，
    #       activity，团簇参与交易的概率 a，
    #       rng，外部随机流。
    # 输出：
    #       D = Σsαφα : 超额需求。
    #       actions : 为每个团簇的行动数组，-1 表示卖出，0 表示不交易，+1 表示买入。

    cluster_count = configuration.cluster_count  # 团簇数 K。
    if cluster_count < 0:
        raise ValueError("cluster_count must be non-negative")
    draws = rng.random(cluster_count)  # 每个团簇抽取一个 [0,1) 随机数。
    actions = np.zeros(cluster_count, dtype=np.int8)  # action，长度 K 的整数数组，-1 表示卖出，0 表示不交易，+1 表示买入。
    actions[draws < activity / 2.0] = -1
    actions[draws >= 1.0 - activity / 2.0] = 1  
    D = int(np.dot(configuration.sizes, actions))  # 点积将同一团簇内共同买卖的节点数计入需求。
    return D, actions

def sample_demands_and_active_counts(
    cluster_sizes: np.ndarray,
    steps: int,
    activity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # 固定团簇规模，用矩阵运算同时抽取多个交易步的需求与活跃团簇数。
    # 由 simulate_quenched() 调用，为固定配置交易提供需求序列。
    # 由 src/local_market/experiments.py 的 run_market_ensemble() 和 src/local_market/network_analysis.py 的 run_network_market_comparison() 为市场系综抽样。
    # 由 tests/test_cb_model.py 的 test_vectorized_demands_report_active_cluster_counts() 引用。
    # 输入：
    #     cluster_sizes 为一维团簇规模数组，每个元素对应一个团簇。
    #     steps 为交易步数 T。
    #     activity 为团簇每步参与交易的概率 a，调用方应保证在 [0,1] 内。
    #     rng 为外部随机数生成器。
    # 输出：
    #     demands 为长度 T 的超额需求数组，D=Σsαφα。
    #     active_counts 为长度 T 的活跃团簇数数组，每项为当步非零行动的团簇数量。

    """Sample excess demand and the number of active clusters per step."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    sizes = np.asarray(cluster_sizes, dtype=np.int64)  # 统一团簇规模的数组类型。
    if sizes.ndim != 1 or np.any(sizes <= 0):
        if sizes.size == 0 and sizes.ndim == 1:
            zeros = np.zeros(steps, dtype=np.int64)
            return zeros, zeros.copy()
        raise ValueError("cluster_sizes must be a one-dimensional positive array")
    # 每行对应一个交易步，每列对应一个团簇，随后按阈值编码成行动矩阵。
    draws = rng.random((steps, sizes.size))
    actions = (draws >= 1.0 - activity / 2.0).astype(np.int8)
    actions -= (draws < activity / 2.0).astype(np.int8)
    demands = actions @ sizes  # 矩阵乘规模向量，逐步计算 Σsαφα。
    active_counts = np.count_nonzero(actions, axis=1).astype(np.int64)  # 逐行统计买卖团簇数。
    return demands, active_counts

def _validate_market_inputs(steps: int, market_depth: float, activity: float) -> None:
    # 校验交易步数非负、市场深度为正、活跃率在 [0,1] 内。
    # 由 simulate_quenched() 和 simulate_annealed() 在抽样前调用，使交易参数合法后进入模拟。
    # 输入：
    #     steps ：交易步数 T。
    #     market_depth ：计算收益率的分母 λ。
    #     activity ：团簇参与交易的概率 a。
    # 输出为 None，参数非法时触发 ValueError。

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if market_depth <= 0.0:
        raise ValueError("market_depth must be positive")
    if not 0.0 <= activity <= 1.0:
        raise ValueError("activity must lie in [0, 1]")

def simulate_quenched(
    configuration: ClusterConfiguration,
    steps: int,
    activity: float,
    market_depth: float,
    rng: np.random.Generator,
    initial_log_price: float = 0.0,
) -> MarketSimulation:
    # 固定一份占据与团簇配置，模拟 steps 步交易，得到需求、收益率和对数价格。
    # 由 scripts/run_small_validation.py 的 main() 对每份抽取的配置调用，随后保存需求与收益率并按 p 汇总。
    # 也由 src/local_market/experiments.py 的 run_baseline() 在 quenched 分支调用。
    # tests/test_cb_model.py 的 test_zero_activity_gives_zero_returns()、test_fixed_seed_is_reproducible() 也有引用。
    # 输入：
    #       configuration ：本次交易期间保持固定的 ClusterConfiguration。
    #       steps ：交易步数 T。
    #       activity ：团簇每步参与交易的概率 a。
    #       market_depth ：市场深度 λ，收益率按 r=D/λ 计算。
    #       rng ：外部随机数生成器。
    #       initial_log_price ：初始对数价格，默认 0，对应初始价格 1。
    # 输出：    
    #       MarketSimulation 对象，字段含义如下。

    """Trade repeatedly on one fixed cluster configuration."""
    _validate_market_inputs(steps, market_depth, activity)  # 检查步数、市场深度和活跃率。
    demands,_ = sample_demands_and_active_counts(configuration.sizes, steps, activity, rng)  # 固定规模，批量抽取需求。
    returns = demands.astype(float) / market_depth
    # cumsum 累计 r 得到对数价格变化，concatenate 在序列开头补入初始对数价格。
    log_prices = np.concatenate(
        ([float(initial_log_price)], float(initial_log_price) + np.cumsum(returns))
    )
    # 构造结果对象，np.full 将固定配置的占据数、团簇数复制到每个交易步。
    return MarketSimulation(
        demands=demands,
        returns=returns,
        log_prices=log_prices,
        occupied_counts=np.full(steps, configuration.occupied_count, dtype=np.int64),
        cluster_counts=np.full(steps, configuration.cluster_count, dtype=np.int64),
    )

def simulate_annealed(
    configuration_factory: Callable[[], ClusterConfiguration],
    steps: int,
    activity: float,
    market_depth: float,
    rng: np.random.Generator,
    initial_log_price: float = 0.0,
) -> MarketSimulation:
    """Regenerate the occupied configuration before every trading step."""
    # annealed 模式的实验：每个交易步重新生成占据状态和团簇，再计算当步交易。
    # 运行 run_baseline.py 时，由 experiments.py 的 run_baseline() 在 annealed 分支调用，并接收其 make_configuration 回调。
    # 输入：
    #       configuration_factory，生成团簇配置的回调函数，
    #       steps， 步数 T，
    #       activity，活跃率 a，
    #       market_depth，市场深度 λ，
    #       rng，随机流，
    #       initial_log_price，初始对数价格，默认 0，对应初始价格 1。
    # 输出：
    #       MarketSimulation对象，需求和收益率长度为 T，对数价格长度为 T+1。

    _validate_market_inputs(steps, market_depth, activity)  # 检查步数、市场深度和活跃率，再预分配逐步记录数组。
    demands = np.empty(steps, dtype=np.int64)
    occupied_counts = np.empty(steps, dtype=np.int64)
    cluster_counts = np.empty(steps, dtype=np.int64)
    for step in range(steps):
        configuration = configuration_factory()  # 回调 experiments.py 的 make_configuration，完成占据抽样与团簇识别。
        demands[step],_ = sample_demands_and_active_once(configuration, activity, rng)  # 对当前全部团簇抽行动，并加权为当步需求。
        occupied_counts[step] = configuration.occupied_count
        cluster_counts[step] = configuration.cluster_count
    returns = demands.astype(float) / market_depth  # 收益率 r = D/λ。
    # 累计收益率，并将初始对数价格放在索引 0。
    log_prices = np.concatenate(
        ([float(initial_log_price)], float(initial_log_price) + np.cumsum(returns))
    )
    return MarketSimulation(  # 将各步结果打包返回给实验入口，再交给脚本保存。
        demands=demands,
        returns=returns,
        log_prices=log_prices,
        occupied_counts=occupied_counts,
        cluster_counts=cluster_counts,
    )
