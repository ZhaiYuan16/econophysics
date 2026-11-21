"""Configuration objects for reproducible baseline experiments."""
# 定义从 YAML 加载并校验的模拟配置。
# BaselineConfig 用于基线，CriticalityConfig 用于格点渗流扫描，MarketAnalysisConfig 用于市场系综，NetworkAnalysisConfig 用于拓扑比较。
# 各类的 from_yaml() 将脚本输入转成对应的数据类。
#
# 命令行入口加载配置后启动模拟，分析和审计脚本也用原始配置确定标准化、尾部和重抽样参数。

from __future__ import annotations  # 延后求值类型注解。

from dataclasses import dataclass  # 标准库数据类装饰器，为各配置生成构造方法并触发初始化校验。
from pathlib import Path  # 标准库路径类，以统一方式打开字符串或 Path 指定的 YAML 文件。
from typing import Any  # 标准库任意类型标注，表示解析后字段值类型各异。

import yaml  # PyYAML 安全解析实验配置。

from .percolation import Boundary  # src/local_market/percolation.py，声明格点配置支持的边界类型。
from .statistics import ClusterExclusion  # src/local_market/statistics.py，声明有限团簇统计的排除规则。


@dataclass(frozen=True)  # 冻结基线配置参数。
class BaselineConfig:
    # 保存基线模拟的输入字段。
    length: int  # 正方格点边长，须为正。
    p: float  # 单格点占据概率，范围 [0, 1]。
    activity: float  # 团簇的交易活跃率 a，范围 [0, 1]。
    market_depth: float  # 市场深度 λ，须为正。
    boundary: Boundary  # 格点边界，取 open 或 periodic。
    mode: str  # 交易模式，取 quenched 或 annealed。
    steps: int  # 交易步数，允许为 0。
    seed: int  # 占据与交易共用随机流的种子。

    def __post_init__(self) -> None:
        # 校验基线模拟配置的字段范围和组合。
        # 引用
        #     BaselineConfig 的数据类构造方法会自动调用。BaselineConfig.from_yaml() 通过该构造方法触发校验。
        # 输入
        #     self 为 BaselineConfig 实例，读取初始化后的配置字段。
        # 输出 None，发现非法字段或组合时抛出 ValueError。
        # 边长须能定义非空方格。
        if self.length <= 0:
            raise ValueError("length must be positive")
        # 占据概率须能用于伯努利抽样。
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must lie in [0, 1]")
        # 团簇交易活跃率须为合法概率。
        if not 0.0 <= self.activity <= 1.0:
            raise ValueError("activity must lie in [0, 1]")
        # 收益率的分母必须为正。
        if self.market_depth <= 0:
            raise ValueError("market_depth must be positive")
        # 边界类型对应两种格点团簇识别方式。
        if self.boundary not in ("open", "periodic"):
            raise ValueError("boundary must be 'open' or 'periodic'")
        # 固定配置与逐步重建配置是两种有效交易模式。
        if self.mode not in ("quenched", "annealed"):
            raise ValueError("mode must be 'quenched' or 'annealed'")
        # 步数允许为零，以返回仅含初始价格的轨迹。
        if self.steps < 0:
            raise ValueError("steps must be non-negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaselineConfig":
        # 从 YAML 加载基线模拟配置。
        # 引用
        #     scripts/simulation/run_baseline.py 的 main() 读取基线格点和交易参数。
        # 输入
        #     cls 为 BaselineConfig 类，用于构造相应配置。
        #     path 为 UTF-8 YAML 文件的字符串路径或 Path。
        # 输出 校验通过的 BaselineConfig 实例。
        # 把字符串或路径统一为 Path 并以 UTF-8 打开。
        with Path(path).open(encoding="utf-8") as stream:
            # 安全解析 YAML 配置为关键字字典。
            data: dict[str, Any] = yaml.safe_load(stream)
        # 用原始字段构造配置，触发初始化校验。
        return cls(**data)


@dataclass(frozen=True)  # 冻结格点渗流扫描配置参数。
class CriticalityConfig:
    # 保存渗流临界性质实验的输入字段。
    lengths: tuple[int, ...]  # 所扫描的格点边长。
    p_c: float  # 用于定义 p/p_c 的参照临界占据概率，范围 (0, 1)。
    p_ratios: tuple[float, ...]  # 升序且无重复的占据概率比 p/p_c。
    samples_per_length: dict[int, int]  # 按边长索引的每个概率点样本数。
    boundary: Boundary  # 开放或周期边界。
    finite_cluster_exclusion: ClusterExclusion  # 计算有限团簇统计的排除规则。
    histogram_p_ratios: tuple[float, ...]  # 报告直方图的 p/p_c 子集。
    bootstrap_replicates: int  # 重抽配置的次数，至少 100。
    seed: int  # 占据与重抽样的基准随机种子。

    def __post_init__(self) -> None:
        # 校验渗流临界性质实验配置的字段范围和组合。
        # 引用
        #     CriticalityConfig 的数据类构造方法会自动调用。CriticalityConfig.from_yaml() 通过该构造方法触发校验。
        # 输入
        #     self 为 CriticalityConfig 实例，读取初始化后的配置字段。
        # 输出 None，发现非法字段或组合时抛出 ValueError。
        # 至少配置一个正边长。
        if not self.lengths or any(length <= 0 for length in self.lengths):
            raise ValueError("lengths must contain positive integers")
        # 参考临界概率须位于开区间。
        if not 0.0 < self.p_c < 1.0:
            raise ValueError("p_c must lie in (0, 1)")
        # 把每个概率比还原成实际占据概率后检查范围。
        if not self.p_ratios or any(
            not 0.0 < ratio * self.p_c <= 1.0 for ratio in self.p_ratios
        ):
            raise ValueError("all p ratios must produce probabilities in (0, 1]")
        # 要求概率比已排序且无重复。
        if sorted(set(self.p_ratios)) != list(self.p_ratios):
            raise ValueError("p_ratios must be sorted and unique")
        # 每个配置尺寸都有且仅有一项样本数设置。
        if set(self.samples_per_length) != set(self.lengths):
            raise ValueError("samples_per_length must define every configured length")
        # 每个边长和概率组合至少抽一个配置。
        if any(samples <= 0 for samples in self.samples_per_length.values()):
            raise ValueError("sample counts must be positive")
        # 边界取值须能分派给格点识别器。
        if self.boundary not in ("open", "periodic"):
            raise ValueError("boundary must be 'open' or 'periodic'")
        # 只接受已定义的三种团簇排除口径。
        if self.finite_cluster_exclusion not in ("largest", "spanning", "none"):
            raise ValueError("invalid finite-cluster exclusion rule")
        # 画直方图的概率点须来自扫描结果。
        if not set(self.histogram_p_ratios).issubset(self.p_ratios):
            raise ValueError("histogram ratios must be included in p_ratios")
        # 连续区间使用至少 100 轮 bootstrap。
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CriticalityConfig":
        # 从 YAML 加载渗流临界性质实验配置。
        # 引用
        #     scripts/simulation/scan_criticality.py 的 main() 加载系综渗流扫描参数。
        # 输入
        #     cls 为 CriticalityConfig 类，用于构造相应配置。
        #     path 为 UTF-8 YAML 文件的字符串路径或 Path。
        # 输出 校验通过的 CriticalityConfig 实例。
        # 以 UTF-8 打开配置文件。
        with Path(path).open(encoding="utf-8") as stream:
            # 安全读取 YAML 配置字段。
            data: dict[str, Any] = yaml.safe_load(stream)
        # 把扫描尺寸转换成整数元组。
        data["lengths"] = tuple(int(value) for value in data["lengths"])
        # 把扫描概率比转换成浮点元组。
        data["p_ratios"] = tuple(float(value) for value in data["p_ratios"])
        # 把直方图概率比转换成浮点元组。
        data["histogram_p_ratios"] = tuple(
            float(value) for value in data["histogram_p_ratios"]
        )
        # 把样本映射的边长与样本数都转为整数。
        data["samples_per_length"] = {
            int(length): int(samples)
            for length, samples in data["samples_per_length"].items()  # 逐个读出按尺寸配置的样本数。
        }
        # 用归一化字段构造并校验扫描配置。
        return cls(**data)


@dataclass(frozen=True)  # 冻结格点市场系综配置参数。
class MarketAnalysisConfig:
    # 保存市场收益率的影响因素实验的输入字段。
    length: int  # 正方格点边长。
    p_c: float  # 参照临界占据概率。
    p_ratios: tuple[float, ...]  # 用于生成 p 的占据概率比。
    activities: tuple[float, ...]  # 团簇交易活跃率列表。
    replicates: int  # 每个占据概率下的独立重复数。
    configurations_per_replicate: int  # 每次重复的占据配置数。
    trades_per_configuration: int  # 每份固定配置的交易笔数。
    market_depth: float  # 收益率分母 λ。
    boundary: Boundary  # 开放或周期边界。
    extreme_thresholds: tuple[float, ...]  # 绝对标准化收益的极端阈值，须含 3、5、8。
    tail_xmin_quantiles: tuple[float, ...]  # 连续尾部下界的候选分位概率。
    tail_minimum_count: int  # 有效尾部最少样本数，至少 50。
    bootstrap_replicates: int  # 重抽拟合次数，至少 100。
    length_sensitivity_fractions: tuple[float, ...]  # 用于样本长度敏感性检查的保留比例。
    seed: int  # 派生占据和交易随机流的实验种子。

    def __post_init__(self) -> None:
        # 校验市场收益率的影响因素实验配置的字段范围和组合。
        # 引用
        #     MarketAnalysisConfig 的数据类构造方法会自动调用。MarketAnalysisConfig.from_yaml() 通过该构造方法触发校验。
        # 输入
        #     self 为 MarketAnalysisConfig 实例，读取初始化后的配置字段。
        # 输出 None，发现非法字段或组合时抛出 ValueError。
        # 格点不能为空。
        if self.length <= 0:
            raise ValueError("length must be positive")
        # 临界参照概率应在 (0, 1)。
        if not 0.0 < self.p_c < 1.0:
            raise ValueError("p_c must lie in (0, 1)")
        # 至少有一个 p/p_c，所有实际占据概率都须合法。
        if not self.p_ratios or any(
            not 0.0 < ratio * self.p_c <= 1.0 for ratio in self.p_ratios
        ):
            raise ValueError("invalid p ratios")
        # 至少有一个团簇活跃率，各值须为合法概率。
        if not self.activities or any(not 0.0 <= value <= 1.0 for value in self.activities):
            raise ValueError("activities must lie in [0, 1]")
        # 重复数、配置数和每份配置的交易笔数均须为正。
        if min(
            self.replicates,
            self.configurations_per_replicate,
            self.trades_per_configuration,
        ) <= 0:
            raise ValueError("replicate and sample counts must be positive")
        # 检查收益率分母。
        if self.market_depth <= 0:
            raise ValueError("market_depth must be positive")
        # 限制团簇识别的边界取值。
        if self.boundary not in ("open", "periodic"):
            raise ValueError("invalid boundary")
        # 极端幅度阈值必须为正。
        if any(value <= 0 for value in self.extreme_thresholds):
            raise ValueError("extreme thresholds must be positive")
        # 汇总程序使用三个固定阈值对应的统计列。
        if not {3.0, 5.0, 8.0}.issubset(self.extreme_thresholds):
            raise ValueError("extreme thresholds must include 3, 5, and 8")
        # 尾下界候选分位概率须严格在 0 和 1 之间。
        if any(not 0.0 < value < 1.0 for value in self.tail_xmin_quantiles):
            raise ValueError("tail xmin quantiles must lie in (0, 1)")
        # 限制尾部样本和重抽样数量下限。
        if self.tail_minimum_count < 50 or self.bootstrap_replicates < 100:
            raise ValueError("tail and bootstrap sample counts are too small")
        # 长度敏感性子样本比例须大于零且至多为一。
        if any(not 0.0 < value <= 1.0 for value in self.length_sensitivity_fractions):
            raise ValueError("length fractions must lie in (0, 1]")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MarketAnalysisConfig":
        # 从 YAML 加载市场收益率的影响因素实验配置。
        # 引用
        #     scripts/analysis/analyze_market_returns.py 的 main() 读取市场系综和尾部拟合参数。
        #     scripts/analysis/audit_downstream_statistics.py 的 main()、scripts/analysis/audit_market_time_series.py 的 main() 和 scripts/analysis/finalize_market_statistics.py 的 main() 用相同配置重算或审计市场结果。
        # 输入
        #     cls 为 MarketAnalysisConfig 类，用于构造相应配置。
        #     path 为 UTF-8 YAML 文件的字符串路径或 Path。
        # 输出 校验通过的 MarketAnalysisConfig 实例。
        # 以 UTF-8 打开配置文件。
        with Path(path).open(encoding="utf-8") as stream:
            # 将 YAML 解析为待归一化的字段字典。
            data: dict[str, Any] = yaml.safe_load(stream)
        # 这些扫描、阈值和子样本参数统一用浮点元组。
        for key in (
            "p_ratios",
            "activities",
            "extreme_thresholds",
            "tail_xmin_quantiles",
            "length_sensitivity_fractions",
        ):
            # 逐个转换当前列表中的数值。
            data[key] = tuple(float(value) for value in data[key])
        # 构造市场配置并运行初始化检查。
        return cls(**data)


@dataclass(frozen=True)  # 冻结网络拓扑比较配置参数。
class NetworkAnalysisConfig:
    # 保存网络拓扑压力测试的输入字段。
    node_count: int  # 每份网络的节点数。
    target_mean_degree: int  # 目标平均度。
    topologies: tuple[str, ...]  # 须覆盖 lattice、er、ws 和 ba。
    network_replicates: int  # 每种拓扑的网络实现数。
    occupancy_samples_per_p: int  # 每份图在每个 p 下的渗流样本数。
    p_grids: dict[str, tuple[float, ...]]  # 按拓扑分配的绝对占据概率扫描网格。
    relative_p_values: tuple[float, ...]  # 用于市场比较的相对有效临界概率。
    market_configurations_per_network: int  # 每市场参数、每网络的配置数。
    trades_per_configuration: int  # 每配置的交易笔数。
    activity: float  # 每步团簇交易活跃率 a。
    market_depth: float  # 收益率分母 λ。
    ws_rewiring_probability: float  # WS 网络的边重连概率。
    path_sources: int  # 近似平均最短路使用的起点数上限。
    extreme_thresholds: tuple[float, ...]  # 绝对 z 的极端阈值，须含 3、5、8。
    tail_xmin_quantiles: tuple[float, ...]  # 尾部下界候选分位概率。
    tail_minimum_count: int  # 最少尾部样本数，至少 50。
    bootstrap_replicates: int  # 重抽样次数，至少 100。
    seed: int  # 派生网络、占据、交易及重抽样随机流的种子。

    def __post_init__(self) -> None:
        # 校验网络拓扑压力测试配置的字段范围和组合。
        # 引用
        #     NetworkAnalysisConfig 的数据类构造方法会自动调用。NetworkAnalysisConfig.from_yaml() 通过该构造方法触发校验。
        # 输入
        #     self 为 NetworkAnalysisConfig 实例，读取初始化后的配置字段。
        # 输出 None，发现非法字段或组合时抛出 ValueError。
        required = {"lattice", "er", "ws", "ba"}
        # 检查压力测试所需的四种拓扑是否齐全。
        if set(self.topologies) != required:
            raise ValueError("topologies must contain lattice, er, ws, and ba")
        # 匹配网络的节点数和目标平均度须为正。
        if self.node_count <= 0 or self.target_mean_degree <= 0:
            raise ValueError("node_count and target_mean_degree must be positive")
        # 图实现、占据配置、交易笔数和最短路起点都需要正样本数。
        if min(
            self.network_replicates,
            self.occupancy_samples_per_p,
            self.market_configurations_per_network,
            self.trades_per_configuration,
            self.path_sources,
        ) <= 0:
            raise ValueError("sample counts must be positive")
        # 每种拓扑都要有自己的绝对概率网格。
        if set(self.p_grids) != required:
            raise ValueError("p_grids must define every topology")
        for grid in self.p_grids.values():  # 逐个检查拓扑网格自身的顺序、唯一性与范围。
            # 各网格须升序、无重复且概率范围合法。
            if sorted(set(grid)) != list(grid) or any(not 0.0 < p <= 1.0 for p in grid):
                raise ValueError("p grids must be sorted, unique, and within (0, 1]")
        # 市场比较使用正的相对概率，乘有效阈值后由模拟截到 1。
        if any(value <= 0 for value in self.relative_p_values):
            raise ValueError("relative p values must be positive")
        # 活跃率须是概率，市场深度须为正。
        if not 0.0 <= self.activity <= 1.0 or self.market_depth <= 0:
            raise ValueError("invalid market activity or depth")
        # WS 重连参数必须为合法概率。
        if not 0.0 <= self.ws_rewiring_probability <= 1.0:
            raise ValueError("WS rewiring probability must lie in [0, 1]")
        # 保留跨拓扑比较需要的固定阈值。
        if not {3.0, 5.0, 8.0}.issubset(self.extreme_thresholds):
            raise ValueError("extreme thresholds must include 3, 5, and 8")
        # 尾部阈值候选是严格介于 0 和 1 的分位概率。
        if any(not 0.0 < q < 1.0 for q in self.tail_xmin_quantiles):
            raise ValueError("tail xmin quantiles must lie in (0, 1)")
        # 检查最少尾样本数与 bootstrap 次数。
        if self.tail_minimum_count < 50 or self.bootstrap_replicates < 100:
            raise ValueError("tail settings are too small")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NetworkAnalysisConfig":
        # 从 YAML 加载网络拓扑压力测试配置。
        # 引用
        #     scripts/simulation/compare_networks.py 的 main() 读取拓扑、扫描和交易参数。
        #     scripts/analysis/audit_downstream_statistics.py 的 main() 用原配置审计网络统计。
        # 输入
        #     cls 为 NetworkAnalysisConfig 类，用于构造相应配置。
        #     path 为 UTF-8 YAML 文件的字符串路径或 Path。
        # 输出 校验通过的 NetworkAnalysisConfig 实例。
        # 统一用 UTF-8 读取配置文件。
        with Path(path).open(encoding="utf-8") as stream:
            # 安全解析 YAML 字段。
            data: dict[str, Any] = yaml.safe_load(stream)
        # 保留配置提供的拓扑遍历顺序。
        data["topologies"] = tuple(data["topologies"])
        # 逐拓扑把概率网格转成浮点元组。
        data["p_grids"] = {
            name: tuple(float(value) for value in grid)
            for name, grid in data["p_grids"].items()  # 逐个读出配置指定的拓扑与概率网格。
        }
        # 将跨拓扑共用的比例、阈值和分位数转换成同一种容器。
        for key in (
            "relative_p_values",
            "extreme_thresholds",
            "tail_xmin_quantiles",
        ):
            # 转换当前参数的所有数值。
            data[key] = tuple(float(value) for value in data[key])
        # 构造网络配置并执行一致性校验。
        return cls(**data)
