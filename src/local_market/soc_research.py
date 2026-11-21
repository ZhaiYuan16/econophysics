"""SOC drive and dissipation dynamics for the report's v1 and v2 experiments."""
# 模拟 SOC 检验实验的缓慢驱动和需求触发耗散。
# ResearchConfig 保存动力学参数，simulate() 运行一条轨迹并记录交易步、雪崩事件和耗散轮次。
#
# 脚本和答辩演示用这些记录分析收敛、雪崩尾部及有限尺寸标度。
from __future__ import annotations  # 延后求值类型注解。

from dataclasses import dataclass  # 标准库数据类装饰器，为 SOC 参数对象生成构造方法等。
import numpy as np  # NumPy 用于占据掩码、随机抽样和团簇行动筛选。
import pandas as pd  # Pandas 用于返回轨迹、事件和轮次表。
from .cb_model import sample_demands_and_action_once  # src/local_market/cb_model.py，抽取逐团簇行动和加权超额需求。
from .percolation import identify_lattice_clusters  # src/local_market/percolation.py，按开放边界识别占据团簇。


@dataclass(frozen=True)  # 冻结 SOC 动力学配置字段的绑定。
class ResearchConfig:
    length: int  # 正方格点边长，至少为 2。
    steps: int  # 模拟步数，至少为 1。
    burn_in: int  # 预热步数，从 0 到 steps−1，供下游筛选记录。
    activity: float = .01  # 团簇交易活跃率 a，默认 0.01。
    theta: float = .08  # 需求绝对值阈值相对格点总数的比例 θ，默认 0.08。
    q: float = .25  # 每个同向团簇格点的耗散概率 q，默认 0.25。
    drive_nodes: int = 1  # 无耗散轮次时填入的最大空位数，默认 1。
    mode: str = "baseline"  # 动力学模式，默认 baseline，另有 cascade、no_dissipation、random_dissipation。
    max_rounds: int = 1000  # 级联中单个事件可执行的最多耗散轮次，默认 1000。

    def __post_init__(self):
        # 校验 SOC 检验实验的模拟参数。
        # 引用
        #     ResearchConfig 的数据类构造方法在初始化字段后自动调用，也覆盖 scripts/simulation/run_soc_research.py 的 main() 和 defense/live.py 的 run_soc() 创建的实例。
        # 输入
        #     self 为已填入初始值的 ResearchConfig 实例，读取尺寸、时间、交易、耗散与驱动参数。
        # 输出 None，非法参数抛出 ValueError。
        if self.length < 2 or self.steps < 1 or not 0 <= self.burn_in < self.steps:
            raise ValueError("invalid lattice or time")
        if not 0 <= self.activity <= 1 or not 0 < self.theta < 1 or not 0 < self.q <= 1:  # 交易参与、高需求触发和移除概率各有独立合法范围。
            raise ValueError("invalid action or dissipation parameter")
        if self.drive_nodes < 1 or self.max_rounds < 1 or self.mode not in {  # 驱动和级联轮数至少为一，并限制可选对照模式。
            "baseline", "cascade", "no_dissipation", "random_dissipation"}:
            raise ValueError("invalid drive or mode")


def simulate(config: ResearchConfig, p0: float, seed: int, path_id: str):
    """Run one path; resample cluster actions after every cascade removal round."""
    # 运行 SOC 检验实验的一条占据与耗散轨迹，级联模式每轮重新划分团簇并抽样行动。
    # 引用
    #     scripts/simulation/run_soc_research.py 的 main() 生成各初始占据率、模式和尺寸的样本表。
    #     defense/live.py 的 run_soc() 生成实时演示的轨迹与事件。
    # 输入
    #     config 为 ResearchConfig，预热步数由下游据此筛选。
    #     p0 为初始独立占据概率，预期在 [0, 1]。
    #     seed 为该路径的 NumPy 随机种子。
    #     path_id 为三张结果表共用的路径标识。
    # 输出依次为
    #     步表，每个交易步一行，记录占据率、首轮需求、驱动与耗散计数。
    #     事件表，每个发生过耗散轮次的交易步一行，记录总移除规模、轮数、等待步数和原因。
    #     轮次表，每个执行过的耗散轮次一行，记录需求、移除数和剩余占据率。无事件时后两张表保留列结构。
    rng = np.random.default_rng(seed)  # 为该路径建立可复现的随机数流。
    n = config.length**2  # 总格点数用于将节点数换成占据率，也用于缩放需求阈值。
    occ = rng.random((config.length, config.length)) < p0  # 按 p0 独立初始化各格点的占据状态。
    rows, events, rounds = [], [], []
    last_event_step = None
    sequence = 0
    for step in range(config.steps):  # 逐步推进占据状态，记录更新前后的统计。
        before = int(occ.sum())  # 在驱动或耗散前记录占据节点数。
        size = duration = first_demand = added = 0
        end_reason = "below_threshold"
        while True:
            clusters = identify_lattice_clusters(occ, "open")  # 开放格点上重新识别本轮团簇及其标签。
            demand, actions = sample_demands_and_action_once(clusters, config.activity, rng)  # 抽取团簇行动，按规模得到本轮净需求。
            if duration == 0:
                first_demand = demand
            if abs(demand) < config.theta*n or demand == 0:  # 绝对需求达到 θL² 且具有方向时才可能耗散。
                end_reason = "below_threshold"
                break
            if config.mode == "no_dissipation":
                end_reason = "dissipation_disabled"
                break
            aligned = np.flatnonzero(actions == np.sign(demand)) + 1  # 挑出与净需求同号的行动，将数组下标转成团簇标签。
            selected = np.isin(clusters.labels, aligned) & (rng.random(occ.shape) < config.q)  # 对同向团簇内各格点独立按 q 决定是否移除。
            count = int(selected.sum())  # 计算本轮拟移除的格点数。
            if config.mode == "random_dissipation" and count:
                available = np.flatnonzero(occ.ravel())  # 随机耗散对照组保留移除数，改从所有占据位置抽样。
                selected = np.zeros(occ.shape, dtype=bool)  # 重建移除掩码，清除对团簇方向的限制。
                selected.ravel()[rng.choice(available, size=count, replace=False)] = True  # 从占据位置不放回选取相同数量的节点。
            occ[selected] = False  # 原地移除选中的占据节点，下一轮使用更新后的格点。
            duration += 1
            size += count
            # 保存本轮需求、移除数量和更新后的占据率。
            rounds.append(dict(path_id=path_id, step=step, round=duration,
                               demand=demand, removed_nodes=count,
                               occupancy_after=float(occ.mean())))  # 以全部格点为分母计算耗散后的占据率。
            if count == 0:
                end_reason = "zero_removal"
                break
            if config.mode != "cascade":
                end_reason = "single_round"
                break
            if duration >= config.max_rounds:
                end_reason = "max_rounds"
                break
        if duration == 0:  # 本步从未执行耗散轮次时才进行缓慢驱动。
            vacancies = np.flatnonzero(~occ.ravel())  # 将空位置转成一维索引，供随机填充。
            if vacancies.size:
                chosen = rng.choice(vacancies, size=min(config.drive_nodes, vacancies.size), replace=False)  # 不放回填充至多 drive_nodes 个空位。
                occ.ravel()[chosen] = True
                added = len(chosen)  # 记录实际新增数，受可用空位数限制。
        if duration:
            sequence += 1
            # 以交易步为单位汇总雪崩事件，包含执行过但移除数为零的事件。
            events.append(dict(path_id=path_id, step=step, event_sequence=sequence,
                size=size, duration=duration,
                waiting_steps=np.nan if last_event_step is None else step-last_event_step,  # 等待时间用相邻事件触发步之差，首个事件无值。
                occupancy_before=before/n, occupancy_after=float(occ.mean()),  # 把耗散前后的占据节点数分别换算成全格点比例。
                trigger_demand=first_demand, trigger_reason="absolute_demand_threshold",
                end_reason=end_reason))
            last_event_step = step
        # 记录每步首轮需求和驱动、耗散后的状态。
        rows.append(dict(path_id=path_id, step=step, occupancy_before=before/n,
            occupancy_after=float(occ.mean()), demand=first_demand,  # 保留更新后的占据率和首轮需求。
            added_nodes=added, removed_nodes=size, duration=duration,
            event_sequence=sequence if duration else 0))
    return (pd.DataFrame(rows),  # 逐步记录总有数据，直接由字典列表构造表。
            pd.DataFrame(events, columns=["path_id", "step", "event_sequence", "size",  # 为无事件的路径也保留事件表列。
                "duration", "waiting_steps", "occupancy_before", "occupancy_after",
                "trigger_demand", "trigger_reason", "end_reason"]),
            pd.DataFrame(rounds, columns=["path_id", "step", "round", "demand",  # 为无耗散轮次的路径也保留轮次表列。
                "removed_nodes", "occupancy_after"]))
