"""Finite-support discrete avalanche tail likelihoods and path bootstrap."""
# 为 SOC 检验实验拟合有限支持上的离散雪崩尾部。
# fit_tail() 拟合给定下界的模型，select_xmin() 用 KS 距离选下界，bootstrap_paths() 按整条轨迹评估不确定性。
#
# 从驱动与耗散轨迹接收雪崩规模或持续轮数，把尾部模型和重抽样结果交给 SOC 分析脚本。
from __future__ import annotations  # 延后求值类型注解。
import numpy as np  # NumPy 用于离散支持、似然和整条路径抽样。
from scipy.optimize import minimize, minimize_scalar  # SciPy 的 minimize 和 minimize_scalar 分别优化双参数和单参数尾部模型。


def fit_tail(values, xmin: int, upper: int, model: str):
    # 在给定离散区间上为 SOC 检验实验拟合纯幂律、指数或截断幂律。
    # 引用
    #     select_xmin() 拟合每个候选下界的纯幂律，bootstrap_paths() 在每轮选出的下界拟合截断幂律。
    #     scripts/analysis/analyze_soc_research.py 的 main() 在选定下界上拟合截断幂律与指数模型。
    #     defense/live.py 的 run_fit() 拟合演示中指定区间的雪崩样本。
    # 输入
    #     values 为雪崩规模或持续轮数，转换为整数。
    #     xmin 为正整数尾下界，包含端点。
    #     upper 为不小于 xmin 的整数支持上界，包含端点。
    #     model 为 power、exponential 或 cutoff。
    # 输出 模型指标字典，含区间、样本数、指数、截断尺度、对数似然、AIC、KS 距离与截断边界标记，空尾部抛出 ValueError。
    x = np.asarray(values, dtype=int)  # 将观测转为整数以匹配离散模型支持。
    x = x[(x >= xmin) & (x <= upper)]  # 只取闭区间内的尾部观测。
    if not len(x):
        raise ValueError("empty tail")
    support = np.arange(xmin, upper+1)  # 枚举模型归一化所需的全部整数规模，含上界。
    logs = np.log(support)  # 缓存支持点对数，供幂律项反复求值。
    def objective(par):
        # 计算有限支持离散模型的负对数似然。
        # 引用
        #     fit_tail() 的 power、exponential 分支用 lambda 把标量候选打包后交给 objective()，由 minimize_scalar 调用 lambda。
        #     fit_tail() 的 cutoff 分支将 objective() 直接交给 minimize 调用。
        # 输入
        #     par 为模型参数序列，纯幂律只含指数，指数模型只含衰减率，截断幂律依次含指数和衰减率。读取外层 model、logs、support 和 x。
        # 输出 给定参数的负对数似然。
        exponent, rate = par if model == "cutoff" else (par[0], 0.) if model == "power" else (0., par[0])  # 纯幂律固定衰减率为零，指数模型固定幂律指数为零。
        lw = -exponent*logs-rate*support  # 按 −指数×log(s)−衰减率×s 计算未归一化对数权重。
        peak = lw.max()  # 取最大对数权重作为数值稳定的平移量。
        logz = peak + np.log(np.exp(lw-peak).sum())  # 在平移后求指数和，再加回平移量得到对数配分函数。
        return float(len(x)*logz + exponent*np.log(x).sum() + rate*x.sum())  # 用样本数乘对数配分函数，加上各样本的负对数权重。
    if model == "power":
        result = minimize_scalar(lambda a: objective([a]), bounds=(.1, 8), method="bounded")  # 在 [0.1, 8] 内优化纯幂律指数。
        exponent, rate = result.x, 0.
        npar = 1
    elif model == "exponential":
        result = minimize_scalar(lambda r: objective([r]), bounds=(1e-9, 2), method="bounded")  # 指数模型只在给定正区间内优化衰减率。
        exponent, rate = 0., result.x
        npar = 1
    else:
        result = minimize(objective, [1.5, .01], bounds=[(.01, 8), (1e-9, 2)], method="L-BFGS-B")  # 有界优化截断幂律的指数与衰减率。
        exponent, rate = result.x
        npar = 2
    weights = support.astype(float)**(-exponent)*np.exp(-rate*(support-xmin))  # 平移指数项以生成数值较稳定的未归一化权重。
    probs = weights/weights.sum()  # 在有限支持上将权重归一化成概率质量。
    observed = np.sort(x)  # 排序观测以便二分计算经验累计频数。
    ecdf = np.searchsorted(observed, support, side="right")/len(observed)  # 逐个支持点计算小于等于该点的样本比例。
    ks = float(np.max(np.abs(ecdf-np.cumsum(probs))))  # 比较经验累计概率和模型累计概率的最大绝对差。
    # 返回模型参数、似然诊断和可识别性标记。
    return dict(model=model, xmin=xmin, upper=upper, tail_count=len(x),
        exponent=float(exponent), cutoff=float(1/rate) if rate else np.inf,  # 将衰减率取倒数得到截断尺度，纯幂律尺度为无穷。
        log_likelihood=-float(result.fun), aic=2*npar+2*float(result.fun), ks=ks,  # 从负对数似然恢复对数似然，并按参数数目计算 AIC。
        boundary=bool(rate < 1e-7 or 1/rate > 10*upper) if model == "cutoff" else False)  # 仅给截断幂律标记极小衰减率或远超支持上界的截断尺度。


def select_xmin(values, upper: int, minimum_count=100, minimum_range=10):
    # 为 SOC 检验实验按最小 KS 距离选择纯幂律尾部下界。
    # 引用
    #     bootstrap_paths() 在每次重抽路径后重选下界。
    #     scripts/analysis/analyze_soc_research.py 的 main() 分别分析持续时间和规模，规模拟合还用于设定备选模型的共同下界。
    # 输入
    #     values 为雪崩观测，取转为整数后的正值。
    #     upper 为离散支持上界，包含端点。
    #     minimum_count 为候选尾部的最少观测数，默认 100。
    #     minimum_range 为样本最大值与候选下界的最小比，默认 10。
    # 输出依次为
    #     KS 最小的拟合字典，无法拟合时为 None。
    #     有效候选拟合列表。
    #     状态字符串，区分成功、样本不足和动态范围或尾样本不足。
    x = np.asarray(values, dtype=int)  # 转换为和离散模型一致的整数样本。
    x = x[(x > 0) & (x <= upper)]
    if len(x) < minimum_count:  # 总样本量不足时直接返回相应状态。
        return None, [], "sample_insufficient"
    candidates = np.unique(x)  # 从样本中提取不同的正观测值作为候选下界。
    candidates = candidates[(candidates <= x.max()/minimum_range)]  # 保留覆盖至少指定最大值比的候选。
    if len(candidates) > 35:  # 候选超过计算上限时按排序位置稀疏取点。
        candidates = np.unique(candidates[np.linspace(0, len(candidates)-1, 35).astype(int)])  # 候选过多时沿排序索引近似等间隔保留至多 35 个。
    rows = []
    for xmin in candidates:
        if (x >= xmin).sum() < minimum_count:  # 按每个候选下界检验尾部样本数量。
            continue
        rows.append(fit_tail(x, int(xmin), upper, "power"))  # 用相同有限上界拟合该候选的纯幂律。
    if not rows:
        return None, [], "dynamic_range_or_tail_insufficient"
    best = min(rows, key=lambda row: row["ks"])  # 选取经验和模型 CDF 差异最小的候选。
    return best, rows, "ok"


def bootstrap_paths(path_values, upper: int, count: int, rng, minimum_count=100,
                    path_groups=None):
    """Resample whole paths within each initial-condition group and refit."""
    # 在 SOC 检验实验的初始条件分组内重抽整条路径，重选下界并重拟合尾部。
    # 引用
    #     scripts/analysis/analyze_soc_research.py 的 main() 为雪崩规模拟合生成路径层 bootstrap 统计。
    # 输入
    #     path_values 为路径标识到该路径的一维规模样本的映射。
    #     upper 为整数支持上界。
    #     count 为 bootstrap 重复次数。
    #     rng 为用于组内有放回抽样的 NumPy 随机数生成器。
    #     minimum_count 为选下界时的最少尾观测数，默认 100。
    #     path_groups 为路径标识到初始条件组的映射，默认全部同组。
    # 输出 每轮一个记录字典，成功时含下界、纯幂律指数和截断幂律参数，失败时仅记录轮次和状态。
    draws = []
    ids = list(path_values)  # 取路径标识作为完整的抽样单位。
    if path_groups is None:
        path_groups = {pid: "all" for pid in ids}
    groups = {group: [pid for pid in ids if path_groups[pid] == group]
              for group in set(path_groups.values())}  # 根据初始条件标签建立路径分层。
    for index in range(count):  # 每一轮独立生成一份分层重抽样本。
        # 各初始条件组独立地有放回抽取与原组相同数量的路径。
        chosen = np.concatenate([rng.choice(members, size=len(members), replace=True)
                                 for members in groups.values()])  # 各初始条件组分别维持原有的路径数。
        values = np.concatenate([path_values[pid] for pid in chosen])  # 合并所抽路径的全部观测，保留路径被抽中的次数。
        pure, _, status = select_xmin(values, upper, minimum_count)  # 在本轮合并样本上重新选择纯幂律下界。
        if status != "ok":
            draws.append(dict(bootstrap=index, status=status))  # 保留本轮下界选择失败的原因。
            continue
        xmin = int(pure["xmin"])  # 把选出的纯幂律下界作为截断拟合的共同起点。
        cutoff = fit_tail(values, xmin, upper, "cutoff")  # 固定本轮选出的下界，拟合截断幂律。
        # 记录本轮成功拟合的下界、两个指数和截断尺度。
        draws.append(dict(bootstrap=index, status="ok", xmin=xmin,
            tau=pure["exponent"], cutoff_tau=cutoff["exponent"],
            cutoff=cutoff["cutoff"], cutoff_boundary=cutoff["boundary"]))
    return draws
