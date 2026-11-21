"""Configuration-level measurements and inference for open square site percolation."""
# 为开放正方格点的渗流临界性质实验构造精细网格并作配置层推断。ratio_grid() 和 scan() 生成观测，summarize()、pseudocritical() 和 crossings() 汇总结构及临界位置。
# pooled_histogram() 归一化规模频数，tail_fits() 选择拟合区间并比较有限支持模型。
#
# 精细扫描脚本在共同占据样本上比较两套团簇排除口径，将临界估计、直方图和尾部模型交给绘图与校验环节。

from __future__ import annotations  # 延后求值类型注解。

from collections import Counter  # 标准库 Counter 按规模计数并池化稀疏直方图。
from dataclasses import dataclass  # 标准库数据类装饰器，为扫描结果生成构造方法。
from time import perf_counter  # 标准库高精度单调计时器，用于记录扫描用时。
import csv  # 标准库 CSV 写出器，用于按配置追加直方图。
from pathlib import Path  # 标准库路径对象，定位可选直方图文件。

import numpy as np  # NumPy 用于概率网格、团簇矩、随机抽样和离散尾部似然。
import pandas as pd  # Pandas 用于组织配置样本、池化频数和结果表。
from scipy.optimize import minimize, minimize_scalar  # SciPy 的 minimize 和 minimize_scalar 分别优化双参数与单参数尾部模型。
from scipy.special import logsumexp  # SciPy 的稳定对数求和用于计算离散模型归一化常数。

from .networks import sample_lattice_occupancy  # src/local_market/networks.py，按 p 独立抽取格点占据状态。
from .percolation import identify_lattice_clusters  # src/local_market/percolation.py，标记开放边界的四邻域团簇及贯通方向。
from .statistics import finite_cluster_sizes  # src/local_market/statistics.py，按最大团簇或贯通标签筛除团簇。


RULES = ("largest_excluded", "all_spanning_excluded")  # 两种有限团簇排除口径，分别移除一个最大团簇或全部贯通团簇。
EVENTS = ("span_x", "span_y", "span_any", "span_both")  # 贯通指标依次为左右、上下、至少一向、两个方向均有贯通。


def ratio_grid(coarse: list[float], center: float, half_width: float,
               common_window: list[float], step: float) -> tuple[float, ...]:
    """Union of original grid, local peak window, and common crossing window."""
    # 为渗流临界性质实验合并粗网格、局部峰窗和公共交叉窗。
    # 引用
    #     scripts/simulation/scan_criticality_refined.py 的 main() 为每个格点尺寸建立扫描概率比。
    # 输入
    #     coarse 为原有的 p/p_c 粗网格。
    #     center 为待细扫的峰所在概率比。
    #     half_width 为正的峰窗口半宽，单位为 p/p_c。
    #     common_window 为两个端点组成的公共窗口，供跨尺寸比较。
    #     step 为正的细网格间距。
    # 输出 按升序排列且去重的概率比元组，各点保留六位小数。
    if step <= 0 or half_width <= 0:
        raise ValueError("grid step and half width must be positive")
    def window(low: float, high: float) -> set[float]:
        # 在给定端点之间生成一个细网格集合。
        # 引用
        #     ratio_grid() 分别调用其内部的 window() 生成峰附近和贯通曲线的公共网格。
        # 输入
        #     low 为窗口下端点。
        #     high 为窗口上端点。读取外层的 step。
        # 输出 以六位小数表示的网格点集合，终点容差取半个步长。
        return {round(float(x), 6) for x in np.arange(low, high + step / 2, step)}  # 扩到上端半个步长，让恰好落在步距上的上端点也被纳入。
    # 将三套概率比统一精度、合并去重并排序。
    return tuple(sorted({round(x, 6) for x in coarse} |
                        window(center - half_width, center + half_width) |
                        window(*common_window)))


@dataclass
class ScanResult:
    samples: pd.DataFrame  # 逐配置的序参量、贯通事件及有限团簇矩。
    histograms: pd.DataFrame  # 逐配置的稀疏规模频数表，流式追加模式在内存中只留 p/p_c≈1 的行。
    runtime_seconds: float  # 本次扫描耗时，单位为秒。
    pooled_counts: pd.DataFrame | None = None  # 跨配置池化的原始频数表，数据类默认 None。


def scan(lengths: list[int], ratios_by_length: dict[int, tuple[float, ...]],
         counts: dict[int, int], p_c: float, seed: int,
         histogram_output: Path | None = None) -> ScanResult:
    """Use each independent occupancy configuration for every observable."""
    # 为渗流临界性质实验用同一份开放格点配置测量序参量、贯通、有限团簇矩和规模频数。
    # 引用
    #     scripts/simulation/scan_criticality_refined.py 的 main() 扫描主网格和猜测峰位附近的补充网格，汇总所有观测。
    # 输入
    #     lengths 为正方格点边长列表。
    #     ratios_by_length 为按边长索引的 p/p_c 网格。
    #     counts 为各边长在每个概率点的配置数。
    #     p_c 为参照临界占据概率，乘比例得到合法 p。
    #     seed 为可复现的主随机种子。
    #     histogram_output 可选逐配置直方图 CSV 路径，默认全部保存在内存。
    # 输出 ScanResult，含逐配置观测、直方图、池化计数和耗时。传入路径时还会追加直方图记录。
    start = perf_counter()  # 用单调计时器记录扫描起点。
    rows = []
    hist = []
    pooled = Counter()  # 为不同尺寸、概率、排除口径和规模累计跨配置频数。
    stream = None
    writer = None
    if histogram_output is not None:
        existed = histogram_output.exists()  # 只在目标文件此前不存在时写列名。
        stream = histogram_output.open("a", newline="", encoding="utf-8")  # 以追加模式打开文件，保留之前扫描的记录。
        writer = csv.writer(stream)  # 建立逐配置频数的 CSV 写出器。
        if not existed:
            # 写下能唯一描述配置、排除口径和团簇规模的列名。
            writer.writerow(("length", "p_ratio", "p", "configuration_id",
                             "exclusion", "cluster_size", "count"))
    for length in lengths:
        for ratio in ratios_by_length[length]:
            p = p_c * ratio
            for sample in range(counts[length]):  # 同一尺寸和概率下每个配置有独立的样本索引。
                # 按种子、尺寸、百万分位概率比和配置号派生独立随机流。
                rng = np.random.default_rng(np.random.SeedSequence(
                    [seed, length, int(round(ratio * 1_000_000)), sample]))
                # 按开放四邻域识别团簇，供两套排除口径共用。
                clusters = identify_lattice_clusters(
                    sample_lattice_occupancy(length, p, rng), "open")  # 在当前 p 抽取一份独立占据阵列。
                x, y = clusters.spans_horizontal, clusters.spans_vertical
                row = {"length": length, "p_ratio": ratio, "p": p,
                       "configuration_id": sample, "occupied_count": clusters.occupied_count,
                       "largest_size": clusters.largest_size,
                       "p_infinity": clusters.p_infinity,  # 最大团簇规模除以 L²。
                       "span_x": int(x), "span_y": int(y),  # 分别记录是否左右贯通和是否上下贯通。
                       "span_any": int(x or y), "span_both": int(x and y)}  # 分别记录至少一个方向贯通和两个方向均有贯通。
                sets = {"all": clusters.sizes,
                        "largest_excluded": finite_cluster_sizes(clusters, "largest"),  # 剔除恰好一个最大团簇，供第一套矩统计使用。
                        "all_spanning_excluded": finite_cluster_sizes(clusters, "spanning")}  # 剔除所有贯通团簇，供第二套矩统计使用。
                for rule in RULES:
                    sizes = sets[rule]
                    row[f"{rule}_m1"] = int(sizes.sum())  # 有限团簇一阶矩为规模和。
                    row[f"{rule}_m2"] = int(np.dot(sizes, sizes))  # 有限团簇二阶矩为规模平方和。
                rows.append(row)  # 保存配置层观测，以便下游联合重抽。
                configuration_hist = []
                for rule, sizes in sets.items():  # 对全部团簇和两套排除口径分别生成直方图。
                    for size, count in Counter(sizes.tolist()).items():  # 对这份配置、这一套口径中的团簇按规模计数。
                        record = (length, ratio, p, sample, rule, size, count)
                        configuration_hist.append(record)  # 保存逐配置的非零频数。
                        pooled[(length, ratio, p, rule, size)] += count
                if writer is not None:
                    writer.writerows(configuration_hist)  # 将当前配置的所有频数直接追加到磁盘。
                    if np.isclose(ratio, 1.0):  # 流式模式仍在内存保留临界概率附近的逐配置数据供尾拟合。
                        hist.extend(configuration_hist)
                else:
                    hist.extend(configuration_hist)  # 纯内存模式保留各概率点的逐配置数据。
    if stream is not None:
        stream.close()  # 释放直方图流并完成缓冲区写出。
    # 为逐配置稀疏频数建立固定列顺序。
    histograms = pd.DataFrame(hist, columns=["length", "p_ratio", "p",
        "configuration_id", "exclusion", "cluster_size", "count"])
    # 把池化计数整理为后续 n_s 归一化所需的长表。
    pooled_frame = pd.DataFrame(
        [(length, ratio, p, rule, size, count)
         for (length, ratio, p, rule, size), count in pooled.items()],  # 遍历按尺寸、概率、排除口径和规模累计的计数。
        columns=["length", "p_ratio", "p", "exclusion", "cluster_size", "C_s"])
    return ScanResult(pd.DataFrame(rows), histograms, perf_counter() - start, pooled_frame)  # 打包配置观测、两种频数表和经过秒数。


def _ratio(second: np.ndarray, first: np.ndarray) -> float:
    # 计算用于渗流临界性质实验的池化易感性。
    # 引用
    #     summarize() 计算原样本各排除口径的 χ，pseudocritical() 在 bootstrap 概率网格上重算 χ 曲线。
    # 输入
    #     second 为各配置有限团簇的平方和数组。
    #     first 为与 second 对齐的各配置有限团簇规模和。
    # 输出 平方和总和除以规模总和，分母为零时返回 0.0。
    denominator = first.sum()  # 先全量合并一阶矩作为池化分母。
    return float(second.sum() / denominator) if denominator else 0.0  # 对汇总量取比值。


def summarize(samples: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Bootstrap whole occupancy configurations, keeping their observables together."""
    # 为渗流临界性质实验按配置重抽，估计序参量、贯通概率和池化易感性的区间。
    # 引用
    #     scripts/simulation/scan_criticality_refined.py 的 main() 保存渗流汇总，传给峰位分析和绘图。
    # 输入
    #     samples 为 scan() 的逐配置样本表，含尺寸、p、p/p_c、序参量、贯通标志和两套团簇矩。
    #     replicates 为大于零的 bootstrap 重复次数。
    #     seed 为重抽随机种子。
    # 输出 每组尺寸和概率一行的汇总表，包含点估计、标准差或二项标准误及 95% 百分位区间。
    rng = np.random.default_rng(seed)  # 建立配置层重抽的随机流。
    rows = []
    for (length, ratio, p), group in samples.groupby(["length", "p_ratio", "p"]):  # 只在同一边长和同一占据概率内重抽。
        n = len(group)
        row = {"length": length, "p_ratio": ratio, "p": p, "samples": n}
        indices = rng.integers(0, n, size=(replicates, n))  # 生成形状 (replicates, n) 的索引矩阵，所有观测量共用。
        for column in ("p_infinity", *EVENTS):
            values = group[column].to_numpy(dtype=float)  # 按配置顺序取当前观测量。
            boots = values[indices].mean(axis=1)  # 每行重抽样本沿配置轴取平均。
            row[f"{column}_mean"] = float(values.mean())  # 原始配置的平均值为点估计。
            row[f"{column}_std"] = float(values.std(ddof=1)) if n > 1 else 0.0  # 保存配置间样本标准差，单个配置时取零。
            row[f"{column}_ci_low"], row[f"{column}_ci_high"] = np.quantile(boots, [0.025, 0.975])  # 用重抽均值的 2.5% 和 97.5% 分位数给出区间。
            if column in EVENTS:
                q = values.mean()  # 二值事件的样本均值就是观测贯通频率。
                row[f"{column}_binomial_se"] = float(np.sqrt(q * (1 - q) / n))  # 用 sqrt(q(1−q)/n) 估计贯通概率的二项标准误。
        for rule in RULES:
            first = group[f"{rule}_m1"].to_numpy(dtype=np.int64)  # 取配置层规模和作为各轮分母的来源。
            second = group[f"{rule}_m2"].to_numpy(dtype=np.int64)  # 取对应平方和作为各轮分子的来源。
            numerator = second[indices].sum(axis=1)  # 按相同配置索引逐轮汇总二阶矩。
            denominator = first[indices].sum(axis=1)  # 按相同配置索引逐轮汇总一阶矩。
            boots = np.divide(numerator, denominator, out=np.zeros(replicates), where=denominator != 0)  # 逐轮取池化矩比值，空分母时取 0。
            row[f"chi_{rule}"] = _ratio(second, first)  # 对原始配置计算池化易感性点估计。
            row[f"chi_{rule}_m1"] = int(first.sum())  # 保留原样本池化分母。
            row[f"chi_{rule}_m2"] = int(second.sum())  # 保留原样本池化分子。
            row[f"chi_{rule}_ci_low"], row[f"chi_{rule}_ci_high"] = np.quantile(boots, [0.025, 0.975])  # 从池化比值的重抽分布读取 95% 百分位区间。
        rows.append(row)  # 保存本尺寸与概率的所有观测量汇总行。
    return pd.DataFrame(rows)  # 将分组记录转换为分析表。


def _peak(p: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    # 取得渗流临界性质实验的网格峰位，并尝试局部二次细化。
    # 引用
    #     pseudocritical() 分别计算各尺寸、各排除口径的原样本易感性峰。
    # 输入
    #     p 为一维非空升序占据概率。
    #     values 为与 p 等长的易感性值。
    # 输出依次为
    #     离散最大值处的 p。
    #     局部三点拟合的抛物线顶点，边界或无有效顶点时为 NaN。
    #     离散网格最大值。
    index = int(np.argmax(values))  # 离散网格中并列最高值取第一个索引。
    grid = float(p[index])  # 将最高值的数组索引映射到实际 p。
    interpolated = np.nan
    if 0 < index < len(p) - 1:  # 只有内部网格点才有可用于插值的双侧邻点。
        a, b, _ = np.polyfit(p[index - 1:index + 2], values[index - 1:index + 2], 2)  # 用最高点及其左右邻点拟合局部二次多项式。
        vertex = -b / (2 * a) if a < 0 else np.nan  # 只在开口向下时计算顶点。
        if np.isfinite(vertex) and p[index - 1] <= vertex <= p[index + 1]:  # 顶点必须有限且留在三个采样点覆盖的区间。
            interpolated = float(vertex)
    return grid, interpolated, float(values[index])  # 同时返回网格位置、可用的插值位置和网格峰高。


def pseudocritical(samples: pd.DataFrame, summary: pd.DataFrame,
                   replicates: int, seed: int) -> pd.DataFrame:
    # 为渗流临界性质实验估计各边长和排除口径的易感性峰位，并用配置层重抽给出不确定性。
    # 引用
    #     scripts/simulation/scan_criticality_refined.py 的 main() 保存峰位，传给精细渗流绘图。
    # 输入
    #     samples 为 scan() 的配置表，供各 p 上重抽。
    #     summary 为 summarize() 的统计表，提供各尺寸的概率和两种 χ 曲线。
    #     replicates 为正的重抽次数。
    #     seed 为峰位重抽的随机种子。
    # 输出 每个尺寸和排除口径一行，含网格峰、二次插值峰、峰高、网格峰置信区间和边界信息。
    rng = np.random.default_rng(seed)  # 创建峰位重抽的随机流。
    rows = []
    for length, group in summary.groupby("length"):  # 每个边长独立估计峰位。
        group = group.sort_values("p")  # 按占据概率排序，为相邻网格插值准备顺序。
        p = group.p.to_numpy()  # 取得一维实际概率网格，每个尺寸至少需要两个点来报告间距。
        subsets = [samples[(samples.length == length) & np.isclose(samples.p, value)] for value in p]  # 为每个网格点找出相同尺寸的可重抽配置。
        for rule in RULES:
            column = f"chi_{rule}"
            grid, vertex, height = _peak(p, group[column].to_numpy())  # 计算原样本的离散峰位、可用的二次顶点和离散峰高。
            estimates = []
            for _ in range(replicates):
                curve = []
                for subset in subsets:
                    indices = rng.integers(0, len(subset), len(subset))  # 在当前网格点有放回重抽整份配置。
                    # 按同一索引重抽一阶和二阶矩，并计算该点的池化易感性。
                    curve.append(_ratio(subset[f"{rule}_m2"].to_numpy()[indices],
                                        subset[f"{rule}_m1"].to_numpy()[indices]))
                estimates.append(float(p[np.argmax(curve)]))  # 以本轮重抽曲线的离散最大值作为峰位样本。
            low, high = np.quantile(estimates, [0.025, 0.975])  # 峰位区间由网格峰 bootstrap 分布的 2.5% 和 97.5% 分位数给出。
            # 保存估计、区间和网格分辨率。
            rows.append({"length": length, "exclusion": rule, "p_grid_peak": grid,
                         "p_quadratic_peak": vertex, "chi_peak": height,
                         "p_ci_low": low, "p_ci_high": high,
                         "minimum_grid_spacing": float(np.min(np.diff(p))),  # 用相邻概率差的最小值表征网格分辨率。
                         "peak_at_grid_boundary": bool(grid in (p[0], p[-1]))})  # 标明最大值是否落在所扫描区间的两端。
    return pd.DataFrame(rows)  # 合并所有尺寸和排除口径的峰位记录。


def crossing(p: np.ndarray, a: np.ndarray, b: np.ndarray, p_c: float) -> float:
    """Linear interpolation in sign-changing brackets, closest to theory if multiple."""
    # 在渗流临界性质实验的公共 p 网格中寻找两条曲线的交叉位置。
    # 引用
    #     crossings() 分别对两尺寸的原始水平贯通曲线和 bootstrap 曲线调用。
    # 输入
    #     p 为非空的一维共同概率网格，按升序排列。
    #     a 为与 p 对齐的第一条贯通概率曲线。
    #     b 为与 p 对齐的第二条贯通概率曲线。
    #     p_c 为理论临界概率，用于多交叉点时择近。
    # 输出 精确网格相等点或线性插值交叉点，无交叉时为 NaN。
    difference = a - b
    candidates = []
    for i in range(len(p) - 1):  # 逐个检查网格点及其后方区间。
        if difference[i] == 0:
            candidates.append(float(p[i]))  # 两曲线恰在网格点相等时直接保留该点。
        elif difference[i] * difference[i + 1] < 0:
            # 相邻差值严格异号时，对差值的零点做线性插值。
            candidates.append(float(p[i] - difference[i] * (p[i + 1] - p[i]) /
                                    (difference[i + 1] - difference[i])))
    if difference[-1] == 0:
        candidates.append(float(p[-1]))  # 循环只访问到倒数第二个网格点，单独纳入末端相等点。
    return min(candidates, key=lambda value: abs(value - p_c)) if candidates else np.nan  # 有多个候选时取与理论临界概率距离最小的点。


def crossings(samples: pd.DataFrame, lengths: list[int], common_ratios: list[float],
              p_c: float, replicates: int, seed: int) -> pd.DataFrame:
    # 为渗流临界性质实验估计相邻尺寸的水平贯通交叉点和重抽样区间。
    # 引用
    #     scripts/simulation/scan_criticality_refined.py 的 main() 保存交叉点，供绘图标注。
    # 输入
    #     samples 为 scan() 的逐配置样本表，使用 length、p_ratio 和 span_x。
    #     lengths 为按比较顺序排列的尺寸列表，每对相邻项构成一对。
    #     common_ratios 为各尺寸共用的升序 p/p_c 网格。
    #     p_c 为把概率比转换为实际概率的临界参照。
    #     replicates 为正的重抽次数。
    #     seed 为重抽随机种子。
    # 输出 逐尺寸对的交叉点与 95% 区间表，含有限交叉的重抽比例，缺少共同网格样本的尺寸对跳过。
    rng = np.random.default_rng(seed)  # 初始化贯通交叉重抽样的随机流。
    rows = []
    ratios = np.asarray(common_ratios)  # 将公共概率比转换成数组，便于生成实际概率。
    p = p_c * ratios
    for left, right in zip(lengths[:-1], lengths[1:]):  # 按照传入顺序配对相邻尺寸。
        data = {}
        for length in (left, right):
            # 在两个尺寸的每个公共概率点取出逐配置水平贯通标志。
            data[length] = [samples[(samples.length == length) &
                 np.isclose(samples.p_ratio, ratio)]["span_x"].to_numpy(dtype=float)
                 for ratio in ratios]
        if any(len(values) == 0 for groups in data.values() for values in groups):  # 任一尺寸的任一概率点缺数据时跳过该尺寸对。
            continue
        # 对每个尺寸、每个概率点取事件均值作为贯通频率。
        means = {length: np.array([values.mean() for values in groups])
                 for length, groups in data.items()}  # 分尺寸保存同一概率网格上的原样本贯通曲线。
        point = crossing(p, means[left], means[right], p_c)  # 在线性连接的两条原样本曲线上找交叉。
        boots = []
        for _ in range(replicates):
            # 每个尺寸和概率下独立有放回重抽配置，再求水平贯通频率。
            curves = {length: np.array([rng.choice(values, len(values)).mean()
                         for values in groups]) for length, groups in data.items()}  # 按尺寸和共同网格点分别汇总重抽曲线。
            boots.append(crossing(p, curves[left], curves[right], p_c))  # 在本轮两条贯通曲线上估计交叉。
        finite = np.asarray(boots)[np.isfinite(boots)]  # 只保留找到交叉的重抽值，用于估计区间。
        low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)  # 对有限交叉值取 95% 百分位区间，没有交叉时两端均为 NaN。
        # 保存交叉点、区间和有效重抽比例。
        rows.append({"length_left": left, "length_right": right, "p_crossing": point,
                     "p_ci_low": low, "p_ci_high": high,
                     "bootstrap_crossing_fraction": len(finite) / replicates,  # 以所有重抽轮数为分母，报告存在交叉的比例。
                     "multiple_crossing_rule": "nearest theoretical p_c"})
    # 返回固定列结构，在没有可比较尺寸对时仍保留列名。
    return pd.DataFrame(rows, columns=["length_left", "length_right", "p_crossing",
        "p_ci_low", "p_ci_high", "bootstrap_crossing_fraction", "multiple_crossing_rule"])


def pooled_histogram(histograms: pd.DataFrame, samples: pd.DataFrame,
                     pooled_counts: pd.DataFrame | None = None) -> pd.DataFrame:
    # 为渗流临界性质实验汇总规模频数，并换算每格点的团簇密度。
    # 引用
    #     scripts/simulation/scan_criticality_refined.py 的 main() 保存 n_s 和 C_s，供规模分布和尾部绘图。
    # 输入
    #     histograms 为逐配置规模频数长表，使用 count 及各分组列。
    #     samples 为逐配置样本表，用于统计各组独立配置数 M。
    #     pooled_counts 为可选的池化频数表，用 C_s 替代逐配置 count，默认用 histograms 汇总。
    # 输出 每尺寸、概率、口径和团簇规模一行，含 C_s、M 及 n_s=C_s/(ML²)。
    if pooled_counts is None:
        # 将逐配置频数按尺寸、概率、排除口径和规模求和，并命名为 C_s。
        counts = histograms.groupby(["length", "p_ratio", "p", "exclusion", "cluster_size"],
                                    as_index=False)["count"].sum().rename(columns={"count": "C_s"})
    else:
        # 对已池化的输入再合并同组行，可合并多个扫描批次。
        counts = pooled_counts.groupby(["length", "p_ratio", "p", "exclusion", "cluster_size"],
                                      as_index=False)["C_s"].sum()
    sizes = samples.groupby(["length", "p_ratio", "p"], as_index=False).size().rename(columns={"size": "M"})  # 按尺寸和概率数出实际配置总数 M。
    counts = counts.merge(sizes, on=["length", "p_ratio", "p"])  # 连接每个频数组所使用的配置数。
    counts["n_s"] = counts.C_s / (counts.M * counts.length**2)  # 以全部配置乘每配置格点数作为密度分母。
    return counts


def finite_powerlaw(values: np.ndarray, xmin: int, xmax: int) -> tuple[float, float, int]:
    """Exact normalized likelihood on the finite integer support, plus KS."""
    # 在闭的整数区间上拟合渗流临界性质实验的有限支持纯幂律。
    # 引用
    #     tail_fits() 用于选下界、重抽配置和固定区间拟合优度检验。
    #     scripts/analysis/criticality_validation_analysis.py 的 tail_audit() 复核团簇尾部。
    # 输入
    #     values 为一维正整数团簇规模数组。
    #     xmin 为正的拟合下界，含端点。
    #     xmax 为拟合上界，含端点，须大于 xmin。
    # 输出依次为
    #     离散幂律指数。
    #     模型与经验 CDF 的 KS 距离。
    #     区间内样本数。区间无跨度或样本少于 30 时前两项为 NaN。
    tail = np.sort(values[(values >= xmin) & (values <= xmax)])  # 保留区间内观测并按规模升序排列。
    if len(tail) < 30 or xmax <= xmin:  # 至少需要 30 个尾部观测和有跨度的整数区间。
        return np.nan, np.nan, len(tail)  # 保留观察数，指数和 KS 使用 NaN 标记。
    support = np.arange(xmin, xmax + 1)  # 明确枚举 xmin 到 xmax 的全部整数支持点。
    logs = np.log(support)  # 缓存支持点对数，以便优化归一化项。
    def nll(alpha: float) -> float:
        # 计算有限支持纯幂律的负对数似然。
        # 引用
        #     finite_powerlaw() 把内部 nll() 交给 scipy.optimize.minimize_scalar() 按候选指数调用。
        # 输入
        #     alpha 为候选幂律指数，读取外层 tail 和 logs。
        # 输出 负对数似然，含全部整数支持点的归一化项。
        return float(alpha * np.log(tail).sum() + len(tail) * logsumexp(-alpha * logs))  # 把观测对数和与对数归一化常数相加。
    fit = minimize_scalar(nll, bounds=(0.01, 6), method="bounded")  # 在给定边界内优化单个幂律指数。
    alpha = float(fit.x)  # 提取最优的离散纯幂律指数。
    model = np.exp(-alpha * logs - logsumexp(-alpha * logs))  # 用稳定对数归一化在每个整数点计算概率。
    empirical = np.searchsorted(tail, support, side="right") / len(tail)  # 按排序位置得到每个支持点的经验累计比例，含相等观测。
    return alpha, float(np.max(np.abs(empirical - np.cumsum(model)))), len(tail)  # 对经验 CDF 和模型累计概率计算最大绝对差。


def finite_cutoff_powerlaw(values: np.ndarray, xmin: int, xmax: int) -> tuple[float, float, float]:
    """Two-parameter cutoff model normalized on the comparison interval."""
    # 在闭的整数区间上拟合渗流临界性质实验的截断幂律。
    # 引用
    #     tail_fits() 分别在完整观察尾部和共同有限区间拟合该模型，供截断尺度估计和模型比较。
    # 输入
    #     values 为一维正整数团簇规模数组，拟合区间内应有观测。
    #     xmin 为正的拟合下界，含端点。
    #     xmax 为拟合区间的包含式整数上界。
    # 输出依次为
    #     截断幂律指数。
    #     正截断尺度 s_c。
    #     优化所得的负对数似然。
    tail = values[(values >= xmin) & (values <= xmax)]  # 只在指定闭区间内估计参数。
    support = np.arange(xmin, xmax + 1)  # 枚举离散归一化支持。
    logs = np.log(support)  # 预计算支持点的对数幂律项。
    def nll(params: np.ndarray) -> float:
        # 计算有限支持截断幂律的负对数似然。
        # 引用
        #     finite_cutoff_powerlaw() 将内部 nll() 交给 scipy.optimize.minimize() 计算优化目标。
        # 输入
        #     params 为含幂律指数和 log(s_c) 的二元素向量，读取外层 logs、support 和 tail。
        # 输出 给定参数的负对数似然。
        alpha, log_sc = params
        weights = -alpha * logs - support / np.exp(log_sc)  # 将幂律项与指数截断项相加得到未归一化对数权重。
        # 用尾样本贡献和稳定计算的配分函数组成负对数似然。
        return float(alpha * np.log(tail).sum() + tail.sum() / np.exp(log_sc) +
                     len(tail) * logsumexp(weights))
    # 在有界指数与对数尺度上拟合，以尾均值提供截断尺度初值。
    fit = minimize(nll, [2.0, np.log(max(float(np.mean(tail)), 2.0))],
                   bounds=[(0.01, 6), (np.log(1), np.log(max(xmax * 100, 2)))])
    return float(fit.x[0]), float(np.exp(fit.x[1])), float(fit.fun)  # 将优化的对数尺度转换回团簇规模单位。


def tail_fits(histograms: pd.DataFrame, ratio: float, thresholds: list[int],
              replicates: int, seed: int) -> pd.DataFrame:
    """Fit finite support and cutoff models, resampling whole configurations."""
    # 为渗流临界性质实验选择有限团簇尾拟合区间，重抽配置并比较纯幂律、指数和截断幂律。
    # 引用
    #     scripts/simulation/scan_criticality_refined.py 的 main() 保存拟合表，供渗流临界报告和尾部图使用。
    # 输入
    #     histograms 为 scan() 的逐配置频数表，含尺寸、p_ratio、排除规则、配置编号、规模和 count。
    #     ratio 为待拟合的 p/p_c，按浮点近似相等匹配。
    #     thresholds 为整数规模频数的筛选阈值列表，各阈值单独确定上界。
    #     replicates 为正的配置 bootstrap 次数，拟合优度另最多做 200 次参数重抽。
    #     seed 为两类重抽样的随机种子。
    # 输出 每尺寸、排除规则和频数阈值一行的拟合表，含成功状态、区间、指数、诊断和模型比较，数据不足的组合记录失败状态。
    rng = np.random.default_rng(seed)  # 建立尾参数和拟合优度重抽使用的随机流。
    rows = []
    selected = histograms[np.isclose(histograms.p_ratio, ratio)]  # 筛出目标占据概率比附近的频数记录。
    for (length, rule), group in selected.groupby(["length", "exclusion"]):  # 按边长和排除口径分别拟合。
        if rule == "all":
            continue
        by_config = {int(sample): part for sample, part in group.groupby("configuration_id")}  # 保留有直方图记录的配置，按配置 ID 索引。
        ids = np.array(sorted(by_config))  # 获取升序配置 ID 作为可重抽的抽样单位。
        def expand(chosen: np.ndarray) -> np.ndarray:
            # 把所选配置的规模频数展开成逐团簇观测。
            # 引用
            #     tail_fits() 先以全部配置生成原样本，再用其内部 expand() 生成每个配置重抽样本。
            # 输入
            #     chosen 为一维配置 ID 序列，允许重复，读取外层 by_config。
            # 输出 一维整数团簇规模，所有重复配置按抽中次数展开，无配置时为空。
            # 在配置内部按计数重复每个规模。
            parts = [np.repeat(by_config[int(i)].cluster_size.to_numpy(dtype=int),
                               by_config[int(i)]["count"].to_numpy(dtype=int)) for i in chosen]
            return np.concatenate(parts) if parts else np.array([], dtype=int)  # 拼接所选配置的观测，空选择用整数空数组。
        values = expand(ids)  # 用全部配置得到原始尾拟合样本。
        if len(values) == 0:
            continue
        size_counts = np.bincount(values)  # 用规模作整数下标，给每种规模计数。
        for threshold in thresholds:
            candidates = np.flatnonzero(size_counts >= threshold)  # 找出达到给定计数要求的规模。
            xmax = int(candidates.max()) if len(candidates) else 0  # 用满足计数阈值的最大规模作为有限窗口上界。
            options = []
            for xmin in (1, 2, 3, 5, 10, 20):
                alpha, ks, n = finite_powerlaw(values, xmin, xmax)  # 在当前候选下界和固定上界之间拟合纯幂律。
                if np.isfinite(alpha) and n >= 100:
                    options.append((ks, xmin, alpha, n))
            if not options:
                rows.append({"length": length, "p_ratio": ratio, "exclusion": rule,  # 记录数据不足的组合和已尝试的上界。
                             "count_threshold": threshold, "s_max_fit": xmax,
                             "status": "insufficient_data"})
                continue
            ks, xmin, alpha, n = min(options)  # 优先取 KS 最小的候选，平局按元组后续字段排序。
            estimates = []
            for _ in range(replicates):
                boot = expand(rng.choice(ids, len(ids), replace=True))  # 有放回抽同数目的配置，并展开对应团簇规模。
                boot_counts = np.bincount(boot) if len(boot) else np.array([])  # 重新计算本轮规模的离散频数。
                eligible = np.flatnonzero(boot_counts >= threshold)  # 在本轮样本中重新应用频数阈值。
                upper = int(eligible.max()) if len(eligible) else 0
                # 在本轮上界下重选下界，要求指数有限且至少有 100 个尾样本。
                possible = [(distance, lower, value) for lower in (1, 2, 3, 5, 10, 20)
                            for value, distance, count in [finite_powerlaw(boot, lower, upper)]  # 对每个候选下界重新拟合有限支持纯幂律。
                            if np.isfinite(value) and count >= 100]  # 只保留有限指数且至少有 100 个尾部观测的候选。
                if possible:
                    estimates.append(min(possible)[2])  # 保存本轮 KS 最小候选的指数。
            low, high = np.quantile(estimates, [0.025, 0.975]) if estimates else (np.nan, np.nan)  # 用有效重抽指数的 95% 百分位区间报告不确定性。
            cutoff_alpha, sc, cutoff_nll = finite_cutoff_powerlaw(values, xmin, int(values.max()))  # 在 xmin 至最大观测规模的完整尾部拟合截断尺度。
            fit_values = values[(values >= xmin) & (values <= xmax)]
            support = np.arange(xmin, xmax + 1)  # 建立模型公平比较所使用的共同离散区间。
            # 在共同区间重算纯幂律的负对数似然，用于计算 AIC。
            power_nll = float(alpha * np.log(fit_values).sum() +
                              len(fit_values) * logsumexp(-alpha * np.log(support)))
            # 在共同区间用一维有界优化拟合指数率。
            exp_fit = minimize_scalar(lambda rate: float(rate * fit_values.sum() +
                len(fit_values) * logsumexp(-rate * support)), bounds=(1e-8, 2), method="bounded")
            finite_cutoff_alpha, finite_sc, finite_cutoff_nll = finite_cutoff_powerlaw(values, xmin, xmax)  # 在共同区间拟合截断幂律，用于比较 AIC。
            # 计算有限支持纯幂律模型的 CDF，供固定区间参数重抽。
            model_cdf = np.cumsum(np.exp(-alpha * np.log(support) -
                                          logsumexp(-alpha * np.log(support))))
            gof_exceedances = 0
            for _ in range(min(replicates, 200)):
                synthetic = rng.choice(support, size=n, p=np.diff(np.r_[0, model_cdf]))  # 由相邻 CDF 差得到概率质量并抽取与原尾样本等量的规模。
                _, synthetic_ks, _ = finite_powerlaw(synthetic, xmin, xmax)  # 保持区间不变，对合成数据重拟合指数并计算 KS。
                gof_exceedances += int(synthetic_ks >= ks)
            gof_replicates = min(replicates, 200)
            # 汇总估计、有效重抽频率及共同区间与完整尾部诊断。
            rows.append({"length": length, "p_ratio": ratio, "exclusion": rule,
                         "count_threshold": threshold, "s_min": xmin, "s_max_fit": xmax,
                         "tail_count": n, "alpha_pdf": alpha, "alpha_ci_low": low,
                         "alpha_ci_high": high, "bootstrap_valid_fraction": len(estimates) / replicates,
                         "ks_distance": ks, "powerlaw_aic": 2 + 2 * power_nll,  # 纯幂律的 AIC 惩罚计一个自由参数。
                         "fixed_interval_gof_p": (gof_exceedances + 1) / (gof_replicates + 1),  # 固定区间 KS 检验用加一修正得 p 值。
                         "gof_replicates": gof_replicates,
                         "exponential_aic_same_interval": 2 + 2 * exp_fit.fun,  # 指数分布在相同区间计一个自由参数。
                         "cutoff_aic_same_interval": 4 + 2 * finite_cutoff_nll,  # 截断幂律在相同区间计两个自由参数。
                         "cutoff_alpha_same_interval": finite_cutoff_alpha,
                         "cutoff_s_c_same_interval": finite_sc,
                         "cutoff_alpha_pdf": cutoff_alpha, "cutoff_s_c": sc,
                         "cutoff_aic_full_tail": 4 + 2 * cutoff_nll,  # 另存完整尾部截断拟合的 AIC，区间可能与纯幂律不同。
                         "status": "ok", "upper_rule": "largest observed s with C_s >= threshold"})
    return pd.DataFrame(rows)  # 将各排除规则和阈值的结果整理为长表。
