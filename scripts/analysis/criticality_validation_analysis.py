"""Read-only audit of completed square-lattice criticality results.

Writes only to results/criticality_validation_20261113. No occupancy sampling.
"""
# 对渗流临界性质实验的方格点精细扫描做构型 bootstrap 和独立诊断。
# bootstrap_curves() 重建观测曲线。pseudocritical_extrapolation()、crossing_audit() 和 symmetry_audit() 分别检验峰位、曲线交点和贯通对称性。
# tail_audit() 复核团簇尾区间。susceptibility_scaling() 和 collapse_audit() 比较易感性标度和跨格长坍缩。

# main() 接收精细扫描及 analyze_criticality_refined.py 的结果。criticality_validation_gridaware.py 读取这里生成的外推区间与逐轮峰位。

from __future__ import annotations  # 延后求值类型注解。

import hashlib  # 计算分析脚本、输入清单及产物的 SHA-256。
import json  # 将审计来源和哈希清单写成 JSON。
from pathlib import Path  # 表示扫描输入与独立验证输出路径。

import matplotlib  # 配置用于批量保存验证图的绘图后端。
matplotlib.use("Agg")  # 在无图形窗口环境生成 PNG。
import matplotlib.pyplot as plt  # 绘制伪临界点外推和坍缩质量对照图。
import numpy as np  # 重采样构型、进行插值与最小二乘并估计区间。
import pandas as pd  # 读取精细扫描表、按条件筛选并保存审计结果。
from scipy.stats import binomtest  # 在横纵贯通结果不一致的构型上做精确二项检验。

ROOT = Path("results/criticality_revision_20261113/full")  # 精细扫描和扫描后分析的归档目录。
OUT = Path("results/criticality_validation_20261113")  # 独立验证的产物目录。
PC = 0.592746  # 二维方格点渗流的理论临界占据概率。
LENGTHS = (32, 64, 128, 256, 512)
RULES = ("largest_excluded", "all_spanning_excluded")  # χ 的两种团簇剔除口径，顺序与 χ 曲线列一致。
B = 1000
RNG = np.random.default_rng(20261113 + 810)  # 按固定种子初始化构型 bootstrap 的模块级随机流。
COLS = ("p_infinity", "chi_largest_excluded", "chi_all_spanning_excluded", "span_x", "span_y")  # 点估计和 bootstrap 曲线共享这五个观测量列。


def get_inputs():
    # 功能：按固定顺序读取渗流临界性质实验的六张归档输入表。
    # 引用：main() 解包六张表，用作曲线、峰位、对称性及尾拟合审计。
    # 输入：无函数参数。从 ROOT/tables 读取。
    # 输出：返回按需读取的生成器，依次产出以下六张 DataFrame。
    # configuration_observables：逐构型占据和团簇观测量。
    # summary：各格长和占据概率条件的汇总。
    # cluster_histogram：团簇规模频数。
    # cluster_tail_fits：按频数门槛保存的团簇尾拟合。
    # pseudocritical：原扫描离散和二次插值峰位。
    # direction_symmetry：横纵贯通的构型配对比较。
    tables = ROOT / "tables"
    # 由迭代者解包时逐一读入归档 CSV。
    return (pd.read_csv(tables / name) for name in (
        "configuration_observables.csv", "summary.csv", "cluster_histogram.csv",
        "cluster_tail_fits.csv", "pseudocritical.csv", "direction_symmetry.csv"))


def bootstrap_curves(samples: pd.DataFrame, summary: pd.DataFrame):
    """Resample configurations at each (L,p), keeping all observables paired."""
    # 功能：在渗流临界性质实验的各格长与概率条件内重抽构型并重建五种观测曲线。
    # 引用：main() 将结果用于 pseudocritical_extrapolation()、crossing_audit()、susceptibility_scaling() 和 collapse_audit()。
    # 输入
    # samples：逐构型表，含格长、占据概率比、构型号、序参量、两方向贯通指示及两种 χ 口径的团簇一、二阶矩。
    # summary：归档条件汇总表，本函数当前只使用 samples。
    # 输出：以格长为键的字典，含一维升序概率比 ratio 和占据概率 p，point 形状为 (概率点数, 5)，boot 为 (B, 概率点数, 5)，列顺序见 COLS。
    curves = {}
    for length in LENGTHS:
        group = samples[samples.length == length]  # 提取当前格长的所有占据条件。
        ratios = np.sort(group.p_ratio.unique())  # 得到按升序排列的不同扫描概率比。
        point = np.empty((len(ratios), len(COLS)))  # 为各概率点分配五项观测量的点估计。
        boots = np.empty((B, len(ratios), len(COLS)))  # 为每轮构型重抽分配整条多观测量曲线。
        for j, ratio in enumerate(ratios):
            subset = group[np.isclose(group.p_ratio, ratio)].sort_values("configuration_id")  # 按概率比取样本，以构型编号固定行次序。
            n = len(subset)
            idx = RNG.integers(0, n, size=(B, n))  # 每轮有放回抽 n 个构型，五项观测共用这一批索引。
            pinf = subset.p_infinity.to_numpy()  # p∞ 是最大团簇规模除以 L²。
            sx = subset.span_x.to_numpy()  # 提取每个构型的横向贯通指示。
            sy = subset.span_y.to_numpy()  # 提取每个构型的纵向贯通指示。
            boots[:, j, 0] = pinf[idx].mean(axis=1)  # 每轮沿构型轴平均 p∞。
            boots[:, j, 3] = sx[idx].mean(axis=1)  # 每轮沿构型轴平均横向贯通指示，得到横向概率。
            boots[:, j, 4] = sy[idx].mean(axis=1)  # 同样计算纵向贯通概率。
            point[j, (0, 3, 4)] = (pinf.mean(), sx.mean(), sy.mean())  # 原构型均值用于这三个观测量的点估计。
            # 两种 χ 分别剔除最大团簇或全部贯通团簇。
            for k, rule in ((1, "largest_excluded"), (2, "all_spanning_excluded")):
                m1 = subset[f"{rule}_m1"].to_numpy()  # 一阶矩 m1 是这一口径保留团簇的规模和。
                m2 = subset[f"{rule}_m2"].to_numpy()  # 二阶矩 m2 是同一批团簇的规模平方和。
                numerator = m2[idx].sum(axis=1)  # 每轮汇总二阶矩作 χ 的分子。
                denominator = m1[idx].sum(axis=1)  # 每轮汇总一阶矩作 χ 的分母。
                # 先合计抽中构型的矩再求比，得到该概率下的 bootstrap χ。
                boots[:, j, k] = np.divide(numerator, denominator,
                    out=np.zeros(B), where=denominator != 0)  # 无保留团簇使分母为零时，该轮 χ 记为 0。
                point[j, k] = m2.sum() / m1.sum() if m1.sum() else 0  # 点估计用原构型的合计矩之比，零分母同样记为 0。
        curves[length] = {"ratio": ratios, "p": PC * ratios, "point": point, "boot": boots}  # 整理概率轴及观测数组，并由 p_c 还原绝对概率。
    return curves


def wls(x, y, sigma):
    # 功能：用已给定的误差尺度作带截距的加权最小二乘直线拟合。
    # 引用：pseudocritical_extrapolation() 用拟合系数和线性算子外推观测峰位及逐轮峰位。
    # 输入
    # x：一维横轴值，至少两个不同的点以确定斜率。
    # y：同 x 逐点对齐的响应。
    # sigma：同 x 逐点对齐的正标准差，用于设置逆方差权重。
    # 输出依次为
    # beta：长度为 2 的数组，依次为截距、斜率。
    # operator：形状为 (2, 点数) 的线性算子，可把同网格的新响应变成拟合系数。
    # chi2：残差除以各自标准差后的平方和。
    # r2：相对加权总离差的 R²，响应恒定且分母为零时为 NaN。
    X = np.column_stack((np.ones(len(x)), x))  # 将常数列和横轴值并为设计矩阵。
    weight = 1 / np.square(sigma)  # 用误差方差倒数加权。
    operator = np.linalg.solve(X.T @ (weight[:, None] * X), X.T * weight)  # 求解加权正规方程得到从响应到系数的算子。
    beta = operator @ y  # 用该算子求当前观测的两项系数。
    residual = y - X @ beta  # 求原响应减预测的残差。
    chi2 = float(np.sum((residual / sigma) ** 2))  # 计算标准化残差平方和。
    weighted_mean = np.average(y, weights=weight)  # 用同一组权重计算响应均值。
    denominator = np.sum(weight * (y - weighted_mean) ** 2)  # 以加权均值为中心求总离差平方和。
    r2 = 1 - np.sum(weight * residual**2) / denominator if denominator else np.nan  # 将加权残差与加权总离差相比。
    return beta, operator, chi2, float(r2)


def pseudocritical_extrapolation(curves, peaks):
    # 功能：复核渗流临界性质实验的易感性峰位，并按 L⁻³/⁴ 外推无限尺寸临界占据概率。
    # 引用：main() 将返回峰高送给 susceptibility_scaling()。scripts/analysis/criticality_validation_gridaware.py 的 main() 读取写出的逐轮峰位与区间。
    # 输入
    # curves：bootstrap_curves() 按格长组织的概率、点估计和重采样曲线。
    # peaks：精细扫描按格长和 χ 剔除口径记录的峰位、区间、最小网格间距等。
    # 输出：形状为 (B, 格长数, 2) 的重抽峰高数组，最后一轴按 RULES 排列。另写峰位核对、逐轮峰位、外推表和图。
    peak_boot = np.empty((B, len(LENGTHS), 2))  # 按重复轮次、格长、χ 口径保存峰位。
    height_boot = np.empty_like(peak_boot)  # 峰高和峰位采用同样的轴顺序。
    point_rows = []
    for i, length in enumerate(LENGTHS):
        data = curves[length]
        for k, rule in enumerate(RULES):
            column = k + 1
            indices = np.argmax(data["boot"][:, :, column], axis=1)  # 每轮在占据概率轴取 χ 的最大值索引，并列最大值取首个。
            peak_boot[:, i, k] = data["p"][indices]  # 用最大值索引从离散网格读出重抽峰位。
            height_boot[:, i, k] = data["boot"][np.arange(B), indices, column]  # 在同一轮的对应扫描点取峰高。
            stored = peaks[(peaks.length == length) & (peaks.exclusion == rule)].iloc[0]  # 查找同格长与剔除口径的原峰位记录。
            # 保存扫描峰位、插值峰位和重新估计的区间。
            point_rows.append({"length": length, "exclusion": rule,
                "stored_grid_peak": stored.p_grid_peak,
                "stored_quadratic_peak": stored.p_quadratic_peak,
                "stored_ci_low": stored.p_ci_low, "stored_ci_high": stored.p_ci_high,
                "rebootstrap_ci_low": np.quantile(peak_boot[:, i, k], .025),  # 用离散重抽峰位的 2.5% 分位作下界。
                "rebootstrap_ci_high": np.quantile(peak_boot[:, i, k], .975),  # 用离散重抽峰位的 97.5% 分位作上界。
                "rebootstrap_sd": np.std(peak_boot[:, i, k], ddof=1),  # 用 B−1 作分母衡量离散重抽峰位的标准差。
                "grid_spacing_min": stored.minimum_grid_spacing})
    # 写出逐格长和口径的峰位核对表。
    pd.DataFrame(point_rows).to_csv(OUT / "peak_bootstrap_audit.csv", index=False)
    # 将逐轮峰位展开，每列对应一个格长和一种口径。
    pd.DataFrame({f"L{length}_{rule}": peak_boot[:, i, k]
        for i, length in enumerate(LENGTHS) for k, rule in enumerate(RULES)}).to_csv(
            OUT / "bootstrap_peak_replicates.csv", index=False)  # 写出供网格修正使用的逐轮峰位表。

    fits = []
    for k, rule in enumerate(RULES):
        # 对比全部五个格长和最大的三个格长的外推。
        for subset_name, indices in (("all_5", np.arange(5)), ("large_3", np.arange(2, 5))):
            lengths = np.array(LENGTHS)[indices]  # 按当前子集取格长。
            x = lengths.astype(float)**(-.75)  # 横轴取 L⁻³/⁴，零对应无限格长。
            # 以原扫描观测的离散峰位作回归响应。
            p = np.array([peaks[(peaks.length == length) &
                         (peaks.exclusion == rule)].p_grid_peak.iloc[0] for length in lengths])
            spread = peak_boot[:, indices, k].std(axis=0, ddof=1)  # 对各格长沿重复轴计算峰位样本标准差。
            # 查找各格长与 χ 口径的最小概率网格间距。
            grid = np.array([peaks[(peaks.length == length) &
                         (peaks.exclusion == rule)].minimum_grid_spacing.iloc[0]
                         for length in lengths])
            sigma = np.sqrt(spread**2 + grid**2 / 12)  # 将构型方差与均匀网格单元的方差 grid²/12 合并。
            beta, operator, chi2, r2 = wls(x, p, sigma)  # 对观测峰位作加权直线拟合并取得线性算子。
            boot_beta = (operator @ peak_boot[:, indices, k].T).T  # 沿用同一算子拟合各轮离散峰位，得到 B 行截距和斜率。
            ci = np.quantile(boot_beta, [.025, .975], axis=0)  # 沿轮次求两项系数各自的 95% 百分位区间。
            # 保存外推点估计、区间、拟合优度和误差尺度。
            fits.append({"exclusion": rule, "subset": subset_name, "size_count": len(lengths),
                "intercept_pc_infinite": beta[0], "intercept_ci_low": ci[0, 0],  # 截距是 L 趋于无穷时估计的临界占据概率。
                "intercept_ci_high": ci[1, 0], "slope": beta[1],
                "slope_ci_low": ci[0, 1], "slope_ci_high": ci[1, 1],
                "chi_squared": chi2, "reduced_chi_squared": chi2 / (len(x) - 2),  # 约化卡方以剩余自由度，即格长数减二，为分母。
                "weighted_r_squared": r2, "theory_pc": PC,
                "grid_floor_applied": True,
                "sigma_min": sigma.min(), "sigma_max": sigma.max(),
                "estimator": "stored discrete grid peaks; configuration bootstrap"})
    # 写出构型 bootstrap 外推结果，供网格修正比较。
    pd.DataFrame(fits).to_csv(OUT / "pseudocritical_extrapolation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)  # 两个口径各占一幅图，共用峰位轴。
    for ax, rule in zip(axes, RULES):
        audit = pd.DataFrame(point_rows).query("exclusion == @rule").sort_values("length")  # 取当前口径并按格长排序。
        x = audit.length.to_numpy(dtype=float)**(-.75)  # 计算峰位图的有限尺寸横轴。
        y = audit.stored_grid_peak.to_numpy()  # 数据点为原扫描网格最大值的位置。
        # 将离散峰位区间换成相对数据点的上下误差长度。
        error = np.vstack((y - audit.rebootstrap_ci_low.to_numpy(),
                           audit.rebootstrap_ci_high.to_numpy() - y))
        ax.errorbar(x, y, yerr=np.maximum(error, 0), fmt="o", capsize=3, label="grid peak, 95% CI")  # 将负误差长度截为零后画出原峰位和区间。
        for row in fits:
            if row["exclusion"] == rule:
                xx = np.linspace(0, max(x), 100)  # 生成从无限格长极限起的直线横坐标。
                ax.plot(xx, row["intercept_pc_infinite"] + row["slope"] * xx,  # 叠加当前口径各格长子集的外推直线。
                        label=row["subset"])
        ax.axhline(PC, color="black", ls="--", label="theory")  # 标出理论临界占据概率。
        ax.set(xlabel=r"$L^{-3/4}$", title=rule)  # 标注有限尺寸横轴与 χ 的剔除口径。
        ax.grid(alpha=.25); ax.legend(fontsize=8)  # 添加读数网格和拟合子集图例。
    axes[0].set_ylabel(r"$p_c(L)$, discrete susceptibility peak")  # 纵轴量是 χ 最大值所在的离散占据概率。
    fig.tight_layout(); fig.savefig(OUT / "pseudocritical_extrapolation.png", dpi=180); plt.close(fig)  # 排版并保存外推图，随后关闭画布。
    # 将逐轮峰高交给易感性指数分析。
    return height_boot


def crossing_candidates(p, left, right):
    # 功能：在共同占据概率网格上枚举两条贯通曲线的交点候选。
    # 引用：crossing_audit() 对每对相邻格长、每个方向的点估计和 bootstrap 曲线求交。
    # 输入
    # p：非空的一维升序占据概率网格。
    # left：左侧格长在这些扫描点上的一维贯通概率。
    # right：右侧格长在相同扫描点上的一维贯通概率。
    # 输出：候选列表，每项依次为交点概率、相等或变号类型、左端网格索引。无交点时为空。
    d = left - right  # 以同一网格上的左曲线减右曲线作为求零函数。
    records = []
    for i in range(len(p) - 1):
        # 绝对差小于 10⁻¹⁴ 的扫描点直接作为交点。
        if abs(d[i]) < 1e-14:
            records.append((float(p[i]), "exact_equal", i))  # 保存近似相等点的概率及其网格索引。
        # 相邻差值异号时，线性插值得到相邻点之间的交点。
        elif d[i] * d[i + 1] < 0:
            records.append((float(p[i] - d[i] * (p[i + 1] - p[i]) /
                                  (d[i + 1] - d[i])), "sign_change", i))
    # 单独检查循环没有覆盖的最后一个网格点。
    if abs(d[-1]) < 1e-14:
        records.append((float(p[-1]), "exact_equal", len(p) - 1))  # 末点相等时，左端索引记为末点本身。
    return records


def crossing_audit(curves):
    # 功能：为渗流临界性质实验核对相邻格长贯通曲线的横纵交点及构型不确定性。
    # 引用：main() 用 bootstrap_curves() 的结果独立生成交点审计表。
    # 输入
    # curves：按格长索引的曲线字典，须含公共概率比网格及横纵贯通的点估计和 B 轮重抽值。
    # 输出：返回 None，写出所有观测交点、逐轮交点及含区间的交点汇总。
    common = np.round(np.arange(.97, 1.030001, .002), 6)  # 在 0.97 到 1.03 的临界邻域选择间隔 0.002 的共同概率比。
    candidate_rows, summary_rows, boot_rows = [], [], []
    # 依格长升序比较每对相邻尺寸。
    for left, right in zip(LENGTHS[:-1], LENGTHS[1:]):
        a, b = curves[left], curves[right]
        ia = np.array([np.flatnonzero(np.isclose(a["ratio"], q))[0] for q in common])  # 在较小格长的曲线中查找共同概率点的索引。
        ib = np.array([np.flatnonzero(np.isclose(b["ratio"], q))[0] for q in common])  # 对较大格长独立查找相同概率点。
        p = PC * common  # 将共同概率比换成实际占据概率。
        selected_boot = {}
        # 按 COLS 的列位置分别比较横向和纵向贯通。
        for direction, col in (("x", 3), ("y", 4)):
            candidates = crossing_candidates(p, a["point"][ia, col], b["point"][ib, col])  # 找到原样本两条曲线的全部交点候选。
            chosen = min(candidates, key=lambda value: abs(value[0] - PC)) if candidates else None  # 取最靠近理论 p_c 的候选作为点估计。
            for value, kind, index in candidates:
                # 将确切或插值候选、其夹住区间及是否入选写表。
                candidate_rows.append({"length_left": left, "length_right": right,
                    "direction": direction, "candidate_p": value, "kind": kind,
                    "bracket_left_p": p[index],
                    "bracket_right_p": p[min(index + 1, len(p) - 1)],  # 候选位于末点时将区间右端限制在末点。
                    "selected": chosen is not None and value == chosen[0]})
            boot = np.full(B, np.nan)  # 为每轮的入选交点预留 NaN，表示没有候选。
            counts = np.zeros(B, dtype=int)  # 记录每轮交点候选的个数。
            for i in range(B):
                # 同轮次比较两个格长在指定方向的重抽曲线。
                possibilities = crossing_candidates(p, a["boot"][i, ia, col],
                                                   b["boot"][i, ib, col])
                counts[i] = len(possibilities)  # 保留本轮全部候选的数量。
                if possibilities:
                    boot[i] = min(possibilities, key=lambda value: abs(value[0] - PC))[0]  # 有候选时也选距理论 p_c 最近的点。
                # 保存逐轮交点和候选数。
                boot_rows.append({"length_left": left, "length_right": right,
                    "direction": direction, "replicate": i, "selected_p": boot[i],
                    "candidate_count": counts[i]})
            selected_boot[direction] = boot
            finite = boot[np.isfinite(boot)]  # 取有有限交点的重抽轮次。
            ci = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]  # 仅用有效轮次求 95% 区间，全失效时记 NaN。
            # 汇总该格长对和贯通方向的交点与诊断比例。
            summary_rows.append({"length_left": left, "length_right": right,
                "direction": direction, "selected_p": chosen[0] if chosen else np.nan,
                "candidate_count": len(candidates), "ci_low": ci[0], "ci_high": ci[1],
                "bootstrap_valid_fraction": len(finite) / B,  # 有效轮次比例以全部 B 轮为分母。
                "bootstrap_multiple_fraction": np.mean(counts > 1),  # 统计存在不止一个候选的轮次比例。
                "selection_rule": "candidate nearest theoretical pc"})
        paired = selected_boot["x"] - selected_boot["y"]  # 横纵贯通用同一批构型重抽，按轮次作交点差。
        paired = paired[np.isfinite(paired)]  # 保留两方向都有有限交点的轮次。
        ci = np.quantile(paired, [.025, .975]) if len(paired) else [np.nan, np.nan]  # 用配对差求 95% 百分位区间。
        # 将同一配对区间附加到横、纵两条汇总记录。
        for row in summary_rows[-2:]:
            row["x_minus_y_crossing_ci_low"] = ci[0]
            row["x_minus_y_crossing_ci_high"] = ci[1]
    # 保存观测曲线的全部候选及选中标记。
    pd.DataFrame(candidate_rows).to_csv(OUT / "crossing_all_candidates.csv", index=False)
    # 保存入选交点的汇总及不确定性。
    pd.DataFrame(summary_rows).to_csv(OUT / "crossing_audit.csv", index=False)
    # 保存每轮重抽的交点与候选数。
    pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap_crossing_replicates.csv", index=False)


def symmetry_audit(samples, stored):
    # 功能：在理论 p_c 处复核渗流临界性质实验的横纵贯通对称性。
    # 引用：main() 传入构型数据和方向对称性归档，本函数将配对区间和检验结果写表。
    # 输入
    # samples：逐构型观测，含格长、概率比及横纵贯通的 0/1 指示量。
    # stored：按格长和概率比保存的横减纵贯通概率差及构型 bootstrap 区间。
    # 输出：返回 None，写出 direction_symmetry_pc.csv，含原区间、精确 p 值及 Holm 校正。
    rows = []
    for length in LENGTHS:
        subset = samples[(samples.length == length) & np.isclose(samples.p_ratio, 1.0)]  # 只取当前格长在理论 p_c 的构型。
        xonly = int(((subset.span_x == 1) & (subset.span_y == 0)).sum())  # 数出仅横向贯通的构型。
        yonly = int(((subset.span_x == 0) & (subset.span_y == 1)).sum())  # 数出仅纵向贯通的构型。
        row = stored[(stored.length == length) & np.isclose(stored.p_ratio, 1.0)].iloc[0]  # 从归档读取同格长临界条件的原差值及区间。
        # 保存两个边际概率、配对区间和仅单向贯通的构型数。
        rows.append({"length": length, "samples": len(subset), "p_x": subset.span_x.mean(),
            "p_y": subset.span_y.mean(), "difference": row.p_x_minus_p_y,
            "paired_bootstrap_ci_low": row.ci_low, "paired_bootstrap_ci_high": row.ci_high,
            "x_only": xonly, "y_only": yonly,
            "mcnemar_exact_p": binomtest(xonly, xonly + yonly, .5).pvalue})  # McNemar 精确检验仅在横纵结果不一致的构型上做成功概率 1/2 的二项检验。
    frame = pd.DataFrame(rows)  # 将各格长检验值整理成表。
    order = np.argsort(frame.mcnemar_exact_p.to_numpy())  # 按原 p 值从小到大计算 Holm 校正。
    adjusted = np.zeros(len(frame))  # 校正值按原格长行号回填。
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(frame) - rank) * frame.mcnemar_exact_p.iloc[index])  # 乘尚未经过校正的检验数，再取截至当前秩的累计最大值。
        adjusted[index] = min(1., running)  # 将校正 p 值上限截为 1。
    frame["holm_adjusted_p"] = adjusted
    # 保存临界点的配对对称性结果。
    frame.to_csv(OUT / "direction_symmetry_pc.csv", index=False)


def tail_audit(histogram, fits):
    # 功能：在渗流临界性质实验中复核团簇规模尾拟合的频数门槛和区间上界敏感性。
    # 引用：main() 用团簇直方图和原尾拟合生成独立诊断表。
    # 输入
    # histogram：各格长、概率及剔除口径的团簇规模与实际频数 C_s。
    # fits：原扫描尾拟合，一行对应一组格长、剔除口径和频数门槛，含尾区间与模型诊断。
    # 输出：返回 None，写出 tail_fit_audit.csv。保守区间无足够数据时，替代指数为 NaN。
    from local_market.criticality_revision import finite_powerlaw  # src/local_market/criticality_revision.py 的 finite_powerlaw 在有限整数区间拟合离散纯幂律。
    rows = []
    # 遍历原扫描拟合的各格长、剔除口径和门槛。
    for row in fits.itertuples():
        # 取相同格长和剔除口径的理论临界点直方图。
        subset = histogram[(histogram.length == row.length) &
            np.isclose(histogram.p_ratio, 1.0) & (histogram.exclusion == row.exclusion)]
        counts = np.zeros(int(subset.cluster_size.max()) + 1, dtype=int)  # 以团簇规模直接索引，未观测规模的频数补零。
        counts[subset.cluster_size.to_numpy(dtype=int)] = subset.C_s.to_numpy(dtype=int)  # 将实际观测频数填到相应规模位置。
        upper = int(row.s_max_fit)
        lower = int(row.s_min)  # 保守重拟合沿用原拟合的下界。
        eligible = np.flatnonzero(counts >= row.count_threshold)  # 找到实际频数达到原拟合门槛的规模。
        previous = int(eligible[-2]) if len(eligible) >= 2 else 0  # 保留倒数第二个合格规模，用于检查尾端点的间隔。
        # 只接受拟合下界以上、且最近 11 个规模位置中至少有 3 个达到门槛的规模。
        close_eligible = [int(s) for s in eligible if s >= lower and
            np.count_nonzero(counts[max(lower, s - 10):s + 1] >= row.count_threshold) >= 3]  # 左侧窗口不低于原拟合下界，并包括当前规模。
        conservative_upper = max(close_eligible) if close_eligible else 0  # 选最大的局部合格规模作为保守拟合上界。
        # 按规模频数恢复逐团簇样本。
        values = np.repeat(subset.cluster_size.to_numpy(dtype=int),
                           subset.C_s.to_numpy(dtype=int))
        alt_alpha, _, alt_count = finite_powerlaw(values, lower, conservative_upper)  # 固定原下界并应用保守上界，重估有限区间指数和区间样本数。
        # 将原拟合、支持稀疏性、保守拟合和理论偏差汇总为一行。
        rows.append({"length": row.length, "exclusion": row.exclusion,
            "count_threshold": row.count_threshold, "s_min": lower, "s_max_fit": upper,
            "tail_count": row.tail_count, "alpha_pdf": row.alpha_pdf,
            "alpha_ci_low": row.alpha_ci_low, "alpha_ci_high": row.alpha_ci_high,
            "ks_distance": row.ks_distance, "fixed_interval_gof_p": row.fixed_interval_gof_p,
            "gof_replicates": row.gof_replicates,
            "delta_aic_cutoff_minus_powerlaw": row.cutoff_aic_same_interval - row.powerlaw_aic,  # 截断幂律减纯幂律的 AIC，基于相同区间。
            "delta_aic_exponential_minus_powerlaw": row.exponential_aic_same_interval - row.powerlaw_aic,  # 指数模型减纯幂律的 AIC，同样用相同区间。
            "observed_max": int(subset.cluster_size.max()),
            "upper_count": counts[upper], "previous_eligible_s": previous,
            "eligible_gap": upper - previous,
            "preceding_10_below_threshold": int(np.count_nonzero(  # 计数上界之前最多 10 个规模中低于频数门槛的位置。
                counts[max(lower, upper - 10):upper] < row.count_threshold)),
            "zero_bin_fraction_in_support": float(np.mean(counts[lower:upper + 1] == 0)),  # 在原拟合的闭区间内统计零频数位置比例。
            "conservative_upper": conservative_upper if conservative_upper else np.nan,
            "conservative_alpha": alt_alpha, "conservative_tail_count": alt_count,
            "distance_to_theory": row.alpha_pdf - 187 / 91})  # 用原拟合 PDF 指数减去二维临界团簇分布的理论指数。
    frame = pd.DataFrame(rows)  # 将逐拟合诊断整理成表。
    frame.to_csv(OUT / "tail_fit_audit.csv", index=False)  # 保存团簇尾拟合审计。


def _slope_bootstrap(lengths, values, boots):
    # 功能：对渗流临界性质实验的标度观测做双对数回归并用重抽分位估计斜率区间。
    # 引用：susceptibility_scaling() 分别在易感性峰位和理论 p_c 拟合指数。
    # 输入
    # lengths：互异的正格长数组，至少有两个格长。
    # values：各格长的正观测值，长度同 lengths。
    # boots：形状为 (重抽轮数, 格长数) 的正观测值数组。
    # 输出依次为
    # slope：点估计的双对数 OLS 斜率。
    # low：重抽斜率的 2.5% 分位。
    # high：重抽斜率的 97.5% 分位。
    x = np.log(np.asarray(lengths, dtype=float))  # 以格长的自然对数作横轴。
    centered = x - x.mean()  # 对横轴中心化，以便直接计算带截距的斜率。
    denom = centered @ centered  # 求横轴离差平方和作共同分母。
    slope = float(centered @ np.log(values) / denom)  # 用中心化横轴和观测对数的内积求点估计斜率。
    slopes = np.log(boots) @ centered / denom  # 对每轮重抽沿格长轴求同一口径的斜率。
    low, high = np.quantile(slopes, [.025, .975])  # 用重抽斜率构造双侧 95% 区间。
    return slope, float(low), float(high)


def susceptibility_scaling(curves, peaks, height_boot):
    # 功能：比较渗流临界性质实验中 χ 峰高及理论 p_c 处 χ 的有限尺寸指数。
    # 引用：main() 将返回的指数表交给 collapse_audit() 做敏感性检查。
    # 输入
    # curves：各格长的点估计和 bootstrap 曲线，必须含 p/p_c=1。
    # peaks：各格长、各剔除口径的原观测 χ 峰高表。
    # height_boot：形状为 (B, 格长数, 2) 的逐轮 χ 峰高，口径按 RULES 排列。
    # 输出：逐剔除口径、格长子集和测量位置的指数点估计及区间表，另写缩放峰高表和指数表。
    rows = []
    exponent_rows = []
    for k, rule in enumerate(RULES):
        for i, length in enumerate(LENGTHS):
            data = curves[length]
            at_pc = int(np.flatnonzero(np.isclose(data["ratio"], 1.0))[0])  # 定位当前格长的理论临界点索引。
            stored = peaks[(peaks.length == length) & (peaks.exclusion == rule)].iloc[0]  # 读取同格长、同剔除口径的观测峰高。
            factor = length**(-43 / 24)  # 以理论 γ/ν=43/24 的反幂移除预期的格长依赖。
            scaled = height_boot[:, i, k] * factor  # 用相同理论因子缩放该格长的重抽峰高。
            # 记录缩放后的观测峰高、区间，以及理论 p_c 处 χ 的对照值。
            rows.append({"length": length, "exclusion": rule,
                "chi_peak": stored.chi_peak, "chi_peak_scaled_theory": stored.chi_peak * factor,
                "scaled_peak_ci_low": np.quantile(scaled, .025),  # 沿重复轮次对缩放峰高取 95% 区间。
                "scaled_peak_ci_high": np.quantile(scaled, .975),
                "chi_at_pc": data["point"][at_pc, k + 1],
                "chi_at_pc_scaled_theory": data["point"][at_pc, k + 1] * factor})
        # 分别用全部五个格长及最大的三个格长拟合。
        for subset, indices in (("all_5", np.arange(5)), ("large_3", np.arange(2, 5))):
            lengths = np.array(LENGTHS)[indices]  # 抽出当前回归使用的格长。
            # 按格长顺序获取观测峰高。
            peak_values = np.array([peaks[(peaks.length == length) &
                (peaks.exclusion == rule)].chi_peak.iloc[0] for length in lengths])
            peak_boot_values = height_boot[:, indices, k]  # 取相同格长和剔除口径的逐轮峰高。
            point_pc = []
            boot_pc = []
            for length in lengths:
                data = curves[length]
                j = int(np.flatnonzero(np.isclose(data["ratio"], 1.0))[0])  # 在当前格长的曲线中找到理论 p_c。
                point_pc.append(data["point"][j, k + 1])  # 添加该格长在理论 p_c 的 χ 点估计。
                boot_pc.append(data["boot"][:, j, k + 1])  # 添加该格长在理论 p_c 的逐轮 χ。
            # 分别组装峰高与理论临界点处观测的回归输入。
            for location, values, boots in (
                ("susceptibility_peak", peak_values, peak_boot_values),
                ("theoretical_pc", np.array(point_pc), np.column_stack(boot_pc))):  # 将各格长的重抽数组按列堆叠。
                slope, low, high = _slope_bootstrap(lengths, values, boots)  # 估计 γ/ν 和按构型重抽的 95% 区间。
                # 保存实测斜率、理论值和拟合范围。
                exponent_rows.append({"exclusion": rule, "subset": subset,
                    "location": location, "fitted_gamma_over_nu": slope,
                    "ci_low": low, "ci_high": high,
                    "theoretical_gamma_over_nu": 43 / 24,
                    "bootstrap_replicates": B})
    # 保存各格长由理论指数缩放后的 χ。
    pd.DataFrame(rows).to_csv(OUT / "susceptibility_scaled_peaks.csv", index=False)
    # 保存两种测量位置与格长子集的指数比较。
    pd.DataFrame(exponent_rows).to_csv(OUT / "susceptibility_exponent_subsets.csv", index=False)
    # 返回指数表供坍缩敏感性分析筛选。
    return pd.DataFrame(exponent_rows)


def collapse_audit(curves, exponent_frame):
    """Dimensionless between-size RMS on one fixed common critical x-window."""
    # 功能：在渗流临界性质实验的共同横轴窗口内，量化不同格长的有限尺寸曲线坍缩程度。
    # 引用：main() 用曲线和 χ 指数表生成理论缩放及拟合指数敏感性结果。
    # 输入
    # curves：各格长上五项观测的点曲线及重抽曲线。
    # exponent_frame：susceptibility_scaling() 返回的按口径、格长子集和测量位置分组的 γ/ν 表。
    # 输出：返回 None，写出理论指数坍缩质量、拟合指数敏感性表及对照图。
    xgrid = np.linspace(-.20, .20, 41)  # 固定 41 个共同无量纲横轴点，覆盖 −0.20 至 0.20。
    # 四种曲线按观测量类型采用各自的理论竖轴缩放幂。
    definitions = (("p_infinity", 0, 5 / 48),
                   ("chi_largest_excluded", 1, -43 / 24),
                   ("chi_all_spanning_excluded", 2, -43 / 24),
                   ("span_x", 3, 0.0))
    rows = []

    def evaluate(column, exponent, subset, *, include_bootstrap):
        # 功能：在同一横轴上插值并缩放多格长曲线，计算格长间相对均方根偏差。
        # 引用：collapse_audit() 在理论指数分支获取点估计与重抽分数，在拟合指数分支只获取点估计。
        # 输入
        # column：观测量在 COLS 中的列号。
        # exponent：纵轴乘以 L 的这一幂次。
        # subset：参与比较的格长序列。
        # include_bootstrap：关键字开关，控制是否插值并汇总 B 条重抽曲线。闭包读取 curves 和 xgrid。
        # 输出依次为
        # score：点曲线相对共同均值曲线的无量纲均方根偏差。
        # boot_scores：每轮一个同口径分数，长度 B。跳过重抽时为 None。
        # normalizer：插值后各格长在 x=0 的点估计均值，计算分数时用其绝对值作分母。
        point_curves, bootstrap_curves = [], []
        for length in subset:
            data = curves[length]
            select = (data["ratio"] >= .97 - 1e-9) & (data["ratio"] <= 1.03 + 1e-9)  # 保留 p/p_c 约 0.97 至 1.03 的近临界扫描点。
            x = (data["p"][select] - PC) * length**.75  # 用 (p−p_c)L³/⁴ 把每个尺度的概率网格变为共同横轴。
            # 共同窗口超出任一格长实际覆盖范围时停止，避免外插。
            if x.min() > xgrid.min() or x.max() < xgrid.max():
                raise ValueError(f"x window not covered at L={length}")
            factor = length**exponent  # 确定当前格长观测量的纵轴因子。
            point_curves.append(np.interp(xgrid, x, data["point"][select, column]) * factor)  # 将点估计曲线插值到 xgrid 后作纵轴缩放。
            if include_bootstrap:
                # 可选地为每个格长建立形状为 (B, 共同点数) 的重抽曲线矩阵。
                bootstrap_curves.append(np.stack([
                    np.interp(xgrid, x, data["boot"][i, select, column]) * factor  # 对该格长的每轮曲线使用相同插值点和纵轴因子。
                    for i in range(B)
                ]))
        point = np.stack(point_curves)  # 汇集成格长乘共同点数的点曲线矩阵。
        normalizer = float(np.mean(point[:, len(xgrid) // 2]))  # 用各格长在 x=0 的缩放点估计均值作归一化标度。
        score = float(np.sqrt(np.mean((point - point.mean(axis=0))**2)) / abs(normalizer))  # 逐横轴点减去跨格长均值，再对所有点和格长求均方根。
        if not include_bootstrap:
            return score, None, normalizer
        boot = np.stack(bootstrap_curves, axis=1)  # 重抽数组按轮次、格长、共同横轴点排列。
        # 每轮减去同轮的跨格长均值，沿格长和横轴求均方根，沿用点估计的归一化标度。
        boot_scores = np.sqrt(np.mean((boot - boot.mean(axis=1, keepdims=True))**2,
                                      axis=(1, 2))) / abs(normalizer)
        return score, boot_scores, normalizer

    all_scores = {}
    # 对比全部格长、去掉最小格长和仅用最大三个格长。
    for subset_name, subset in (("all_5", LENGTHS),
                                ("without_32", LENGTHS[1:]),
                                ("large_3", LENGTHS[2:])):
        for name, column, exponent in definitions:
            score, boot_scores, normalizer = evaluate(column, exponent, subset, include_bootstrap=True)  # 用理论指数计算坍缩点分数、逐轮分数与归一化值。
            all_scores[(name, subset_name)] = boot_scores  # 保留各观测量和格长子集的逐轮分数，便于配对比较。
            low, high = np.quantile(boot_scores, [.025, .975])  # 取重抽分数的双侧 95% 百分位区间。
            # 保存理论缩放的质量与窗口设置。
            rows.append({"observable": name, "subset": subset_name,
                "exponent_source": "theory", "vertical_L_power": exponent,
                "dimensionless_rms": score, "bootstrap_ci_low": low,
                "bootstrap_ci_high": high, "normalizer_at_x0": normalizer,
                "common_x_min": xgrid.min(), "common_x_max": xgrid.max(),
                "x_grid_points": len(xgrid), "bootstrap_replicates": B,
                "probability_large3_better_than_all5": float(np.mean(  # 以同轮分数更低为改进，比较最大三个格长与全部五个格长。
                    all_scores[(name, "large_3")] < all_scores[(name, "all_5")]))
                    if subset_name == "large_3" else np.nan})
    # 写出理论指数下的坍缩质量表。
    pd.DataFrame(rows).to_csv(OUT / "collapse_quality_theory.csv", index=False)

    sensitivity = []
    for subset_name, subset in (("all_5", LENGTHS), ("without_32", LENGTHS[1:]),
                                ("large_3", LENGTHS[2:])):
        for name, column in (("chi_largest_excluded", 1),
                             ("chi_all_spanning_excluded", 2)):
            # 对两种 χ 分别尝试由不同格长子集得到的拟合指数。
            for fit_subset in ("all_5", "large_3"):
                # 读取同一剔除口径在理论临界点拟合的 γ/ν。
                fit = exponent_frame[(exponent_frame.exclusion == name.removeprefix("chi_")) &
                    (exponent_frame.subset == fit_subset) &
                    (exponent_frame.location == "theoretical_pc")].iloc[0]
                exponent = -fit.fitted_gamma_over_nu  # 用拟合指数的负值缩放 χ 纵轴。
                score, _, _ = evaluate(column, exponent, subset, include_bootstrap=False)  # 只计算点分数，用于区分拟合指数和格长范围的影响。
                # 记录曲线子集、指数拟合子集和对应的坍缩质量。
                sensitivity.append({"observable": name, "curve_subset": subset_name,
                    "exponent_fit_subset": fit_subset, "vertical_L_power": exponent,
                    "dimensionless_rms": score,
                    "interpretation": "diagnostic only; fitted exponent is not a theory replacement"})
    # 保存拟合指数敏感性结果。
    pd.DataFrame(sensitivity).to_csv(OUT / "collapse_exponent_sensitivity.csv", index=False)

    frame = pd.DataFrame(rows)  # 将理论指数的诊断行整理成绘图表。
    fig, ax = plt.subplots(figsize=(8.5, 4.6))  # 创建坍缩质量对照图。
    x = np.arange(3)  # 为三个格长子集设置离散横轴位置。
    for j, name in enumerate(("p_infinity", "chi_largest_excluded",
                              "chi_all_spanning_excluded", "span_x")):
        # 按固定的子集顺序取当前观测量分数。
        group = frame[frame.observable == name].set_index("subset").loc[
            ["all_5", "without_32", "large_3"]]
        ax.plot(x, group.dimensionless_rms, "o-", label=name)  # 比较当前观测量在三个格长子集上的无量纲偏差。
    ax.set_xticks(x, ["all five", "L≥64", "L≥128"])  # 标出各横轴位置使用的格长范围。
    ax.set(ylabel="dimensionless between-size RMS", title=r"Theory-exponent collapse on $|x|\leq0.20$")  # 轴标签和标题注明质量指标与共同窗口。
    ax.grid(alpha=.25); ax.legend(fontsize=8)  # 添加参考网格与观测量图例。
    fig.tight_layout(); fig.savefig(OUT / "collapse_quality.png", dpi=180); plt.close(fig)  # 排版、保存理论指数坍缩图并关闭画布。


def main():
    # 功能：从精细扫描归档完成渗流临界性质实验的独立临界性验证。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。使用 ROOT 作为精细扫描目录，使用 OUT 作为新的验证结果目录。
    # 输出：返回 None，写出验证表、图、来源和产物哈希。OUT 已存在时抛出 FileExistsError。
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing validation output: {OUT}")
    OUT.mkdir(parents=True)  # 建立新验证结果目录。
    # 读取逐构型观测、汇总、团簇直方图、尾拟合、峰位和方向对称性。
    samples, summary, histogram, fits, peaks, stored_symmetry = get_inputs()
    # 按构型重抽，建立后续验证共用的观测曲线。
    curves = bootstrap_curves(samples, summary)
    # 外推伪临界概率并保留逐轮 χ 峰高。
    height_boot = pseudocritical_extrapolation(curves, peaks)
    # 核对相邻格长的贯通交点。
    crossing_audit(curves)
    # 在理论临界点核对横纵贯通的配对对称性。
    symmetry_audit(samples, stored_symmetry)
    # 检查团簇尾拟合的支持和区间敏感性。
    tail_audit(histogram, fits)
    # 估计 χ 峰值与理论临界点的尺寸指数。
    exponent_frame = susceptibility_scaling(curves, peaks, height_boot)
    # 比较理论和实测指数下不同格长的曲线坍缩。
    collapse_audit(curves, exponent_frame)
    # 保存扫描来源、bootstrap 单位、种子和脚本版本。
    metadata = {"source_root": str(ROOT), "theoretical_pc": PC,
        "bootstrap_replicates": B, "bootstrap_unit": "independent occupancy configuration",
        "bootstrap_seed": 20261113 + 810,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),  # 计算当前分析脚本全部字节的哈希。
        "source_manifest_sha256": hashlib.sha256(  # 将原扫描的产物清单字节纳入来源校验。
            (ROOT / "metadata/sha256_manifest.json").read_bytes()).hexdigest(),
        "no_new_occupancy_sampling": True}
    # 将验证的输入来源及参数写入元数据。
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # 按文件名收集验证目录中现有产物的哈希，排除校验清单本身。
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
                if p.is_file() and p.name != "sha256_manifest.json"}
    # 写出验证产物的校验清单。
    (OUT / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # 报告已归档的验证产物数量和目录。
    print("saved", len(manifest), "validation artifacts to", OUT)


if __name__ == "__main__":
    # 执行精细扫描的独立验证。
    main()
