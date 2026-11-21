#!/usr/bin/env python3
"""CCDF interval OLS for the market and network results in the submitted report."""
# 用经验 CCDF 的双对数普通最小二乘回归比较市场和网络收益尾部。
# absolute_returns() 和 empirical() 准备曲线。market_fit() 在市场尾部选宽区间，network_fit() 在网络曲线上筛选多个稳定下降区间。

# main() 读取市场收益率的影响因素实验与网络拓扑压力测试的逐笔收益归档，输出条件汇总和网络候选区间表，供报告的市场与网络区间拟合图使用。

from pathlib import Path  # 定位仓库根目录及输入输出文件。

import numpy as np  # 筛选有限样本、累计经验频数并进行双对数回归。
import pandas as pd  # 读取逐笔交易、按实验条件分组和保存拟合结果。

BASE = Path(__file__).resolve().parents[2]  # 相对于脚本定位仓库目录，输入不依赖当前工作目录。
OUT = BASE / "results/loglog_ols_comparison"


def absolute_returns(group: pd.DataFrame, exclude_raw_zeros: bool) -> np.ndarray:
    """Keep the raw-return zero rule separate from log-axis positivity."""
    # 功能：按指定的原始零收益口径筛选样本并取标准化收益绝对值。
    # 引用：main() 为市场和网络条件分别准备双对数拟合样本。
    # 输入
    # group：一个条件的逐笔交易表，包含原始收益 return 和标准化收益 z。
    # exclude_raw_zeros：为真时剔除原始收益为零的交易。
    # 输出：一维 |z| 数组，长度为通过有限值和零收益筛选的笔数。允许 |z|=0 或返回空数组。
    raw = group["return"].to_numpy(dtype=float)  # 提取原始收益，用来按交易是否为零筛选。
    z = group["z"].to_numpy(dtype=float)  # 提取与原始收益逐笔对齐的标准化收益。
    keep = np.isfinite(raw) & np.isfinite(z)  # 同时要求两列均为有限值。
    if exclude_raw_zeros:
        keep &= raw != 0  # 根据原始收益列识别零交易。
    return np.abs(z[keep])  # 保留通过筛选的绝对标准化收益。


def empirical(values):
    # 功能：以正且有限的样本为分母，构造逐阈值的经验 CCDF。
    # 引用：market_fit() 和 network_fit() 用本函数得到曲线及对应的右尾计数。
    # 输入
    # values：一维绝对标准化收益样本，允许零和非有限值。
    # 输出：无正有限样本时为 None，否则依次返回三个等长一维数组。
    # x：升序的不同正样本值。
    # ccdf：从当前阈值起的右尾样本数除以正有限样本总数。
    # remaining：大于等于各 x 的计数，含重复样本。
    v = np.asarray(values, dtype=float)  # 转为数值数组。
    v = v[np.isfinite(v) & (v > 0)]  # 排除无效值和无法取对数的零值。
    if not len(v):
        return None
    x, counts = np.unique(v, return_counts=True)  # 将重复幅度合并为一个阈值并保留频数。
    remaining = np.cumsum(counts[::-1])[::-1]  # 从高向低累计频数，使每个阈值含等于该值的观测。
    return x, remaining / remaining[0], remaining  # 用首个阈值覆盖的样本数作为分母。


def market_interval_fit(x, ccdf, remaining):
    """Preserve the original market fit: 70th-percentile search and wide near-best R²."""
    # 功能：为市场收益率的影响因素实验选取尾部近最佳 R² 中跨度最大的 OLS 区间。
    # 引用：market_fit() 传入由经验分位门槛截出的尾部曲线。
    # 输入
    # x：升序的一维正阈值。
    # ccdf：各 x 对应的经验右尾概率，分母为正有限样本总数。
    # remaining：各 x 对应的右尾计数。三个数组逐点对齐。
    # 输出依次为
    # fit：成功时为区间、斜率、截距、指数、稳定性及状态字典。否则为 None。
    # reason：成功时为 None，否则给出曲线点数或区间宽度和样本数不足的原因。
    if len(x) < 14:
        return None, "too_few_distinct_points"
    # 将 100 个几何间隔目标映射到实际阈值并去重。
    ix = np.unique(np.clip(np.searchsorted(x, np.geomspace(x[0], x[-1], 100)), 0, len(x)-1))
    xx, yy, rr = x[ix], ccdf[ix], remaining[ix]  # 以相同索引抽取阈值、概率和计数。
    good = (rr >= 20) & (yy > 0) & (xx > 0)  # 只保留尾部至少有 20 笔且可取双对数的点。
    xx, yy, rr = xx[good], yy[good], rr[good]
    if len(xx) < 14:
        return None, "too_few_tail_points"
    lx, ly = np.log10(xx), np.log10(yy)  # 转成以 10 为底的对数，横轴跨度以十倍程计。
    candidates = []
    for i in range(len(lx)-13):
        for j in range(i+13, len(lx)):
            # 至少覆盖一个十倍程，且左端至右端之前至少有 100 笔。
            if lx[j]-lx[i] < 1 or rr[i]-rr[j] < 100:
                continue
            X, Y = lx[i:j+1], ly[i:j+1]  # 两端都纳入区间回归。
            slope, intercept = np.polyfit(X, Y, 1)  # 对区间曲线点等权回归。
            fitted = intercept+slope*X  # 计算各点的拟合对数概率。
            sst = np.sum((Y-Y.mean())**2)  # 以对数概率的总离差平方和作 R² 分母。
            r2 = 1-np.sum((Y-fitted)**2)/sst if sst > 0 else -np.inf  # 平坦区间的 R² 记作负无穷。
            candidates.append((r2, lx[j]-lx[i], i, j, intercept, slope))  # 保留区间跨度和端点索引以供排序。
    if not candidates:
        return None, "no_interval_meets_width_and_count"
    best_r2 = max(row[0] for row in candidates)  # 先确定候选中的最高 R²。
    eligible = [row for row in candidates if row[0] >= best_r2-.002]  # 保留距最佳 R² 至多 0.002 的区间。
    r2, width, i, j, intercept, slope = max(eligible, key=lambda row: (row[1], row[0]))  # 优先选择跨度最大的区间，跨度相同再比 R²。
    mid = (i+j)//2
    left = np.polyfit(lx[i:mid+1], ly[i:mid+1], 1)[0]  # 左半段含中点，单独求斜率。
    right = np.polyfit(lx[mid:j+1], ly[mid:j+1], 1)[0]  # 右半段含中点，单独求斜率。
    drift = abs(left-right)/max(abs(slope), .01)  # 用全段斜率幅度衡量两半段的相对漂移，最小分母为 0.01。
    # 线性候选须满足 R²、负斜率和半段斜率稳定性三项判据。
    status = "visible_linear_candidate" if r2 >= .98 and slope < 0 and drift <= .25 else "no_clear_linear_scaling"
    # 回传区间端点、点数、事件数和拟合诊断。
    return dict(xmin=xx[i], xmax=xx[j], n_points=j-i+1, n_interval=int(rr[i]-rr[j]),
                slope=slope, intercept=intercept, r2=r2, ccdf_alpha=-slope,  # CCDF 指数等于负斜率。
                pdf_tau=1-slope, span_decades=width,  # PDF 指数比 CCDF 指数大 1。
                slope_half_relative_difference=drift, status=status), None


def market_fit(values):
    # 功能：用经验上尾确定市场收益率的影响因素实验的 OLS 搜索起点并汇总拟合。
    # 引用：main() 将返回字典并入市场各占据概率与活动率条件的汇总。
    # 输入
    # values：一维绝对标准化收益样本。
    # 输出：含搜索起点、全曲线点数和拟合字段或失败状态的字典。无正样本时仅返回状态。
    e = empirical(values)  # 构造以正样本为分母的经验曲线。
    if e is None:
        return {"status": "no_positive_data"}
    x, ccdf, remaining = e
    lower = np.flatnonzero(remaining <= .30*remaining[0])  # 查找剩余样本数不超过总数 30% 的阈值。
    floor = x[lower[0]] if len(lower) else x[-1]  # 从首个达到门槛的点开始搜索。没有达到门槛时取末点。
    keep = x >= floor
    fit, reason = market_interval_fit(x[keep], ccdf[keep], remaining[keep])  # 保留原概率分母，对门槛以上区间执行 OLS 搜索。
    return {"scanned_from": floor, "curve_points": len(x),  # 将全曲线点数和扫描门槛随拟合结果保存。
            **(fit if fit is not None else {"status": reason})}


def network_candidates(x, ccdf, remaining):
    """Preserve the original refined network interval selection."""
    # 功能：为网络拓扑压力测试枚举并排序稳定下降的双对数 OLS 区间。
    # 引用：network_fit() 用首个候选作条件汇总，并保留完整候选列表。
    # 输入
    # x：非空的一维升序正阈值。
    # ccdf：各 x 对应的经验右尾概率。
    # remaining：各 x 对应的右尾样本数。三个数组逐点对齐。
    # 输出依次为
    # candidates：最多五个按等级、R² 和跨度排序的拟合字典。失败时为空列表。
    # reason：成功时为 None，否则为点数、尺度范围、事件数或稳定性不足的状态。
    ix = np.unique(np.clip(np.searchsorted(x, np.geomspace(x[0], x[-1], 145)), 0, len(x)-1))  # 将 145 个几何间隔目标映射到实际阈值并去重。
    xx, yy, rr = x[ix], ccdf[ix], remaining[ix]  # 对三个数组应用共同的采样索引。
    good = (rr >= 10) & (yy > 0)  # 排除右尾少于 10 笔或概率非正的点。
    xx, yy, rr = xx[good], yy[good], rr[good]
    if len(xx) < 7:
        return [], "too_few_curve_points"
    lx, ly = np.log10(xx), np.log10(yy)  # 用双对数坐标衡量区间宽度与概率降幅。
    all_rows = []
    for i in range(len(xx)-6):
        # 跳过接近 CCDF 平台顶部的区间起点。
        if yy[i] > .98:
            continue
        for j in range(i+6, len(xx)):
            span, nobs, drop = lx[j]-lx[i], rr[i]-rr[j], ly[i]-ly[j]  # 计算区间跨度、端点间样本数和对数概率降幅。
            # 三项都达到最小值才作回归。
            if span < .65 or nobs < 100 or drop < .15:
                continue
            X, Y = lx[i:j+1], ly[i:j+1]
            slope, intercept = np.polyfit(X, Y, 1)  # 逐点等权求区间的对数斜率和截距。
            # 排除平坦或上升的拟合。
            if slope >= 0:
                continue
            fitted = intercept+slope*X  # 计算区间拟合值。
            sst = np.sum((Y-Y.mean())**2)  # 计算对数概率的总离差平方和。
            r2 = 1-np.sum((Y-fitted)**2)/sst if sst > 0 else -np.inf  # 常数区间的 R² 记为负无穷。
            one, two = i+(j-i)//3, i+2*(j-i)//3  # 按点索引将区间分成三段。
            slopes, drops = [], []
            for a, b in ((i, one), (one, two), (two, j)):
                if b-a < 2:
                    break
                slopes.append(np.polyfit(lx[a:b+1], ly[a:b+1], 1)[0])  # 求每个局部区间的回归斜率。
                drops.append(ly[a]-ly[b])  # 记录每个局部区间的对数概率降幅。
            # 要求三段都有足够点数与实际降幅。
            if len(slopes) < 3 or min(drops) < .015:
                continue
            drift = (max(slopes)-min(slopes))/max(abs(slope), .02)  # 三段斜率极差按整体斜率幅度归一化，分母至少 0.02。
            stable = all(s < 0 for s in slopes) and drift < .35  # 三段皆为负斜率且相对极差小于 0.35 时判为稳定。
            tier = "standard" if span >= 1 and len(X) >= 14 and r2 >= .98 and stable else "local"  # 标准档同时要求跨度、点数、R² 和稳定性，其他先列为局部档。
            # 局部档仍须达到较宽松的 R² 和斜率漂移要求。
            if tier == "local" and (r2 < .965 or drift > .85):
                continue
            # 记录端点、计数、拟合参数、指数和区间等级。
            all_rows.append(dict(xmin=xx[i], xmax=xx[j], n_points=len(X),
                n_interval=int(nobs), n_above_upper=int(rr[j]), start_ccdf=yy[i], end_ccdf=yy[j],
                log_drop=drop, span_decades=span, slope=slope, intercept=intercept, r2=r2,
                ccdf_alpha=-slope, pdf_tau=1-slope, third_slope_spread=drift, tier=tier))
    if not all_rows:
        available = np.log10(xx[-1]/xx[0]) if xx[-1] > xx[0] else 0  # 无候选时先检查现有曲线的总跨度。
        if available < .65:
            return [], "insufficient_scale_range"
        # 检查是否存在跨度和区间事件量均达标的窗口，以区分失败原因。
        populated = any(lx[j]-lx[i] >= .65 and rr[i]-rr[j] >= 100
                        for i in range(len(lx)) for j in range(i+1, len(lx)))
        return [], "no_stable_decline" if populated else "insufficient_events_in_window"
    all_rows.sort(key=lambda row: (row["tier"] != "standard", -row["r2"], -row["span_decades"]))  # 按标准档优先、R² 降序、跨度降序排列。
    chosen = []
    for row in all_rows:
        # 两端与已选区间均相差不到 0.22 个十倍程时视为近重复。
        if any(abs(np.log10(row["xmin"]/other["xmin"])) < .22 and
               abs(np.log10(row["xmax"]/other["xmax"])) < .22 for other in chosen):
            continue
        chosen.append(row)  # 保留排名靠前的新区间，最多五个。
        if len(chosen) == 5:
            break
    return chosen, None


def network_fit(values):
    # 功能：汇总网络拓扑压力测试单个条件的最佳区间和备选区间。
    # 引用：main() 采用汇总字典写条件统计，并逐行写候选排名。
    # 输入
    # values：一维绝对标准化收益样本，调用时已剔除原始零收益。
    # 输出依次为
    # result：曲线点数、候选数、最佳区间字段及状态。无正数据时仅含状态。
    # candidates：排序后的候选区间列表，失败或无数据时为空。
    e = empirical(values)  # 构造正样本的经验曲线。
    if e is None:
        return {"status": "no_positive_data"}, []
    x, ccdf, remaining = e
    candidates, reason = network_candidates(x, ccdf, remaining)  # 在全经验曲线上筛选稳定区间。
    result = {"curve_points": len(x), "candidate_count": len(candidates)}
    if candidates:
        best = candidates[0]
        result.update(best)  # 将排名第一的区间参数并入条件汇总。
        result["status"] = "standard_candidate" if best["tier"] == "standard" else "exploratory_local_only"  # 区分标准区间和仅供探索的局部区间。
    else:
        result["status"] = reason
    return result, candidates


def main():
    # 功能：批量复算市场和网络收益经验 CCDF 的区间回归。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。从 BASE 下读取两份逐笔收益归档。
    # 输出：返回 None，将条件汇总和网络候选写入 OUT，并打印主要拟合列。
    rows, candidate_rows = [], []
    # 配置两个实验的输入文件、分组键及原始零收益口径。
    families = (
        ("market", "results/tables/market_returns.csv.gz", ["p_ratio", "activity"], False),
        ("network", "results/tables/network_market_returns.csv.gz", ["topology", "relative_p"], True),
    )
    for kind, source, keys, exclude_zeros in families:
        data = pd.read_csv(BASE/source, usecols=keys+["return", "z"])  # 仅读取分组及尾部计算所需的列。
        for key, group in data.groupby(keys, sort=True):  # 分市场条件或网络条件独立分析。
            label = (f"p/pc={key[0]:g}, a={key[1]:g}" if kind == "market"
                     else f"{key[0].upper()} p/pc={key[1]:g}")
            values = absolute_returns(group, exclude_zeros)  # 提取该分组的有效绝对收益。
            # 记录拟合对象、条件标签和样本筛选统计。
            base = dict(model="CB_market" if kind == "market" else "CB_network_market",
                kind=kind, condition=label, fitted_distribution="CCDF", label=label,
                n_observations=len(group), zero_return_count=int((group["return"] == 0).sum()),  # 零收益计数来自原始收益列。
                n_raw=int(np.count_nonzero(np.isfinite(values) & (values > 0))),  # n_raw 记录最终可取对数的有限正幅度数。
                zero_rule="exclude_raw_zero" if exclude_zeros else "include_raw_zero",
                source=source)
            if kind == "market":
                rows.append(base | market_fit(values))  # 汇总市场上尾的区间拟合。
            else:
                fit, candidates = network_fit(values)  # 获取网络条件的最佳结果和备选区间。
                rows.append(base | fit)
                # 按排序结果给网络候选编号。
                candidate_rows.extend(dict(kind=kind, label=label, rank=rank, **candidate)
                                      for rank, candidate in enumerate(candidates, 1))
    OUT.mkdir(parents=True, exist_ok=True)  # 建立回归对比输出目录。
    summary = pd.DataFrame(rows)  # 合并两个实验的条件汇总。
    summary.to_csv(OUT/"ols_summary_v2.csv", index=False)  # 写出每个条件的主结果。
    pd.DataFrame(candidate_rows).to_csv(OUT/"candidate_intervals_v2.csv", index=False)  # 单独写出网络候选的排名和区间参数。
    print(summary[["kind", "label", "status", "xmin", "xmax", "slope", "pdf_tau", "r2"]].to_string(index=False))  # 在终端展示主要拟合参数和状态。


if __name__ == "__main__":
    # 启动归档 CCDF 对比。
    main()
