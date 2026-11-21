#!/usr/bin/env python3
"""Recompute SOC diagnostics from versioned path and event CSV files."""
# 从 SOC 检验实验的归档轨迹和雪崩事件重算临界窗口内的驻留比例、雪崩分布及尾部诊断。
# main() 汇总轨迹与系综、比较初始占据率、拟合尾部、执行路径 bootstrap，并可生成规模标度结果。

# 本脚本接收 scripts/simulation/run_soc_research.py 的归档。scripts/analysis/analyze_soc_scaling.py 可从这里生成的尾部和持续时间表独立完成标度分析。
from __future__ import annotations  # 延后求值类型注解。
import argparse  # 解析归档目录和是否跳过规模标度的选项。
from pathlib import Path  # 拼接归档和诊断文件路径。
import matplotlib.pyplot as plt  # 绘制占据率轨迹、事件分布与尾拟合诊断。
import numpy as np  # 排序事件、计算经验概率并提供重采样随机数。
import pandas as pd  # 读取轨迹和事件表、按独立路径汇总并保存诊断。
from scipy.stats import ks_2samp  # 计算两组占据率经验分布的 KS 距离。
from local_market.soc_scaling import (  # src/local_market/soc_scaling.py 提供共享的归档设置与规模标度入口。
    read_soc_run_settings,  # 恢复各格长的实际步数、预热期和路径重采样设置。
    write_size_scaling,  # 生成雪崩规模的有限尺寸标度产物。
)
from local_market.soc_tail import bootstrap_paths, fit_tail, select_xmin  # src/local_market/soc_tail.py 的 bootstrap_paths 按路径重采样，fit_tail 拟合指定模型，select_xmin 搜索尾部起点。


def main():
    # 功能：从 SOC 检验实验的已有路径和事件重新计算占据率、雪崩尾部和标度诊断。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。--root 必填，须含 config.yaml、critical_windows.csv 及条件子目录。有 run.json 时优先读取实际运行设置。--skip-size-scaling 默认关闭。
    # 输出：返回 None，在归档目录写逐路径和系综统计、分布距离、尾拟合与 bootstrap 等表和图，默认同时写规模标度。
    # 建立 SOC 归档分析的命令行解析器。
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)  # 必须指定已有实验归档。
    # 允许把规模标度留给独立入口。
    parser.add_argument(
        "--skip-size-scaling",
        action="store_true",
        help="Leave size diagnostics to analyze_soc_scaling.py.",
    )
    args = parser.parse_args()  # 解析实际运行选项。
    root = args.root
    win = pd.read_csv(root/"critical_windows.csv").set_index("length")  # 按格长索引静态临界窗口。
    timing_by_length, boot_count, simulation_seed = read_soc_run_settings(root, win.index.to_list())  # 从归档恢复时间和重采样设置。
    steps_by_length = {length: steps for length, (steps, _) in timing_by_length.items()}
    burn_in_by_length = {length: burn_in for length, (_, burn_in) in timing_by_length.items()}
    recomputed = []
    # 遍历存在轨迹归档的条件子目录。
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/"paths.csv.gz").exists():
            continue
        raw = pd.read_csv(folder/"paths.csv.gz")  # 载入当前条件的逐步轨迹。
        for pid, group in raw.groupby("path_id", sort=False):  # 以 path_id 为单位计算轨迹统计。
            length = int(group.length.iloc[0])  # 从同一路径的首行读取格长。
            steps, burn = timing_by_length[length]  # 采用该格长的实际步数和预热期。
            actual_steps = len(group)  # 以归档中的实际轨迹长度检查预热期。
            if burn >= actual_steps:
                raise ValueError(f"{folder.name}/{pid}: burn-in {burn} must be smaller than {actual_steps} recorded steps")
            if group.step.duplicated().any():  # 旧归档的路径编号冲突可能把同一步号记录多次。
                raise ValueError(f"{folder.name}/{pid}: repeated step numbers for this path ID")
            post = group[group.step.between(burn, steps, inclusive="left")]  # 保留预热后到运行结束前的交易步。
            if post.empty:
                raise ValueError(f"{folder.name}/{pid}: no recorded steps in the analysis window [{burn}, {steps})")
            p = post.occupancy_after.to_numpy()  # 以每步更新后的占据率作为热化后统计样本。
            half = len(p)//2  # 用样本中点切分前半段与后半段。
            bounds = win.loc[length]  # 查询当前格长的临界窗口。
            # 保存路径标识、起始条件、种子以及热化后统计。
            recomputed.append(dict(condition=folder.name, length=length,
                initial_occupancy=float(group.initial_occupancy.iloc[0]),
                replicate=int(group.replicate.iloc[0]), path_id=pid,
                seed=int(group.seed.iloc[0]), steps=actual_steps, burn_in=burn,
                mean_occupancy=p.mean(), occupancy_sd=p.std(ddof=1),  # 同一路径的占据率均值和样本标准差。
                drift=p[half:].mean()-p[:half].mean(),  # 漂移取后半段均值减前半段均值。
                window_fraction=np.mean((p >= bounds.window_low)&(p <= bounds.window_high)),  # 临界窗口比例以全部热化后步数为分母并包含端点。
                event_rate=np.mean(post.duration > 0),  # 事件率以热化后交易步为分母，持续时间大于零计为事件。
                event_count=int((post.duration > 0).sum()),  # 统计热化后出现事件的步数。
                mean_event_size=post.loc[post.removed_nodes > 0, "removed_nodes"].mean()))  # 仅在移除节点数为正的步上求平均规模。
    summary = pd.DataFrame(recomputed)  # 将逐路径记录合并成表。
    summary.to_csv(root/"path_summary_recomputed.csv", index=False)  # 保存逐路径重算结果。
    figure_dir = root/"figures"
    figure_dir.mkdir(exist_ok=True)  # 准备诊断图目录。
    # 按模拟条件、格长和初始占据率汇总独立路径。
    summary.groupby(["condition", "length", "initial_occupancy"]).agg(
        paths=("path_id", "nunique"), mean_p=("mean_occupancy", "mean"),  # 路径数按不同 path_id 统计，mean_p 等权平均各路径均值。
        sd_p=("mean_occupancy", "std"), window_fraction=("window_fraction", "mean"),  # sd_p 衡量路径均值之间的离散，window_fraction 是各路径窗口比例的均值。
        drift=("drift", "mean"), event_rate=("event_rate", "mean")  # 对路径漂移与事件率等权平均。
    ).reset_index().to_csv(root/"ensemble_summary.csv", index=False)  # 将分组键恢复为普通列并保存系综表。
    distribution_rows = []
    tail_rows, scan_rows, boot_rows, duration_rows = [], [], [], []
    rng = np.random.default_rng(simulation_seed + 771)  # 使用由模拟种子派生的固定重采样序列。
    cascade_events = pd.DataFrame()  # 保留预热后的级联事件供持续时间和尾模型诊断共用。
    # 分别诊断 baseline 和 cascade 归档。
    for condition in ("baseline", "cascade"):
        folder = root/condition
        if not folder.exists():
            continue
        all_paths = pd.read_csv(folder/"paths.csv.gz")  # 保留未裁剪轨迹供后面展示全过程。
        try:
            events = pd.read_csv(folder/"events.csv.gz")  # 读取同条件的雪崩事件。
        except pd.errors.EmptyDataError:
            events = pd.DataFrame()  # 旧版无事件归档可能没有表头。
        if events.empty:
            events = pd.DataFrame(columns=["step", "length", "size", "duration"])
        # 轨迹和事件均按格长选择同一段预热后的观测区间。
        paths = all_paths[all_paths.step.between(
            all_paths.length.map(burn_in_by_length), all_paths.length.map(steps_by_length), inclusive="left"
        )]
        events = events[events.step.between(
            events.length.map(burn_in_by_length), events.length.map(steps_by_length), inclusive="left"
        )]
        if condition == "cascade":
            cascade_events = events
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))  # 三个面板分别画占据率、规模 CCDF 和规模 PDF。
        for l in sorted(paths.length.unique()):  # 各格长独立估计分布。
            group = paths[paths.length == l]  # 保留当前格长的所有路径样本。
            for p0 in sorted(group.initial_occupancy.unique()):  # 占据率分布按初始占据率分别作图。
                sub = group[group.initial_occupancy == p0]  # 汇集同一初始占据率的热化后时间点。
                # 按时间点绘制占据率密度直方图。
                axes[0].hist(sub.occupancy_after, bins=40, density=True, histtype="step",
                             label=f"L={l}, p0={p0}")
            # 遍历不同初始占据率的无序配对。
            for a in sorted(group.initial_occupancy.unique()):
                for b in sorted(group.initial_occupancy.unique()):
                    if a >= b:
                        continue
                    va = group[group.initial_occupancy == a].occupancy_after  # 为 KS 距离提取较低初始占据率的占据率样本。
                    vb = group[group.initial_occupancy == b].occupancy_after  # 提取较高初始占据率的占据率样本。
                    distribution_rows.append(dict(condition=condition, length=l, initial_a=a,
                        initial_b=b, ks_descriptive=ks_2samp(va, vb).statistic,  # 合并轨迹时间点有依赖，KS 只作描述性距离。
                        note="pooled time points dependent; no iid p-value"))
            ev = events[(events.length == l) & (events["size"] > 0)]  # 仅以当前格长实际移除节点的事件画规模分布。
            if len(ev):
                x = np.sort(ev["size"].to_numpy())  # 排序后用样本秩构造经验右尾。
                axes[1].step(x, 1-np.arange(len(x))/len(x), label=f"L={l}")  # CCDF 的分母为当前格长全部筛选后的正事件。
                axes[2].hist(x, bins=np.geomspace(1, max(x)+1, 25), density=True,  # 用几何分箱和密度归一化画规模 PDF。
                             histtype="step", label=f"L={l}")
            # 只为有正事件的 cascade 条件拟合尾部。
            if condition != "cascade" or len(ev) == 0:
                continue
            duration_fit, _, duration_status = select_xmin(ev.duration, 1000)  # 用 1000 为上界检查持续时间能否选出尾部阈值。
            # 保存持续时间事件数、最大轮数、拟合状态和选中阈值。
            duration_rows.append(dict(length=l, event_count=len(ev),
                max_duration=int(ev.duration.max()), status=duration_status,
                xmin=duration_fit["xmin"] if duration_fit is not None else None))
            upper = l*l  # 规模上界取格点总数 L²。
            pure, scan, status = select_xmin(ev["size"], upper)  # 按离散纯幂律为正规模选择尾部阈值。
            for row in scan:
                scan_rows.append(dict(condition=condition, length=l, **row))  # 保留各候选阈值诊断供敏感性图使用。
            if status != "ok":
                # 阈值条件不满足时保存状态和正事件数，并跳过本格长的后续拟合。
                tail_rows.append(dict(condition=condition, length=l, status=status,
                                      positive_events=len(ev)))
                continue
            xmin = int(pure["xmin"])  # 将纯幂律选中的阈值作为三个模型的共同下界。
            fits = [
                pure,
                fit_tail(ev["size"], xmin, upper, "cutoff"),  # 在相同尺度支持上拟合截断幂律。
                fit_tail(ev["size"], xmin, upper, "exponential"),  # 在相同尺度支持上拟合指数模型。
            ]
            for row in fits:
                # 给三种模型的结果附上条件、格长和样本数。
                tail_rows.append(dict(condition=condition, length=l, status=status,
                                      positive_events=len(ev), **row))
            # 按独立路径整理事件数组，保留没有正事件的路径。
            path_values = {pid: ev.loc[ev.path_id == pid, "size"].to_numpy(dtype=int)
                           for pid in group.path_id.unique()}
            # 建立路径到初始占据率的映射供分层抽样。
            path_groups = {pid: float(sub.initial_occupancy.iloc[0])
                           for pid, sub in group.groupby("path_id")}
            # 每次重抽整条路径并重选尾阈值，保留初始占据率分层。
            for row in bootstrap_paths(path_values, upper, boot_count, rng,
                                       path_groups=path_groups):
                boot_rows.append(dict(condition=condition, length=l, **row))  # 将每轮拟合结果标记条件和格长后收集。
        axes[0].set_xlabel("occupancy p")  # 第一幅图的横轴为更新后的占据率。
        axes[1].set(xscale="log", yscale="log", xlabel="S", ylabel="CCDF")  # 第二幅图用双对数坐标展示规模 CCDF。
        axes[2].set(xscale="log", yscale="log", xlabel="S", ylabel="PDF")  # 第三幅图用双对数坐标展示密度。
        for ax in axes:
            if ax.has_data():
                ax.legend(fontsize=6)  # 只为有观测数据的面板显示格长和初始占据率标签。
        fig.tight_layout()  # 调整分布图排版。
        fig.savefig(figure_dir/f"{condition}_distributions.png", dpi=160)  # 按模拟条件保存分布图。
        plt.close(fig)  # 关闭当前分布图。
        # 取最大格长且重复编号为 0 的完整轨迹。
        sample = all_paths[(all_paths.length == all_paths.length.max()) & (all_paths.replicate == 0)]
        fig, ax = plt.subplots(figsize=(9,3))  # 创建占据率时序图。
        for p0, group in sample.groupby("initial_occupancy"):
            ax.plot(group.step, group.occupancy_after, lw=.6, label=f"p0={p0}")  # 以真实模拟步为横轴，为每个初始占据率绘制轨迹。
        w = win.loc[sample.length.iloc[0]]  # 用完整轨迹的格长查找对应临界窗口。
        ax.axhspan(w.window_low, w.window_high, color="gray", alpha=.2)  # 在轨迹图上用色带标出临界窗口。
        ax.set(xlabel="true step", ylabel="occupancy p")  # 设置真实步数和占据率轴标签。
        ax.legend()  # 显示初始占据率图例。
        fig.tight_layout()  # 调整轨迹图排版。
        fig.savefig(figure_dir/f"{condition}_trajectories.png", dpi=160)  # 按条件保存轨迹图。
        plt.close(fig)  # 关闭轨迹图。
    # 即使没有样本或拟合记录也写出表头，独立标度等下游可以直接读取空表。
    pd.DataFrame(distribution_rows, columns=[
        "condition", "length", "initial_a", "initial_b", "ks_descriptive", "note",
    ]).to_csv(root/"distribution_distances.csv", index=False)
    model_columns = [
        "model", "xmin", "upper", "tail_count", "exponent", "cutoff",
        "log_likelihood", "aic", "ks", "boundary",
    ]
    fits = pd.DataFrame(tail_rows, columns=["condition", "length", "status", "positive_events", *model_columns])
    fits.to_csv(root/"tail_fits.csv", index=False)  # 保存三种尾部模型和跳过拟合的原因。
    scan = pd.DataFrame(scan_rows, columns=["condition", "length", *model_columns])
    scan.to_csv(root/"xmin_scan.csv", index=False)  # 保存候选阈值与各自的拟合诊断。
    bootstrap = pd.DataFrame(boot_rows, columns=[
        "condition", "length", "bootstrap", "status", "xmin", "tau", "cutoff_tau", "cutoff", "cutoff_boundary",
    ])
    bootstrap.to_csv(root/"tail_bootstrap.csv", index=False)  # 保存按路径重采样的逐轮结果和状态。
    pd.DataFrame(duration_rows, columns=["length", "event_count", "max_duration", "status", "xmin"]).to_csv(
        root/"duration_diagnostics.csv", index=False
    )
    if duration_rows:
        fig, ax = plt.subplots()  # 创建持续时间图。
        for l, group in cascade_events.groupby("length"):
            durations = np.sort(group.duration.to_numpy())  # 排序已经按实际预热期筛选的持续时间，单位为松弛轮数。
            if len(durations):
                ax.step(durations, 1-np.arange(len(durations))/len(durations), label=f"L={l}")  # 用各格长保留的全部持续时间为分母画经验 CCDF。
        ax.set(xlabel="relaxation rounds T", ylabel="empirical CCDF")  # 标注持续轮数横轴和经验 CCDF 纵轴。
        ax.legend()  # 显示格长图例。
        fig.savefig(figure_dir/"duration_ccdf.png", dpi=160)  # 保存持续时间分布图。
        plt.close(fig)  # 关闭持续时间图。
    if not bootstrap.empty:
        intervals = []
        # 只按成功拟合的轮次汇总各格长的区间。
        for l, group in bootstrap[bootstrap.status == "ok"].groupby("length"):
            intervals.append(dict(length=l, successful=len(group),
                tau_low=group.tau.quantile(.025), tau_high=group.tau.quantile(.975),  # 从成功轮次取 τ 的双侧 95% 百分位区间。
                cutoff_low=group.cutoff.quantile(.025), cutoff_high=group.cutoff.quantile(.975),  # 从相同轮次取截断尺度的双侧 95% 百分位区间。
                cutoff_boundary_fraction=group.cutoff_boundary.mean()))  # 计算成功拟合中截断参数贴边的比例。
        pd.DataFrame(intervals).to_csv(root/"tail_intervals.csv", index=False)  # 保存尾部参数区间。
        fig, ax = plt.subplots()  # 创建尾指数重采样分布图。
        for l, group in bootstrap[bootstrap.status == "ok"].groupby("length"):
            ax.hist(group.tau, bins=25, histtype="step", label=f"L={l}")  # 按格长画成功轮次的 τ 频数。
        ax.set(xlabel="PDF exponent tau", ylabel="path bootstrap count")  # 标明横轴为 PDF 指数，纵轴为 bootstrap 计数。
        ax.legend()  # 显示格长图例。
        fig.savefig(figure_dir/"bootstrap_tau.png", dpi=160)  # 保存尾指数的采样分布图。
        plt.close(fig)  # 关闭尾指数图。
    if not scan.empty:
        fig, ax = plt.subplots()  # 创建阈值敏感性图。
        for l, group in scan.groupby("length"):  # 按格长整理候选阈值的诊断。
            ax.plot(group.xmin, group.ks, label=f"L={l}")  # 绘制变化尾部下界时经验和模型 CCDF 的 KS 距离。
        ax.set(xlabel="S_min", ylabel="CCDF KS distance")  # 标明候选下界和距离的轴含义。
        ax.legend()  # 显示格长图例。
        fig.savefig(figure_dir/"xmin_scan.png", dpi=160)  # 保存阈值扫描图。
        plt.close(fig)  # 关闭阈值扫描图。
    if not fits.empty:
        # 逐格长处理 cascade 的成功拟合，要求三种模型齐全。
        for l, subset in fits[(fits.condition == "cascade") & (fits.status == "ok")].groupby("length"):
            if len(subset) != 3:
                continue
            xmin = int(subset.xmin.iloc[0])  # 读取三个模型共用的下界。
            upper = int(l*l)  # 将物理支持上界取为格点总数。
            x = cascade_events.loc[
                (cascade_events.length == l) & (cascade_events["size"] >= xmin), "size"
            ].to_numpy(dtype=int)  # 从相同的预热后样本中选取超过共同尾部阈值的规模。
            if len(x) == 0:
                continue
            support = np.arange(xmin, upper+1)  # 枚举从阈值到 L² 的离散支持。
            fig, ax = plt.subplots()  # 创建当前格长的尾模型图。
            observed = np.sort(x)  # 排序实际尾部规模，供经验 CCDF 使用。
            ax.step(observed, 1-np.arange(len(observed))/len(observed), label="empirical")  # 按保留尾样本的秩画条件经验 CCDF。
            for row in subset.itertuples():
                rate = 0 if row.model == "power" else 1/row.cutoff  # 纯幂律的截断率为零，其余取截断尺度的倒数。
                weights = support.astype(float)**(-row.exponent)*np.exp(-rate*(support-xmin))  # 在离散支持上计算幂次项与指数衰减项的乘积。
                probs = weights/weights.sum()  # 归一化使支持上的质量和为 1。
                ax.plot(support, np.cumsum(probs[::-1])[::-1], label=row.model)  # 从最大规模反向累加概率质量，得到模型 CCDF。
            ax.set(xscale="log", yscale="log", xlabel="avalanche size S", ylabel="CCDF")  # 以双对数坐标展示规模与右尾概率。
            ax.legend()  # 标明经验曲线和各模型。
            fig.savefig(figure_dir/f"L{l}_tail_models.png", dpi=160)  # 按格长保存尾模型对照图。
            plt.close(fig)  # 关闭当前格长的尾模型图。
    if not args.skip_size_scaling:
        write_size_scaling(root, fits, standalone=False, timing_by_length=timing_by_length)  # 为规模标度传入同一套实际运行时间设置。


if __name__ == "__main__":
    # 执行 SOC 归档分析入口。
    main()
