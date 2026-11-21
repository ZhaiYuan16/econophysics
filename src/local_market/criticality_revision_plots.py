"""Figures for the revised square-lattice criticality scan."""
# 用 make_figures() 绘制开放方格精细扫描的临界观测、峰位、贯通、数据塌缩和团簇规模尾部。
#
# 渗流临界性质实验把精细扫描和配置重抽样产生的表交给本模块，得到用于报告的 PNG 图。

from pathlib import Path  # 标准库路径类，用于指定图像目录。

import matplotlib  # Matplotlib 的后端接口，用于启用无窗口绘图。
matplotlib.use("Agg")  # 启用 Agg 后端，允许命令行批量出图。
import matplotlib.pyplot as plt  # Matplotlib 绘图接口，用于创建、保存和关闭图形。
import numpy as np  # NumPy 用于区间误差、临界行筛选和模型曲线计算。
import pandas as pd  # Pandas 提供精细扫描结果表的类型与数据操作。


def make_figures(summary: pd.DataFrame, peaks: pd.DataFrame,
                 crossings: pd.DataFrame, histogram: pd.DataFrame,
                 fits: pd.DataFrame, p_c: float, path: Path) -> None:
    # 为渗流临界性质实验从精细扫描汇总表生成一组开放方格诊断图。
    # 引用
    #     scripts/simulation/scan_criticality_refined.py 的 main() 在生成渗流汇总、峰位、交叉和尾部模型表后调用。
    # 输入
    #     summary 为分尺寸与 p 的观测汇总表，含序参量、两套 χ 和方向贯通统计。
    #     peaks 为两种排除口径的网格峰位及区间表。
    #     crossings 为相邻边长的水平贯通交叉估计表，可为空。
    #     histogram 为含 p_ratio、规模、排除规则、计数 C_s 和密度 n_s 的池化直方图。
    #     fits 为有限支持尾部模型拟合表。
    #     p_c 为理论临界占据概率，用于参考线与无量纲缩放。
    #     path 为目标图像目录，可自动创建。
    # 输出 None，创建目录并写出八张 PNG 图。
    path.mkdir(parents=True, exist_ok=True)  # 创建图像目录及缺失的父目录。
    def save(fig, name):
        # 统一排版、保存并关闭当前图。
        # 引用
        #     make_figures() 内部的 save() 由临界观测、峰位、水平贯通、方向对比、两种塌缩、规模分布与尾部拟合八处绘图调用。
        # 输入
        #     fig 为待保存的 Matplotlib Figure。
        #     name 为相对文件名，读取外层 path 作为目录。
        # 输出 None，以 180 DPI 写图并释放图形。
        fig.tight_layout()  # 在写图前压紧布局，以容纳各子图的轴标签。
        fig.savefig(path / name, dpi=180)  # 将指定图保存到目标目录。
        plt.close(fig)  # 释放当前图的资源，供批量绘图继续使用。
    def lines(ax, column, x="p"):
        # 在一个子图上绘制按边长分组的扫描曲线。
        # 引用
        #     make_figures() 的四联临界观测图和垂直贯通图调用其内部 lines()。
        # 输入
        #     ax 为待绘制的 Matplotlib Axes。
        #     column 为外层 summary 中纵坐标的列名。
        #     x 为横坐标列名，默认 p。
        # 输出 None，向 ax 添加各边长的曲线、网格和图例。
        for length, group in summary.groupby("length"):  # 每个边长绘一条独立曲线。
            group = group.sort_values(x)  # 按横轴列排序，连线沿占据概率或指定横轴前进。
            ax.plot(group[x], group[column], "o-", markersize=2, label=f"L={length}")  # 用小标记显示离散采样点，并以边长标识曲线。
        ax.grid(alpha=.2)  # 添加浅网格帮助读取曲线。
        ax.legend(fontsize=7)  # 列出各曲线对应的格点边长。
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 四个面板分别显示序参量、两种易感性及水平贯通概率。
    # 将四个观测字段和轴标签逐一分配到面板。
    for ax, col, label in zip(axes.flat,
            ("p_infinity_mean", "chi_largest_excluded", "chi_all_spanning_excluded", "span_x_mean"),
            (r"$P_\infty=\langle s_{max}/L^2\rangle$", r"$\chi$: one largest excluded",
             r"$\chi$: all spanning excluded", "$P_x$: left to right")):
        lines(ax, col); ax.set_ylabel(label); ax.set_xlabel("p")  # 画当前观测量的多尺寸曲线，并分别标记纵轴观测量和横轴 p。
    save(fig, "criticality_observables_revised.png")  # 保存四联观测图。

    fig, ax = plt.subplots(figsize=(7, 4))  # 新建峰位随尺寸变化的对照图。
    for rule, group in peaks.groupby("exclusion"):  # 按有限团簇排除规则区分峰位序列。
        ax.errorbar(group.length + (0 if rule == "largest_excluded" else 5), group.p_grid_peak,  # 第二套排除规则的点横向错开 5，避免遮挡。
            yerr=[np.maximum(0, group.p_grid_peak - group.p_ci_low),  # 从下界到峰位构造非负下误差。
                  np.maximum(0, group.p_ci_high - group.p_grid_peak)],  # 从峰位到上界构造非负上误差。
            fmt="o-", capsize=2, label=rule)
    ax.axhline(p_c, color="black", ls="--", label="theoretical $p_c$")  # 画理论临界概率的水平参照线。
    ax.set(xlabel="L", ylabel="grid peak p"); ax.legend(fontsize=8); ax.grid(alpha=.2)  # 标注峰位和格点边长，显示排除口径图例与网格。
    save(fig, "criticality_pseudocritical_revised.png")  # 保存伪临界峰位图。

    fig, ax = plt.subplots(figsize=(8, 5))  # 为水平贯通曲线和交叉点创建图形。
    for length, group in summary.groupby("length"):  # 每个尺寸单独绘制一条贯通曲线。
        group = group.sort_values("p")  # 按实际 p 排序以连续绘图。
        ax.plot(group.p, group.span_x_mean, "o-", ms=2, label=f"L={length}")  # 绘制左右贯通概率的估计曲线。
        ax.fill_between(group.p, group.span_x_ci_low, group.span_x_ci_high, alpha=.1)  # 用半透明色带表示配置 bootstrap 区间。
    for value in crossings.p_crossing.dropna() if not crossings.empty else []:  # 跳过未估出交叉的尺寸对。
        ax.axvline(value, color="gray", alpha=.25)  # 用灰色竖线标出估计的交叉位置。
    ax.axvline(p_c, color="black", ls="--"); ax.set(xlabel="p", ylabel="$P_x$: left to right")  # 用理论 p_c 标注参照竖线，同时标明 p 和水平贯通概率两轴。
    ax.legend(fontsize=8); ax.grid(alpha=.2)  # 显示尺寸图例和读数网格。
    save(fig, "criticality_spanning_probability_revised.png")  # 保存含交叉点的水平贯通图。

    fig, ax = plt.subplots(figsize=(8, 5))  # 创建垂直贯通对照图。
    lines(ax, "span_y_mean")  # 画每个边长的上下贯通概率。
    ax.set(xlabel="p", ylabel="$P_y$: top to bottom")  # 标记占据概率及垂直事件方向。
    save(fig, "criticality_direction_symmetry_revised.png")  # 保存垂直贯通图，供与水平贯通相对照。

    fig, ax = plt.subplots(figsize=(8, 5))  # 创建水平贯通缩放图。
    for length, group in summary.groupby("length"):  # 为各格点边长绘制重标度后的样本点。
        x = (group.p - p_c) * length**.75  # 按二维渗流 1/ν=3/4 缩放离临界点的距离。
        ax.plot(x, group.span_x_mean, "o", ms=3, label=f"L={length}")  # 在缩放后的横轴上使用原始水平贯通概率。
    ax.set(xlabel=r"$(p-p_c)L^{3/4}$, theoretical $p_c$", ylabel="$P_x$")  # 标记理论临界参照和缩放坐标。
    ax.legend(fontsize=8); ax.grid(alpha=.2)  # 显示尺寸图例和背景网格。
    save(fig, "criticality_spanning_collapse_revised.png")  # 保存水平贯通塌缩图。

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 创建序参量和两种易感性的三个缩放面板。
    for length, group in summary.groupby("length"):  # 每个尺寸为三个观测量分别添加一组点。
        x = (group.p - p_c) * length**.75  # 使用与贯通塌缩相同的无量纲横坐标。
        for ax, col, power in zip(axes,
                ("p_infinity_mean", "chi_largest_excluded", "chi_all_spanning_excluded"),
                (5/48, -43/24, -43/24)):
            ax.plot(x, group[col] * length**power, "o", ms=2, label=f"L={length}")  # 用对应的理论幂次重标度每个纵轴观测。
    for ax, label in zip(axes, (r"$P_\infty L^{5/48}$", r"$\chi_A L^{-43/24}$", r"$\chi_B L^{-43/24}$")):  # 按观测量标注三个纵轴各自的缩放口径。
        ax.set(xlabel=r"$(p-p_c)L^{3/4}$", ylabel=label); ax.grid(alpha=.2)  # 标明通用横坐标及各观测缩放口径，并给每幅子图加网格。
    axes[0].legend(fontsize=7)  # 在共享的第一幅面板列出各尺寸图例。
    save(fig, "criticality_finite_size_scaling_revised.png")  # 保存三个观测量的有限尺寸塌缩图。

    critical = histogram[np.isclose(histogram.p_ratio, 1.0)]  # 取参照临界概率处的规模频数和密度。
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 为两种有限团簇排除口径分别创建面板。
    for ax, rule in zip(axes, ("largest_excluded", "all_spanning_excluded")):  # 两种排除规则分别使用一个密度面板。
        for length, group in critical[critical.exclusion == rule].groupby("length"):  # 只比较当前口径下的各尺寸临界直方图。
            ax.loglog(group.cluster_size, group.n_s, ".", ms=2, label=f"L={length}")  # 在双对数轴画按总格点数归一化的规模频数。
        ax.set(xlabel="cluster size s", ylabel=r"$n_s=C_s/(ML^2)$", title=rule)  # 标明横轴规模、纵轴每格点密度及排除规则。
        ax.legend(fontsize=7)  # 为每个面板添加尺寸图例。
    save(fig, "criticality_cluster_distributions_revised.png")  # 保存临界点团簇密度图。

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 新建原始规模频数与尾部拟合的两联图。
    for ax, rule in zip(axes, ("largest_excluded", "all_spanning_excluded")):  # 两种排除规则分别使用一个尾拟合面板。
        subset = critical[critical.exclusion == rule]
        for length, group in subset.groupby("length"):  # 将所选排除口径的临界频数按格点尺寸分组。
            ax.loglog(group.cluster_size, group.C_s, ".", ms=2, label=f"L={length}")  # 在双对数轴上绘制未经密度归一化的 C_s。
        # 只绘制当前口径下频数阈值为 10 且拟合成功的模型。
        for row in fits[(fits.exclusion == rule) & (fits.count_threshold == 10) &
                        (fits.status == "ok")].itertuples():
            ax.axvline(row.s_max_fit, color="gray", alpha=.25)  # 标出纯幂律拟合的有限上界。
            support = np.arange(int(row.s_min), int(row.s_max_fit) + 1)  # 用闭区间内整数规模构造纯幂律支持。
            if len(support) > 1:
                weights = support ** (-row.alpha_pdf)
                ax.loglog(support, row.tail_count * weights / weights.sum(), "-", lw=1)  # 在有限支持上归一化纯幂律，并乘尾样本数换算为预测计数。
            observed = subset[subset.length == row.length]
            if not observed.empty and np.isfinite(row.cutoff_s_c):
                upper = int(observed.cluster_size.max())  # 完整截断拟合的上界取当前尺寸最大观测规模。
                support = np.arange(int(row.s_min), upper + 1)  # 枚举截断幂律完整尾部的整数支持。
                weights = support ** (-row.cutoff_alpha_pdf) * np.exp(-support / row.cutoff_s_c)  # 组合幂律和指数衰减，得到未归一化权重。
                tail_count = observed[observed.cluster_size >= row.s_min].C_s.sum()  # 用下界以上所有观测计数为完整尾部拟合定标。
                ax.loglog(support, tail_count * weights / weights.sum(), "--", lw=1)  # 将截断幂律预测频数画为虚线。
        ax.axhline(10, color="black", ls="--", alpha=.5)  # 以水平虚线标出计数阈值 10。
        ax.set(xlabel="cluster size s", ylabel="raw count $C_s$",  # 标明规模和原始频数两轴。
               title=f"{rule}: solid finite power law, dashed cutoff")
        ax.legend(fontsize=7, loc="lower left", ncol=2)  # 用两列图例标识各尺寸的经验曲线。
    save(fig, "criticality_cluster_tail_fits_revised.png")  # 保存团簇尾拟合图。
