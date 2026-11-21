"""Publication-style plotting helpers for simulation results."""
# 将模拟与分析表绘成报告图。
# plot_critical_observables()、plot_cluster_distributions()、plot_pseudocritical_estimates() 和 plot_finite_size_scaling() 展示格点渗流。
# plot_market_ensemble_samples() 和 plot_market_time_series() 分别展示系综样本与固定格点轨迹。
# plot_return_distributions()、plot_activity_statistics()、plot_active_clusters_by_size() 和 plot_tail_exponents() 展示分布、活跃率、团簇数标度和拟合诊断。
# plot_network_structure_metrics() 和 plot_network_degree_distributions() 展示整图结构，plot_network_criticality() 展示渗流。
# plot_network_market_comparison()、plot_network_return_ccdfs() 和 plot_network_tail_exponents() 展示网络收益统计、分布和拟合尾指数。
#
# 本模块接收渗流临界性质实验、市场收益率的影响因素实验和网络拓扑压力测试产出的统计表或轨迹对象，写出可由命令行批量生成的图。

from __future__ import annotations  # 延后求值类型注解。

from pathlib import Path  # 标准库路径类，用于创建图像输出目录和保存文件。

import matplotlib  # Matplotlib 后端接口，用于选择无窗口绘图。

matplotlib.use("Agg")  # 启用 Agg 后端，支持无图形界面的命令行运行。
import matplotlib.pyplot as plt  # Matplotlib 绘图接口，用于创建、绘制、写出和关闭图形。
import numpy as np  # NumPy 用于浮点条件筛选、理论曲线和误差条计算。
import pandas as pd  # Pandas 提供绘图所读统计表及数据整形操作。
from scipy import stats  # SciPy 统计模块为收益分布图提供标准正态 PDF 与生存函数。


def _finish_figure(fig: plt.Figure, path: Path) -> None:
    # 创建输出目录，再按统一分辨率保存并关闭图形。
    # 引用
    #     plot_critical_observables()、plot_cluster_distributions()、plot_pseudocritical_estimates() 和 plot_finite_size_scaling() 用它保存渗流图。
    #     plot_market_ensemble_samples()、plot_market_time_series()、plot_return_distributions()、plot_activity_statistics()、plot_tail_exponents() 和 plot_active_clusters_by_size() 用它保存格点市场图。
    #     plot_network_structure_metrics()、plot_network_degree_distributions()、plot_network_criticality()、plot_network_market_comparison()、plot_network_return_ccdfs() 和 plot_network_tail_exponents() 用它保存网络图。
    # 输入
    #     fig 为已经绘制的 Matplotlib Figure。
    #     path 为目标图像文件路径，后缀由调用方指定。
    # 输出 None，创建缺失目录，以 200 DPI 写图并关闭 fig。
    path.parent.mkdir(parents=True, exist_ok=True)  # 按需创建图像目标的父目录。
    fig.savefig(path, dpi=200, bbox_inches="tight")  # 以紧凑边框保存当前图，减少周围空白。
    plt.close(fig)  # 关闭已保存的 Figure，释放批量绘图资源。


def plot_critical_observables(
    summary: pd.DataFrame, p_c: float, path: Path
) -> None:
    """Plot order parameter, finite-cluster susceptibility, and spanning rate."""
    # 为渗流临界性质实验绘制序参量、池化易感性和贯通概率。
    # 引用
    #     scripts/simulation/scan_criticality.py 的 main() 用扫描汇总绘制多尺寸临界观测图。
    # 输入
    #     summary 为分边长和概率比的扫描汇总表，含 P∞ 与标准误、χ 与 bootstrap 区间及贯通概率。
    #     p_c 为参考临界占据概率，显示在图例中。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存三联图。
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))  # 给序参量、易感性和贯通概率各分配一个横向面板。
    for length, group in summary.groupby("length", sort=True):  # 每个边长分别绘制三类观测量。
        group = group.sort_values("p_ratio")  # 按无量纲占据概率比排序，保证连线顺序。
        label = f"L={length}"
        # 绘制 P∞ 曲线，用配置平均值的标准误画误差条。
        axes[0].errorbar(
            group["p_ratio"],
            group["p_infinity"],
            yerr=group["p_infinity_sem"],  # 纵向误差为样本标准误。
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            label=label,
        )
        # 绘制全部配置规模矩取比所得到的池化易感性。
        axes[1].plot(
            group["p_ratio"],
            group["susceptibility_pooled"],
            marker="o",
            markersize=3,
            linewidth=1,
            label=label,
        )
        # 在池化易感性曲线上叠加配置 bootstrap 置信带。
        axes[1].fill_between(
            group["p_ratio"],
            group["susceptibility_ci_low"],  # 给出置信带下边界。
            group["susceptibility_ci_high"],  # 给出置信带上边界。
            alpha=0.12,
        )
        # 绘制在相同扫描点估计的贯通概率。
        axes[2].plot(
            group["p_ratio"],
            group["spanning_probability"],
            marker="o",
            markersize=3,
            linewidth=1,
            label=label,
        )
    axes[0].set_ylabel(r"$P_\infty=\langle s_{\max}/L^2\rangle$")  # 用最大团簇占整格点的比例标记序参量。
    axes[1].set_ylabel(r"$\chi$ (largest cluster removed)")  # 按当前图样式标记池化易感性。
    axes[2].set_ylabel("Spanning probability")  # 标记用于比较贯通概率的面板。
    for axis in axes:
        axis.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)  # 在每个面板标出参照临界位置 p/p_c=1。
        axis.set_xlabel(r"$p/p_c$")  # 统一使用概率比作横轴。
        axis.grid(alpha=0.25)  # 添加浅网格便于跨面板读数。
    axes[0].legend(title=f"open boundary\n$p_c={p_c:.6f}$", fontsize=8)  # 在第一面板显示尺寸和理论临界概率。
    fig.suptitle("Square-lattice site percolation: finite-size observables")  # 给三联图设置统一的临界观测主题。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭临界三联观测图。


def plot_cluster_distributions(
    histogram: pd.DataFrame,
    selected_ratios: tuple[float, ...],
    exclusion: str,
    path: Path,
) -> None:
    """Plot finite-cluster n_s per lattice site for each system size."""
    # 为渗流临界性质实验按边长绘出有限团簇规模密度。
    # 引用
    #     scripts/simulation/scan_criticality.py 的 main() 传入扫描频数和配置中的直方图概率比。
    # 输入
    #     histogram 为规模直方图长表，含 length、p_ratio、cluster_size 和 finite_n_s_per_site。
    #     selected_ratios 为要显示的 p/p_c 元组。
    #     exclusion 为标题中显示的有限团簇排除规则，筛选已经体现在 histogram 里。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存各尺寸的双对数分布面板。
    lengths = sorted(histogram["length"].unique())  # 从长表提取并排序有记录的格点边长。
    ncols = 2
    nrows = int(np.ceil(len(lengths) / ncols))  # 按两列面板计算容纳所有边长所需的行数。
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows), squeeze=False)  # 按尺寸分配面板，并始终保留二维轴阵列。
    for axis, length in zip(axes.flat, lengths, strict=False):  # 每个边长配一个面板，允许有空面板。
        for ratio in selected_ratios:
            group = histogram[
                (histogram["length"] == length)
                & np.isclose(histogram["p_ratio"], ratio)  # 用浮点近似相等对齐所选 p/p_c。
                & (histogram["finite_n_s_per_site"] > 0)  # 只绘制能显示在对数纵轴上的正密度。
            ]
            # 在双对数轴画该尺寸和概率比的规模密度散点。
            axis.loglog(
                group["cluster_size"],  # 横轴使用整数团簇规模。
                group["finite_n_s_per_site"],  # 纵轴使用除以配置数和每配置格点数后的频数。
                marker="o",
                linestyle="none",
                markersize=2.2,
                alpha=0.75,
                label=f"p/pc={ratio:g}",
            )
        axis.set_title(f"L={length}")  # 标记当前面板的格点边长。
        axis.set_xlabel("Cluster size s")  # 标记横轴团簇规模。
        axis.set_ylabel(r"$n_s$ per lattice site")  # 标记纵轴所用的每格点密度。
        axis.grid(which="both", alpha=0.2)  # 显示对数主次刻度网格。
        axis.legend(fontsize=7)  # 用图例区分占据概率比。
    for axis in axes.flat[len(lengths) :]:  # 遍历布局中余下的轴。
        axis.set_visible(False)  # 隐藏扩展成两列后未分配尺寸的空面板。
    fig.suptitle(f"Finite-cluster distributions (exclusion: {exclusion})")  # 报告绘图数据所用的有限团簇排除口径。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭有限团簇分布图。


def plot_pseudocritical_estimates(
    estimates: pd.DataFrame, p_c: float, path: Path
) -> None:
    # 为渗流临界性质实验比较两种网格伪临界估计随尺寸的变化。
    # 引用
    #     scripts/simulation/scan_criticality.py 的 main() 绘制扫描得到的峰位与斜率估计。
    # 输入
    #     estimates 为每个边长的 p_chi_peak 和 p_max_order_parameter_slope 表。
    #     p_c 为占据概率的理论临界参照。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存以 1/L 为横轴的估计对照图。
    fig, axis = plt.subplots(figsize=(6.5, 4.5))  # 为两个伪临界位置序列建立一个共同坐标系。
    inverse_length = 1.0 / estimates["length"]  # 取边长倒数作为有限尺寸横轴。
    # 画池化易感性的网格峰位。
    axis.plot(
        inverse_length,
        estimates["p_chi_peak"],
        "o-",
        label=r"peak of $\chi$",
    )
    # 画序参量对 p 的最大斜率对应的网格位置。
    axis.plot(
        inverse_length,
        estimates["p_max_order_parameter_slope"],
        "s-",
        label=r"max slope of $P_\infty$",
    )
    axis.axhline(p_c, color="black", linestyle="--", label=f"theory {p_c:.6f}")  # 叠加理论临界概率的水平线。
    axis.set_xlabel(r"$1/L$")  # 标记横轴为边长倒数。
    axis.set_ylabel("Grid-based pseudocritical p")  # 标明纵轴为基于网格的伪临界 p。
    axis.grid(alpha=0.25)  # 显示读数网格。
    axis.legend()  # 用图例区分两个估计和理论参照。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭伪临界估计对照图。


def plot_finite_size_scaling(
    summary: pd.DataFrame,
    scaling: pd.DataFrame,
    critical_ratio: float,
    path: Path,
) -> None:
    # 为渗流临界性质实验将测得的有限尺寸标度与拟合及理论斜率比较。
    # 引用
    #     scripts/simulation/scan_criticality.py 的 main() 把临界观测和回归表交给该函数。
    # 输入
    #     summary 为按尺寸和 p_ratio 汇总的观测表。
    #     scaling 为含观测名、拟合斜率、截距和理论斜率的标度表。
    #     critical_ratio 为选择观测行的 p/p_c，脚本使用 1。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存序参量和易感性的双对数标度图。
    # 选择与参照概率比相近的观测并按边长排序。
    critical = summary[np.isclose(summary["p_ratio"], critical_ratio)].sort_values(
        "length"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))  # 给序参量和池化易感性分别建立一个双对数面板。
    specifications = (
        (axes[0], "p_infinity", r"$P_\infty$"),
        (axes[1], "susceptibility_pooled", r"$\chi$"),
    )
    for axis, column, ylabel in specifications:
        name = "p_infinity" if column == "p_infinity" else "susceptibility"
        fit = scaling.loc[scaling["observable"] == name].iloc[0]  # 读取当前观测量的拟合参数和理论指数。
        lengths = critical["length"].to_numpy(dtype=float)  # 将边长转换为浮点，为幂律计算准备横坐标。
        values = critical[column].to_numpy(dtype=float)  # 取该观测量随边长变化的模拟值。
        fitted = np.exp(fit["intercept"]) * lengths ** fit["fitted_exponent"]  # 把对数空间拟合恢复为原尺度上的幂律。
        theoretical = values[0] * (lengths / lengths[0]) ** fit["theoretical_exponent"]  # 将理论幂律锚定到最小尺寸的实测值，以对比斜率。
        axis.loglog(lengths, values, "o", label="simulation")  # 用散点绘制实际模拟观测。
        # 绘制由回归指数和截距决定的拟合线。
        axis.loglog(
            lengths,
            fitted,
            "-",
            label=f"fit exponent={fit['fitted_exponent']:.3f}",
        )
        # 绘制二维理论指数对应的参考线。
        axis.loglog(
            lengths,
            theoretical,
            "--",
            label=f"2D theory={fit['theoretical_exponent']:.3f}",
        )
        axis.set_xlabel("System length L")  # 标记实际边长 L。
        axis.set_ylabel(ylabel)  # 给当前面板标注对应的观测量。
        axis.grid(which="both", alpha=0.25)  # 显示对数主次刻度网格。
        axis.legend(fontsize=8)  # 在图例中区别模拟、拟合和理论。
    fig.suptitle(r"Finite-size scaling at $p/p_c=1$")  # 在标题中标记脚本采用的临界概率比。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭有限尺寸标度图。


def plot_market_ensemble_samples(returns: pd.DataFrame, path: Path) -> None:
    """Show ordered ensemble draws without implying a price trajectory."""
    # 为市场收益率的影响因素实验展示临界占据概率处按顺序排放的系综交易样本。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 画系综示例，scripts/analysis/audit_market_time_series.py 的 main() 用临界子集复核样本结构。
    # 输入
    #     returns 为逐笔交易表，使用 activity、p_ratio、replicate、step 和 return。
    #     path 为输出图像路径。
    # 输出 None，在 path 按活跃率保存重复编号 0 的前至多 3000 个系综样本。
    activities = sorted(returns["activity"].unique())  # 得到各个活跃率，供分面排序。
    fig, axes = plt.subplots(len(activities), 1, figsize=(10, 2.4 * len(activities)), squeeze=False)  # 给每个活跃率分配一行，同时保留二维轴阵列。
    for row, activity in enumerate(activities):  # 将活跃率映射到各自的面板行。
        group = returns[
            np.isclose(returns["p_ratio"], 1.0)  # 只取近似处于临界占据概率的交易。
            & np.isclose(returns["activity"], activity)  # 用近似相等匹配当前活跃率。
            & (returns["replicate"] == 0)
        ].sort_values("step").head(3000)  # 按系综样本次序取前 3000 笔。
        axes[row, 0].plot(group["step"], group["return"], linewidth=0.55)  # 画样本索引对应的原始对数收益率。
        axes[row, 0].set_ylabel(f"a={activity:g}\nreturn")  # 标出该面板的活跃率和收益纵轴。
        axes[row, 0].grid(alpha=0.2)  # 为该行添加浅网格。
    axes[-1, 0].set_xlabel("Ensemble sample index (not time)")  # 将横轴标为系综样本索引。
    fig.suptitle("CB ensemble samples at p/pc=1; blocks share an occupied configuration")  # 在标题中注明临界概率和分块共用占据配置。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭系综样本图。


def plot_market_time_series(simulations: dict[float, object], path: Path) -> None:
    """Plot only genuine time-ordered MarketSimulation trajectories."""
    # 为市场收益率的影响因素实验绘制固定配置交易的收益率和对数价格轨迹。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 作轨迹展示，scripts/analysis/audit_market_time_series.py 的 main() 绘制固定配置对照图。
    # 输入
    #     simulations 为活跃率到 MarketSimulation 实例的非空字典。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存各活跃率的时序图，输入为空或值类型不符时抛出 TypeError。
    from .cb_model import MarketSimulation  # src/local_market/cb_model.py，提供固定配置交易结果类型用于入口校验。

    if not simulations or any(not isinstance(x, MarketSimulation) for x in simulations.values()):  # 要求每个值是 MarketSimulation，并至少有一条轨迹。
        raise TypeError("time-series plots require MarketSimulation trajectories")
    fig, axes = plt.subplots(len(simulations), 2, figsize=(12, 2.8 * len(simulations)), squeeze=False)  # 每个活跃率画一行，左边收益率，右边对数价格。
    for row, (activity, simulation) in enumerate(sorted(simulations.items())):  # 按活跃率排序，以固定两列面板的行顺序。
        times = np.arange(1, simulation.returns.size + 1)  # 对第一笔收益从时间 1 开始编号。
        axes[row, 0].plot(times, simulation.returns, linewidth=0.55)  # 绘制每步对数收益率。
        axes[row, 1].plot(np.arange(simulation.log_prices.size), simulation.log_prices, linewidth=0.8)  # 对数价格包含 t=0 的初始值和每步更新后的值。
        axes[row, 0].set_ylabel(f"a={activity:g}\nreturn")  # 标记每一行轨迹对应的活跃率。
        # 收益率与对数价格面板都使用浅网格，便于对齐交易时间与数值。
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    # 在底部给两列统一标记模拟时间。
    axes[-1, 0].set_xlabel("Simulation time t")
    axes[-1, 1].set_xlabel("Simulation time t")  # 为价格列底部标注同一个时间尺度。
    axes[0, 0].set_title("Log return")  # 标明收益率所在的左列。
    axes[0, 1].set_title("Cumulative log return")  # 标明逐步累积形成的对数价格所在的右列。
    fig.suptitle("Quenched CB trajectory at p/pc=1; independent actions, no volatility memory")  # 注明图所展示的临界固定配置场景。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭固定配置轨迹图。


def plot_return_distributions(distributions: pd.DataFrame, path: Path) -> None:
    """Plot standardized signed PDFs and absolute-return CCDFs."""
    # 为市场收益率的影响因素实验并排绘制有符号标准化收益 PDF 和绝对收益 CCDF。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 和 scripts/analysis/finalize_market_statistics.py 的 main() 用汇总分布表绘制分析图和最终图。
    # 输入
    #     distributions 为含 activity、p_ratio、distribution、x、value 的长表，当前轴索引需要至少两个活跃率。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存多活跃率、不同占据概率比的两类分布图。
    activities = sorted(distributions["activity"].unique())  # 提取并排序所有活跃率作为面板行。
    fig, axes = plt.subplots(len(activities), 2, figsize=(12, 3.2 * len(activities)))  # 每个活跃率横向并排放置 PDF 和 CCDF。
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, distributions["p_ratio"].nunique()))  # 按概率比数从颜色表中选取可区分的颜色。
    ratios = sorted(distributions["p_ratio"].unique())  # 提取排序后的 p/p_c 与颜色对应。
    for row, activity in enumerate(activities):  # 依次设置每个活跃率的面板行。
        for color, ratio in zip(colors, ratios, strict=True):  # 一组概率比在所有面板中使用同一种颜色。
            selected = distributions[
                np.isclose(distributions["activity"], activity)  # 从分布表匹配当前活跃率。
                & np.isclose(distributions["p_ratio"], ratio)  # 同时匹配当前占据概率比。
            ]
            pdf = selected[selected["distribution"] == "signed_pdf"]
            ccdf = selected[selected["distribution"] == "absolute_ccdf"]
            # 用对数纵轴显示带符号收益的密度。
            axes[row, 0].semilogy(
                pdf["x"],
                pdf["value"],
                color=color,
                marker=".",
                linestyle="none",
                markersize=2.5,
                label=f"p/pc={ratio:g}",
            )
            # 在双对数轴显示绝对标准化收益的超越概率。
            axes[row, 1].loglog(
                ccdf["x"], ccdf["value"], color=color, linewidth=1, label=f"p/pc={ratio:g}"
            )
        x_pdf = np.linspace(-6, 6, 300)  # 在负到正的等距网格上准备正态 PDF 参照。
        axes[row, 0].semilogy(x_pdf, stats.norm.pdf(x_pdf), "k--", linewidth=0.9, label="Gaussian")  # 用标准正态密度叠加比较有符号分布。
        x_ccdf = np.geomspace(0.05, 8, 200)  # 为绝对值尾部概率生成正的等比网格。
        # 用标准正态单侧生存概率的两倍作为绝对值 CCDF 参照。
        axes[row, 1].loglog(
            x_ccdf, 2 * stats.norm.sf(x_ccdf), "k--", linewidth=0.9, label="Gaussian"
        )
        axes[row, 0].set_xlim(-10, 10)  # 将有符号 PDF 的显示横轴限制在 −10 到 10。
        axes[row, 0].set_ylim(bottom=1e-7)  # 保留对数密度纵轴的显示下限。
        axes[row, 0].set_ylabel(f"a={activity:g}\nPDF")  # 为该行标明活跃率及 PDF 口径。
        axes[row, 1].set_ylabel(r"$P(|z|\geq x)$")  # 标明 CCDF 包含正负两端且阈值处取等号。
        axes[row, 0].grid(alpha=0.2)  # 给半对数 PDF 添加网格。
        axes[row, 1].grid(which="both", alpha=0.2)  # 给双对数 CCDF 添加主次刻度网格。
        axes[row, 1].legend(fontsize=7)  # 用图例区分概率比与正态参照。
    axes[-1, 0].set_xlabel("Standardized return z")  # 标明 PDF 横轴为有符号 z。
    axes[-1, 1].set_xlabel("Absolute standardized return |z|")  # 标明 CCDF 横轴为绝对 z。
    fig.suptitle("CB return distributions: PDF and absolute-return CCDF")  # 标记总体比较的两种分布。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭收益分布对照图。


def plot_activity_statistics(summary: pd.DataFrame, path: Path) -> None:
    # 为市场收益率的影响因素实验比较不同活跃率下的波动率、峰度、极端概率和活跃团簇数。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 将池化收益统计交给此函数绘图。
    # 输入
    #     summary 为按 p_ratio 和 activity 的市场汇总表，含所绘四项指标与前三项的重复间标准差。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存四联活跃率对照图。
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))  # 给四个交易指标各分配一幅面板。
    specifications = (
        ("return_std", "replicate_return_std_sd", "Return standard deviation"),
        ("return_excess_kurtosis", "replicate_kurtosis_sd", "Excess kurtosis"),
        ("p_abs_z_gt_5", "replicate_p_abs_z_gt_5_sd", r"$P(|z|>5)$"),
        ("mean_active_cluster_count", None, "Mean active clusters"),
    )
    for ratio, group in summary.groupby("p_ratio", sort=True):  # 按占据概率比分组，每组在各面板一条曲线。
        group = group.sort_values("activity")  # 按活跃率递增排列连线的数据点。
        for axis, (column, error_column, ylabel) in zip(  # 将各指标的数据和误差定义分配给对应面板。
            axes.flat, specifications, strict=True
        ):
            errors = group[error_column] if error_column is not None else None
            # 画当前指标随活跃率变化的曲线。
            axis.errorbar(
                group["activity"],
                group[column],
                yerr=errors,  # 有误差字段时显示重复之间的标准差。
                marker="o",
                capsize=2,
                label=f"p/pc={ratio:g}",
            )
            axis.set_xscale("log")  # 活跃率跨度较大，横轴使用对数尺度。
            axis.set_xlabel("Activity a")  # 统一标注团簇活跃率 a。
            axis.set_ylabel(ylabel)  # 标示当前面板对应的收益或活动指标。
            axis.grid(alpha=0.25)  # 添加浅网格。
    axes[0, 0].legend(fontsize=8)  # 在首个面板列出概率比图例。
    fig.suptitle("Market activity and return statistics")  # 标记总体比较主题。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭活跃率统计图。


def plot_tail_exponents(tail_fits: pd.DataFrame, path: Path) -> None:
    """Show fitted exponents as labels; use a marker only for validated fits."""
    # 为市场收益率的影响因素实验标注描述性尾指数，并用颜色和符号表示拟合诊断。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main()、scripts/analysis/audit_market_tails_visual.py 的 main() 和 scripts/analysis/finalize_market_statistics.py 的 main() 用它呈现拟合结果及诊断。
    # 输入
    #     tail_fits 为含 activity、p_ratio、alpha 或 alpha_pdf 以及 power_law_reliable 的拟合表。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存尾指数标签和诊断符号图，缺少诊断列时抛出 ValueError。
    if "power_law_reliable" not in tail_fits:
        raise ValueError("tail exponent plot requires power_law_reliable diagnoses")
    alpha_column = "alpha" if "alpha" in tail_fits else "alpha_pdf"
    ratios = sorted(tail_fits.p_ratio.unique())  # 按占据概率比排序，给纵轴分配离散位置。
    fig, axis = plt.subplots(figsize=(9, 4.8))  # 建立活跃率与占据概率比的诊断平面。
    for row, ratio in enumerate(ratios):  # 为每个概率比确定对应的纵轴行。
        group = tail_fits[np.isclose(tail_fits.p_ratio, ratio)]  # 用浮点近似相等选出该行的拟合结果。
        for _, result in group.iterrows():  # 逐个市场场景添加拟合标记。
            reliable = bool(result.power_law_reliable)  # 把拟合诊断字段转为布尔值以选择显示样式。
            color = "tab:blue" if reliable else "crimson"
            # 通过诊断的拟合画圆点，其他拟合画叉号。
            axis.scatter(result.activity, row, marker="o" if reliable else "x",
                         color=color, s=70, zorder=3)
            # 将描述性尾指数标记在点的上方。
            axis.annotate(f"{result[alpha_column]:.2f}", (result.activity, row),
                          xytext=(0, 10), textcoords="offset points",
                          ha="center", fontsize=9, color=color)
    # 添加通过纯 Pareto 诊断的样式图例。
    axis.scatter([], [], marker="o", color="tab:blue",
                 label="passes basic pure Pareto diagnostics")
    # 添加纯 Pareto 诊断未通过的样式图例。
    axis.scatter([], [], marker="x", color="crimson",
                 label="pure Pareto unsupported; alpha descriptive only")
    activities = sorted(tail_fits.activity.unique())  # 收集扫描实际用过的活跃率刻度。
    axis.set(xscale="log", ylim=(-.5, len(ratios)-.5), xlabel="Activity a", ylabel="p/pc")  # 用对数活跃率作横轴，用离散概率比位置作纵轴。
    axis.set_yticks(range(len(ratios)), [f"{ratio:g}" for ratio in ratios])  # 在每个纵轴位置显示实际 p/p_c。
    axis.set_xticks(activities, [f"{activity:g}" for activity in activities])  # 用实验活跃率作为横轴刻度。
    axis.grid(axis="x", alpha=.3)  # 只在横轴添加网格，便于比较活跃率位置。
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.15), fontsize=8)  # 将诊断图例放在轴外下方。
    axis.set_title("Market tail exponent reliability (labels: descriptive fits)")  # 提示标签是描述性尾拟合估计。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭格点尾部拟合诊断图。


def plot_active_clusters_by_size(activity_scaling: pd.DataFrame, path: Path) -> None:
    # 为市场收益率的影响因素实验画出活跃团簇期望数随边长的变化。
    # 引用
    #     scripts/analysis/analyze_market_returns.py 的 main() 用格点团簇数统计推得的活跃团簇期望值绘图。
    # 输入
    #     activity_scaling 为含 length、activity、expected_active_cluster_count 及其 SEM 的标度表。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存各活跃率的标度曲线和标准误。
    fig, axis = plt.subplots(figsize=(7, 4.8))  # 建立一个坐标系供不同活跃率的曲线比较。
    for activity, group in activity_scaling.groupby("activity", sort=True):  # 按团簇活跃率分组。
        group = group.sort_values("length")  # 各曲线按边长升序连线。
        # 绘制期望活跃团簇数及其均值标准误。
        axis.errorbar(
            group["length"],
            group["expected_active_cluster_count"],  # 纵坐标由调用者按 a 乘平均团簇总数得到。
            yerr=group["expected_active_cluster_count_sem"],  # 用期望活跃团簇数的 SEM 画误差条。
            marker="o",
            capsize=3,
            label=f"a={activity:g}",
        )
    axis.set_xscale("log", base=2)  # 尺寸轴用以 2 为底的对数尺度。
    axis.set_yscale("log")  # 期望活跃团簇数也用对数尺度。
    axis.set_xlabel("System length L")  # 标注格点边长。
    axis.set_ylabel(r"Expected active clusters $a\langle K\rangle$")  # 标注期望值口径 a⟨K⟩。
    axis.grid(which="both", alpha=0.25)  # 添加对数主次刻度网格。
    axis.legend()  # 用图例标明各线对应的活跃率。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭活跃团簇标度图。


def plot_network_structure_metrics(metrics: pd.DataFrame, path: Path) -> None:
    # 为网络拓扑压力测试比较占据前各类网络的结构指标。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 将逐图结构指标绘成拓扑对照图。
    # 输入
    #     metrics 为按拓扑和实现记录的全图平均度、聚类系数、最大连通分量比例及近似平均最短路表。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存网络实现均值和实现间标准差组成的四联柱状图。
    columns = (
        ("mean_degree", "Mean degree"),
        ("clustering_coefficient", "Clustering coefficient"),
        ("largest_connected_component_fraction", "Largest component fraction"),
        ("average_shortest_path_length_approx", "Approx. mean shortest path"),
    )
    order = ["lattice", "er", "ws", "ba"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))  # 给四种整图指标各分配一幅面板。
    for axis, (column, label) in zip(axes.flat, columns, strict=True):  # 把指标列及显示标签分派到对应轴。
        grouped = metrics.groupby("topology")[column]  # 按拓扑汇总当前指标的各个图实现。
        means = grouped.mean().reindex(order)  # 以实现均值作柱高，统一四种拓扑的显示顺序。
        errors = grouped.std().reindex(order).fillna(0)  # 以实现间样本标准差作误差，缺失值改为零。
        axis.bar(order, means, yerr=errors, capsize=3)  # 把均值和误差画成各拓扑的柱状图。
        axis.set_ylabel(label)  # 标记面板对应的整图结构指标。
        axis.grid(axis="y", alpha=0.25)  # 添加水平网格便于比较柱高。
    fig.suptitle("Matched network structure metrics")  # 标注匹配网络这一比较场景。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭整图结构指标图。


def plot_network_degree_distributions(degrees: pd.DataFrame, path: Path) -> None:
    # 为网络拓扑压力测试绘制按拓扑池化的经验度分布。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 从逐图度频数绘图，scripts/figures/plot_network_degree.py 的模块顶层从保存数据重绘。
    # 输入
    #     degrees 为含 topology、degree、node_count 的逐实现度频数表。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存各拓扑的双对数度分布，零度节点计入归一化分母。
    fig, axis = plt.subplots(figsize=(7, 4.8))  # 用一个图比较各拓扑的度分布。
    # 按拓扑和度跨网络实现将节点数池化。
    aggregated = (
        degrees.groupby(["topology", "degree"], as_index=False)["node_count"].sum()
    )
    totals = aggregated.groupby("topology")["node_count"].transform("sum")  # 取得每种拓扑所有度共同使用的总节点数，含零度。
    aggregated["fraction"] = aggregated["node_count"] / totals
    for topology, group in aggregated.groupby("topology", sort=False):  # 按拓扑绘制独立曲线。
        # 孤立节点计入总节点数。对数坐标轴只显示正度。
        visible = group[group["degree"] > 0]
        axis.loglog(visible["degree"], visible["fraction"], "o-", markersize=3, label=topology.upper())  # 仅绘制正度点，节点比例沿用完整节点分母。
    axis.set_xlabel("Degree k")  # 标记整数节点度。
    axis.set_ylabel("Node fraction")  # 标注纵轴为该拓扑下节点比例。
    axis.grid(which="both", alpha=0.25)  # 在双对数坐标显示主次网格。
    axis.legend()  # 用图例标记各拓扑曲线。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭网络度分布图。


def plot_network_criticality(
    summary: pd.DataFrame, estimates: pd.DataFrame, path: Path
) -> None:
    # 为网络拓扑压力测试绘制各拓扑的序参量、池化易感性和有效临界位置。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 将网络渗流汇总和有效临界估计绘成对照图。
    # 输入
    #     summary 为各拓扑随实际占据概率变化的 p_infinity 与 susceptibility_pooled 表。
    #     estimates 为每拓扑的 effective_pc 估计表。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存双纵轴的四联渗流图。
    order = ["lattice", "er", "ws", "ba"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))  # 格点、ER、WS 和 BA 各占一个面板。
    for axis, topology in zip(axes.flat, order, strict=True):  # 按固定拓扑顺序对应面板。
        group = summary[summary["topology"] == topology].sort_values("p")  # 取当前拓扑的观测并按实际 p 排序。
        # 查找当前拓扑由池化易感性峰定义的有效临界概率。
        pc = float(
            estimates.loc[estimates["topology"] == topology, "effective_pc"].iloc[0]
        )
        axis.plot(group["p"], group["p_infinity"], "o-", color="tab:blue", label=r"$P_\infty$")  # 用左轴绘制原图归一化的序参量。
        axis.set_xlabel("Occupation probability p")  # 标实际占据概率 p。
        axis.set_ylabel(r"$P_\infty$", color="tab:blue")  # 将左轴颜色与序参量曲线对应。
        twin = axis.twinx()  # 建立与左轴共用 p 的右纵轴。
        # 用右轴绘制池化有限团簇易感性。
        twin.plot(
            group["p"],
            group["susceptibility_pooled"],
            "s-",
            color="tab:red",
            markersize=3,
            label=r"$\chi$",
        )
        twin.set_ylabel(r"$\chi$", color="tab:red")  # 用与易感性曲线相同的颜色标右轴。
        axis.axvline(pc, color="black", linestyle="--", linewidth=1)  # 画有效临界概率的竖线。
        axis.set_title(f"{topology.upper()}, effective pc={pc:.3f}")  # 标题同时标明网络拓扑和有效临界估计。
        axis.grid(alpha=0.2)  # 添加浅网格便于读数。
    fig.suptitle("Numerical site-percolation thresholds by topology")  # 标明四幅图比较的是数值渗流阈值。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭网络渗流双轴图。


def plot_network_market_comparison(summary: pd.DataFrame, path: Path) -> None:
    # 为网络拓扑压力测试比较各拓扑的超额峰度和标准化极端事件概率。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 将池化市场统计绘成极端收益对照图。
    # 输入
    #     summary 为含 topology、relative_p、超额峰度、双侧 5σ 和 8σ 事件概率及单侧崩盘概率的汇总表。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存四项极端收益指标的比较图。
    specifications = (
        ("return_excess_kurtosis", "Excess kurtosis"),
        ("p_abs_z_gt_5", r"$P(|z|>5)$"),
        ("p_abs_z_gt_8", r"$P(|z|>8)$"),
        ("negative_crash_probability", r"$P(z<-5)$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))  # 为峰度与三种极端概率各建立一幅面板。
    for topology, group in summary.groupby("topology", sort=False):  # 把不同网络拓扑画成不同曲线。
        group = group.sort_values("relative_p")  # 按相对各拓扑有效临界点的概率比排序。
        for axis, (column, ylabel) in zip(axes.flat, specifications, strict=True):  # 逐个将指标列分配给对应面板。
            axis.plot(group["relative_p"], group[column], "o-", label=topology.upper())  # 绘制该拓扑的指标随相对概率变化。
            axis.set_xlabel(r"$p/p_c^{eff}$")  # 标明横轴使用拓扑自身的有效临界概率归一。
            axis.set_ylabel(ylabel)  # 标明面板上的收益统计或相应阈值。
            axis.grid(alpha=0.25)  # 为该子图加网格。
    axes[0, 0].legend()  # 在首个面板统一给出拓扑图例。
    fig.suptitle("Network topology and standardized extreme events")  # 标注总体比较主题。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭网络极端收益对照图。


def plot_network_return_ccdfs(distributions: pd.DataFrame, path: Path) -> None:
    # 为网络拓扑压力测试比较不同网络的绝对标准化收益 CCDF。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 将按相对临界概率生成的 CCDF 表绘成面板图。
    # 输入
    #     distributions 为含 relative_p、topology、x 与 value 的绝对收益 CCDF 长表。
    #     path 为输出图像路径。
    # 输出 None，在 path 按相对概率比保存各拓扑的分布和标准正态基准。
    ratios = sorted(distributions["relative_p"].unique())  # 排序不同相对概率比，固定面板顺序。
    fig, axes = plt.subplots(1, len(ratios), figsize=(5 * len(ratios), 4.5), squeeze=False)  # 为每个概率比建立一幅横向排列的面板。
    for axis, ratio in zip(axes.flat, ratios, strict=True):  # 把横向各轴与概率比分配对应。
        selected = distributions[np.isclose(distributions["relative_p"], ratio)]  # 按浮点近似相等筛出当前概率比的分布。
        for topology, group in selected.groupby("topology", sort=False):  # 为当前场景下各拓扑单独画尾部。
            axis.loglog(group["x"], group["value"], label=topology.upper())  # 在双对数坐标绘制经验绝对收益 CCDF。
        gaussian_x = np.geomspace(0.05, 8, 200)  # 创建正阈值的等比网格供正态参照使用。
        axis.loglog(gaussian_x, 2 * stats.norm.sf(gaussian_x), "k--", label="Gaussian")  # 用两倍标准正态生存概率画绝对收益 CCDF 参照。
        axis.set_title(f"p/pc_eff={ratio:g}")  # 标题给出相对拓扑有效临界点的概率比。
        axis.set_xlabel("|z|")  # 标注横轴为标准化收益的绝对值。
        axis.set_ylabel(r"$P(|z|\geq x)$")  # 标注纵轴为绝对 z 大于等于阈值的概率。
        axis.grid(which="both", alpha=0.2)  # 添加对数轴的主次网格。
        axis.legend(fontsize=8)  # 用图例区别拓扑与正态基准。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭网络绝对收益分布图。


def plot_network_tail_exponents(tails: pd.DataFrame, path: Path) -> None:
    # 为网络拓扑压力测试比较各拓扑和概率比的连续尾 PDF 指数。
    # 引用
    #     scripts/simulation/compare_networks.py 的 main() 将尾指数与分层 bootstrap 区间绘图。
    # 输入
    #     tails 为含 topology、relative_p、alpha_pdf、alpha_ci_low 和 alpha_ci_high 的拟合表。
    #     path 为输出图像路径。
    # 输出 None，在 path 保存不同拓扑的尾指数与区间对照图。
    fig, axis = plt.subplots(figsize=(7, 4.8))  # 建立一个面板用于比较各网络场景的尾指数。
    for topology, group in tails.groupby("topology", sort=False):  # 按拓扑划分尾指数曲线。
        group = group.sort_values("relative_p")  # 对各拓扑按相对占据概率排序。
        # 画各拓扑的尾部点估计与不对称纵向误差条。
        axis.errorbar(
            group["relative_p"],
            group["alpha_pdf"],
            yerr=np.vstack(  # 把上下误差叠成形状为 (2, 场景数) 的数组。
                [
                    group["alpha_pdf"] - group["alpha_ci_low"],  # 用指数点估计减区间下界得到下误差。
                    group["alpha_ci_high"] - group["alpha_pdf"],  # 用区间上界减点估计得到上误差。
                ]
            ),
            marker="o",
            capsize=3,
            label=topology.upper(),
        )
    axis.set_yscale("log")  # 尾指数用对数纵轴比较。
    axis.set_xlabel(r"$p/p_c^{eff}$")  # 横轴按拓扑有效临界概率归一。
    axis.set_ylabel("Absolute-return PDF tail exponent alpha")  # 纵轴标明连续尾密度指数。
    axis.grid(alpha=0.25)  # 添加浅网格便于比较。
    axis.legend()  # 显示网络拓扑图例。
    fig.tight_layout()  # 调整边距，避免图例和轴标签被裁切。
    _finish_figure(fig, path)  # 按统一设置保存并关闭网络尾指数比较图。
