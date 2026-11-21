"""Write finite-size avalanche scaling results and diagnostic figures."""
# read_soc_run_settings() 读取归档实际使用的时间和重采样设置，write_size_scaling() 从雪崩尺度拟合估计有限尺寸指数并写出诊断。
#
# SOC 检验实验的综合分析和独立标度脚本共用此入口，把不同格点尺寸的截断尺度和预热后的雪崩记录汇成标度诊断。

from __future__ import annotations  # 延后求值类型注解。

import json  # 标准库，用于读取模拟记录的运行元数据。
from pathlib import Path  # 标准库路径对象，用于定位配置、事件表和输出文件。

import matplotlib.pyplot as plt  # Matplotlib 绘制截断标度和数据塌缩诊断。
import numpy as np  # NumPy 用于对数回归输入和恢复原尺度拟合曲线。
import pandas as pd  # Pandas 读取事件表、过滤拟合并写出汇总 CSV。
import yaml  # PyYAML 读取归档配置和旧运行的分析参数。
from scipy.stats import linregress  # SciPy 的线性回归用于估计 log(S_c) 对 log(L) 的斜率。


def read_soc_run_settings(
    root: Path,
    lengths: list[int],
) -> tuple[dict[int, tuple[int, int]], int, int]:
    # 恢复 SOC 检验实验按格长实际使用的步数和预热期，以及路径重采样参数。
    # 引用
    #     scripts/analysis/analyze_soc_research.py 的 main() 用这些设置重算路径、事件和尾部诊断。
    #     本文件的 write_size_scaling() 独立运行时读取相同的时间设置。
    # 输入
    #     root 为模拟归档目录，含 config.yaml，新归档还含 run.json。
    #     lengths 为当前分析所需的格点边长列表。
    # 输出依次为以格长为键的 (总步数, 预热步数) 字典、路径重采样次数、配置中的模拟主种子。
    config = yaml.safe_load((root / "config.yaml").read_text())  # 归档配置提供老版本的分析参数和主种子。
    metadata_path = root / "run.json"
    run = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}  # 有运行记录时优先采用实际值。
    recorded_steps = run.get("steps_by_length", {})
    recorded_burn_in = run.get("burn_in_by_length", {})
    configured_steps = config.get("steps_by_length", {})
    configured_burn_in = config.get("burn_in_by_length", {})
    timing_by_length = {}
    for length in lengths:
        # JSON 把格长键存成字符串，归档 YAML 使用整数键。
        steps = int(recorded_steps.get(str(length), configured_steps.get(length, config.get("steps"))))
        burn_in = int(recorded_burn_in.get(str(length), configured_burn_in.get(length, config.get("burn_in"))))
        if not 0 <= burn_in < steps:
            raise ValueError(f"L={length}: burn-in {burn_in} must be smaller than the run length {steps}")
        timing_by_length[length] = (steps, burn_in)

    bootstrap_replicates = run.get("bootstrap_replicates")
    if bootstrap_replicates is None:
        # 旧版冒烟运行只记录模式，正式运行的重采样次数保存在归档配置中。
        bootstrap_replicates = 10 if run.get("smoke", False) else config["bootstrap_replicates"]
    return timing_by_length, int(bootstrap_replicates), int(config["seed"])


def write_size_scaling(
    root: Path,
    fits: pd.DataFrame,
    *,
    standalone: bool,
    timing_by_length: dict[int, tuple[int, int]] | None = None,
) -> None:
    """Write the size diagnostics in the format of the calling analysis."""
    # 为 SOC 检验实验写出截断尺度的有限尺寸回归和雪崩数据塌缩诊断。
    # 引用
    #     scripts/analysis/analyze_soc_research.py 的 main() 传入实际时间设置并写综合分析格式。
    #     scripts/analysis/analyze_soc_scaling.py 的 main() 从已有拟合写独立标度格式。
    # 输入
    #     root 为试验输出目录，含 cascade/events.csv.gz，独立运行时读取其中的运行元数据和归档配置。
    #     fits 为按尺寸保存的尾部拟合表，使用 model、length、cutoff、boundary、exponent，独立格式还使用 status。
    #     standalone 为布尔值，控制成功拟合筛选、CSV 字段及尺度图样式。
    #     timing_by_length 为可选字典，键为格长，值依次为总步数和预热步数，缺省时读归档。
    # 输出 None，在 root 下写 size_scaling.csv，至少三条有效截断记录时再写两张诊断图。
    has_models = "model" in fits and fits["model"].notna().any()  # 只有表头或跳过原因时视为无拟合模型。
    cuts = pd.DataFrame()  # 预设空筛选结果，以处理无有效模型的输入。
    if has_models:
        cuts = fits[fits["model"] == "cutoff"]
        if standalone:
            cuts = cuts[(cuts["status"] == "ok") & cuts["boundary"].eq(False)]  # 独立报告仅纳入成功且截断尺度可识别的拟合。
            cuts = cuts.sort_values("length")  # 按尺度排序，保证独立回归图连线的顺序。
        else:
            cuts = cuts[~cuts["boundary"].astype(bool)]  # 综合报告排除标记为边界的截断拟合。

    table = root / "size_scaling.csv"
    size_count = len(cuts)  # 计入回归用的截断拟合记录条数。
    if size_count < 3:
        if standalone:
            row = dict(status="insufficient_identified_cutoffs", size_count=size_count, required=3)  # 独立格式用 size_count 和 required 记录规模不足。
        else:
            status = "insufficient_identified_cutoffs" if has_models else "no_valid_tail_fits"
            row = dict(status=status, identified_sizes=size_count, required_sizes=3)  # 综合格式区分缺模型与无足够可识别截断值。
        pd.DataFrame([row]).to_csv(table, index=False)  # 有效记录不足时也写出一行可读的状态表。
        return

    regression = linregress(np.log(cuts["length"]), np.log(cuts["cutoff"]))  # 在双对数空间拟合 S_c 随 L 的幂律，斜率为 D_s。
    if standalone:
        # 独立格式报告 D_s、标准误和 R²。
        row = dict(
            status="preliminary_three_size_fit",
            D_s=regression.slope,
            standard_error=regression.stderr,
            r2=regression.rvalue**2,  # 相关系数平方给出回归的 R²。
            size_count=size_count,
        )
    else:
        # 综合格式输出指数和说明字段。
        row = dict(
            exponent_Ds=regression.slope,
            r2=regression.rvalue**2,  # 相关系数平方给出回归的 R²。
            size_count=size_count,
            note="preliminary three-size regression",
        )
    pd.DataFrame([row]).to_csv(table, index=False)  # 按调用方格式写出标度回归摘要。

    figures = root / "figures"
    figures.mkdir(exist_ok=True)  # 准备存储两张标度诊断图的目录。
    fig, ax = plt.subplots()  # 建立截断尺度对格点长度图。
    if standalone:
        ax.plot(cuts["length"], cuts["cutoff"], "o")  # 用点标出估计的各尺度截断值。
        ax.plot(cuts["length"], np.exp(regression.intercept) * cuts["length"]**regression.slope)  # 通过指数变换把对数回归线画回原尺度。
    else:
        ax.plot(cuts["length"], cuts["cutoff"], "o-")  # 用连线显示相邻格点尺寸的拟合截断。
    ax.set(xscale="log", yscale="log", xlabel="L", ylabel="S_c")  # 把 L 与 S_c 都设置为对数轴。
    fig.savefig(figures / "size_scaling.png", dpi=160)  # 保存截断尺度图。
    plt.close(fig)  # 释放截断尺度图。

    if timing_by_length is None:
        timing_by_length, _, _ = read_soc_run_settings(root, cuts["length"].to_list())  # 独立标度使用与综合分析相同的归档时间设置。
    events = pd.read_csv(root / "cascade" / "events.csv.gz")  # 加载级联模式逐事件雪崩记录。
    pure = fits[fits["model"] == "power"]
    if standalone:
        tau = pure.set_index("length").loc[cuts["length"], "exponent"].mean()  # 按可识别截断尺寸索引纯幂律指数并取均值。
    else:
        tau = pure.loc[pure["length"].isin(cuts["length"]), "exponent"].mean()  # 筛选相同尺度的纯幂律拟合，取指数均值。

    fig, ax = plt.subplots()  # 为不同尺度的重标度曲线建立共同坐标轴。
    for length in cuts["length"]:
        steps, burn_in = timing_by_length[length]  # 按当前格长取总步数和预热步数。
        in_analysis_window = events["step"].between(burn_in, steps, inclusive="left")  # 步号从零起，保留预热后到模拟结束前的事件。
        sizes = events.loc[(events["length"] == length) & in_analysis_window, "size"]
        counts = sizes.value_counts().sort_index()  # 对预热后雪崩按整数规模计数并从小到大排列。
        sizes_in_curve = counts.index.to_numpy()  # 取得有样本的雪崩规模作为曲线的离散支持。
        probability = counts.to_numpy() / len(sizes)  # 规模计数除以该尺度预热后事件总数。
        # 把不同 L 的曲线画在同一重标度坐标系中。
        ax.plot(
            sizes_in_curve / length**regression.slope,  # 横轴除以 L 的 D_s 次幂。
            sizes_in_curve**tau * probability,  # 纵轴用 S 的平均尾指数次幂乘经验规模概率。
            ".",
            label=f"L={length}",
        )
    ax.set(xscale="log", yscale="log", xlabel="S/L^Ds", ylabel="S^tau P(S;L)")  # 在双对数轴上比较各个 L 的塌缩程度。
    ax.legend()  # 用图例标明各条曲线的格点长度。
    fig.savefig(figures / "size_collapse_diagnostic.png", dpi=160)  # 保存塌缩诊断图。
    plt.close(fig)  # 释放塌缩诊断图。
