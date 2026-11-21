#!/usr/bin/env python3
"""Add scaling and diagnostic tables to a completed refined scan."""
# 为渗流临界性质实验的精细扫描补充有限尺寸标度、方向对称性和尾部截断诊断。
# analyze() 从归档扫描表生成这三类诊断，并为结果目录更新来源说明和哈希清单。

# 本脚本在 scripts/simulation/scan_criticality_refined.py 写出扫描产物后运行，使用已有构型和拟合表。

from __future__ import annotations  # 延后求值类型注解。

import argparse  # 解析命令行的扫描结果目录。
import hashlib  # 计算脚本、绘图源文件和结果文件的 SHA-256。
import json  # 将来源信息和校验清单序列化为 JSON。
from pathlib import Path  # 拼接扫描目录及其表格、元数据路径。

import numpy as np  # 对数组取对数、从构型重采样并计算分位数。
import pandas as pd  # 读写扫描表，按格长和占据概率分组。
from scipy import stats  # 对临界点的观测量进行线性回归。


def analyze(root: Path) -> None:
    # 功能：在精细扫描结果目录中补写标度、对称性和截断尺度诊断，专用于渗流临界性质实验。
    # 引用：模块命令行入口把 --root 传入本函数，默认处理 results/criticality_revision_20261113/full。
    # 输入
    # root：扫描结果根目录，tables/ 下须有构型、汇总、团簇直方图及尾拟合表，metadata/ 用于保存来源。
    # 输出：返回 None，写出三张诊断 CSV、来源 JSON 和 SHA-256 清单，并在终端打印标度表。
    tables = root / "tables"
    # 读取逐占据构型观测，用于配对比较横向与纵向贯通。
    samples = pd.read_csv(tables / "configuration_observables.csv")
    # 读取各格长和概率条件的汇总量，用于在理论临界点估计指数。
    summary = pd.read_csv(tables / "summary.csv")
    # 读取团簇规模频数，用于确定各尾拟合条件的最大观测规模。
    histogram = pd.read_csv(tables / "cluster_histogram.csv")
    # 读取截断幂律拟合结果，用其 s_c 检查优化上界。
    fits = pd.read_csv(tables / "cluster_tail_fits.csv")

    # 取 p/p_c 近似为 1 的行，按格长排列后拟合。少于三个格长时停止。
    critical = summary[np.isclose(summary.p_ratio, 1.0)].sort_values("length")
    if len(critical) < 3:
        raise ValueError("at least three lattice sizes at p_c are needed for scaling")
    scaling_rows = []
    # 分别检查最大团簇占全部格点的比例，以及剔除最大团簇和剔除全部贯通团簇两种易感性口径。
    for name, column, theory in (
        ("p_infinity", "p_infinity_mean", -5 / 48),
        ("chi_largest_excluded", "chi_largest_excluded", 43 / 24),
        ("chi_all_spanning_excluded", "chi_all_spanning_excluded", 43 / 24),
    ):
        # 双对数回归斜率是观测量对格长 L 的标度指数。
        regression = stats.linregress(np.log(critical.length), np.log(critical[column]))
        # 保存拟合指数、标准误、理论指数及尺度点数。
        scaling_rows.append({"observable": name, "p": float(critical.p.iloc[0]),
            "fitted_exponent": regression.slope, "exponent_standard_error": regression.stderr,
            "theoretical_exponent": theory, "r_squared": regression.rvalue**2,
            "size_count": len(critical), "fit_method": "log-log OLS at theoretical p_c"})
    # 将三种观测量的指数汇总落盘。
    pd.DataFrame(scaling_rows).to_csv(tables / "scaling_exponents.csv", index=False)

    # 使用固定随机种子，使方向对称性的重采样可重复。
    rng = np.random.default_rng(20261113 + 5)
    symmetry_rows = []
    # 在相同格长和占据概率下比较两个方向。
    for (length, ratio, p), group in samples.groupby(["length", "p_ratio", "p"]):
        # 对同一构型的横纵贯通指示量作差，保留配对。
        difference = (group.span_x - group.span_y).to_numpy(dtype=float)
        # 每轮有放回抽取同样数量的构型，再沿构型轴求平均。
        boot = difference[rng.integers(0, len(group), size=(1000, len(group)))].mean(axis=1)
        # 取 1000 轮平均差的双侧 95% 百分位区间。
        low, high = np.quantile(boot, [.025, .975])
        # 同时保存原样本平均差和 bootstrap 区间。
        symmetry_rows.append({"length": length, "p_ratio": ratio, "p": p,
            "p_x_minus_p_y": difference.mean(), "ci_low": low, "ci_high": high,
            "configurations": len(group), "bootstrap_replicates": 1000})
    # 将各条件的方向差和区间写入独立表。
    pd.DataFrame(symmetry_rows).to_csv(tables / "direction_symmetry.csv", index=False)

    # 对理论临界点按格长和剔除口径取团簇规模最大值。
    maxima = histogram[np.isclose(histogram.p_ratio, 1.0)].groupby(
        ["length", "exclusion"]).cluster_size.max()
    diagnostics = []
    # 按已有尾拟合行查回同一格长和剔除口径的最大值。
    for row in fits.itertuples():
        observed_max = int(maxima.loc[(row.length, row.exclusion)])  # 将索引结果转为整数团簇规模。
        upper_bound = observed_max * 100  # 截断尺度优化上界等于最大观测团簇规模的 100 倍。
        diagnostics.append({"length": row.length, "exclusion": row.exclusion,
            "count_threshold": row.count_threshold,
            "full_tail_observed_max": observed_max,
            "cutoff_scale_optimizer_upper_bound": upper_bound,
            "cutoff_scale_at_upper_bound": bool(row.cutoff_s_c >= .99 * upper_bound),  # 到达上界的 99% 时标记为贴边。
            "cutoff_scale_status": "not_identified" if row.cutoff_s_c >= .99 * upper_bound
                                   else "finite_optimum"})
    # 保存截断尺度的识别状态。
    pd.DataFrame(diagnostics).to_csv(tables / "cutoff_diagnostics.csv", index=False)

    metadata = root / "metadata"
    # 记录分析脚本自身的源码哈希。
    provenance = {"analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "figures_regenerated_after_scan": True,
                  "figure_source_sha256": hashlib.sha256(  # 同时记录重绘所用的源码版本。
                      Path("src/local_market/criticality_revision_plots.py").read_bytes()).hexdigest(),
                  "inputs": ["configuration_observables.csv", "summary.csv",
                             "cluster_histogram.csv", "cluster_tail_fits.csv"],
                  "direction_bootstrap_unit": "independent occupancy configuration",
                  "cutoff_flag_rule": "fitted s_c >= 0.99 * optimizer upper bound (100 * maximum observed cluster size)"}
    # 写出分析输入、重采样单位和上界判定规则。
    (metadata / "postanalysis.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # 遍历结果目录并以相对路径为键计算哈希，排除将被更新的清单本身。
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*")) if path.is_file() and path.name != "sha256_manifest.json"}
    # 写回校验清单。
    (metadata / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # 在终端展示标度回归结果，便于快速核对。
    print(pd.DataFrame(scaling_rows).to_string(index=False))


if __name__ == "__main__":
    # 用模块说明作为命令行帮助简介。
    parser = argparse.ArgumentParser(description=__doc__)
    # 允许指定精细扫描目录，默认取完整扫描。
    parser.add_argument("--root", type=Path,
                        default=Path("results/criticality_revision_20261113/full"))
    # 解析命令行目录并启动扫描后分析。
    analyze(parser.parse_args().root)
