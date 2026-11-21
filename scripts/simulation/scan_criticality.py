#!/usr/bin/env python3
"""Scan square-lattice site percolation around the critical point."""
# 运行渗流临界性质实验的基准扫描，估计方格点阵的伪临界位置和有限尺寸标度。
# parse_args() 读取运行配置与输出选项，main() 是命令行入口，组织扫描、汇总和绘图。
#
# 本脚本生成逐配置观测和按尺寸与占据概率汇总的渗流数据。伪临界峰位用于精细扫描选点，
# 临界观测汇总供市场收益率的影响因素实验构造活跃率的尺寸缩放。

from __future__ import annotations  # 延后求值函数的类型注解。

import argparse  # 标准库，用于声明并解析命令行选项。
import json  # 标准库，用于将配置、统计定义和输出路径写为 JSON。
import platform  # 标准库，读取运行时 Python 版本。
from dataclasses import asdict  # 标准库，将渗流配置数据类转换成元数据字典。
from pathlib import Path  # 标准库，用路径对象处理配置与输出目录。

import matplotlib  # 第三方 Matplotlib 绘图库，这里记录绘图环境版本。
import numpy as np  # 第三方 NumPy 数值计算库，这里记录数值计算环境版本。
import pandas as pd  # 第三方 pandas 表格处理库，这里记录表格处理环境版本。
import scipy  # 第三方 SciPy 科学计算库，这里记录科学计算环境版本。

from local_market.config import CriticalityConfig  # 本地 src/local_market/config.py，读取并校验基准渗流扫描的 YAML 配置。
from local_market.experiments import estimate_finite_size_scaling, run_criticality_scan  # 本地 src/local_market/experiments.py，estimate_finite_size_scaling 拟合标度，run_criticality_scan 生成扫描样本与统计。
from local_market.plotting import (  # 本地 src/local_market/plotting.py，基准渗流实验的绘图函数。
    plot_cluster_distributions,  # 绘制所选占据概率下的团簇规模分布。
    plot_critical_observables,  # 绘制序参量、易感性等观测量随占据概率的变化。
    plot_finite_size_scaling,  # 绘制临界观测及其随格点尺寸变化的拟合趋势。
    plot_pseudocritical_estimates,  # 绘制离散扫描网格上的伪临界估计。
)


def parse_args() -> argparse.Namespace:
    # 解析渗流临界性质实验基准扫描的命令行选项。
    # 由本文件的 main() 调用，用于定位配置和输出，并控制扫描进度显示。
    # 输入无函数参数，从进程命令行读取选项。
    # 输出 argparse.Namespace，含必填的配置路径 config、默认 results 的 output_root 和默认关闭的 quiet 开关。
    parser = argparse.ArgumentParser(description=__doc__)  # 用原模块说明作为命令行帮助的简介。
    parser.add_argument("--config", type=Path, required=True)  # 要求提供基准扫描的 YAML 配置路径。
    parser.add_argument("--output-root", type=Path, default=Path("results"))  # 指定表格和图的共同父目录。
    parser.add_argument("--quiet", action="store_true")  # 请求关闭扫描进度显示。
    return parser.parse_args()  # 返回解析结果，参数不合法时由 argparse 退出。


def main() -> None:
    # 运行渗流临界性质实验的基准扫描，保存逐配置观测、汇总、伪临界估计和标度拟合。
    # 由本文件的 __main__ 入口调用。scripts/simulation/scan_criticality_refined.py 的 main() 读取伪临界峰位，
    # scripts/analysis/analyze_market_returns.py 的 main() 读取汇总表构造随尺寸变化的市场活跃率。
    # 输入无函数参数，扫描配置和结果目录来自命令行。
    # 输出 None。写出扫描 CSV、图和运行元数据 JSON，并在终端显示估计结果。
    args = parse_args()  # 获取基准扫描的配置路径、输出位置和进度开关。
    config = CriticalityConfig.from_yaml(args.config)  # 加载格点尺寸、采样次数、临界概率及团簇排除规则等配置。
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)  # 创建 CSV 和元数据目录，允许复用。
    figures.mkdir(parents=True, exist_ok=True)  # 创建绘图目录，允许复用。

    # 扫描多个格点尺寸和占据概率，依次取得逐配置样本、分组汇总、团簇频数与伪临界估计。
    samples, summary, histogram, estimates = run_criticality_scan(
        config, show_progress=not args.quiet  # 使用配置构建扫描网格，只有 quiet 未开启时才显示进度。
    )
    scaling = estimate_finite_size_scaling(summary)  # 在 p/p_c = 1 对序参量和池化易感性分别做对数回归，估计它们随格点边长变化的幂律指数。
    paths = {
        "samples": tables / "criticality_samples.csv",
        "summary": tables / "criticality_summary.csv",
        "histogram": tables / "criticality_cluster_histogram.csv",
        "estimates": tables / "criticality_pseudocritical.csv",
        "scaling": tables / "criticality_scaling.csv",
    }
    samples.to_csv(paths["samples"], index=False)  # 保存逐格点配置的观测数据。
    summary.to_csv(paths["summary"], index=False)  # 保存按尺寸与占据概率汇总的渗流统计。
    histogram.to_csv(paths["histogram"], index=False)  # 保存团簇规模频数和归一化密度。
    estimates.to_csv(paths["estimates"], index=False)  # 保存池化易感性峰和序参量最大斜率处的扫描占据概率及对应观测值。
    scaling.to_csv(paths["scaling"], index=False)  # 保存有限尺寸标度拟合。

    # 绘制各格点尺寸的临界观测，并以理论临界概率 p_c 作参照。
    plot_critical_observables(
        summary, config.p_c, figures / "criticality_observables.png"
    )
    # 绘制有限团簇的规模分布。
    plot_cluster_distributions(
        histogram,
        config.histogram_p_ratios,  # 指定需要绘图的 p/p_c。
        config.finite_cluster_exclusion,  # 用实际的有限团簇排除规则标注分布图。
        figures / "criticality_cluster_distributions.png",
    )
    # 对照各尺寸的伪临界估计和理论临界概率。
    plot_pseudocritical_estimates(
        estimates, config.p_c, figures / "criticality_pseudocritical.png"
    )
    # 绘制随尺寸变化的临界观测与标度拟合。
    plot_finite_size_scaling(
        summary,
        scaling,
        1.0,  # 选用 p/p_c = 1 的观测绘制临界点标度。
        figures / "criticality_finite_size_scaling.png",
    )

    # 记录复现扫描所需的配置、统计口径、软件版本和表格位置。
    metadata = {
        "config": asdict(config),  # 将配置字段转成 JSON 可序列化的字典。
        "definitions": {
            "p_infinity": "mean(largest_size / total_lattice_sites), zero if empty",  # 序参量分母为全部格点数，随后对配置取均值。
            "susceptibility_pooled": (
                "sum over realizations and retained clusters of s^2 divided by "
                "the corresponding sum of s"
            ),
            "finite_cluster_exclusion": config.finite_cluster_exclusion,  # 记录易感性和有限团簇直方图使用的排除规则。
            "pseudocritical": "discrete scan-grid locations; no interpolation",
            "cluster_histogram": "n_s is cluster count per lattice site and realization",
        },
        "software": {
            "python": platform.python_version(),  # 记录运行本扫描的 Python 版本。
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    # 以 UTF-8 写入本次基准扫描的元数据。
    (tables / "criticality_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(estimates.to_string(index=False))  # 展示扫描网格上的伪临界估计。
    print(scaling.to_string(index=False))  # 展示各观测量的有限尺寸标度拟合结果。
    print(f"saved stage-2 outputs under {args.output_root}")  # 报告结果根目录。


if __name__ == "__main__":
    main()  # 直接运行脚本时执行基准渗流扫描。
