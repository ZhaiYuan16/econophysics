#!/usr/bin/env python3
"""Independent finite-size avalanche scaling analysis from saved tail fits."""
# 以 main() 为独立入口，汇总雪崩持续时间的标度资格并重建雪崩规模的有限尺寸标度结果。

# 本脚本用于 SOC 检验实验，读取 scripts/analysis/analyze_soc_research.py 生成的持续时间诊断和尾拟合表。
from __future__ import annotations  # 延后求值类型注解。

import argparse  # 解析必填的 SOC 分析目录参数。
from pathlib import Path  # 表示输入和输出结果的文件路径。

import pandas as pd  # 读取 SOC 诊断表并写出持续时间分析状态。

from local_market.soc_scaling import write_size_scaling  # src/local_market/soc_scaling.py 的 write_size_scaling 根据已拟合的尾部分布生成规模标度结果。


def main():
    # 功能：读取 SOC 检验实验的归档尾拟合，独立计算规模标度并标记持续时间标度的准备状态。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。从命令行读取必填的 --root，目录下须有 duration_diagnostics.csv 和 tail_fits.csv，允许旧版无表头的空诊断文件。
    # 输出：返回 None，在 --root 下写出 duration_scaling.csv，并由 write_size_scaling() 写出 size_scaling.csv。至少三条可识别的截断拟合才绘制规模标度与坍缩图。
    # 建立独立标度分析的命令行解析器。
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)  # 分析根目录必须显式指定。
    args = parser.parse_args()  # 读取用户提供的目录。
    root = args.root
    # 读取上游诊断。旧版在无事件时可能生成没有列名的空文件。
    diagnostics = {}
    for name in ("duration_diagnostics", "tail_fits"):
        try:
            diagnostics[name] = pd.read_csv(root / f"{name}.csv")
        except pd.errors.EmptyDataError:
            diagnostics[name] = pd.DataFrame()
    durations = diagnostics["duration_diagnostics"]
    # 筛出状态为 ok 的行。旧表无状态列时，合格集为空。
    qualified = durations[durations.status == "ok"] if "status" in durations else pd.DataFrame()
    # 写出合格行数及门槛。满三行时仍标记需要拟合持续时间截断。
    pd.DataFrame([dict(status="insufficient_duration_dynamic_range" if len(qualified) < 3
        else "duration_cutoff_fit_required", qualified_sizes=len(qualified), required=3)]).to_csv(
        root/"duration_scaling.csv", index=False)
    # 以独立分析模式生成雪崩规模的有限尺寸标度结果。
    write_size_scaling(root, diagnostics["tail_fits"], standalone=True)


if __name__ == "__main__":
    # 启动已归档 SOC 结果的标度分析。
    main()
