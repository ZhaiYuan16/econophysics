#!/usr/bin/env python3
"""Run one small CB baseline experiment and save machine-readable output."""

from __future__ import annotations  # 延后求值类型注解，使返回类型可以引用当前配置类。

import argparse  # 标准库，ArgumentParser 定义命令行选项，Namespace 保存解析后的参数。
import json  # 标准库，dumps 将参数及统计字典编码为 JSON 文本。
from pathlib import Path  # 标准库，Path 表示路径，并提供创建目录、拼接路径和写文件的方法。

import pandas as pd  # 第三方库，DataFrame 将各条时间序列整理成表格，to_csv 保存表格。

from local_market.config import BaselineConfig  # 本地 src/local_market/config.py，读取、保存和校验基准参数。
from local_market.experiments import run_baseline  # 本地 src/local_market/experiments.py，按配置组织 CB 模拟。


def parse_args() -> argparse.Namespace:
    # 读取命令行中的配置路径和输出目录。
    # 由 main() 调用，为后续读取配置和保存结果准备路径。
    # 输入：
    #       来自进程命令行。--config 为必需的 YAML 路径，--output-dir 默认为 results/tables。
    # 输出：
    #       Namespace 对象，其 config、output_dir 属性均为 Path 对象。

    parser = argparse.ArgumentParser(description=__doc__)   
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    return parser.parse_args()                          # 解析实际命令行，缺少必需参数时由 argparse 报错并退出。


def main() -> None:
    # 串联配置读取、单次模拟和结果保存三项流程。
    # 由文件末尾的脚本入口调用。
    # 无显式输入参数，返回 None。

    args = parse_args()                                 # 读取命令行中的 YAML 配置路径和输出目录。
    config = BaselineConfig.from_yaml(args.config)      # 从 YAML 文件读取实验配置并校验。
    simulation = run_baseline(config)                   # 按配置运行单次 CB 模拟；返回结果对象，包含逐步数据。
    args.output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录，允许父目录缺失或目录已存在。

    stem = f"baseline_{config.mode}_L{config.length}_seed{config.seed}"
    
    frame = pd.DataFrame(                               # 各数组每个元素对应一步交易。DataFrame 将它们按列对齐。
        {
            "step": range(config.steps),
            "demand": simulation.demands,
            "return": simulation.returns,
            "log_price": simulation.log_prices[1:],     # 去掉初始值，使每行对应本步交易后的对数价格。
            "occupied_count": simulation.occupied_counts,
            "cluster_count": simulation.cluster_counts,
        }
    )
    csv_path = args.output_dir / f"{stem}.csv"          # 将逐步数据写入 CSV
    metadata_path = args.output_dir / f"{stem}.json"    # 将参数与汇总指标写入 JSON。
    frame.to_csv(csv_path, index=False)
    
    metadata = config.to_dict() | {                     # to_dict 将配置转为普通字典，再加入收益率均值、样本标准差与非零比例。
        "output_csv": str(csv_path),
        "return_mean": float(simulation.returns.mean()) if config.steps else None,
        "return_std": float(simulation.returns.std(ddof=1)) if config.steps > 1 else None,
        "nonzero_return_fraction": (                    # 零步时均值和非零比例记为 None，步数不足两步时样本标准差记为 None。
            float((simulation.returns != 0).mean()) if config.steps else None
        ),
    }
    
    metadata_path.write_text(                           # dumps 保留中文并缩进排版，write_text 以 UTF-8 写入 JSON 文件。
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {csv_path}")                          
    print(f"saved {metadata_path}")


if __name__ == "__main__":
    main()  # 文件作为脚本执行时启动完整实验流程。
