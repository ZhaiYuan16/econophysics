#!/usr/bin/env python3
"""Run one small CB baseline experiment and save machine-readable output."""
# 运行一次 Cont-Bouchaud 格点市场基线模拟。parse_args() 读取配置路径与输出目录，main() 是命令行入口，
# 调用市场模拟并保存逐步交易表和运行摘要。
#
# 本脚本把格点与交易参数转为单条基线轨迹，供检查超额需求、对数收益率、对数价格与团簇结构。

from __future__ import annotations  # 延后求值函数的类型注解。

import argparse  # 标准库，用于声明并解析命令行选项。
import json  # 标准库，用于将配置和交易摘要写为 JSON。
from dataclasses import asdict  # 标准库，将基线配置数据类转换为元数据字典。
from pathlib import Path  # 标准库，用路径对象接收参数和构造输出文件名。

import pandas as pd  # 第三方 pandas，用于组织并导出逐步交易表。

from local_market.config import BaselineConfig  # 本地 src/local_market/config.py，读取并校验基线 YAML 配置。
from local_market.experiments import run_baseline  # 本地 src/local_market/experiments.py，按配置运行一条格点市场基线轨迹。


def parse_args() -> argparse.Namespace:
    # 解析基线模拟的命令行选项。
    # 由本文件的 main() 调用，用于定位配置文件和结果目录。
    # 输入无函数参数，从进程命令行读取选项。
    # 输出 argparse.Namespace，含必填的 config 和默认值为 results/tables 的 output_dir，均为路径。
    parser = argparse.ArgumentParser(description=__doc__)  # 用原模块说明作为命令行帮助的简介。
    parser.add_argument("--config", type=Path, required=True)  # 要求提供基线 YAML 配置路径。
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))  # 允许将本次输出写入指定目录。
    return parser.parse_args()  # 返回解析后的路径，参数不合法时由 argparse 退出。


def main() -> None:
    # 运行并保存一次基线模拟，包括固定团簇或逐步重抽团簇两种配置模式。
    # 由本文件的 __main__ 入口调用，供命令行运行。
    # 输入无函数参数，配置文件和输出目录来自命令行。
    # 输出 None。写出逐步交易 CSV 和配置及收益率摘要 JSON，并打印文件路径。
    args = parse_args()  # 取得本次运行的配置路径和输出目录。
    config = BaselineConfig.from_yaml(args.config)  # 从 YAML 读取格点大小、交易步数、模式与随机种子等模拟参数。
    simulation = run_baseline(config)  # 根据模式运行交易，取得逐步需求、收益率、价格与格点结构记录。
    args.output_dir.mkdir(parents=True, exist_ok=True)  # 创建结果目录，允许复用已有目录。

    stem = f"baseline_{config.mode}_L{config.length}_seed{config.seed}"  # 用模式、格点边长和种子区分基线结果。
    # 将各时间序列按交易步对齐，每笔交易占一行。
    frame = pd.DataFrame(
        {
            "step": range(config.steps),  # 交易步从 0 编号。
            "demand": simulation.demands,  # 超额需求按团簇规模加权求得。
            "return": simulation.returns,  # 对数收益率等于超额需求除以市场深度。
            "log_price": simulation.log_prices[1:],  # 跳过初始对数价格，保留每步交易后的值。
            "occupied_count": simulation.occupied_counts,  # 当前交易配置中的占据格点总数。
            "cluster_count": simulation.cluster_counts,  # 当前交易配置中的团簇总数。
        }
    )
    csv_path = args.output_dir / f"{stem}.csv"
    metadata_path = args.output_dir / f"{stem}.json"
    frame.to_csv(csv_path, index=False)  # 导出逐步交易数据，省略 pandas 行索引。
    metadata = asdict(config) | {  # 在基线配置字段外补充输出位置和收益率统计。
        "output_csv": str(csv_path),
        "return_mean": float(simulation.returns.mean()) if config.steps else None,  # 交易步数为零时没有可报告的均值。
        "return_std": float(simulation.returns.std(ddof=1)) if config.steps > 1 else None,  # 样本标准差用 T−1 作分母，至少需要两步交易。
        "nonzero_return_fraction": (
            float((simulation.returns != 0).mean()) if config.steps else None  # 分母包括所有交易步，零步时记为空值。
        ),
    }
    # 将配置和摘要写入同名 JSON，保留 UTF-8 文本。
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {csv_path}")  # 报告逐步交易表的位置。
    print(f"saved {metadata_path}")  # 报告运行摘要的位置。


if __name__ == "__main__":
    main()  # 直接运行脚本时执行基线模拟。
