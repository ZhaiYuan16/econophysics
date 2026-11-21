#!/usr/bin/env python3
"""Run the stage-1 three-point validation without a full parameter scan."""
# 对配置给定的少量占据概率抽取格点，并在每份团簇配置上运行固定团簇交易。
# parse_args() 读取配置与输出目录，main() 是命令行入口，保存逐配置结构、逐笔交易及按占据概率汇总的结果。
#
# 本脚本用于项目快速入门的三点校验，先检查格点抽样、团簇识别和 Cont-Bouchaud 交易能否连通。

from __future__ import annotations  # 延后求值函数的类型注解。

import argparse  # 标准库，用于声明并解析命令行选项。
import json  # 标准库，用于将本次校验配置写为 JSON。
from pathlib import Path  # 标准库，用路径对象定位配置和结果文件。

import numpy as np  # 第三方 NumPy，为配置抽样和固定团簇交易提供随机数生成器。
import pandas as pd  # 第三方 pandas，用于组织逐配置、逐笔交易和汇总表。
import yaml  # 第三方 PyYAML，用于读取小规模校验的 YAML 配置。

from local_market.cb_model import simulate_quenched  # 本地 src/local_market/cb_model.py，在固定团簇配置上生成交易轨迹。
from local_market.networks import sample_lattice_occupancy  # 本地 src/local_market/networks.py，按占据概率抽取方格点阵的占据状态。
from local_market.percolation import identify_lattice_clusters  # 本地 src/local_market/percolation.py，按边界条件识别四邻域团簇。


def parse_args() -> argparse.Namespace:
    # 解析三点校验的命令行选项。
    # 由本文件的 main() 调用，用于定位配置文件和结果目录。
    # 输入无函数参数，从进程命令行读取选项。
    # 输出 argparse.Namespace，含必填的 config 和默认值为 results/tables 的 output_dir，均为路径。
    parser = argparse.ArgumentParser(description=__doc__)  # 用原模块说明作为命令行帮助的简介。
    parser.add_argument("--config", type=Path, required=True)  # 要求提供校验用 YAML 配置路径。
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))  # 允许将校验结果写入指定目录。
    return parser.parse_args()  # 返回解析后的路径，参数不合法时由 argparse 退出。


def main() -> None:
    # 按占据概率重复抽取格点、识别团簇，并在每份配置上运行固定团簇交易。
    # 由本文件的 __main__ 入口调用，供项目快速入门流程执行少量扫描点的校验。
    # 输入无函数参数，格点、交易和抽样参数来自命令行指定的 YAML 配置。
    # 输出 None。写入逐配置团簇表、逐笔交易表、按占据概率汇总的表和配置 JSON，并打印汇总。
    args = parse_args()  # 取得配置路径和结果目录。
    with args.config.open(encoding="utf-8") as stream:  # 以 UTF-8 打开配置文件，并在读取后自动关闭。
        config = yaml.safe_load(stream)  # 读取占据概率、边界条件、重复数和交易参数。

    length = int(config["length"])  # 格点边长 L，点阵形状为 L×L。
    activity = float(config["activity"])  # 团簇每步参与交易的概率 a。
    market_depth = float(config["market_depth"])  # 市场深度 λ，将超额需求换算成对数收益率。
    boundary = config["boundary"]  # 格点的开放或周期边界条件。
    configurations_per_p = int(config["configurations_per_p"])  # 每个占据概率下抽取的独立格点配置数。
    trades_per_configuration = int(config["trades_per_configuration"])  # 每份配置固定团簇后模拟的交易步数。
    rng = np.random.default_rng(int(config["seed"]))  # 由配置种子创建同一次校验共用的随机数生成器。
    args.output_dir.mkdir(parents=True, exist_ok=True)  # 创建结果目录，允许复用已有目录。

    cluster_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    for p in map(float, config["p_values"]):  # 按配置中的顺序遍历占据概率 p。
        for sample in range(configurations_per_p):
            occupied = sample_lattice_occupancy(length, p, rng)  # 按 p 独立抽取 L×L 格点的占据状态。
            clusters = identify_lattice_clusters(occupied, boundary)  # 按指定边界识别四邻域团簇，得到团簇结构和贯通或绕行标记。
            # 每份配置记录一行结构统计，sample 是在当前 p 下从 0 开始的编号。
            cluster_rows.append(
                {
                    "p": p,
                    "sample": sample,
                    "occupied_count": clusters.occupied_count,
                    "cluster_count": clusters.cluster_count,
                    "largest_size": clusters.largest_size,
                    "largest_occupied_fraction": clusters.largest_occupied_fraction,  # 分母为已占据格点数，空配置记为 0。
                    "spans": clusters.spans,  # 开放边界记录是否贯通，周期边界记录是否绕行。
                }
            )
            # 固定这份团簇配置，在每个交易步重抽各团簇的买卖行动。
            simulation = simulate_quenched(
                clusters,
                steps=trades_per_configuration,  # 这份配置要生成的交易步数。
                activity=activity,  # 用团簇活跃率控制每步参与买卖的概率。
                market_depth=market_depth,  # 用市场深度将超额需求换算成对数收益率。
                rng=rng,
            )
            # 把一份配置的交易轨迹展开，每笔交易记录一行需求和收益率。
            return_rows.extend(
                {
                    "p": p,
                    "sample": sample,
                    "trade": trade,  # 在当前配置内从 0 开始的交易编号。
                    "demand": int(demand),
                    "return": float(return_),
                }
                for trade, (demand, return_) in enumerate(
                    zip(simulation.demands, simulation.returns, strict=True)  # 逐步配对需求和对数收益率，长度不一致时直接报错。
                )
            )

    clusters_frame = pd.DataFrame(cluster_rows)  # 将配置记录组装为逐配置表。
    returns_frame = pd.DataFrame(return_rows)  # 将交易记录组装为逐笔表。
    # 按占据概率分别汇总格点配置和交易，再合并两张汇总表。
    summary = (
        clusters_frame.groupby("p", as_index=False)
        .agg(
            mean_occupied=("occupied_count", "mean"),
            mean_cluster_count=("cluster_count", "mean"),
            mean_largest_size=("largest_size", "mean"),
            mean_largest_occupied_fraction=("largest_occupied_fraction", "mean"),  # 对逐配置的最大团簇占据比例取均值。
            spanning_frequency=("spans", "mean"),  # 布尔标记取均值，分母是该 p 的全部配置数。
        )
        .merge(
            returns_frame.groupby("p", as_index=False).agg(
                return_mean=("return", "mean"),  # 合并该 p 所有配置的逐笔收益后取均值。
                return_std=("return", "std"),  # 对该 p 的逐笔收益计算样本标准差，分母为交易笔数减一。
                # 匿名函数计算非零收益比例，由 pandas 的 GroupBy.agg() 对当前 p 的收益调用。
                # 输入 values 为当前 p 的一维逐笔收益 Series，合并该 p 下所有配置。
                # 输出非零收益笔数除以该组交易总笔数。
                nonzero_return_fraction=("return", lambda values: (values != 0).mean()),
            ),
            on="p",  # 只保留同时有结构汇总和交易汇总的 p。
        )
    )

    clusters_path = args.output_dir / "validation_small_clusters.csv"
    returns_path = args.output_dir / "validation_small_returns.csv"
    summary_path = args.output_dir / "validation_small_summary.csv"
    metadata_path = args.output_dir / "validation_small_metadata.json"
    clusters_frame.to_csv(clusters_path, index=False)  # 保存逐配置结构，便于按配置复核团簇统计。
    returns_frame.to_csv(returns_path, index=False)  # 保存逐笔需求和收益率，便于按配置复核交易。
    summary.to_csv(summary_path, index=False)  # 保存按占据概率合并的校验结果。
    # 将本次读取的配置保存为 UTF-8 JSON，便于重现。
    metadata_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))  # 在终端展示按占据概率汇总的结果。
    print(f"saved {summary_path}")  # 报告汇总表路径。


if __name__ == "__main__":
    main()  # 直接运行脚本时执行三点校验。
