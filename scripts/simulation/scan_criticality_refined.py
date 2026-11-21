#!/usr/bin/env python3
"""Run revised square-lattice criticality smoke, pilot, or full scan."""
# 运行渗流临界性质实验的精细扫描。main() 是命令行入口，按冒烟、先导或完整模式构造概率网格，
# 采集开放边界格点观测，并汇总两种团簇排除口径下的易感性、伪临界峰位、贯通交叉点和团簇规模分布。
#
# 完整模式用基准扫描的峰位选择细扫区间。精细扫描表供渗流后处理与验证脚本读取，
# 其中静态水平贯通概率及易感性也用于 SOC 检验实验构造有限尺寸窗口。

from __future__ import annotations  # 延后求值函数的类型注解。

import argparse  # 标准库，用于声明并解析命令行选项。
import hashlib  # 标准库，用 SHA-256 记录源文件和产物的内容摘要。
import json  # 标准库，用于将运行元数据和产物清单写为 JSON。
import platform  # 标准库，用于记录 Python 版本。
import resource  # 标准库，读取本进程的资源使用信息。
import subprocess  # 标准库，运行 Git 命令读取 HEAD 提交。
from datetime import datetime, timezone  # 标准库，用 datetime 和 timezone 记录 UTC 时间及运行时长。
from pathlib import Path  # 标准库，用路径对象定位配置、源码和结果文件。

import matplotlib  # 第三方 Matplotlib 绘图库，这里记录绘图环境版本。
import numpy as np  # 第三方 NumPy，用于生成概率比网格并处理易感性分母为零的点。
import pandas as pd  # 第三方 pandas，用于读取基准峰位、合并扩展扫描记录和保存输出表。
import scipy  # 第三方 SciPy 科学计算库，这里记录数值计算环境版本。
import yaml  # 第三方 PyYAML，用于读取精细扫描配置。

from local_market.criticality_revision import (  # 本地 src/local_market/criticality_revision.py，开放边界扫描及其统计函数。
    crossings, pooled_histogram, pseudocritical, ratio_grid, scan, summarize,  # crossings 求贯通交叉点，pooled_histogram 汇总规模频数，pseudocritical 求峰位，ratio_grid 生成网格，scan 采样，summarize 汇总观测。
    tail_fits,  # 拟合临界点团簇规模的尾部模型。
)
from local_market.criticality_revision_plots import make_figures  # 本地 src/local_market/criticality_revision_plots.py，按精细扫描表生成图。


def main() -> None:
    # 运行渗流临界性质实验的精细扫描，完整模式还会在易感性峰位靠近局部网格边缘时补点。
    # 由本文件的 __main__ 入口调用。scripts/analysis/analyze_criticality_refined.py 的 analyze() 读取扫描表作标度分析，
    # scripts/analysis/criticality_validation_analysis.py 的 main() 和 scripts/analysis/criticality_validation_gridaware.py 的 main() 读取结果作验证。
    # scripts/simulation/run_soc_research.py 的 main() 用完整模式的静态汇总表构造 SOC 窗口。
    # 输入无函数参数，从命令行读取 YAML 配置、运行模式和输出根目录。
    # 输出 None。在该模式的新目录中写入表格、图、运行元数据和校验清单。目录已存在时抛出异常。
    parser = argparse.ArgumentParser(description=__doc__)  # 用原模块说明作为命令行帮助的简介。
    parser.add_argument("--config", type=Path, default=Path("configs/criticality_refined.yaml"))  # 默认读取仓库的精细扫描配置。
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), required=True)  # 必须选择冒烟、先导或完整运行模式。
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/criticality_revision_20261113"))  # 三种模式分别写入此根目录的对应子目录。
    args = parser.parse_args()  # 解析运行选项，模式不合法时由 argparse 退出。
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))  # 读取完整的精细扫描及重抽样参数。
    p_c = float(config["p_c"])  # p_c 为理论参照，实际占据概率 p 等于 p_c 乘扫描概率比。
    seed = int(config["seed"])  # 提取扫描与重抽样的主种子。
    step = float(config["ratio_step"])  # 公共交叉窗口与局部峰窗的细网格步长，单位为 p/p_c。
    low, high = map(float, config["spanning_window"])  # 取跨尺寸比较的公共窗口端点。
    common = [round(float(x), 6) for x in np.arange(low, high + step / 2, step)]  # 将网格点舍入到六位小数，给上端点留半步容差。
    if 1.0 not in common:  # 公共窗口必须实际包含 p/p_c = 1 这一网格点。
        raise ValueError("common window must contain p_c")
    if args.mode == "smoke":
        lengths = [8, 12]  # 冒烟运行使用两个小点阵。
        ratios = {length: (0.98, 1.0, 1.02) for length in lengths}  # 每个冒烟尺寸扫临界点和其左右各一个点。
        counts = {length: 4 for length in lengths}  # 冒烟模式每个尺寸和概率点抽取四份配置。
        common_used = list(ratios[lengths[0]])  # 两个尺寸共用冒烟网格，用于贯通交叉估计。
        bootstraps = 100  # 冒烟统计使用 100 次重抽样。
    elif args.mode == "pilot":
        lengths = [512]  # 先导运行只检查 L = 512。
        common_used = list(map(float, config["pilot_ratios"]))  # 由配置选择大尺寸先导扫描的概率比。
        ratios = {512: tuple(common_used)}
        counts = {512: int(config["pilot_samples"])}  # 读取先导扫描每个概率点的配置数。
        bootstraps = int(config["bootstrap_replicates"])  # 使用配置指定的先导重抽样次数。
    else:
        lengths = list(map(int, config["lengths"]))  # 读取完整扫描覆盖的格点边长。
        estimates = pd.read_csv(config["baseline_estimates"])  # 读取基准渗流扫描保存的伪临界估计。
        centers = {int(row.length): float(row.p_ratio_chi_peak) for row in estimates.itertuples()}  # 用各尺寸的易感性峰位 p/p_c 作局部细扫中心。
        centers[512] = float(config["peak_center_512"])  # 对 L = 512 使用配置给定的峰窗中心。
        # 为各尺寸合并粗网格、局部峰窗和公共贯通窗口，按概率比去重排序。
        ratios = {length: ratio_grid(config["coarse_ratios"], centers[length],
            float(config["peak_half_width"]), config["spanning_window"], step)  # 指定局部峰窗半宽和公共窗口，两个窗口都采用细网格步长。
            for length in lengths}
        counts = {int(key): int(value) for key, value in config["samples_per_length"].items()}  # 配置数按格点边长给定，应用于该尺寸的每个概率点。
        common_used = common  # 完整模式用公共贯通窗口比较不同尺寸。
        bootstraps = int(config["bootstrap_replicates"])  # 使用配置指定的完整扫描重抽样次数。
    out = args.output_root / args.mode
    out.mkdir(parents=True, exist_ok=False)  # 创建模式专属目录，防止覆盖已有运行结果。
    tables, figures, metadata = (out / name for name in ("tables", "figures", "metadata"))
    for path in (tables, figures, metadata):
        path.mkdir()  # 分别创建表格、图和元数据子目录。
    (metadata / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")  # 保存原始配置文本，供复现负载和扫描网格。
    started = datetime.now(timezone.utc)  # 记录扫描开始的 UTC 时刻。
    streamed_histogram = tables / "configuration_cluster_counts.csv" if args.mode == "full" else None  # 完整模式把逐配置规模频数追加到 CSV，其余模式将其保留在内存。
    scanned = scan(lengths, ratios, counts, p_c, seed, streamed_histogram)  # 对开放格点采样，得到逐配置观测、规模频数、池化计数和扫描耗时。
    extension_notes = []
    if args.mode == "full":  # 仅完整运行根据初扫的易感性曲线向外扩展局部峰窗。
        # 记录各尺寸待检查的局部峰窗上下界，补点后会更新。
        local_bounds = {length: [centers[length] - float(config["peak_half_width"]),
                                 centers[length] + float(config["peak_half_width"])]
                        for length in lengths}
        for _ in range(3):  # 最多执行三轮补点。
            additions = {}
            for length in lengths:
                group = scanned.samples[scanned.samples.length == length]  # 取该边长已完成的全部逐配置扫描记录。
                # 在每个 p/p_c 处，分别按两套排除口径跨配置求团簇规模和及规模平方和。
                pooled = group.groupby("p_ratio").agg({
                    "largest_excluded_m1": "sum", "largest_excluded_m2": "sum",
                    "all_spanning_excluded_m1": "sum", "all_spanning_excluded_m2": "sum"})
                left, right = local_bounds[length]
                desired = set()
                for rule in ("largest_excluded", "all_spanning_excluded"):  # 分别检查排除一个最大团簇和排除全部贯通团簇的曲线。
                    curve = pooled[f"{rule}_m2"] / pooled[f"{rule}_m1"].replace(0, np.nan)  # 用池化二阶矩除以池化一阶矩，分母为零的点记为 NaN。
                    peak = float(curve.fillna(0).idxmax())  # 以离散网格上易感性最大的概率比定位峰，NaN 按零参与寻峰。
                    if peak <= left + 1.5 * step:  # 峰至左界不超过 1.5 个步长，或落到左界之外时，向低概率补点。
                        desired.update(round(float(x), 6) for x in np.arange(left - .02, left, step))  # 从左界减 0.02 处开始，按细网格步长取左界之前的点。
                    if peak >= right - 1.5 * step:  # 峰至右界不超过 1.5 个步长，或落到右界之外时，向高概率补点。
                        desired.update(round(float(x), 6) for x in np.arange(right + step, right + .02 + step / 2, step))  # 从右界外一个步长处开始，向外取约 0.02 范围内的点，末端使用半步容差。
                additions[length] = tuple(sorted(desired - set(ratios[length])))  # 去掉已经扫描过的概率比，只留下新增点。
            if not any(additions.values()):  # 所有尺寸都无需新点时提前结束补扫。
                break
            extra_lengths = [length for length in lengths if additions[length]]  # 只选择仍有新增概率比的尺寸。
            extra = scan(extra_lengths, additions, counts, p_c, seed, streamed_histogram)  # 用原配置数、临界参照和种子扫描新点，完整频数继续追加到同一文件。
            scanned.samples = pd.concat([scanned.samples, extra.samples], ignore_index=True)  # 合并主扫描和补扫的逐配置渗流观测。
            scanned.histograms = pd.concat([scanned.histograms, extra.histograms], ignore_index=True)  # 合并内存保留的临界点附近逐配置规模频数。
            scanned.pooled_counts = pd.concat([scanned.pooled_counts, extra.pooled_counts], ignore_index=True)  # 合并主扫描和补扫跨配置池化的规模计数。
            scanned.runtime_seconds += extra.runtime_seconds  # 累加扫描耗时，用于运行元数据。
            for length in extra_lengths:
                ratios[length] = tuple(sorted(set(ratios[length]) | set(additions[length])))  # 更新该尺寸已扫描的概率比全集，避免下一轮重复采样。
                local_bounds[length][0] = min(local_bounds[length][0], min(additions[length]))  # 把临时峰窗下界扩至本轮最小新增概率比。
                local_bounds[length][1] = max(local_bounds[length][1], max(additions[length]))  # 把临时峰窗上界扩至本轮最大新增概率比。
                extension_notes.append({"length": length, "added_ratios": list(additions[length])})  # 保存本轮补扫的尺寸与概率比，供复核扩展过程。
    summary = summarize(scanned.samples, bootstraps, seed + 1)  # 按尺寸和概率汇总序参量、贯通概率及两套易感性，并做重抽样推断。
    peaks = pseudocritical(scanned.samples, summary, bootstraps, seed + 2)  # 定位两套易感性的离散网格峰和局部插值峰，并估计网格峰区间。
    cross = crossings(scanned.samples, lengths, common_used, p_c, bootstraps, seed + 3)  # 用共享网格估计各尺寸对的贯通曲线交叉点和重抽样区间。
    histogram = pooled_histogram(scanned.histograms, scanned.samples, scanned.pooled_counts)  # 按配置数和 L² 归一化池化规模频数，得到每格点团簇密度 n_s。
    if args.mode == "full":
        # 完整模式在 p/p_c = 1 拟合团簇规模尾部，按尺寸、排除口径和频数阈值比较结果。
        fits = tail_fits(scanned.histograms, 1.0, config["tail_count_thresholds"],
                         int(config["tail_bootstrap_replicates"]), seed + 4)  # 使用尾部拟合专用的重抽样次数和种子偏移。
    else:
        fits = pd.DataFrame(columns=["length", "exclusion", "count_threshold", "status"])  # 为冒烟和先导模式保留带列名的空拟合表。
    # 为逐配置观测、频数表、汇总和推断结果指定输出表名。
    frames = {"configuration_observables": scanned.samples,
              "configuration_cluster_counts": scanned.histograms,
              "summary": summary, "pseudocritical": peaks, "crossings": cross,
              "cluster_histogram": histogram, "cluster_tail_fits": fits}
    for name, frame in frames.items():
        if name == "configuration_cluster_counts" and streamed_histogram is not None:  # 完整模式的逐配置频数已经流式写出，跳过该表以保留完整记录。
            continue
        frame.to_csv(tables / f"{name}.csv", index=False)  # 将当前结果表写为 CSV，省略行索引。
    make_figures(summary, peaks, cross, histogram, fits, p_c, figures)  # 根据精细扫描汇总、峰位、交叉点、规模分布和尾拟合生成图。
    ended = datetime.now(timezone.utc)  # 记录表格与绘图完成后的 UTC 时刻。
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # 读取当前 HEAD 提交，和实际文件内容摘要共同标识运行源码。
    # 选取本入口及扫描、绘图、团簇识别和统计相关源码，供内容校验。
    source_files = [Path(__file__), Path("src/local_market/criticality_revision.py"),
                    Path("src/local_market/criticality_revision_plots.py"),
                    Path("src/local_market/percolation.py"), Path("src/local_market/statistics.py")]
    # 汇集运行范围、耗时、网格扩展、统计定义、版本和源码摘要。
    meta = {"mode": args.mode, "started_utc": started.isoformat(), "ended_utc": ended.isoformat(),
            "scan_wall_seconds": scanned.runtime_seconds,  # 仅累计抽样扫描调用的耗时。
            "total_wall_seconds": (ended - started).total_seconds(),  # 包括扫描、整理表和绘图在内的耗时。
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,  # 读取进程的峰值常驻内存，Linux 下单位为 KiB。
            "sample_count": len(scanned.samples), "grid_points_by_length":
            {str(length): len(ratios[length]) for length in lengths},  # 记录完成补扫后每个尺寸的概率点数。
            "peak_window_extensions": extension_notes,
            "seed": seed, "bootstrap_replicates": bootstraps, "code_head": revision,
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()  # 对列出的实际源码字节计算 SHA-256。
                              for path in source_files},
            "definitions": {"p_infinity": "s_max/L**2, all configurations",
                "chi_largest_excluded": "pooled sum(s^2)/sum(s), one largest cluster per configuration excluded",
                "chi_all_spanning_excluded": "pooled sum(s^2)/sum(s), all open-boundary spanning clusters excluded",
                "span_x": "left-to-right", "span_y": "top-to-bottom",
                "span_any": "x or y", "span_both": "x and y",
                "n_s": "C_s/(M*L**2)"},  # 将规模 s 的池化频数 C_s 除以样本数 M 与每份配置的格点数 L²。
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__}}
    (metadata / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")  # 保存本次精细扫描的运行元数据。
    # 按相对于本次运行目录的路径计算已写出文件的 SHA-256。
    manifest = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(out.rglob("*")) if path.is_file()}  # 仅为文件计算摘要，清单自身此时尚未写出。
    (metadata / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")  # 保存产物校验清单。
    # 在终端输出结果目录、扫描及汇总用时、峰值内存和样本数。
    print(json.dumps({"output": str(out), "scan_seconds": scanned.runtime_seconds,
                      "total_seconds": meta["total_wall_seconds"],
                      "max_rss_kib": meta["max_rss_kib"], "sample_count": len(scanned.samples)}, indent=2))


if __name__ == "__main__":
    main()  # 直接运行脚本时执行所选模式的精细渗流扫描。
