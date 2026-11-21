#!/usr/bin/env python3
"""Run versioned SOC paths and derive finite-size windows from static P_x."""
# 运行 SOC 检验实验的版本化模拟。windows() 从静态水平贯通概率反查有限尺寸占据率窗口，
# main() 是命令行入口，按条件和初始占据率模拟路径，保存逐步、逐事件、逐轮及逐路径汇总。
#
# 本脚本接收渗流临界性质实验的精细静态扫描汇总。SOC 后处理读取窗口和路径、事件表，
# 用于复算预热后的占据率诊断、分析雪崩尾部和比较有限尺寸标度。
from __future__ import annotations  # 延后求值窗口函数的类型注解。
import argparse  # 标准库，用于声明并解析命令行选项。
import json  # 标准库，用于编码运行元数据。
import time  # 标准库，用单调计时器记录运行耗时。
from pathlib import Path  # 标准库，用路径对象定位配置与各条件的结果目录。
import numpy as np  # 第三方 NumPy，用于反查窗口、派生随机种子和汇总逐步统计。
import pandas as pd  # 第三方 pandas，用于读入静态扫描并拼接路径、事件及轮次表。
import yaml  # 第三方 PyYAML，用于读取 SOC 实验配置。
from local_market.soc_research import ResearchConfig, simulate  # 本地 src/local_market/soc_research.py，ResearchConfig 保存动力学参数，simulate 生成一条路径的三类记录。


def windows(summary: pd.DataFrame, lengths: list[int]) -> pd.DataFrame:
    # 为 SOC 检验实验按格点尺寸 L 提取静态水平贯通概率 P_x 从 0.1 到 0.9 对应的占据率窗口，并记录易感性峰位。
    # 由本文件的 main() 调用。它按尺寸取窗口上下界，统计预热后落在闭区间内的模拟步比例。
    # scripts/analysis/analyze_soc_research.py 的 main() 读取写出的 critical_windows.csv，按同一窗口复算该比例。
    # 输入
    #     summary 为静态渗流扫描汇总表，包含尺寸 length、占据率 p、水平贯通概率 span_x_mean，
    #     以及排除所有贯通团簇后的易感性 chi_all_spanning_excluded。每个请求尺寸须有扫描记录，且 P_x 须覆盖两个端点。
    #     lengths 为请求的格点边长列表，单位为格点数。列表顺序决定输出行序。
    # 输出为 DataFrame，每个请求尺寸对应一行，依次包含 length、窗口下界 window_low、上界 window_high、
    #     易感性最大处的扫描占据率 susceptibility_peak 和方法名 method。缺失尺寸或端点覆盖不足时抛出 ValueError，空尺寸列表返回无行无列的表。
    out = []
    for length in lengths:
        group = summary[summary.length == length].sort_values("p")  # 按边长选出静态扫描记录，再沿占据率 p 从小到大排列，为反查阈值建立顺序。
        x = group.p.to_numpy()  # 取一维占据率数组，每项对应该尺寸的一条扫描记录。
        y = group.span_x_mean.to_numpy()  # 取同序的一维水平贯通概率数组，P_x 为该 p 下水平贯通的样本比例。
        if not (y.min() <= .1 and y.max() >= .9):  # 用扫描值的极值检查两个阈值是否被覆盖，避免在扫描区间外给出窗口边界。
            raise ValueError(f"static P_x does not bracket both endpoints for L={length}")
        # Isotonic envelope removes minor binomial noise before inversion.
        y = np.maximum.accumulate(y)  # 沿 p 取累计最大值，压平 P_x 的局部回落，使贯通概率可用于反查占据率。
        low, high = np.interp([.1, .9], y, x)  # 在单调化的 P_x 序列上做线性插值，依次反查 10% 和 90% 贯通概率对应的 p。
        peak = group.loc[group.chi_all_spanning_excluded.idxmax(), "p"]  # 在当前尺寸的原始扫描点中找到排除所有贯通团簇后易感性最大的一行，取该点的 p。
        # 收集当前尺寸的窗口、易感性峰位和插值方法，供模拟和后续分析按尺寸查找。
        out.append(dict(length=length, window_low=low, window_high=high,
                        susceptibility_peak=peak, method="P_x isotonic linear interpolation"))
    return pd.DataFrame(out)  # 按 lengths 的顺序把各尺寸结果组装成表。


def main():
    # 运行 SOC 检验实验的各动力学条件，计算预热后的占据率和耗散诊断。
    # 由本文件的 __main__ 入口调用。scripts/analysis/analyze_soc_research.py 的 main() 读取输出目录重算轨迹与事件诊断，
    # scripts/analysis/analyze_soc_scaling.py 的 main() 通过有限尺寸标度函数分析事件表。
    # 输入无函数参数。命令行提供配置路径、必填的新输出目录和可选的快速冒烟开关。
    # 输出 None。复制配置，保存窗口、逐步路径、事件、轮次、逐路径摘要和运行元数据。输出目录已存在时抛出异常。
    parser = argparse.ArgumentParser()  # 创建 SOC 实验的命令行解析器。
    parser.add_argument("--config", type=Path, default=Path("configs/soc_research_v1.yaml"))  # 默认读取 v1 动力学配置，也可传入其他版本的配置。
    parser.add_argument("--output", type=Path, required=True)  # 要求为本次运行提供输出目录。
    parser.add_argument("--smoke", action="store_true")  # 开启冒烟模式以缩短路径和减少条件及重复数。
    args = parser.parse_args()  # 解析运行选项，缺少必填参数时由 argparse 退出。
    args.output.mkdir(parents=True, exist_ok=False)  # 创建新运行目录，防止覆盖同目录的已有结果。
    cfg = yaml.safe_load(args.config.read_text())  # 将 YAML 配置读成后续分组与动力学设置。
    (args.output / "config.yaml").write_text(args.config.read_text())  # 归档原配置，供下游读取主种子和实验设置。
    lengths = [32, 64, 128]  # 两种运行模式都为这三个格点边长计算静态窗口。
    win = windows(pd.read_csv(cfg["static_summary"]), lengths)  # 读取静态扫描汇总，为各格点边长计算有限尺寸占据率窗口。
    win.to_csv(args.output / "critical_windows.csv", index=False)  # 保存窗口及峰位，供下游按尺寸复算窗口内比例。
    # 模拟和 run.json 共用各格长的实际步数和预热期，冒烟模式统一为 60 步和 20 步。
    steps_by_length = {
        length: 60 if args.smoke else int(cfg.get("steps_by_length", {}).get(length, cfg["steps"]))
        for length in lengths
    }
    burn_in_by_length = {
        length: 20 if args.smoke else int(cfg.get("burn_in_by_length", {}).get(length, cfg["burn_in"]))
        for length in lengths
    }
    reps = 1 if args.smoke else cfg["replicates"]  # 冒烟模式每个组合模拟一次，正式运行使用配置重复数。
    bootstrap_replicates = 10 if args.smoke else cfg["bootstrap_replicates"]  # 保存后处理应使用的路径重采样次数。
    start = time.perf_counter()  # 记录开始时刻，后续报告已用秒数。
    summary_rows = []
    for condition_index, condition in enumerate(cfg["conditions"]):  # 条件在配置中的位置参与种子派生。
        if args.smoke and condition["name"] not in ("baseline", "cascade"):  # 冒烟模式只运行 baseline 和 cascade 两个条件。
            continue
        name = condition["name"]
        folder = args.output / name
        folder.mkdir()  # 为当前动力学条件创建独立目录。
        trace_files = []
        event_files = []
        round_files = []
        for l in condition.get("lengths", lengths):  # 条件可以指定运行尺寸子集，默认使用静态窗口覆盖的三个尺寸。
            local_steps = steps_by_length[l]  # 读取该格长的实际模拟步数。
            local_burn = burn_in_by_length[l]  # 读取该格长的实际预热步数。
            for initial in condition.get("initial_occupancies", cfg["initial_occupancies"]):  # 条件可以覆盖全局初始占据率列表。
                initial = float(initial)
                probability_key = int(np.float64(initial).view(np.uint64))  # 沿用答辩演示的浮点编码，近邻初始占据率使用不同随机流。
                for rep in range(reps):
                    sequence = np.random.SeedSequence([cfg["seed"], condition_index, l, probability_key, rep])  # 用主种子和实验坐标派生路径随机流。
                    seed = int(sequence.generate_state(1, dtype=np.uint64)[0])  # 取一个 64 位整数作为 simulate() 的种子并写入归档。
                    path_id = f"{name}_L{l}_p{initial}_r{rep:02d}"  # 用可读的小数表示初始占据率，保留区分不同浮点值所需的精度。
                    # 装入格点大小、时间、交易活跃率和驱动耗散参数。
                    conf = ResearchConfig(l, local_steps, local_burn, cfg["activity"],
                        condition.get("theta", cfg["theta"]), condition.get("q", cfg["q"]),  # 优先采用当前条件的需求触发阈值 θ 和同向格点耗散概率 q。
                        condition.get("drive_nodes", cfg["drive_nodes"]), condition["mode"])  # 优先使用条件指定的驱动节点数，并选择该条件的动力学模式。
                    trace, event, rounds = simulate(conf, initial, seed, path_id)  # 依次取得逐步轨迹、执行过耗散的事件记录和逐耗散轮次记录。
                    # 为三张表附上相同的实验坐标，便于在合并后按路径或条件筛选。
                    for frame in (trace, event, rounds):
                        frame["condition"] = name
                        frame["length"] = l
                        frame["initial_occupancy"] = initial
                        frame["replicate"] = rep
                        frame["seed"] = seed
                    trace_files.append(trace)  # 保留本路径的逐交易步轨迹。
                    event_files.append(event)  # 保留本路径的逐事件记录，包括零移除的触发事件。
                    round_files.append(rounds)  # 保留本路径的逐耗散轮次记录。
                    ss = trace[trace.step >= local_burn]  # 保留步号不小于预热步数的记录。
                    low, high = win.set_index("length").loc[l, ["window_low", "window_high"]]  # 按格点边长取静态窗口的占据率上下界。
                    p = ss.occupancy_after.to_numpy()  # 将每步驱动或耗散后的占据率转为一维序列。
                    half = len(p)//2  # 以序列向下取整的一半为分界，供前后半段均值比较。
                    # 每条路径写一行预热后的占据率和事件摘要，同时保留实验坐标。
                    summary_rows.append(dict(condition=name, length=l, initial_occupancy=initial,
                        replicate=rep, path_id=path_id, seed=seed, steps=local_steps,
                        burn_in=local_burn, mean_occupancy=p.mean(),  # 占据率均值的分母为预热后步数。
                        occupancy_sd=p.std(ddof=1), drift=p[half:].mean()-p[:half].mean(),  # 标准差用样本口径，漂移为后半段减去前半段的平均占据率。
                        window_fraction=np.mean((p >= low)&(p <= high)),  # 以全部预热后步数为分母，计入落在闭窗口内的更新后占据率。
                        event_rate=np.mean(ss.duration > 0), event_count=int((ss.duration > 0).sum()),  # 有耗散轮次的步视为事件，事件率分母为全部预热后步数。
                        mean_event_size=ss.loc[ss.removed_nodes > 0, "removed_nodes"].mean()))  # 只在实际移除了节点的步上平均移除节点数，无这类步时为 NaN。
        for filename, parts in (("paths.csv.gz", trace_files), ("events.csv.gz", event_files),
                                ("rounds.csv.gz", round_files)):
            pd.concat(parts, ignore_index=True).to_csv(folder / filename, index=False, compression="gzip")  # 将同一条件和同一记录类型的路径表纵向合并并保存为 gzip 压缩 CSV。
        print(f"{name}: {len(trace_files)} paths, {time.perf_counter()-start:.1f}s", flush=True)  # 报告当前条件完成的路径数和自模拟开始经过的秒数。
    pd.DataFrame(summary_rows).to_csv(args.output / "path_summary.csv", index=False)  # 保存跨条件的逐路径预热后摘要，供比较汇聚和耗散行为。
    # 记录实际使用的运行模式、步数、预热和重采样次数，供下游从元数据恢复分析设置。
    (args.output / "run.json").write_text(json.dumps(dict(smoke=args.smoke, seconds=time.perf_counter()-start,
        steps_by_length=steps_by_length, burn_in_by_length=burn_in_by_length,
        bootstrap_replicates=bootstrap_replicates,
        replicates=reps, config=str(args.config)), indent=2))


if __name__ == "__main__":
    main()  # 直接运行脚本时执行 SOC 检验实验。
