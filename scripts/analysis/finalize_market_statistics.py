#!/usr/bin/env python3
"""Build the market chapter's four core figures from archived observations."""
# 为市场收益率的影响因素实验汇编四张核心图。main() 校验并汇总归档数据，plot_archived_time_paths() 绘制固定构型的时间轨迹，plot_market_activity() 比较系综统计。

# 本脚本接收归档市场收益率、audit_market_time_series.py 的时间轨迹和 audit_market_tails_visual.py 的尾部诊断，产出同一口径的图表。
from __future__ import annotations  # 延后求值类型注解。

import argparse  # 解析输入表目录和最终产物目录。
import hashlib  # 计算输入文件的 SHA-256。
import json  # 将归档来源说明写成 JSON。
from pathlib import Path  # 表示归档输入和输出路径。

import matplotlib  # 使用 Matplotlib 配置无窗口绘图后端。
matplotlib.use("Agg")  # 允许在无图形界面的环境中生成 PNG。
import matplotlib.pyplot as plt  # 创建轨迹和活动率统计图。
import numpy as np  # 计算逐步绝对收益、样本方差并核对数组。
import pandas as pd  # 读取归档 CSV、按条件汇总和写出表格。

from local_market.config import MarketAnalysisConfig  # src/local_market/config.py 的 MarketAnalysisConfig 提供系综条件及重复次数。
from local_market.experiments import summarize_market_returns  # src/local_market/experiments.py 的 summarize_market_returns 重算分组标准化和汇总。
from local_market.plotting import plot_return_distributions, plot_tail_exponents  # src/local_market/plotting.py 的 plot_return_distributions 绘制 PDF 和 CCDF，plot_tail_exponents 绘制尾指数。
from local_market.statistics import return_distribution_table  # src/local_market/statistics.py 的 return_distribution_table 生成收益率分布表。


def arguments() -> argparse.Namespace:
    # 功能：解析市场收益率的影响因素实验最终汇编所需的输入及输出路径。
    # 引用：main() 用返回的路径读取归档数据并确定图表目录。
    # 输入：无函数参数。读取命令行中的路径选项。
    # 输出：argparse.Namespace，含 source、time_tables、tail_table 和 output_root。
    # 以模块说明构造命令行帮助。
    parser = argparse.ArgumentParser(description=__doc__)
    # 设置逐笔市场收益率归档的默认位置。
    parser.add_argument("--source", type=Path,
                        default=Path("results/tables/market_returns.csv.gz"))
    # 设置固定构型时间轨迹表的默认目录。
    parser.add_argument("--time-tables", type=Path,
                        default=Path("results/market_time_audit_20261113/tables"))
    # 设置已经审计的尾部拟合表的默认位置。
    parser.add_argument("--tail-table", type=Path,
                        default=Path("results/market_tail_audit_20261114/tables/market_tail_diagnostics.csv"))
    # 设置四张核心图和配套表格的输出根目录。
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/market_statistics_final_20261114"))
    # 解析路径参数，未指定时取以上默认归档。
    return parser.parse_args()


def plot_archived_time_paths(tables: Path, activities: tuple[float, ...], path: Path) -> None:
    # 功能：绘制市场收益率的影响因素实验在 p/p_c=1 下的固定构型收益率时间轨迹。
    # 引用：main() 调用本函数生成 market_time_series.png。
    # 输入
    # tables：含 quenched_a_<activity>.csv 的目录，每张表包含 time、return 和 log_price。
    # activities：要绘图的活动率 a，按传入顺序各占一行，至少有一个活动率。
    # path：PNG 输出路径，父目录由调用者建立。
    # 输出：返回 None，将图保存到 path 后关闭画布。时间或累计收益校验失败时抛出断言异常。
    # 每个活动率一行，三列依次画收益率、绝对收益率和累计对数收益率。
    fig, axes = plt.subplots(len(activities), 3, figsize=(15, 2.75*len(activities)), squeeze=False)
    # 遍历活动率并确定各自的子图行号。
    for row, activity in enumerate(activities):
        # 读取该活动率的单条固定构型轨迹。
        trace = pd.read_csv(tables / f"quenched_a_{activity:g}.csv")
        # 验证时间索引从第 1 步开始连续递增。
        assert np.array_equal(trace.time.to_numpy(), np.arange(1,len(trace)+1))
        # 验证归档对数价格以初始值为零，等于对数收益率累计和。
        np.testing.assert_allclose(trace["return"].cumsum(), trace.log_price, atol=1e-12)
        t = trace.time.to_numpy()  # 横轴用归档的交易步。
        axes[row,0].plot(t,trace["return"],lw=.5)  # 第一列保留收益正负号。
        axes[row,1].plot(t,np.abs(trace["return"]),lw=.5,color="tab:orange")  # 第二列显示逐步波动幅度。
        axes[row,2].plot(t,trace.log_price,lw=.8,color="tab:green")  # 第三列显示相对初始值的对数价格。
        for axis in axes[row]:
            axis.grid(alpha=.2)  # 为三列提供读数网格。
            axis.set_xlabel("simulation time t")  # 三列共用交易步口径。
        axes[row,0].set_ylabel(f"a={activity:g}\nr(t)")  # 在第一列标识本行活动率和收益量。
        axes[row,1].set_ylabel("|r(t)|")  # 标识第二列为绝对对数收益。
        axes[row,2].set_ylabel("cumulative log return")  # 标识第三列为累计对数收益。
    # 首行标题给出三列的变量含义。
    for axis,title in zip(axes[0],("log return","absolute log return","log price relative to t=0")):
        axis.set_title(title)
    # 标注归档时间轨迹的构型条件和抽样设定。
    fig.suptitle("One fixed p/pc=1 occupancy; independent cluster actions each step")
    fig.tight_layout()  # 调整子图和文字间距。
    fig.savefig(path,dpi=180,bbox_inches="tight")  # 将轨迹图以 180 dpi 写入指定路径。
    plt.close(fig)  # 释放轨迹图占用的绘图资源。


def plot_market_activity(summary: pd.DataFrame, path: Path) -> None:
    # 功能：在市场收益率的影响因素实验中绘制六项系综统计量随活动率的变化。
    # 引用：main() 用本函数将补齐方差和绝对收益均值的汇总表绘成核心图。
    # 输入
    # summary：每个 p/p_c 与活动率组合一行的系综统计表，须包含 specs 指定的六列。
    # path：统计汇总图的 PNG 路径。
    # 输出：返回 None，写出图并关闭画布。
    # 六个面板依次用原始收益均值、样本方差、绝对收益均值、标准化极端概率、超额峰度和平均活跃团簇数。
    specs=(("return_mean","Mean r"),("return_variance","Var(r)"),
           ("mean_absolute_return","Mean |r|"),("p_abs_z_gt_5",r"P(|z|>5)"),
           ("return_excess_kurtosis","Excess kurtosis"),
           ("mean_active_cluster_count","Mean active clusters"))
    fig,axes=plt.subplots(3,2,figsize=(11,10))  # 将六个统计量排成三行两列。
    for ratio,group in summary.groupby("p_ratio",sort=True):  # 每个 p/p_c 条件画一条曲线。
        group=group.sort_values("activity")  # 让每条线按活动率从低到高连接。
        # 遍历面板及对应统计列。
        for ax,(column,label) in zip(axes.flat,specs):
            ax.plot(group.activity,group[column],"o-",label=f"p/pc={ratio:g}")  # 以活动率为横轴，保留占据概率条件的图例标签。
            ax.set(xscale="log",xlabel="activity a",ylabel=label)  # 活动率使用对数轴，纵轴保持各统计量口径。
            ax.grid(alpha=.25)  # 加浅色网格便于读数。
    axes[0,0].axhline(0,color="gray",lw=.8)  # 给带符号的收益均值添加零基准。
    axes[0,0].legend(fontsize=8)  # 在第一幅子图汇总占据概率图例。
    fig.suptitle("CB ensemble statistics: 400 configurations and 500 actions each per condition")  # 标注每个条件的构型数和每个构型的抽样次数。
    fig.tight_layout()  # 调整统计图文字和子图间距。
    fig.savefig(path,dpi=180,bbox_inches="tight")  # 保存统计图为 180 dpi PNG。
    plt.close(fig)  # 释放统计图资源。


def sha256(path: Path) -> str:
    # 功能：分块计算归档文件的 SHA-256。
    # 引用：main() 为来源 JSON 中的每个输入文件调用本函数。
    # 输入
    # path：要校验的、可按二进制读取的文件路径。
    # 输出：64 个十六进制字符组成的 SHA-256 字符串。文件读取异常向上传递。
    digest=hashlib.sha256()  # 建立可增量更新的摘要对象。
    with path.open("rb") as source:  # 按原始字节读取，避免文本解码或换行转换。
        for chunk in iter(lambda: source.read(1<<20),b""):  # 每次最多读 1 MiB，空字节串表示文件结束。
            digest.update(chunk)  # 依读取顺序累计文件内容。
    return digest.hexdigest()  # 把最终摘要转成十六进制字符串。


def main() -> None:
    # 功能：重建市场收益率的影响因素实验的四张核心图及其归档统计表。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。命令行提供归档路径和输出目录，系综条件来自 configs/market_analysis.yaml。
    # 输出：返回 None，写出四张 PNG、三张 CSV 和来源 JSON。归档口径校验失败时抛出断言异常。
    # 读取归档路径与输出选项。
    args=arguments()
    # 加载用于复核样本数和标准化的实验参数。
    config=MarketAnalysisConfig.from_yaml("configs/market_analysis.yaml")
    figures=args.output_root/"figures"
    tables=args.output_root/"tables"
    # 建立 PNG 输出目录。
    figures.mkdir(parents=True,exist_ok=True)
    # 建立统计表和来源说明的输出目录。
    tables.mkdir(parents=True,exist_ok=True)
    # 读取归档逐笔收益。
    raw=pd.read_csv(args.source)
    # 按配置重算标准化收益和市场条件统计。
    returns,summary,_=summarize_market_returns(raw,config)
    # 验证重算 z 与归档 z 逐笔一致。
    np.testing.assert_allclose(raw.z.to_numpy(),returns.z.to_numpy(),atol=1e-12)
    # 核对条件数为占据概率条件数与活动率条件数之积。
    assert len(summary)==len(config.p_ratios)*len(config.activities)==12
    # 核对每个条件的交易数为重复数、每重复构型数和每构型交易数的乘积。
    assert (summary.observations==config.replicates*config.configurations_per_replicate*
            config.trades_per_configuration).all()
    distribution_rows=[]
    absolute_means=[]
    # 逐格长、占据概率和活动率建立分布表。
    for key,group in returns.groupby(["length","p_ratio","p","activity"],sort=True):
        raw_return=group["return"].to_numpy(dtype=float)  # 提取原始收益率，用于未标准化矩和零收益判定。
        z=group.z.to_numpy(dtype=float)  # 提取同条件下已中心化、标准化的收益率。
        absolute_means.append({"p_ratio":key[1],"activity":key[3],
                               "mean_absolute_return":float(np.abs(raw_return).mean()),  # 对所有交易求绝对收益均值。
                               "return_variance":float(np.var(raw_return,ddof=1)),  # 以 n−1 作分母计算原始收益样本方差。
                               "nonzero_returns":int(np.count_nonzero(raw_return))})  # 统计原始收益严格非零的交易笔数。
        # 将市场条件键附在每一行分布统计上。
        distribution_rows.extend(dict(zip(("length","p_ratio","p","activity"),key,strict=True)) | row
                                 for row in return_distribution_table(z,raw_returns=raw_return))  # 用全部 z 计算分布及归一化分母，原始收益用于限定 CCDF 阈值范围。
    # 将新增统计量按占据概率比和活动率一对一并入汇总。
    summary=summary.merge(pd.DataFrame(absolute_means),on=["p_ratio","activity"],validate="one_to_one")
    # 核对方差与原汇总的标准差平方一致。
    np.testing.assert_allclose(summary.return_variance,summary.return_std**2,rtol=1e-12)
    distribution=pd.DataFrame(distribution_rows)  # 将各市场条件的分布行合成绘图表。
    # 读取审计通过的尾诊断表。
    tail=pd.read_csv(args.tail_table)
    # 核对尾诊断表覆盖所有市场条件。
    assert len(tail)==len(config.p_ratios)*len(config.activities)
    for _,row in tail.iterrows():
        # 按占据概率比和活动率查找同条件的系综统计行。
        check=summary[np.isclose(summary.p_ratio,row.p_ratio)&
                      np.isclose(summary.activity,row.activity)].iloc[0]
        # 验证尾诊断选择的非零交易数与原始收益一致。
        assert row.n_nonzero==check.nonzero_returns
    # 保存活动率统计。
    summary.to_csv(tables/"market_activity_statistics.csv",index=False)
    # 保存 PDF 与 CCDF 的输入数据。
    distribution.to_csv(tables/"market_pdf_ccdf.csv",index=False)
    # 保存原尾部审计值。
    tail.to_csv(tables/"market_tail_diagnostics.csv",index=False)
    # 由各活动率的单构型轨迹绘制时间序列面板。
    plot_archived_time_paths(args.time_tables,config.activities,figures/"market_time_series.png")
    # 由系综汇总绘制活动率统计面板。
    plot_market_activity(summary,figures/"market_activity_statistics.png")
    # 绘制含原始零收益的无条件分布。
    plot_return_distributions(distribution,figures/"market_pdf_ccdf.png")
    # 用已审计的结果绘制尾指数和区间。
    plot_tail_exponents(tail,figures/"market_tail_exponents.png")
    # 确定来源清单所覆盖的收益率、尾部和时间轨迹文件。
    inputs=[args.source,args.tail_table]+[
        args.time_tables/f"quenched_a_{activity:g}.csv" for activity in config.activities]
    # 记录输入摘要与系综、标准化、零收益及尾拟合口径。
    metadata={"input_sha256":{str(path):sha256(path) for path in inputs},
              "ensemble":"4 repeats x 100 independent configurations x 500 actions per condition",
              "time_paths":"archived separate quenched paths at p/pc=1; not the ordered ensemble",
              "normalization":"r=D/N, N=16384; z=(r-mean_all)/std_all(ddof=1)",
              "zero_rule":"signed PDF and unconditional CCDF include raw zero returns; tail fits exclude raw return==0 before |z|",
              "tail_fit":"audited 120 configuration-bootstrap intervals and 100-draw GOF are reused without recomputation"}
    # 将来源说明随统计表写出。
    (tables/"provenance.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n")
    # 报告核心图表的输出目录。
    print(f"Wrote 4 core figures and 3 tables to {args.output_root}")


# 直接运行脚本时执行最终汇编。
if __name__=="__main__":main()
