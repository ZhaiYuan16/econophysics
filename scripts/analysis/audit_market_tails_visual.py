#!/usr/bin/env python3
"""Reproduce visual and numerical diagnostics for the archived CB market tails."""
# 审计市场收益率的影响因素实验的尾拟合，比较原始零收益筛选后的指数、模型选择及可靠性。
# truncated_fit() 和 model_curves() 计算条件尾模型。corrected_uncertainty() 估计构型抽样误差与拟合优度。main() 生成逐条件诊断图和表。

# 本脚本读取 analyze_market_returns.py 归档的交易与原拟合。finalize_market_statistics.py 用输出的诊断表绘制最终尾指数图。
from __future__ import annotations  # 延后求值类型注解。

import json  # 把尾部诊断口径保存为 JSON。
from pathlib import Path  # 表示审计目录及文件路径。

import matplotlib  # 配置批量绘图的后端。
matplotlib.use("Agg")  # 在无窗口环境保存审计图像。
import matplotlib.pyplot as plt  # 生成右尾曲线、秩图、离散性图和阈值扫描图。
import numpy as np  # 用数组处理尾样本和支持，并提供 bootstrap 随机数。
import pandas as pd  # 读归档交易表、按条件分组并输出拟合诊断。
from scipy.integrate import quad  # 数值积分计算截断幂律的归一化常数及右尾积分。
from scipy.optimize import minimize  # 有界优化截断幂律的负对数似然。
from scipy.stats import halfnorm  # 提供半正态密度作为高指数条件的参考曲线。

from local_market.tail_fitting import select_pareto_tail  # src/local_market/tail_fitting.py 的 select_pareto_tail 扫描阈值并选择纯 Pareto 拟合。
from local_market.hierarchical_statistics import block_histogram, resampled_counts, weighted_pareto_fit  # src/local_market/hierarchical_statistics.py 的 block_histogram 压缩构型频数、resampled_counts 聚合抽中构型、weighted_pareto_fit 用加权样本重选阈值。
from local_market.statistics import nonzero_absolute_standardized_returns  # src/local_market/statistics.py 的 nonzero_absolute_standardized_returns 从原始非零交易提取 |z|。
from local_market.plotting import plot_tail_exponents  # src/local_market/plotting.py 的 plot_tail_exponents 按条件展示指数及诊断。

ROOT = Path("results/market_tail_audit_20261114")  # 固定的市场尾部审计归档目录。
FIG = ROOT / "figures"
TAB = ROOT / "tables"
QUANTILES = (.70, .80, .85, .90, .925, .95)  # Pareto 下界从这些经验分位数对应的幅度中选择。


def truncated_fit(tail: np.ndarray, xmin: float) -> tuple[float, float, float]:
    """Continuous x^-alpha exp[-lambda(x-xmin)] density, conditional x>=xmin."""
    # 功能：在市场收益率的影响因素实验中拟合从 xmin 开始的连续指数截断幂律。
    # 引用：corrected_uncertainty() 用拟合似然给 bootstrap 模型排名。main() 同时使用参数画曲线。
    # 输入
    # tail：非空的一维尾样本数组，元素均应 ≥ xmin。
    # xmin：正的连续尾部阈值，与 tail 使用相同幅度尺度。
    # 输出依次为
    # alpha：密度的幂律指数。
    # rate：每单位原始幅度的指数截断率。
    # log_likelihood：优化返回参数处的条件对数似然。
    y = tail / xmin  # 将幅度缩放为 y=x/xmin。
    logy_sum = np.log(y).sum()  # 累计幂律项所需的 log(y)。
    excess_sum = (y - 1).sum()  # 累计截断项所需的缩放超额 y−1。
    n = len(y)

    def objective(theta: np.ndarray) -> float:
        # 功能：计算截断幂律在给定参数下的负对数似然。
        # 引用：truncated_fit() 将本回调传入 scipy.optimize.minimize()，由优化器反复调用。
        # 输入
        # theta：长度为 2，依次为幂指数与按 xmin 缩放后的截断率。闭包读取 xmin、logy_sum、excess_sum、n。
        # 输出：缩放后积分换回原始幅度的条件负对数似然。
        alpha, rate = theta
        # 在 u=log(y) 坐标积分密度形状，其中 +u 来自变量替换的雅可比。
        norm = quad(lambda u: np.exp(-alpha*u-rate*(np.exp(u)-1)+u),
                    0, min(40., np.log(max(1e3, 100/rate))), epsrel=2e-6)[0]  # 积分从尾部阈值对应的零开始，随截断率调节上界并限在 40。
        return alpha*logy_sum + rate*excess_sum + n*np.log(norm*xmin)  # 两项样本贡献加 n 倍的归一化对数，乘 xmin 恢复原始尺度。

    # 用有界 L-BFGS-B 搜索幂指数与正的缩放截断率。
    fit = minimize(objective, [min(2., 1+len(y)/max(logy_sum, 1)), .05],
                   bounds=[(0, 250), (1e-8, 250)], method="L-BFGS-B")  # 限制两个参数的搜索范围。
    return float(fit.x[0]), float(fit.x[1]/xmin), float(-fit.fun)  # 将截断率除以 xmin 换回原尺度，并把目标函数取负。


def model_curves(x: np.ndarray, xmin: float, alpha: float,
                 truncated_alpha: float, rate: float, exp_rate: float):
    # 功能：在市场收益率的影响因素实验中生成三种以 xmin 为条件的连续尾部 CCDF。
    # 引用：main() 将各曲线乘观测尾部比例后与经验 CCDF 对照。
    # 输入
    # x：一维绘图阈值数组，各值应 ≥ xmin。
    # xmin：正的共同尾部阈值。
    # alpha：纯 Pareto 的 PDF 指数，应大于 1。
    # truncated_alpha：截断幂律的 PDF 幂指数。
    # rate：截断幂律在原始幅度尺度上的截断率。
    # exp_rate：指数模型在原始幅度尺度上的衰减率。
    # 输出依次为三个与 x 同形状的一维数组
    # pareto：纯 Pareto 的右尾概率。
    # truncated：指数截断幂律的右尾概率。
    # exponential：条件指数模型的右尾概率。
    pareto = (x/xmin)**(1-alpha)  # 从纯 Pareto PDF 的指数换算解析 CCDF。
    exponential = np.exp(-exp_rate*(x-xmin))  # 用相对 xmin 的超额幅度计算指数尾概率。
    # Integrate on log x, retaining the actual fitted continuous cutoff model.
    def integral(lower):
        # 功能：计算截断幂律从指定幅度起的未归一化右尾积分。
        # 引用：model_curves() 在共同尾部阈值和各绘图阈值上调用本函数。
        # 输入
        # lower：积分下界，按原始幅度给出。闭包读取 xmin、truncated_alpha、rate。
        # 输出：同一对数变换及积分上界下的浮点积分值。
        return quad(lambda u: np.exp((1-truncated_alpha)*u-rate*xmin*(np.exp(u)-1)),  # 在对数尺度积分截断模型形状，并包含雅可比。
                    np.log(lower/xmin), min(40., np.log(max(1e3,100/max(rate*xmin,1e-9)))),  # 将幅度下界映为 log(lower/xmin)，按截断率截取积分上界。
                    epsrel=2e-5)[0]
    norm = integral(xmin)  # 以 xmin 处积分作为条件分布的分母。
    truncated = np.array([integral(v)/norm for v in x])  # 对每个阈值独立求积分并归一化。
    return pareto, truncated, exponential


def save(fig, name):
    # 功能：排版并保存市场尾部审计图。
    # 引用：main() 将四组尾部面板以及可能出现的高指数对照图传入。
    # 输入
    # fig：待保存的 Matplotlib Figure。
    # name：相对于 FIG 的目标文件名。
    # 输出：返回 None，写出图像后关闭该画布。
    fig.tight_layout()  # 调整各面板和轴文字的间距。
    fig.savefig(FIG/name, dpi=170)  # 以 170 dpi 写入审计图目录。
    plt.close(fig)  # 释放画布资源。


def corrected_uncertainty(group: pd.DataFrame, best, seed: int):
    """Configuration bootstrap, excluding exact zero raw returns before each fit."""
    # 功能：为市场收益率的影响因素实验估计剔除原始零收益后的尾指数区间、拟合优度与模型胜率。
    # 引用：main() 将返回值写入诊断表并用于判定纯幂律拟合的可靠性。
    # 输入
    # group：某个市场条件的逐笔交易，含 return、replicate 和 configuration。实现按 4 个重复、每重复 100 个构型编码。
    # best：原数据选中的 Pareto 拟合，读取 xmin、alpha_pdf 和 ks_distance。
    # seed：可复现构型重采样与零假设模拟的随机种子。
    # 输出依次为
    # ci：长度为 2 的数组，保存 PDF 指数的 2.5% 与 97.5% 分位。
    # gof：按模拟 KS 超过或等于观测距离的频次作加一校正的 p 值。
    # model_freq：三种模型在 120 轮中 AIC 最低的频率字典。
    ids = group.replicate.to_numpy(dtype=int)*100+group.configuration.to_numpy(dtype=int)  # 将重复号和构型号编码为唯一的块 ID。
    support, blocks = block_histogram(group["return"].to_numpy(dtype=float), ids)  # 将每个构型的原始收益压缩到共同支持上的计数。
    rng = np.random.default_rng(seed)  # 初始化可重复的抽样流。
    estimates = []
    model_wins = {"pure power law":0,"truncated power law":0,"exponential":0}
    for _ in range(120):
        picked = np.concatenate([r*100+rng.integers(0,100,100) for r in range(4)])  # 在四个重复内分别有放回抽 100 个构型。
        weights = resampled_counts(blocks,picked)  # 将抽中构型的交易频数累加。
        mean = np.dot(support,weights)/weights.sum()  # 用抽中交易的频数重算全样本收益均值。
        std = np.sqrt(np.dot((support-mean)**2,weights)/(weights.sum()-1))  # 用所有抽中交易和 n−1 分母重算标准差。
        absolute = np.abs((support[support!=0]-mean)/std)  # 标准化后，只为原始收益非零的支持点计算绝对幅度。
        fit = weighted_pareto_fit(absolute,weights[support!=0],QUANTILES,500)  # 按非零支持的频数加权重选尾阈值，尾部至少 500 笔。
        estimates.append(fit["alpha_pdf"])  # 收集本轮纯幂律的 PDF 指数。
        # 将阈值以上支持按样本频数展开供备选模型拟合。
        tail = np.repeat(absolute[absolute>=fit["xmin"]],
                         weights[support!=0][absolute>=fit["xmin"]])
        n,cut,alpha = len(tail),fit["xmin"],fit["alpha_pdf"]  # 记录本轮的尾部笔数、共同阈值和纯幂律指数。
        exp_rate = 1/np.mean(tail-cut)  # 指数模型的率是平均超额的倒数。
        _,_,ll_trunc = truncated_fit(tail,cut)  # 对共同阈值以上的尾样本拟合截断幂律。
        ll_pure = n*np.log((alpha-1)/cut)-alpha*np.log(tail/cut).sum()  # 计算纯 Pareto 在共同尾部支持上的条件对数似然。
        ll_exp = n*np.log(exp_rate)-exp_rate*(tail-cut).sum()  # 计算指数模型在同一尾样本上的条件对数似然。
        # 纯幂律与指数各计一个参数，截断幂律计两个参数。
        aics={"pure power law":2-2*ll_pure,
              "truncated power law":4-2*ll_trunc,"exponential":2-2*ll_exp}
        model_wins[min(aics,key=aics.get)]+=1  # 给 AIC 最小的模型累计一次胜出。
    ci = np.quantile(estimates,[.025,.975])  # 对 120 个指数估计取双侧 95% 百分位区间。
    # Semiparametric null: resample configuration tail counts, replace tail
    # magnitudes under the fitted Pareto, and reselect xmin in each simulation.
    original = np.asarray(blocks.sum(axis=0)).ravel()  # 合并原归档的构型频数。
    mean = np.dot(support,original)/original.sum()  # 用原归档的全部交易求中心。
    std = np.sqrt(np.dot((support-mean)**2,original)/(original.sum()-1))  # 以原归档样本方差标准化。
    absz = np.abs((support-mean)/std)  # 将原收益支持转成标准化绝对幅度。
    eligible = support!=0  # 零收益排除使用原始支持。
    body = eligible & (absz<best.xmin)  # 体部为原始非零且小于所选阈值的点。
    tail_mask = eligible & (absz>=best.xmin)  # 尾部为原始非零且达到阈值的点。
    exceed = 0
    for _ in range(100):
        picked = np.concatenate([r*100+rng.integers(0,100,100) for r in range(4)])  # 在每个重复内重抽构型，以保留构型的体尾计数变化。
        counts = resampled_counts(blocks,picked)  # 合并抽中构型的支持频数。
        ntail = int(counts[tail_mask].sum())  # 求本轮须替换的尾交易数。
        synthetic_tail = best.xmin*(1-rng.random(ntail))**(-1/(best.alpha_pdf-1))  # 用 Pareto 逆分布生成同样数量的尾幅度。
        # 组合重采样体部与模拟尾部，并再次选择阈值。
        synthetic = weighted_pareto_fit(np.concatenate([absz[body],synthetic_tail]),
                      np.concatenate([counts[body],np.ones(ntail,dtype=np.int64)]),QUANTILES,500)  # 体部沿用频数权重，逐个模拟的尾观测权重为 1。
        exceed += synthetic["ks_distance"]>=best.ks_distance  # 若重选阈值后的 KS 至少与观测值一样大，累加超越次数。
    # 返回区间、加一校正后的检验概率和各模型胜率。
    return ci,(exceed+1)/101,{key:value/120 for key,value in model_wins.items()}


def main():
    # 功能：重做市场收益率的影响因素实验的尾拟合及可视诊断。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。读取 results/tables 下的逐笔收益归档和原尾拟合表。
    # 输出：返回 None，保存尾诊断、阈值扫描、模型对照图和方法 JSON。
    # 建立图像输出目录。
    FIG.mkdir(parents=True, exist_ok=True)
    # 建立表格和方法输出目录。
    TAB.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv("results/tables/market_returns.csv.gz")  # 读入用于重拟合的逐笔收益。
    old = pd.read_csv("results/tables/market_tail_fits.csv")  # 读取口径修正前的模型、阈值及区间。
    scan_rows = []
    # 为 12 个市场条件创建每类诊断的面板。
    panels = {kind: plt.subplots(3, 4, figsize=(19, 13)) for kind in
              ("ccdf", "zipf", "discreteness", "xmin_scan")}
    rows = []
    anomaly = []
    # 将占据概率比和活动率的每个组合独立处理。
    for index, ((ratio, activity), group) in enumerate(data.groupby(["p_ratio", "activity"], sort=True)):
        oldrow = old[(np.isclose(old.p_ratio,ratio)) & (np.isclose(old.activity,activity))].iloc[0]  # 按市场条件匹配原拟合行。
        raw = group["return"].to_numpy(dtype=float)  # 提取原始收益用于剔除零交易。
        x = nonzero_absolute_standardized_returns(raw, group.z.to_numpy(dtype=float))  # 对原始非零交易取归档标准化收益的绝对值。
        best,candidates = select_pareto_tail(x,QUANTILES,500)  # 用候选分位数和最少 500 笔尾样本选择纯 Pareto 拟合。
        scan_rows.extend({"p_ratio":ratio,"activity":activity,**c} for c in candidates)  # 为各候选阈值记录市场条件标签。
        ci,gof,model_freq = corrected_uncertainty(group,best,20261114+index)  # 按当前条件估计指数区间、拟合优度与三种模型胜率。
        xmin = float(best.xmin)
        tail = np.sort(x[x>=xmin])  # 排序共同阈值以上的尾样本。
        unique, count = np.unique(tail, return_counts=True)  # 分别取不同幅度和完全相同数值的频数。
        alpha = float(best.alpha_pdf)
        exrate = 1 / np.mean(tail-xmin)  # 以平均尾部超额的倒数估计指数率。
        ta, rate, ll_trunc = truncated_fit(tail, xmin)  # 在相同尾样本上估计截断模型的两个参数。
        ll_pure = len(tail)*np.log((alpha-1)/xmin)-alpha*np.log(tail/xmin).sum()  # 计算纯幂律尾条件对数似然。
        ll_exp = len(tail)*np.log(exrate)-exrate*(tail-xmin).sum()  # 计算指数尾条件对数似然。
        # 对三种模型按参数个数计算 AIC。
        aics = {"pure power law":2-2*ll_pure, "truncated power law":4-2*ll_trunc,
                "exponential":2-2*ll_exp}
        preferred = min(aics, key=aics.get)  # 选择原尾样本上 AIC 最低的模型。
        dynamic = tail[-1]/xmin  # 用尾最大幅度与阈值之比表示动态范围。
        # A failure on any criterion means that an exponent is descriptive only.
        # 检查拟合优度、样本数、动态范围、不同幅度数及纯幂律的相对 AIC。
        reliable = bool(gof>=.1 and
                        len(tail)>=1000 and dynamic>=10 and len(unique)>=30 and
                        aics["pure power law"]<=min(aics.values())+2)
        # 保存原口径和修正口径的点估计、区间和模型诊断。
        rows.append({"p_ratio":ratio,"activity":activity,"x_min":xmin,"x_max":tail[-1],
                     "n_nonzero":len(x),"n_tail":len(tail),"unique_tail":len(unique),
                     "x_max/x_min":dynamic,"alpha":alpha,
                     "old_alpha_CI":f"[{oldrow.alpha_ci_low:.3g}, {oldrow.alpha_ci_high:.3g}]",
                     "alpha_CI":f"[{ci[0]:.3g}, {ci[1]:.3g}]",
                     "KS":best.ks_distance,"GOF_p":gof,
                     "old_alpha":oldrow.alpha_pdf,"old_xmin":oldrow.xmin,
                     "preferred_model":preferred,"power_law_reliable":reliable,
                     "aic_pure":aics["pure power law"],"aic_truncated":aics["truncated power law"],
                     "aic_exponential":aics["exponential"],"truncated_alpha":ta,
                     "truncated_lambda":rate,"exponential_lambda":exrate,
                     "max_value_frequency":int(count[-1]),"largest_frequency_fraction":count.max()/len(tail)})  # 同时记录最大幅度频数和最常见幅度所占比例。
        rows[-1].update({"bootstrap_preferred_"+k.replace(" ","_"):v for k,v in model_freq.items()})  # 将模型 bootstrap 胜率添加到本条件的诊断行。
        title = f"p/pc={ratio:g}, a={activity:g}"
        ix = np.unravel_index(index,(3,4))  # 将条件编号映射到面板行列。
        ax = panels["ccdf"][1][ix]
        survival = np.cumsum(count[::-1])[::-1]/len(x)  # 从高幅度累计频数，用全部非零交易作分母。
        ax.step(unique, survival, where="post", color="black",lw=1,label="empirical")  # 阶梯显示经验尾概率。
        grid = np.geomspace(xmin,tail[-1],120)  # 在尾部上下界之间建立几何绘图网格。
        curves = model_curves(grid,xmin,alpha,ta,rate,exrate)  # 计算三种以 xmin 为条件的模型右尾曲线。
        for curve,label,color in zip(curves,("Pareto","truncated","exponential"),("tab:blue","tab:orange","tab:green")):
            ax.plot(grid,len(tail)/len(x)*curve,color=color,lw=1.3,label=label)  # 乘以尾部交易占比，换成经验曲线的概率口径。
        ax.axvline(xmin,color="gray",ls=":",label="xmin")  # 标记共同的尾部阈值。
        ax.set(xscale="log",yscale="log",title=title,xlabel="|z|",ylabel="P(|z| >= x)")  # 按双对数轴展示绝对标准化收益和右尾概率。
        # 在面板标注指数、阈值、尾部笔数、拟合优度和动态范围。
        ax.text(.03,.04,f"alpha={alpha:.2f}, xmin={xmin:.3g}\nn={len(tail)}, GOF p={gof:.3g}\nrange={dynamic:.2f}x",transform=ax.transAxes,fontsize=8,
                bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
        ax = panels["zipf"][1][ix]
        rank = np.arange(1,len(tail)+1)  # 生成从 1 开始的尾部样本秩。
        ax.plot(rank,tail[::-1],".",ms=1,color="black",label="raw ordered tail")  # 按降序绘制含重复值的尾样本。
        ref = xmin*(len(tail)/rank)**(1/max(alpha-1,1e-8))  # 用 Pareto CCDF 的逆函数求各秩的理论幅度。
        ax.plot(rank,ref,color="tab:blue",lw=1,label="Pareto reference")  # 将理论秩曲线叠在经验点上。
        ax.set(xscale="log",yscale="log",title=title,xlabel="rank k",ylabel="x(k)")  # 设置双对数的秩与幅度坐标。
        ax.text(.03,.04,f"n={len(tail)}, unique={len(unique)}",transform=ax.transAxes,fontsize=8)  # 标注总体尾样本量及不同幅度数。
        ax = panels["discreteness"][1][ix]
        ax.scatter(unique,count,s=7,color="tab:purple")  # 为每种不同幅度画一个实际计数点。
        ax.set(xscale="log",yscale="log",title=title,xlabel="distinct |z|",ylabel="frequency")  # 在双对数轴查看幅度频数。
        # 标注不同幅度数与总尾样本数。
        ax.text(.03,.04,f"unique_tail={len(unique)}\nn_tail={len(tail)}",transform=ax.transAxes,fontsize=8,
                bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
        ax = panels["xmin_scan"][1][ix]
        c = pd.DataFrame(candidates).sort_values("xmin")  # 按候选阈值排序以便绘制敏感性曲线。
        ax.plot(c.xmin,c.ks_distance,"o-",color="tab:blue",label="KS")  # 左轴画阈值对应的 KS 距离。
        ax2=ax.twinx(); ax2.plot(c.xmin,c.alpha_pdf,"s--",color="tab:orange",ms=3,label="alpha")  # 新建右轴并画 PDF 指数随阈值变化。
        ax3=ax.twinx(); ax3.spines["right"].set_position(("axes",1.13))  # 为尾样本数建立另一条右轴并向外移开。
        ax3.plot(c.xmin,c.tail_count,"^:",color="tab:green",ms=3,label="n tail")  # 在外侧右轴画阈值对应的尾样本数。
        ax.axvline(xmin,color="gray",ls=":")  # 标记选中的阈值。
        ax.set(title=title,xlabel="candidate xmin",ylabel="KS",xscale="log")  # 标明阈值横轴为对数轴，主纵轴为 KS。
        ax2.set_ylabel("alpha",color="tab:orange",fontsize=8)  # 给指数右轴加变量名。
        ax3.set_ylabel("n tail",color="tab:green",fontsize=8)  # 给样本数右轴加变量名。
        if alpha>10: anomaly.append((title,x,tail,xmin,alpha,ta,rate,exrate))  # 收集 PDF 指数大于 10 的条件供密度对照。
        print(title,"alpha",round(alpha,2),"range",round(dynamic,2),"unique",len(unique),"model",preferred,flush=True)  # 在终端报告本条件拟合诊断。
    for kind,(fig,axes) in panels.items():
        if kind=="ccdf": axes[0,0].legend(fontsize=7)  # CCDF 首幅图标明经验和模型曲线。
        if kind=="zipf": axes[0,0].legend(fontsize=7)  # 秩图首幅图标明样本点和理论参考。
        # 为每类诊断设置对应的图标题。
        fig.suptitle({"ccdf":"Tail CCDF and conditional fits","zipf":"Rank-size: raw repeated observations remain visible",
                      "discreteness":"Exact tail value frequencies","xmin_scan":"xmin selection sensitivity: KS, alpha, n tail"}[kind])
        save(fig,f"market_tail_{kind}.png")  # 保存并关闭当前诊断面板。
    if anomaly:
        fig,axs=plt.subplots(len(anomaly),2,figsize=(13,4*len(anomaly)),squeeze=False)  # 每个高指数条件一行，分别作两种纵轴尺度的对照。
        for row,(title,x,tail,xmin,alpha,ta,rate,exrate) in enumerate(anomaly):
            bins=np.geomspace(max(x.min(),xmin/10),x.max(),70)  # 采用几何分箱以呈现高幅度尾部的直方密度。
            for col in range(2):
                ax=axs[row,col]
                ax.hist(x,bins=bins,density=True,histtype="step",color="black",label="observed")  # 用全部非零绝对幅度画经验密度。
                grid=np.geomspace(xmin,x.max(),150)  # 在共同尾部支持上建立模型密度绘图点。
                norm=quad(lambda u: np.exp((1-ta)*u-rate*xmin*(np.exp(u)-1)),0,40,epsrel=1e-5)[0]*xmin  # 在对数幅度上积分，乘 xmin 换回原尺度的截断归一化常数。
                curves=((alpha-1)/xmin*(grid/xmin)**(-alpha),  # 在共同阈值上计算归一化的纯 Pareto 密度。
                        (grid/xmin)**(-ta)*np.exp(-rate*(grid-xmin))/norm,  # 用数值积分常数归一化截断幂律。
                        exrate*np.exp(-exrate*(grid-xmin)),  # 计算以 xmin 起步的指数密度。
                        halfnorm.pdf(grid,scale=1))  # 用单位尺度半正态作参照。
                for y,label in zip(curves,("Pareto tail","truncated tail","exponential tail","half normal reference")):
                    ax.plot(grid,y*len(tail)/len(x) if label!="half normal reference" else y,lw=1,label=label)  # 三种条件尾密度按尾部比例缩放，半正态沿用其原密度。
                ax.axvline(xmin,color="gray",ls=":")  # 标记共同尾阈值。
                ax.set(xscale="log",yscale="log" if col else "linear",title=f"{title}: {'log-log' if col else 'log-linear'}",xlabel="|z|",ylabel="density")  # 两列都按对数幅度绘制，纵轴分别取线性和对数。
                if row==0: ax.legend(fontsize=7)  # 用首行图例识别各密度。
        save(fig,"market_tail_high_alpha_histograms.png")  # 保存并关闭高指数条件对照图。
    summary=pd.DataFrame(rows)  # 将逐条件诊断整理成表。
    summary.to_csv(TAB/"market_tail_diagnostics.csv",index=False)  # 保存供核心图汇编使用的尾诊断。
    pd.DataFrame(scan_rows).to_csv(TAB/"market_tail_xmin_scan.csv",index=False)  # 保存所有扫描阈值的诊断结果。
    plot_tail_exponents(summary,FIG/"market_tail_exponents.png")  # 由审计汇总绘制尾指数和区间。
    # 保存样本口径、重采样设置和模型可靠性规则。
    (TAB/"method.json").write_text(json.dumps({"source":"archived 200000 returns per combination",
      "continuous_approximation":"|z| is discrete because D is integer; continuous model curves are diagnostic approximations",
      "gof":"selection-aware, within-replicate configuration-block semiparametric null, 100 draws; zeros removed before xmin selection",
      "confidence_interval":"120 bootstrap draws: within each of 4 replicates resample 100 configuration blocks; restandardize and reselect xmin; zeros excluded; refit all three conditional tail models and record AIC winner each draw",
      "reliability_rule":"GOF p>=0.1, n_tail>=1000, range>=10, unique_tail>=30, pure Pareto AIC within 2 of best",
      "preferred_model":"minimum continuous conditional AIC at original selected xmin; interpret as descriptive when discrete peaks appear"},indent=2)+"\n")

# 直接运行脚本时执行尾部审计。
if __name__=="__main__": main()
