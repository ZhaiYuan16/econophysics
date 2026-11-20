# 经济物理学 Project 6 · Cont–Bouchaud 市场模型

本仓库提供《经济物理学》Project 6 课程报告的 Python 源码与实验配置。项目实现 Cont–Bouchaud（CB）渗流市场模型，研究投资者团簇如何影响收益率分布，比较规则、随机、小世界和无标度网络，并检验引入驱动和损耗后的模型是否表现出自组织临界性（SOC）。报告 PDF 随课程材料单独提交。

## 安装与快速验证

在仓库根目录运行以下命令。请使用 Python 3.11 或更新版本。完整流程建议使用 Linux 或 WSL。精细扫描依赖 Python 的 Unix 模块 `resource`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

[requirements.txt](requirements.txt) 锁定直接依赖版本，[pyproject.toml](pyproject.toml) 声明 Python 版本和安装配置。

先运行小规模验证。它在开放边界的 32×32 方格上使用三个占据概率，检查团簇统计和收益率模拟。

```bash
python scripts/simulation/run_small_validation.py --config configs/validation_small.yaml --output-dir results/quickstart
```

运行后，终端会显示三行汇总。`results/quickstart/` 下会生成团簇和收益率 CSV、汇总表 `validation_small_summary.csv` 以及记录本次配置的 JSON。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [src/local_market/](src/local_market/) | CB 交易、渗流团簇、网络生成、统计分析与 SOC 模型 |
| [configs/](configs/) | 实验参数与随机种子 |
| [scripts/simulation/](scripts/simulation/) | 模拟与参数扫描 |
| [scripts/analysis/](scripts/analysis/) | 统计分析、稳健性检查与尾部拟合 |
| [scripts/figures/](scripts/figures/) | 报告图表的绘制脚本 |

## 模型约定

- `p` 表示节点的占据概率，`a` 表示团簇在一个时间步交易的概率。每个团簇以 `a/2` 的概率买入，以 `a/2` 的概率卖出，以 `1−a` 的概率保持不活跃。
- 对数收益率等于按团簇规模加权的超额需求除以市场深度。二维方格使用四邻域，边界条件由实验配置指定。
- 随机实验使用配置中的种子。尾部分析的零收益处理随统计口径而异，见下文。

## 复现报告图表

按各节的命令顺序运行。原始模拟数据和派生图表写入 `results/`，由脚本生成。已有对应 CSV 时，可以从读取该文件的分析步骤开始。首次完整运行中，精细渗流扫描、相关长度计算和 SOC 长路径较耗时。

精细扫描、临界性验证和 SOC 模拟需要新的输出目录。复跑正式配置前，请保留已有结果，并为对应运行准备干净的输出路径。下游分析和绘图默认读取下列命令使用的路径。

### 渗流（图 1–5、表 1–2）

```bash
python scripts/simulation/scan_criticality.py --config configs/criticality.yaml
python scripts/simulation/scan_criticality_refined.py --config configs/criticality_refined.yaml --mode full
python scripts/analysis/analyze_criticality_refined.py
python scripts/analysis/criticality_validation_analysis.py
python scripts/analysis/criticality_validation_gridaware.py
python scripts/figures/plot_correlation_length_preview.py
python scripts/figures/plot_criticality_spanning.py
python scripts/figures/plot_cluster_ols.py
```

基准扫描产生 `results/tables/criticality_pseudocritical.csv`，供精细扫描读取。精细扫描产生 `results/criticality_revision_20261113/full/tables/` 中的汇总、配置观测及团簇直方图。后处理产生 `results/figures/criticality_*.png`、`results/tables/cluster_ols_intervals.csv` 及验证表。

### 市场与网络（图 6–12、表 3）

```bash
python scripts/analysis/analyze_market_returns.py --config configs/market_analysis.yaml
python scripts/analysis/audit_market_time_series.py
python scripts/simulation/compare_networks.py --config configs/network_analysis.yaml
python scripts/analysis/audit_downstream_statistics.py
python scripts/analysis/audit_market_tails_visual.py
python scripts/analysis/finalize_market_statistics.py
python scripts/analysis/loglog_ols.py
python scripts/figures/plot_market_ols.py
python scripts/figures/plot_network_degree.py
python scripts/figures/plot_network_degree_outcomes.py
python scripts/figures/plot_network_ols_report.py
```

市场和网络模拟分别产生 `results/tables/market_returns.csv.gz` 和 `results/tables/network_market_returns.csv.gz`。尾部审计产生市场尾诊断与网络崩盘频率的分层区间；市场汇总脚本产生图 6–8 和表 3。OLS 入口只读取两张收益表，产生 `results/loglog_ols_comparison/ols_summary_v2.csv` 和 `candidate_intervals_v2.csv`，供图 9、11、12 使用。

市场 OLS 保留中心化后非零的原始零收益；网络 OLS 在构造 `|z|` 的 CCDF 前排除原始零收益。市场 Pareto 尾模型也排除原始零收益。两种 OLS 的零值规则分别遵循报告第 6.4、7.2 节。

### 自组织临界性（图 13–14、表 4–5）

```bash
python scripts/simulation/run_soc_research.py --config configs/soc_research_v1.yaml --output results/soc_research_v1_full
python scripts/analysis/analyze_soc_research.py --root results/soc_research_v1_full
python scripts/simulation/run_soc_research.py --config configs/soc_research_v2.yaml --output results/soc_research_v2_full
python scripts/analysis/analyze_soc_research.py --root results/soc_research_v2_full
python scripts/analysis/analyze_soc_scaling.py --root results/soc_research_v2_full
python scripts/figures/plot_soc_mechanisms.py
```

SOC 模拟读取精细扫描的 `results/criticality_revision_20261113/full/tables/summary.csv` 计算静态临界窗口。v1 保存参数敏感性及对照路径；报告主图使用 v2 长路径。正式事件规模分析由 `src/local_market/soc_tail.py` 比较纯幂律、截断幂律和指数模型，输出位于 `results/soc_research_v2_full/`。无事件时间步参与事件率统计，尾模型只使用实际发生的正规模事件。

已有模拟 CSV 时，可从相应章节的分析与绘图命令开始。报告中的具体数值可能随模拟样本及网络零值修正而变化。
