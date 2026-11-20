# Project 6：Cont–Bouchaud 市场模型

本目录是《报告提交版.pdf》的代码。请从本目录执行命令，使用 Python 3.11 或更新版本。安装依赖：

```bash
python -m pip install -e .
```

结果写入 `results/`，该目录默认不随代码提交。已有对应 CSV 时，可跳过生成它的模拟。首次完整模拟较耗时，尤其是精细渗流扫描、相关长度计算和 SOC 长路径。`criticality_validation_analysis.py` 和 SOC 模拟要求各自的输出目录尚不存在。

## 渗流：图 1–5、表 1–2

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

## 市场与网络：图 6–12、表 3

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

## SOC：图 13–14、表 4–5及尾模型比较

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
