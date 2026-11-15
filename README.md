# Cont–Bouchaud 渗流市场模型

经济物理课程 Project 6 的代码仓库。实现 Cont–Bouchaud（CB）模型：投资者在网络上形成连通团簇，团簇共同买入、卖出或保持静默，订单汇总后产生市场收益率。代码还包含二维格点的渗流临界性分析、市场活性实验、网络拓扑比较，以及驱动与耗散的 SOC 扩展。

本仓库仅提交代码和实验配置。运行生成的数据、统计表和图保存在 `results/`，不随仓库提交。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `src/local_market/` | CB 模型、渗流、网络、统计和 SOC 的实现 |
| `configs/` | 模拟参数与随机种子 |
| `scripts/simulation/` | 模拟入口 |
| `scripts/analysis/` | 统计、拟合与诊断 |
| `scripts/figures/` | 从结果表生成图像 |
| `tests/` | 单元测试 |

## 环境配置

需要 Python **3.11 或更新版本**。项目使用的依赖版本固定在 `requirements.txt`；建议在独立虚拟环境中安装。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Windows PowerShell 将激活命令换成 `.venv\Scripts\Activate.ps1`。安装完成后可用以下命令检查环境：

```bash
python --version
pytest -q
```

所有命令均从仓库根目录运行。`pip install -e .` 使 `local_market` 包可被脚本导入；`--no-deps` 避免重复解析已按固定版本安装的依赖。

## 快速运行

先运行小规模验证，检查团簇识别和交易模拟是否能正常生成结果：

```bash
python scripts/simulation/run_small_validation.py --config configs/validation_small.yaml
```

输出写入 `results/tables/`。该命令用于环境检查；下列正式实验使用各自的配置和样本量，计算时间会明显更长。

## 正式实验入口

| 研究内容 | 命令 | 主要输出 |
| --- | --- | --- |
| 二维格点临界性 | `python scripts/simulation/scan_criticality.py --config configs/criticality.yaml` | `results/tables/`、`results/figures/` |
| 精细有限尺寸扫描 | `python scripts/simulation/scan_criticality_refined.py --config configs/criticality_refined.yaml --mode full` | `results/criticality_revision_20261113/full/` |
| 市场收益率与活性 | `python scripts/analysis/analyze_market_returns.py --config configs/market_analysis.yaml` | `results/tables/`、`results/figures/` |
| 四类网络比较 | `python scripts/simulation/compare_networks.py --config configs/network_analysis.yaml` | `results/tables/`、`results/figures/` |
| 驱动耗散 SOC 扩展 | `python scripts/simulation/run_soc_research.py --config configs/soc_research_v2.yaml --output results/soc_research_v2_full` | `results/soc_research_v2_full/` |

按表格顺序运行可满足主要依赖：精细扫描读取基础扫描的临界点估计，SOC 扩展读取精细扫描的静态临界性汇总。精细扫描还支持 `--mode smoke` 和 `--mode pilot`；首次检查可先运行 `smoke`。`run_soc_research.py` 要求 `--output` 指向**尚不存在**的目录，重复运行时请指定新的输出路径。

模拟完成后，可运行相应的后处理脚本，例如：

```bash
python scripts/analysis/analyze_criticality_refined.py --root results/criticality_revision_20261113/full
python scripts/analysis/analyze_soc_research.py --root results/soc_research_v2_full
```

绘图脚本位于 `scripts/figures/`，读取 `results/` 中的模拟或分析产物。完整实验涉及大量蒙特卡洛采样和 Bootstrap 重采样；建议先执行测试及小规模验证，再按所需章节运行正式任务。
