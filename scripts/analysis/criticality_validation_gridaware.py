"""Propagate discrete peak-cell uncertainty into WLS extrapolation intervals."""
# 为渗流临界性质实验的伪临界外推补充扫描网格引起的峰位不确定性。
# jitter_peak_cells() 扰动各格长的离散峰位，main() 拟合外推直线并比较加入网格扰动前后的区间。

# 本脚本读取精细扫描保存的峰位和网格，以及 criticality_validation_analysis.py 生成的 bootstrap 峰位和外推区间。

import hashlib  # 计算脚本及验证产物的 SHA-256。
import json  # 将分析来源与哈希清单写成 JSON。
from pathlib import Path  # 拼接扫描和验证的结果路径。

import numpy as np  # 操作扫描网格、执行重采样并求解加权最小二乘。
import pandas as pd  # 读取已保存的峰位和汇总表并写出区间。

SOURCE = Path("results/criticality_revision_20261113/full/tables")  # 精细扫描的输入表目录。
OUT = Path("results/criticality_validation_20261113")  # 临界性验证的现有结果及新增报告目录。
PC = 0.592746  # 方格点渗流的理论临界占据概率。
LENGTHS = (32, 64, 128, 256, 512)  # 已保存扫描的五个格长。


def jitter_peak_cells(draws: np.ndarray, grid: np.ndarray, rng) -> np.ndarray:
    """Uniformly locate a grid maximum within its neighbouring midpoint cell."""
    # 功能：为渗流临界性质实验的 bootstrap 峰位加入有限扫描网格的定位不确定性。
    # 引用：main() 对每个格长调用本函数，将扰动峰位送入加权最小二乘外推。
    # 输入
    # draws：一维 bootstrap 峰位数组，各元素是在扫描网格上取到峰值的占据概率 p。
    # grid：与 draws 同一格长的升序扫描点，至少含两个点。
    # rng：提供逐元素均匀抽样的 NumPy 随机数生成器，抽样会推进其状态。
    # 输出：与 draws 同形状的峰位数组，每个元素在对应网格单元内独立均匀抽取。draws 为空时返回空数组。
    # 用升序网格的插入位置找每个离散峰位对应的格点。
    indices = np.searchsorted(grid, draws)
    # 核对插入位置对应的扫描点。若位置越过网格上界，索引时会触发 IndexError。位置有效但不近似相等时抛出 ValueError。
    if not np.all(np.isclose(grid[indices], draws)):
        raise ValueError("bootstrap peak outside the stored scan grid")
    # 下界取左邻点与峰位的中点，最左扫描点向外延伸半个首段间距。
    low = np.where(indices > 0, (grid[np.maximum(indices - 1, 0)] + draws) / 2,
                   draws - (grid[1] - grid[0]) / 2)
    # 上界取峰位与右邻点的中点，最右扫描点向外延伸半个末段间距。
    high = np.where(indices < len(grid) - 1,
                    (grid[np.minimum(indices + 1, len(grid) - 1)] + draws) / 2,
                    draws + (grid[-1] - grid[-2]) / 2)
    # 在各自的上下界之间抽取连续峰位，不确定性由相邻扫描点间距决定。
    return rng.uniform(low, high)


def main():
    # 功能：在渗流临界性质实验中，将离散扫描分辨率传递到伪临界点的加权最小二乘外推区间。
    # 引用：模块命令行入口调用 main()。
    # 输入：无函数参数。从 SOURCE 读取离散峰位和网格，从 OUT 读取构型 bootstrap 峰位和原外推结果。
    # 输出：返回 None，写出网格修正区间表、分析元数据及哈希清单，并打印区间表。区间表已存在时抛出 FileExistsError。
    destination = OUT / "pseudocritical_extrapolation_gridaware.csv"
    if destination.exists():
        raise FileExistsError(destination)
    # 读取各格长和易感性剔除口径的观测峰位。
    peaks = pd.read_csv(SOURCE / "pseudocritical.csv")
    # 读取每个格长实际扫描的占据概率。
    summary = pd.read_csv(SOURCE / "summary.csv")
    # 读取构型重采样得到的离散峰位，每列对应一种格长和剔除口径。
    boots = pd.read_csv(OUT / "bootstrap_peak_replicates.csv")
    # 读取只考虑构型抽样误差的原区间，作并列比较。
    original = pd.read_csv(OUT / "pseudocritical_extrapolation.csv")
    # 固定峰位重采样和连续扰动的随机序列。
    rng = np.random.default_rng(20261113 + 811)
    rows = []
    for rule in ("largest_excluded", "all_spanning_excluded"):
        grid_values = []
        for length in LENGTHS:
            # 按格长提取升序占据概率网格。
            grid = np.sort(summary[summary.length == length].p.to_numpy())
            # 取该格长和剔除口径的全部 bootstrap 峰位。
            draws = boots[f"L{length}_{rule}"].to_numpy()
            grid_values.append((grid, draws))  # 将扫描网格和离散重复峰位成对保存。
        # 分别用全部五个格长和最大的三个格长外推。
        for subset, indices in (("all_5", np.arange(5)), ("large_3", np.arange(2, 5))):
            lengths = np.array(LENGTHS)[indices]  # 取当前外推使用的格长。
            # 用二维渗流的 1/ν=3/4 构造横轴，横轴为零时对应无限格长。
            x = lengths.astype(float)**(-.75)
            # 原扫描的离散峰位用于计算点估计。
            observed = np.array([peaks[(peaks.length == length) &
                (peaks.exclusion == rule)].p_grid_peak.iloc[0] for length in lengths])
            # 用各格长 bootstrap 峰位的样本标准差描述构型误差。
            spread = np.array([grid_values[i][1].std(ddof=1) for i in indices])
            # 用最小网格间距除以 √12 表示均匀单元的分辨率误差。
            grid_floor = np.array([np.diff(grid_values[i][0]).min() / np.sqrt(12)
                for i in indices])
            # 按两种误差的合计方差倒数加权。
            weight = 1 / (spread**2 + grid_floor**2)
            # 构造截距列和有限尺寸修正列。
            X = np.column_stack([np.ones(len(x)), x])
            # 求解加权正规方程得到线性算子，用于批量从各格长峰位求截距和斜率。
            operator = np.linalg.solve(X.T @ (weight[:, None] * X), X.T * weight)
            # 有放回抽 5000 个重复行索引，各格长使用同一组索引。
            sampled_rows = rng.integers(0, len(boots), 5000)
            # 将逐格长扰动结果堆成矩阵，每行是一轮重复。
            jittered = np.column_stack([
                jitter_peak_cells(grid_values[i][1][sampled_rows], grid_values[i][0], rng)  # 对当前格长的选中峰位在对应网格单元内均匀抽样。
                for i in indices])
            # 每轮分别求截距和斜率。
            coefficients = (operator @ jittered.T).T
            # 沿重复轮次取两列系数的双侧 95% 百分位区间。
            intervals = np.quantile(coefficients, [.025, .975], axis=0)
            # 取相同剔除口径和格长子集的原外推行。
            old = original[(original.exclusion == rule) & (original.subset == subset)].iloc[0]
            # 每种口径和子集保存一个外推结果。
            rows.append({"exclusion": rule, "subset": subset,
                "point_intercept": (operator @ observed)[0],  # 无限格长临界概率的点估计仍取离散观测峰位。
                "gridaware_intercept_ci_low": intervals[0, 0],
                "gridaware_intercept_ci_high": intervals[1, 0],
                "point_slope": (operator @ observed)[1],  # 有限尺寸修正斜率同样基于离散观测峰位。
                "gridaware_slope_ci_low": intervals[0, 1],
                "gridaware_slope_ci_high": intervals[1, 1],
                "config_only_intercept_ci_low": old.intercept_ci_low,
                "config_only_intercept_ci_high": old.intercept_ci_high,
                "theory_pc": PC, "draws": len(sampled_rows),
                "method": "configuration peak bootstrap + uniform neighboring grid midpoint cell"})
    # 写出加入网格不确定性后的区间表。
    pd.DataFrame(rows).to_csv(destination, index=False)
    # 记录脚本版本、点估计口径及扰动方法。
    (OUT / "gridaware_metadata.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),  # 哈希覆盖当前脚本的全部源字节。
        "new_simulations": False, "point_estimator": "stored discrete scan-grid peak",
        "grid_uncertainty": "uniform within adjacent scan-point midpoints"}, indent=2) + "\n")
    # 为验证目录中的文件计算哈希，排除已有清单。
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
                if p.is_file() and not p.name.startswith("sha256_manifest")}
    # 写出新的独立校验清单。
    (OUT / "sha256_manifest_v2.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # 在终端展示所有口径和格长子集的外推结果。
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    # 启动离散扫描分辨率的敏感性分析。
    main()
