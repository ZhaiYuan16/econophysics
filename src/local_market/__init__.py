"""Local-information financial market models."""
# 集中导出格点占据抽样、团簇识别和两种 Cont-Bouchaud 交易入口。
# sample_lattice_occupancy() 抽取占据格点，identify_lattice_clusters() 生成 ClusterConfiguration。
# simulate_quenched() 固定团簇配置，simulate_annealed() 每步重建配置，两者都返回 MarketSimulation。
#
# 包级接口将格点生成、团簇识别和交易模拟串在一起，供调用者按这三个步骤组成基线模拟。

from .cb_model import MarketSimulation, simulate_annealed, simulate_quenched  # src/local_market/cb_model.py，导出交易结果容器和固定配置、逐步更新配置两种模拟函数。
from .networks import sample_lattice_occupancy  # src/local_market/networks.py，导出正方格点占据状态抽样函数。
from .percolation import ClusterConfiguration, identify_lattice_clusters  # src/local_market/percolation.py，导出格点团簇配置和四邻域识别函数。

# 列出包级公开名称，供通配导入按这一范围导出。
__all__ = [
    "ClusterConfiguration",
    "MarketSimulation",
    "identify_lattice_clusters",
    "sample_lattice_occupancy",
    "simulate_annealed",
    "simulate_quenched",
]
