# Tremor Stabilizer：课设交付版 v1

本工程交付：**五个因果震颤估计器、可追溯的信号对比数据、MuJoCo两自由度浅勺颗粒运输/倾倒实验**。
默认算法为ABPF，HuberBMFLC保留为备选；所有正式参数和工况冻结。

## 快速开始

从工程根目录运行，Python 3.11。当前本机环境为`me5418`，新环境按`requirements.txt`安装。

```bash
conda activate me5418
python -m delivery check
python -m delivery demo
# 相同seed、同一原生MuJoCo摄像机
python -m delivery view --method NoComp
python -m delivery view --method ABPF
```

先完成约11秒相同舀取，再平滑加入震颤，比较运输/倾倒效果。窗口需桌面OpenGL；其余检查和出图可无窗口运行。
可加`--speed 0.5`或`--seed 3`。可选方法为NoComp、FixedBPF、ABPF、OriginalBMFLC、HuberBMFLC、DampedBMFLC、Oracle。

## 交付内容

| 内容 | 入口 |
|---|---|
| 队友调用、单位、校准、重跑说明 | [delivery/README.md](delivery/README.md) |
| 公共算法API（仅依赖NumPy） | [gimbal/estimators.py](gimbal/estimators.py) |
| 冻结参数 | [delivery/config.json](delivery/config.json) |
| 最终实验报告 | [课设交付实验报告](reports/final_delivery/课设交付实验报告.md) |
| 信号图：估计、主动重建、PSD、rho、污染 | [signal_figures](reports/final_delivery/signal_figures/) |
| 系统图：保持/转移、姿态、控制代价 | [system_figures](reports/final_delivery/system_figures/) |
| 汇总与逐次数据 | [final_delivery](experiments/results/final_delivery/) |
| 数据/代码校验 | [delivery/manifest.json](delivery/manifest.json) |

## 在线调用示例

```python
from gimbal.estimators import EstimatorConfig, create_estimator

# 仿真参考标定。自有硬件请显式提供自己的Fs、轴数、单位及独立校准参数。
config = EstimatorConfig.reference(unit="rad/s")
estimator = create_estimator("ABPF", config)
estimator.reset()
estimated_tremor, diagnostics = estimator.step([0.1, -0.05])
```

输入/输出均为角速度，shape=(axes,)。不在每个sample重置，不把算法输出直接当作角度或关节命令。

## 维护与复现

```bash
# 整理已保存证据、核对历史信号输出、重画图；不重跑MuJoCo
python -m delivery build
# 在新目录重跑全部物理实验（保留交付结果）
OPENBLAS_NUM_THREADS=1 python -m delivery system --workers 4
# 生成可直接转交队友的白名单移交包
python -m delivery package
```

正式主表采用独立验证种子3–9；0–2仅用于NoComp/Oracle工况开发。
该场景是等装载运输诊断，不能代替全程震颤进食验证。当前单来源工况支持工程选型，不宣称跨人群最优。
完整历史代码和结果保留；旧模型的系统排名不与最终场景合并。历史说明见[历史存档](reports/history/工程历史说明.md)。
`AGENTS.md`保留早期设计边界，其中Phase 1目标已完成；本次交付工作遵循后续已授权的算法比较和系统功能验证任务。
