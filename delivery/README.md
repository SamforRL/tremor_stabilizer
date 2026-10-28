# 队友交付指南（冻结版 v1）

本版本交付三部分：可独立流式调用的五算法、信号对比证据、MuJoCo浅勺洒漏实验。
默认 ABPF，HuberBMFLC 为强主动干扰下的备选。无补偿与 Oracle 仅为仿真边界，不是可调用的现实估计器。

## 安装与最短运行

在解压后的工程根目录运行，Python 3.11：

```bash
# 本机已有环境可直接激活；新环境安装 requirements.txt
conda activate me5418
python -m pip install -r requirements.txt
python -m delivery check
python -m delivery demo
python -m delivery view --method NoComp
python -m delivery view --method ABPF
```

`check`核查五算法reset、因果前缀、维度、单位等价、原实现输出一致性、模型编译及交付文件哈希。
原生MuJoCo窗口需要桌面OpenGL；无需显卡的运行/出图/检查不打开窗口。
视窗关闭即退出；`--speed 0.5`半速。默认代表seed=3由NoComp送达数量中位数自动选择，七方法共用。

## 队友算法调用

仅流式算法依赖NumPy，不需要加载MuJoCo或数据集：

```python
import numpy as np
from gimbal.estimators import EstimatorConfig, create_estimator

# 已冻结仿真参考标定，不是对新设备/新用户自动标定
cfg = EstimatorConfig.reference(unit="rad/s")
estimator = create_estimator("ABPF", cfg)
estimator.reset()  # 每条独立记录开始时调用，不要每个sample重置

for gyro in your_gyro_stream:  # shape=(2,), rad/s, uniform cfg.sample_rate_hz
    disturbance, diagnostics = estimator.step(np.asarray(gyro))
    reconstructed_task_rate = gyro - disturbance
```

可选名称：FixedBPF、ABPF、OriginalBMFLC、HuberBMFLC、DampedBMFLC。
输入和输出同单位，都是角速度，不是角度。时间步由Fs显式确定；输入必须有限，错误维度/NaN/Inf会拒绝。
所有轴独立，没有PCA或轴间融合。输出使用本次更新前权重；返回的权重诊断是更新后状态，统一为配置单位。

自有设备要明确Fs、轴数、单位以及独立校准结果：

```python
cfg = EstimatorConfig(
    sample_rate_hz=100.0, axes=2, unit="rad/s",
    modal_frequency_hz=(5.0, 6.0),  # 示例占位，替换为独立校准结果
    huber_delta_deg_s=(8.0, 9.0),   # 仅Huber需要；这里固定以deg/s配置
)
estimator = create_estimator("ABPF", cfg)
```

这些示例数字不是新设备的验证参数。ABPF必须逐轴提供modal；Huber必须逐轴提供正delta。
参考配置在`delivery/config.json`，课设结果不能通过改配置再沿用原数据标签。
算法不直接输出gimbal关节命令；仿真使用共同坐标映射、积分和控制器。一般gyro轴分量不是Euler角速度，实际硬件需先约定坐标与安装方向。

## 结果与图表

- 主报告：`reports/final_delivery/课设交付实验报告.md`
- 总表：`experiments/results/final_delivery/final_summary.csv`
- 信号：`.../signals/metrics.csv`、`traces/`、`provenance.csv`、`audit.json`
- 系统：`.../system/per_run_metrics.csv`、`aggregate_metrics.csv`、`paired_differences.csv`、`particle_outcomes.csv`
- 图片：`reports/final_delivery/signal_figures/`、`system_figures/`
- 库版本与检查：`experiments/results/final_delivery/verification.json`
- 冻结文件清单：`delivery/manifest.json`

CSV系统汇总的主表取`split=validation`（种子3–9）；development是0–2，all是全部10个。
种子代表颗粒初态，不是独立受试者。示例NPZ所有读取均不需要allow_pickle。

## 实验分工与限制

信号比较沿用既有p03_1记录代理和已设计主动信号：稳态、快速动作、rho扫描。
递推输出与历史核对一致；仅将权重诊断改成完整182项与统一单位。历史错误诊断没有覆盖。

最终系统使用4mm勺、180粒源容器；真实舀起约40粒。相同舀取完成后，从11.2s开始平滑加入震颤并启用补偿，15.75s主动倾倒。
这验证等装载运输，不是全程震颤患者进食的完整模型。七方法启用前物理状态和picked ID集合完全一致。
NoComp/Oracle用开发种子选定工况后冻结，没有看ABPF结果选场景；模型、控制器和其他算法也未按最终结果调参。
R_retain以实际舀起数为分母，R_transfer以这些颗粒最终进入目标的数量计算；未舀起的源容器库存不算洒漏。

ABPF不是所有信号工况最优；Original稳态精度更高，Huber在极高rho更稳。旧托盘/旧mocap系统结果属于历史探索，不与本交付混合排名。

## 重画图与重跑

```bash
# 审计/重放已有信号输入、整理已有系统结果、重画图并冻结清单；不重跑MuJoCo
python -m delivery build
# 真正重新仿真；70个带震颤run + 10个无震颤run，输出到独立reproduction_system/
OPENBLAS_NUM_THREADS=1 python -m delivery system --workers 4
# 小型运行检查：七条件一个seed，另一个无震颤基线
OPENBLAS_NUM_THREADS=1 python -m delivery system --workers 4 --seeds 1
# 验证后生成完整移交ZIP（不包含原始Dataset/个人配置）
python -m delivery package
```

重跑结果位于`experiments/results/final_delivery/reproduction_system/`，不会覆盖冻结交付结果。
在原生视窗查看某算法/某seed：`python -m delivery view --method HuberBMFLC --seed 3`。
算法切换不改变摄像机、物理参数或控制器。

## 文献追溯

- `analysis/tremor_baselines/REPRODUCTION_NOTES.md`：固定BPF、ABPF、BMFLC的实现与文献映射。
- `experiments/results/final_delivery/signals/historical_Damped_BMFLC_reproduction_notes.md`：2026 Damped论文公式与明确偏差。
- Huber是工程鲁棒更新变体，不标为原BMFLC论文的原始公式。
- 记录代理来自公开记录的离线频带提取，不是独立测得的生理震颤真值。

`experiments/results/final_delivery/mechanism/`另外沿用Round3的ABPF固定/自适应中心消融及频率响应表。
`signal_figures/abpf_fixed_adaptive_mechanism.png`用于解释频率反馈的作用和限幅；该独立100Hz合成验证不与当前148Hz记录代理实验混合统计。
