# Tremor Stabilizer — 课设交付版

MuJoCo两自由度主动稳勺：五种因果震颤估计器、信号对比结果、浅勺颗粒运输与倾倒实验。
默认使用 **ABPF**，保留Huber-BMFLC作为强主动干扰下的备选。

## 看效果

- [无补偿视频](MuJoCo_NoComp.mp4)
- [ABPF补偿视频](MuJoCo_ABPF.mp4)

同一seed、同一摄像机，21秒。前11.2秒共同舀取，随后平滑加入震颤并启用补偿。

## 运行

Python 3.11；从仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m delivery check
python -m delivery view --method NoComp
python -m delivery view --method ABPF
```

视窗需要桌面OpenGL。支持 `--speed 0.5`；仓库提供全部七方法的同一代表seed=3回放。
重新仿真：`OPENBLAS_NUM_THREADS=1 python -m delivery system --workers 4`，结果存入独立目录，不覆盖交付数据。

## 调用算法

```python
from gimbal.estimators import EstimatorConfig, create_estimator
est = create_estimator("ABPF", EstimatorConfig.reference(unit="rad/s"))
est.reset()  # 每条独立记录开始时调用
estimated_tremor, diagnostics = est.step([0.1, -0.05])
```

可选 FixedBPF、ABPF、OriginalBMFLC、HuberBMFLC、DampedBMFLC。算法调用只依赖NumPy，输入/输出为相同单位的角速度、shape=(axes,)。
参考配置使用148.148Hz及独立p03_2校准；自有设备需用EstimatorConfig显式提供Fs、轴数、单位和标定，不能直接照搬患者参数。

## 结果

| 方法 | 运输保持率 | 最终转移率 | 主动倾倒RMSE |
|---|---:|---:|---:|
| NoComp | 36.8% | 56.4% | 2.32° |
| FixedBPF | 78.0% | 89.2% | 1.85° |
| **ABPF** | **99.7%** | **100%** | **0.96°** |
| OriginalBMFLC | 9.8% | 10.4% | 9.17° |
| HuberBMFLC | 97.6% | 99.0% | 5.37° |
| DampedBMFLC | 99.7% | 100% | 8.97° |
| Oracle | 100% | 100% | 0.80° |

主表为未参与工况选择的7个种子（3–9）；0–2用于NoComp/Oracle工况开发，所有10种子的完整指标均保留。
原始算法参数未按最终结果调整。ABPF与Damped转移率接近，但ABPF主动倾倒误差更小。

- [最终汇总CSV](experiments/results/final_delivery/final_summary.csv)
- [信号指标与快动作/rho实验](experiments/results/final_delivery/signals/metrics.csv)
- [系统逐次指标](experiments/results/final_delivery/system/per_run_metrics.csv) · [汇总](experiments/results/final_delivery/system/aggregate_metrics.csv)
- [信号对比图](reports/final_delivery/signal_figures/) · [系统图](reports/final_delivery/system_figures/)

分母是实际舀起的颗粒数，不是180粒源容器库存。离勺颗粒仍可能进入目标，因此运输保持率和最终转移率分开报告。
信号对比沿用冻结记录代理与已设计主动信号；记录代理不是纯生理震颤标签。Original稳态精度更高，Huber在极高rho更稳，ABPF并非所有信号工况第一。
本场景是已知时序的等装载运输诊断、单一记录来源；种子是颗粒初态而非受试者。不能据此推断临床效果或跨人群最优。
