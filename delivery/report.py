"""Chinese handoff report generated only from curated measured results."""
import numpy as np
from .common import ROOT,DATA,REPORT,CONFIG,read_csv,write_csv,write_json


def write_report():
    signals=read_csv(DATA/'signals/metrics.csv');system=read_csv(DATA/'system/aggregate_metrics.csv')
    def sig(case,m,rho=None):
        return next(r for r in signals if r['case']==case and r['method']==m and (rho is None or float(r['nominal_rho'])==rho))
    def sys(m,split='validation'):return next(r for r in system if r['method']==m and r['split']==split)
    def f(r,k,d=3):return f'{float(r[k]):.{d}f}'
    def ms(r,k,scale=1,d=1):return f'{float(r[k+"_mean"])*scale:.{d}f} ± {float(r[k+"_sd"])*scale:.{d}f}'
    summary=[]
    for m in CONFIG['methods']:
        a=sys(m);s=sig('steady',m);fast=sig('fast_transition',m);high=sig('amplitude_ratio',m,16.)
        summary.append(dict(method=m,steady_nrmse=float(s['nrmse']),fast_nrmse=float(fast['nrmse']),rho16_nrmse=float(high['nrmse']),
            validation_retention_percent=100*float(a['transport_retention_ratio_mean']),
            validation_transfer_percent=100*float(a['transfer_ratio_of_picked_mean']),
            validation_unload_rmse_deg=float(a['unload_pitch_rmse_deg_mean']),
            validation_intent_amplitude_ratio=float(a['intentional_pitch_amplitude_ratio_mean']),
            validation_torque_cost_Nm2_s=float(a['torque_integral_Nm2_s_mean'])))
    write_csv(DATA/'final_summary.csv',summary)
    write_json(DATA/'frozen_delivery_config.json',CONFIG)
    actual=sig('fast_transition','ABPF')['actual_event_rho']
    lines=['# 课设交付版：算法比较与浅勺功能验证','',
    '## 1. 交付结论','',
    '**默认估计器采用 ABPF，保留 HuberBMFLC 作为强主动干扰下的备选。** 选择依据是当前信号精度、主动动作鲁棒性、实际颗粒运输和主动倾倒保留的综合表现，不构造加权总分，也不宣称 ABPF 在所有指标上第一。',
    '',
    '本次复用并逐样本审计历史信号数据，补齐浅勺场景缺失的四种方法，共形成五算法信号证据和七条件 × 10 paired seeds 的系统证据。估计器递推、参数、4 mm 勺面、控制器和震颤工况均未根据此次结果调整。',
    '',
    '## 2. 算法模块与冻结边界','',
    '`gimbal/estimators.py` 提供 create_estimator / step / reset；输入为每个时刻的一维轴向角速度数组，输出为同物理量的震颤估计。支持 deg/s 和 rad/s，采样率与轴数显式配置；轴独立处理。默认参考配置为 p03_2 校准的二轴、148.148 Hz，不能直接视为新用户或新设备已完成标定。',
    '',
    'FixedBPF：3–12 Hz，Butterworth prototype order=4，最终数字带通阶数=8；ABPF：独立 modal=[4.60046,8.53419] Hz，步长0.1 Hz、modal±1.5 Hz；Original/Huber：3–12 Hz、0.1 Hz 网格、mu=0.01；Huber delta=[8.87985,9.82088] deg/s。Damped：沿用2026复现 eta=.02、lambda_ref=.99999@1kHz、kdmp=350、xdmp=.009 rad/s，其内部状态保持 SI 量纲。',
    '',
    '输出使用更新前权重，诊断权重为更新后状态；Fourier 排列为全部 sine 后接全部 cosine，共182项，bias不计入震颤输出。公共 API 将各方法诊断权重统一为所选信号单位，未改算法。',
    '',
    '## 3. 信号级实验：沿用数据，不重新选参','',
    '来源：Pinheiro p03_1、20–32 s、X/Y 两轴记录角速度的离线3–12 Hz代理；独立校准沿用 p03_2 的0–15 s。主要图表使用历史 moderate 条件（alpha=1.12764494）；strong 的历史表也随包保留。录制代理不是独立生理震颤标签。',
    '',
    '稳态输入仅为记录代理；快速动作使用原最小跃度10°动作的角速度，动作在3.0–3.35 s，污染评价段为3–5 s；幅值比扫描使用相同波形与相同记录代理，rho∈{0.5,1,2,4,8,16}，定义为3–5 s内双轴合并RMS(v)/RMS(d)。',
    '',
    f'原快速动作文件名中的 rho1 是历史标签，实际该案例 rho={float(actual):.3f}，本交付不将它误写成严格rho=1；严格ratio对比来自独立幅值比扫描。全部源输入、时间戳、真值和历史哈希在 provenance.csv 中保存。',
    '',
    '| 方法 | Steady nRMSE | Fast nRMSE | rho=16 nRMSE | Steady corr | Steady phase° |',
    '|---|---:|---:|---:|---:|---:|']
    for m in CONFIG['methods']:
        s=sig('steady',m);fast=sig('fast_transition',m);high=sig('amplitude_ratio',m,16)
        lines.append(f'| {m} | {f(s,"nrmse")} | {f(fast,"nrmse")} | {f(high,"nrmse")} | {f(s,"correlation")} | {f(s,"phase_error_deg",1)} |')
    lines+=['',
    'nRMSE为逐轴 RMSE/RMS(注入代理) 后的双轴平均，包含启动，不做时间对齐。相关系数与幅值比也按轴计算后平均；相位为各轴真实代理主频处的频谱相位差，不是全频率统一延迟。重建主动信号使用 mixture−estimate；由于未加测量噪声，它与震颤误差具有代数关系，不能当作另一份独立证据。',
    '',
    '复用审计：对已有轨迹逐样本重放公共接口，要求输出与历史保存结果完全相同；40条比较的nRMSE均与历史表一致。未保存轨迹的rho档位仅从既有rho=16的主动分量按已冻结比例重放，并核对历史指标。未采集新数据、未重新调参。',
    '',
    '诊断修正：旧统计曾对已排除bias的权重再次截去一项，且Damped权重为rad/s而其他方法为deg/s。新权重污染图统一使用182项与deg/s，并以同方法纯扰动参考的权重差范数定义。只修正诊断，不改震颤输出；旧文件保留。',
    '',
    '解释：Original BMFLC 稳态精度最高，但在强主动 transient 下污染严重；Huber在极高rho下优于ABPF。ABPF因此是当前工况的工程折中，不能声称对任意主动幅值均最稳。',
    '',
    '机制补充直接沿用Round3的fixed-center/自适应ABPF消融（独立100Hz合成验证，不混入当前数据表）：缓漂RMSE约0.966→0.455 deg/s，阶跃约1.190→0.482 deg/s；中心仍受modal±1.5Hz限制，并非总能到达真实频率。五人记录代理180个配对案例中的相对改善中位数仅约1.59%，说明不能将全部优势归因于频率反馈；宽通带结构及独立校准本身很重要。原表、响应表和频率轨迹已复制至mechanism/，另附统一机制图。',
    '',
    '## 4. 最终系统场景与公平性','',
    '4 mm半椭球壳勺面，180粒源容器，真实接触舀取后运输到目标并主动pitch倾倒；21 s，1 ms物理步长、两自由度gimbal、±45°关节范围、±.15 Nm力矩限幅。手柄为动态自由根并通过weld跟随mocap，接触获得真实运动速度。算法、控制器、颗粒初态与摄像机完全共用。',
    '',
    '0–11.2 s：所有条件同样无震颤、无补偿舀取，估计器可以看到历史，补偿积分固定为零。11.2 s后：0.5 s quintic窗平滑加入相同p03_2记录代理，alpha=5，同时启用补偿；15.75 s后主动pitch倾倒。这是已知任务时序的**等装载运输诊断**，不表示患者震颤只在运输时发生，也不是为某个估计器设计的在线gate。',
    '',
    'NoComp/Oracle上下界只用开发种子0–2选定深度与强度，之后工况冻结。主结论使用未参与工况选择的7个种子3–9；全部10种子另存。新增四算法只补齐相同冻结条件，逐帧确认七方法在11.2 s前的根位姿、关节角、全部颗粒位置和实际舀起ID完全一致。',
    '',
    '估计器输入来自实际MuJoCo手柄世界角速度，经解析坐标导数映射到现有roll/pitch坐标；输出经共同deg/s→rad/s、采样保持、1kHz梯形积分、IK与PD链。Oracle也经过同链，其真值是实际投影角速度减已知主动角速度，包含物理驱动/接触偏差；其他方法不读取该真值。',
    '',
    '## 5. 最终系统结果：独立验证种子3–9','',
    '| 条件 | 保持率% | 转移率% | Roll/Pitch RMSE° | 倾倒RMSE° | 主动pitch幅值比 | 全程力矩积分Nm²s |',
    '|---|---:|---:|---:|---:|---:|---:|']
    for m in CONFIG['system']['all_methods']:
        a=sys(m)
        lines.append(f'| {m} | {ms(a,"transport_retention_ratio",100)} | {ms(a,"transfer_ratio_of_picked",100)} | {f(a,"transport_roll_rmse_deg_mean")} / {f(a,"transport_pitch_rmse_deg_mean")} | {f(a,"unload_pitch_rmse_deg_mean")} | {f(a,"intentional_pitch_amplitude_ratio_mean")} | {f(a,"torque_integral_Nm2_s_mean",5)} |')
    lines+=['',
    '零震颤10种子均全部保持并送达，初始服务份量约39–45粒；180粒是源容器库存，未舀起的颗粒不记作洒漏。R_retain=N(舀起且卸料前仍在勺中)/N_picked；R_transfer=N(舀起且最终在目标)/N_picked。最终洒漏比例记为1−R_transfer，运输离勺数量则为N_picked−N_retained。已离勺颗粒仍可能落入目标，因此两种损失不可混称。',
    '',
    '额外保存：逐颗粒ID结果、运输/倾倒失败数量、方向误差峰值/p95、相对NoComp的3–12Hz Welch功率衰减、关节速度、全程力矩积分/峰值/饱和、分阶段采样力矩和饱和、全程估计nRMSE/相关/幅值/相位，以及Fourier污染。',
    '',
    '全程约8.35%力矩饱和来自共同舀取阶段；运输/倾倒的100Hz控制日志中均未出现饱和。分阶段采样统计与1kHz全程统计分别命名，不能将前者冒充完整物理步检查。',
    '',
    '## 6. 最终选择','',
    'ABPF与Damped在颗粒转移上接近，但Damped主动倾倒幅值比约1.18、倾倒误差约8.97°；ABPF约1.002和0.96°。ABPF因此更能同时满足运输、主动倾倒与较低姿态误差。Huber转移率约99%，在极高rho信号实验更稳，适合作为备选；其当前倾倒误差仍高于ABPF。Original BMFLC不能因稳态nRMSE最低就选为当前默认方法。',
    '',
    'ABPF的过零/阻尼机制是否贡献了全部收益，不由这次场景实验单独证明；本次没有新增fixed-center消融。既有Round3机制研究保持独立。新模型当前仅一个记录来源和一种冻结运输强度，7个种子是颗粒初态重复，不是7名受试者；不作普遍最优或临床推断。',
    '',
    '## 7. 交付材料与证据分工','',
    '| 材料 | 目的 | 使用方式 |','|---|---|---|',
    '| gimbal/estimators.py + delivery/config.json | 队友在线调用 | 统一接口，显式单位/采样率/标定 |',
    '| signals/metrics.csv + signal_figures/ | 稳态、主动污染、rho鲁棒性对比 | 沿用历史信号输出，统一图表/诊断 |',
    '| system/aggregate_metrics.csv + system_figures/ | 当前最终洒漏与控制对比 | 以validation七种子为主表 |',
    '| system/particle_outcomes.csv | 解释掉粒、目标接收与卸料失败 | 按ID自动统计，非人工数视频 |',
    '| 原生MuJoCo回放 | 展示功能结局与机械动作 | 相同NoComp中位数代表seed=3 |',
    '| 历史旧托盘与旧mocap系统结果 | 探索记录 | 不与最终4mm场景合并排名 |',
    '',
    '数据根目录：`experiments/results/final_delivery/`；统一图根目录：`reports/final_delivery/`。完整可移交包由`python -m delivery package`生成至`dist/tremor_stabilizer_course_v1.zip`，包含必要代码、模型、参考处理数据、逐次结果、信号输入、回放状态和清单，不包含原始Dataset目录或个人配置。',
    '',
    '详细调用、检查、重画图、重跑与回放命令见工程README及`delivery/README.md`。图中所有正式结果保持因果时序，没有事后相位对齐。',
    '']
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'课设交付实验报告.md').write_text('\n'.join(lines),encoding='utf-8')
