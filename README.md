# Lithium Battery State Estimation & Diagnostics Framework

> **A Physics-Data Hybrid Twin Approach (V14.3 Real-Time HIL Ready)**
> 基于“机理 + 数据”融合的锂电池数字孪生、故障诊断与 SOX 估算系统。
> 采用离线/在线分离架构，已做好 MCU 迁移准备（MCU-Ready）。

## 更新记录 / Change Log
- 2026-01-08: add SOH estimation NaN handling, scatter fallback residuals, safety-limit risk plots, and README notes.
- 2026-01-07：修复 `src/data_schema.py` 字段字符串语法错误；`OnlineBMSEngine` 采样间隔改为时间戳/`dt_s` 推导；添加更新记录说明。
- 2026-01-07：阈值标定仅用健康运营数据并截尾；严重度放大且过滤校准/老化；机械压力估计平滑；软传感去偏仅用健康段；SOH 演示收敛减缓；散点图阈值自动标定。
- 说明：本节记录日常增量改动（按日期倒序）；大版本里程碑请写在下方“版本演进”章节。

## 1. 项目概述 (Overview)

本项目构建了一套车规级的锂电池全生命周期健康监测算法框架。针对“内部状态不可测”和“老化状态难估计”的行业痛点，采用 **物理模型 (Physics-based)** 与 **数据驱动 (Data-driven)** 的深度融合策略。

在仅依赖外部可测信号（V, I, T_surf, ε_case）前提下，实现从毫秒级故障诊断到长周期寿命预测的全栈能力。

**核心能力：**
- **多维状态反演**：基于混合训练策略的软传感器，反演内部气压 (P_gas) 与核心温度 (T_core)。
- **物理/传感故障解耦**：利用物理残差 (P_mech - P_soft) 与测量残差 (P_meas - P_soft) 的正交性，区分物理故障 (Cold Swelling) 与传感器漂移 (Sensor Drift)。
- **高鲁棒 SOX 估算**：内置安时积分 SOC 与基于 RLS 的自适应 SOH 估算，在物理闭环仿真中实现容量从 100% 收敛到 90% 的精准跟踪。
- **量化安全评估**：引入归一化严重度 (Normalized Severity) 指标与鲁棒阈值标定，提供红/橙/蓝分级预警。
- **实时 HIL 仿真**：提供独立的 `OnlineBMSEngine`，支持逐点流式计算，可模拟真实 BMS 固件运行环境。

-----

## 2. 代码结构与功能 (File Structure & Functions)

### 2.1 主程序 (Applications)

| 文件 | 功能 | 作用 |
| :--- | :--- | :--- |
| `main_pipeline.py` | 离线全链路仿真主程序 | 生成合成数据、训练 D2/D3 模型、执行 SOH 物理闭环验证、自动标定诊断阈值，并输出所有图表 (`outputs/`) |
| `realtime_monitor.py` | 实时 HIL 仿真上位机 | 模拟传感器数据流，实时调用在线引擎，动态展示压力/温度/SOC/SOH 及故障严重度报警 |
| `analyze_metrics.py` | 离线指标分析工具 | 批量计算并导出诊断指标统计表 (`outputs/diagnosis_indicators_table.csv`) |
| `plot_2d_scatter.py` | 散点图绘制工具 | 绘制故障解耦二维散点图 |

### 2.2 核心算法库 (src/)

| 模块 | 角色 | 功能 |
| :--- | :--- | :--- |
| `online_engine.py` | RT 在线引擎 | 封装 SOC/SOH 估算、软传感推理与诊断逻辑；支持逐点计算，便于移植 C |
| `mech_strain_pressure.py` | D2 力学模型 | 定义应变与气压的非线性本构，含热膨胀补偿；提供拟合与预测接口 |
| `soft_sensor.py` | D3 软传感器 | 基于 XGBoost 的数据驱动模型，含滑窗/微分特征工程，反演 P 与 T_core |
| `thermal_model_rc.py` | D1 热模型 | 基于 3-Node RC 网络的热模型，用于生成核心温度真值 |
| `diagnostics.py` | D4 诊断工具 | 计算残差指标、聚合特征，IsolationForest + RF 判别 |
| `synthetic_data.py` | 数据工厂 | 生成含正常/过压/传感器漂移/加速老化的高保真合成数据 |
| `data_loader.py` | 数据适配 | 读取外部实验数据，列映射、单位换算、时间轴对齐与重采样 |
| `preprocessing.py` | 信号预处理 | 低通滤波、异常值标记与插值修复 |
| `data_schema.py` | 数据字典 | 标准字段命名与单位定义，保证语义一致 |

-----

## 3. 快速开始 (Quick Start)

### 3.1 环境准备
```bash
pip install -r requirements.txt
```

### 3.2 模式一：离线全链路分析
```bash
python main_pipeline.py
```
输出：`outputs/` 目录生成故障解耦图、SOH 收敛曲线及车队健康报告。

### 3.3 模式二：实时 HIL 仿真
```bash
python realtime_monitor.py
```
操作：弹出实时动态窗口，展示压力、温度、SOC/SOH 与严重度。

-----

## 4. 关键结果 (Key Results)

- **故障解耦诊断**：`outputs/plot_fault_*.png` 展示 Cold Swelling 与 Sensor Drift 的残差分区。
- **SOH 收敛**：`outputs/plot_sox_estimation.png` 中估计 SOH 从 100% 平滑收敛到 90%。
- **车队健康评估**：`outputs/risk_score*.png` 展示归一化严重度及分级预警。
- **Safety-limit views**: outputs/risk_score_fleet.png and outputs/risk_score_full.png add a dashed safety limit line for operational vs full data.
- **Scatter fallback**: outputs/plot_metrics_scatter.png uses measurement residual when P_gas_kPa exists, otherwise it falls back to truth residual.

-----

## 5. 版本演进 (Version History)

- **[v14.3] Real-Time HIL Ready**：封装 `OnlineBMSEngine`，新增 HIL 仿真与实时看板。
- **[v13.0] Ultimate Refactoring**：重构老化物理闭环生成逻辑；软传感去偏/平滑。
- **[v9.0 - v11.0] Engineering Refactoring**：引入归一化严重度指标与阈值标定。
