

-----

# Lithium Battery State Estimation & Diagnostics Framework

> **A Physics-Data Hybrid Twin Approach**
>
> 基于“机理+数据”融合驱动的锂电池数字孪生与预警系统

## 1\. 项目概述 (Overview)

本项目构建了一套完整的、逻辑自洽的锂电池健康监测算法框架。针对电池\*\*“内部状态（核心温度、气压）不可直接测量”\*\*的行业痛点，本系统采用 **物理模型 (Physics-based)** 与 **数据驱动 (Data-driven)** 相结合的“双轮驱动”策略。

在仅依赖外部易测信号（电压、电流、表面温度、壳体应变）的前提下，系统实现了对电池内部状态的**毫秒级软测量**与**故障解耦诊断**。

**核心能力：**

  * **软测量精度**：$R^2 > 0.99$ (MAE $\approx$ 0.1 kPa)
  * **故障隔离**：有效区分 **真实物理故障 (Overpressure)** 与 **传感器漂移 (Sensor Fault)**。
  * **分级预警**：基于 Isolation Forest 的无监督异常评分，实现红/蓝分明的风险分级。

-----

## 2\. 技术架构 (System Architecture)

系统采用模块化分层设计，核心代码位于 `src/` 目录下：

| 层级 | 模块文件 | 功能定义 | 核心技术 |
| :--- | :--- | :--- | :--- |
| **D0** | `data_schema.py` | **数据底座** | 统一数据语义，严格隔离物理真值 (True)、测量值 (Meas) 与估计值 (Est)。 |
| **D1** | `thermal_model_rc.py` | **热观测器** | 基于 3-Node RC 热网络 + SciPy 加速，实时反演核心温度 ($T_{core}$)。 |
| **D2** | `mech_strain_pressure.py` | **力学观测器** | 壳体应变-压力非线性映射（含热补偿），支持从标定数据辨识参数。 |
| **D3** | `soft_sensor.py` | **AI 软测量** | 基于 XGBoost 的高精度回归模型，学习物理规律，反演内部气压。 |
| **D4** | `diagnostics.py` | **诊断大脑** | 计算物理一致性残差 ($P_{mech} - P_{soft}$)，利用 Isolation Forest 进行异常检测。 |
| **Data** | `synthetic_data.py` | **数据工厂** | 生成包含正常、过压、传感器故障的多场景合成数据，注入真实噪声。 |
| **Main** | `main_pipeline.py` | **全链路集成** | 一键运行数据生成、模型训练、推理评估与可视化报告。 |

-----

## 3\. 快速开始 (Quick Start)

### 3.1 环境准备

确保已安装 Python 3.9+，建议使用 Conda 环境：

```bash
conda create -n battery_env python=3.9
conda activate battery_env
pip install -r requirements.txt
```

### 3.2 运行全链路仿真

该脚本将自动生成数据、训练模型、进行故障注入并输出诊断图表：

```bash
python main_pipeline.py
```

### 3.3 查看结果

运行结束后，请查看 `outputs/` 目录：

  * **`soft_sensor_metrics.csv`**: 软测量模型精度评估报告。
  * **`plot_fault_overpressure.png`**: **[物理故障验证]** 绿线（物理模型）跟随真值，红线（软测量）滞后，形成残差。
  * **`plot_fault_sensor.png`**: **[传感器故障隔离]** 仅测量值漂移，绿线与红线保持一致且正常，成功识别误报。
  * **`risk_score.png`**: 全 Fleet 风险评分，红色柱状代表模型判定的异常循环。

-----

## 4\. 真实数据适配 (Real-world Integration)

为支持从仿真走向实测，V4.3 版本已预置真实数据适配接口：

  * **数据加载** (`src/data_loader.py`):
      * 支持 CSV/Excel 导入。
      * 自动处理列名映射、单位换算与时间轴去重/对齐。
  * **预处理** (`src/preprocessing.py`):
      * 提供低通滤波 (Low-pass Filter)。
      * 支持异常值标记 (Outlier Flagging) 而非简单剔除，保留故障证据。

-----

## 5\. 版本演进 (Version History)

本项目经历了多次严谨的工程迭代，最终形成 V4.3 Golden Baseline。

### [v4.3] - 2025-11-28 (Golden Baseline)

  * **Final Polish**: 重构了可视化逻辑，风险图颜色由**模型判决结果**决定，而非上帝视角标签，实现了真正的盲测验证。
  * **Robustness**: 增加了软测量 $R^2$ 指标的自动化检查与警告机制。
  * **Docs**: 集中化管理输出路径配置，规范了工程结构。

### [v4.2] - Physics Logic Fix

  * **物理语义修正**: 重新定义了 `sensor_fault`。现在该故障只影响**测量层 ($P_{meas}$)**，保持**物理真值 ($P_{true}$)** 不变，彻底解决了此前逻辑中的物理悖论。
  * **Diagnostics**: 在诊断特征中增加了 `has_P_meas`，帮助 Isolation Forest 有效区分标定电池（有传感器）和量产电池（无传感器）。

### [v3.0 - v4.0] - The Closed Loop

  * **物理闭环**: 实现了完整的 "Physics-Data Loop"。利用标定数据反向拟合 D2 参数，再应用于量产电池，模拟真实的模型误差 (Model Mismatch)。
  * **防泄露**: 软测量特征工程采用 Group-wise (按循环分组) 处理，彻底消除了时序数据泄露风险。
  * **语义清晰化**: 建立了 `_true` (真值), `_meas` (测量), `_est` (估计) 的严格命名规范。

-----

## 6\. 未来规划 (Roadmap)

  * **V5.0 Performance**: 优化 D4 诊断模块，支持 `score_table` 批处理接口，移除诊断循环中的 `iterrows` 以提升大规模数据处理速度。
  * **V5.1 Infra**: 在入口处增加 SciPy/XGBoost 的环境依赖检查与优雅降级策略。
  * **V5.2 Feature**: 支持变采样率 ($dt_s$) 场景的特征自适应工程。
