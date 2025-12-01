````markdown
# Lithium Battery State Estimation & Diagnostics Framework

> **A Physics-Data Hybrid Twin Approach**
>
> 基于“机理+数据”融合驱动的锂电池数字孪生与预警系统

## 1. 项目概述 (Overview)

本项目构建了一套完整的、逻辑自洽的锂电池健康监测算法框架。针对电池**“内部状态（核心温度、气压）不可直接测量”**的行业痛点，本系统采用 **物理模型 (Physics-based)** 与 **数据驱动 (Data-driven)** 相结合的“双轮驱动”策略。

在仅依赖外部易测信号（电压、电流、表面温度、壳体应变）的前提下，系统实现了对电池内部状态的**毫秒级软测量**与**故障解耦诊断**。

**核心能力：**

* **高精度软测量**：基于 XGBoost 的非线性回归，反演精度 $R^2 > 0.99$ (MAE $\approx$ 0.1 kPa)。
* **故障解耦**：有效区分 **真实物理故障 (Overpressure)** 与 **传感器漂移 (Sensor Fault)**，解决了传统BMS“误报率高”的难题。
* **分级预警**：基于 Isolation Forest 的无监督异常评分，无需故障样本训练即可识别异常。

-----

## 2. 技术架构 (System Architecture)

系统采用模块化分层设计，核心代码位于 `src/` 目录下：

| 层级 | 模块文件 | 功能定义 | 核心技术 |
| :--- | :--- | :--- | :--- |
| **D0** | `data_schema.py` | **数据底座** | 统一数据语义，严格隔离物理真值 (True)、测量值 (Meas) 与估计值 (Est)。 |
| **D1** | `thermal_model_rc.py` | **热观测器** | 基于 3-Node RC 热网络 + SciPy 加速，实时反演核心温度 ($T_{core}$)。 |
| **D2** | `mech_strain_pressure.py` | **力学观测器** | 壳体应变-压力非线性映射（含热补偿），支持从标定数据辨识参数。 |
| **D3** | `soft_sensor.py` | **AI 软测量** | 基于 XGBoost 的数据驱动模型，学习物理规律，反演内部气压。 |
| **D4** | `diagnostics.py` | **诊断大脑** | 计算物理一致性残差 ($P_{mech} - P_{soft}$) 与测量残差，利用 Isolation Forest 异常检测。 |
| **Data** | `synthetic_data.py` | **数据工厂** | 生成包含正常、过压、传感器故障的多场景合成数据，注入真实噪声。 |
| **Main** | `main_pipeline.py` | **全链路集成** | 一键运行数据生成、模型训练、推理评估与可视化报告。 |

-----

## 3. 快速开始 (Quick Start)

### 3.1 环境准备

确保已安装 Python 3.9+，建议使用 Conda 环境：

```bash
conda create -n battery_env python=3.9
conda activate battery_env
pip install -r requirements.txt
````

### 3.2 运行全链路仿真

该脚本将自动生成数据、训练模型、进行故障注入并输出诊断图表：

```bash
python main_pipeline.py
```

### 3.3 生成论文级图表

运行以下脚本，生成故障解耦的二维散点图：

```bash
python plot_2d_scatter.py
```

-----

## 4\. 结果解读 (Results Interpretation)

运行结束后，`outputs/` 目录将生成以下关键图表，验证了算法的有效性：

### 4.1 物理故障验证 (`plot_fault_overpressure.png`)

  * **现象**：物理模型估计（绿线）跟随真值（黑线）跳变，而AI软测量（红线）滞后。
  * **结论**：**物理一致性残差增大**。系统判定为物理结构异常（如鼓包），而非电气故障。

### 4.2 传感器故障隔离 (`plot_fault_sensor.png`)

  * **现象**：传感器读数（青点）大幅漂移，但物理模型与AI模型保持一致且正常。
  * **结论**：**测量残差增大，物理一致性正常**。系统判定为传感器故障，避免误报。

### 4.3 故障解耦图 (`plot_metrics_scatter.png`)

  * **现象**：真实物理故障（Overpressure）和传感器故障（Sensor Drift）在二维残差空间中呈**正交分布**。
  * **结论**：证明了本系统具备极强的故障分类能力。

### 4.4 风险评分 (`risk_score.png`)

  * **现象**：正常电池评分为负（蓝色），故障电池评分为正（红色）。
  * **结论**：无监督算法有效实现了红/绿灯式的预警。

-----

## 5\. 真实数据适配 (Real-world Integration)

为支持从仿真走向实测，系统预置了真实数据适配接口：

  * **数据加载** (`src/data_loader.py`):
      * 支持 CSV/Excel 导入，自动处理列名映射与单位换算。
      * 具备鲁棒的时间轴对齐与重采样功能。
  * **预处理** (`src/preprocessing.py`):
      * 提供低通滤波 (Low-pass Filter) 消除信号毛刺。
      * 支持异常值标记 (Outlier Flagging) 而非简单剔除，保留故障特征。

-----

## 6\. 版本演进 (Version History)

本项目经历了多次严谨的工程迭代，最终形成 V5.0 Stable Release。

### [v5.0] - Engineering Robustness (Approved)

  * **Soft Sensor Fix**: 修复了缺失值填充逻辑。采用 `interpolate` 替代 `fillna(0)`，消除了 0K 温度和 0% SOC 导致的物理谬误，提升了抗丢包能力。
  * **Mech Model Fix**: 引入了参数归一化 (Normalization)。解决了 `curve_fit` 在微小应变 ($10^{-6}$) 与高压 ($10^2$) 混合拟合时的数值不稳定问题。

### [v4.3.3] - Final Rigor

  * **Scientific Rigor**: 在 D3 训练中引入 `Stable Split` (排序+固定随机种)，确保论文结果可复现。
  * **Validation**: 增强了数据输入校验（零方差特征剔除、样本量检查）。
  * **Visualization**: 统一了图表术语（Residual），添加了关键点标注。

### [v4.0 - v4.2] - Physics Logic Fix

  * **物理语义修正**: 重新定义了 `sensor_fault`。该故障只影响测量层，物理真值不变，解决了物理悖论。
  * **Diagnostics**: 增加了 `has_P_meas` 特征，帮助算法区分标定电池与量产电池。

### [v3.0] - The Closed Loop

  * **物理闭环**: 实现了完整的 "Physics-Data Loop"。利用标定数据反向拟合 D2 参数，再应用于量产电池。

-----

**© 2025 Battery State Estimation Project**

```
```