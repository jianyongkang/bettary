

-----

````markdown
# Lithium Battery State Estimation & Diagnostics Framework

> **A Physics-Data Hybrid Twin Approach (V14.3 Real-Time HIL Ready)**
>
> 基于“机理+数据”融合驱动的锂电池数字孪生、故障诊断与 SOX 估算系统。
> **本项目采用离线/在线分离架构，已做好 MCU 移植准备 (MCU-Ready)。**

## 1. 项目概述 (Overview)

本项目构建了一套**车规级**的锂电池全生命周期健康监测算法框架。针对电池**“内部状态不可测”**与**“老化状态难估计”**的行业痛点，本系统采用了 **物理模型 (Physics-based)** 与 **数据驱动 (Data-driven)** 深度融合的策略。

在仅依赖外部易测信号（$V, I, T_{surf}, \epsilon_{case}$）的前提下，系统实现了从**毫秒级故障诊断**到**长周期寿命预测**的全栈能力，并提供了**离线训练**与**在线推理 (Real-time Inference)** 两套完整架构。

**核心能力：**

* **多维状态反演**：基于 **混合训练策略** 的软传感器，精准反演**内部气压 ($P_{gas}$)** 与 **核心温度 ($T_{core}$)**。
* **物理/传感故障解耦**：利用物理残差 ($P_{mech}-P_{soft}$) 与测量残差 ($P_{meas}-P_{soft}$) 的正交特性，有效区分 **真实物理故障 (Cold Swelling)** 与 **传感器漂移 (Sensor Drift)**。
* **高鲁棒 SOX 估算**：内置安时积分 SOC 与 **基于 RLS 的自适应 SOH 估算**，在物理闭环仿真中实现了容量从 100% 向 90% 真值的精准收敛。
* **量化安全评估**：引入 **归一化严重度 (Normalized Severity)** 指标与 **鲁棒阈值标定**，提供红/橙/蓝分级预警。
* **实时 HIL 仿真**：提供独立的 `OnlineBMSEngine`，支持逐点流式计算，模拟真实 BMS 固件运行环境。

-----

## 2. 技术架构 (System Architecture)

系统采用模块化分层设计，核心算法位于 `src/` 目录：

### 2.1 核心算法库 (Core Library)

| 层级 | 模块文件 | 功能定义 | 核心技术 |
| :--- | :--- | :--- | :--- |
| **D0** | `data_schema.py` | **数据底座** | 统一数据语义，严格隔离物理真值 (True)、测量值 (Meas) 与估计值 (Est)。 |
| **D1** | `thermal_model_rc.py` | **热观测器** | 基于 3-Node RC 热网络，提供热力学基础状态。 |
| **D2** | `mech_strain_pressure.py` | **力学观测器** | 建立壳体应变-压力的非线性本构方程，作为物理一致性基准。 |
| **D3** | `soft_sensor.py` | **AI 软测量** | **[离线训练]** 采用“标定+车队”混合训练的多目标 XGBoost 模型，解决 OOD 问题。 |
| **D4** | `diagnostics.py` | **诊断工具** | **[离线分析]** 基于 Median+MAD 的鲁棒统计标定与严重度计算工具。 |
| **RT** | `online_engine.py` | **在线引擎** | **[MCU 原型]** 轻量级推理引擎。脱离 Pandas 依赖，支持 `step(sample)` 逐点调用，易于移植 C 代码。 |

### 2.2 应用程序 (Applications)

| 文件 | 用途 | 说明 |
| :--- | :--- | :--- |
| `main_pipeline.py` | **离线全链路仿真** | 执行数据生成、模型训练、参数标定及论文级图表绘制。 |
| `realtime_monitor.py` | **实时 HIL 上位机** | **[演示工具]** 模拟传感器数据流，调用 `OnlineBMSEngine`，提供动态黑客风仪表盘。 |

-----

## 3. 快速开始 (Quick Start)

### 3.1 环境准备

```bash
pip install -r requirements.txt
````

### 3.2 模式一：离线全链路分析 (Offline Pipeline)

用于算法验证、参数标定及生成分析报告。

```bash
python main_pipeline.py
```

**输出**：`outputs/` 目录下生成故障解耦图、SOH 收敛曲线及车队健康报表。

### 3.3 模式二：实时 HIL 仿真 (Real-time Dashboard)

模拟真实 BMS 运行环境，体验流式数据处理与动态报警。

```bash
python realtime_monitor.py
```

**操作**：运行后将弹出实时动态窗口，展示压力、温度、SOC/SOH 及故障严重度的实时变化曲线。

-----

## 4\. 关键结果解读 (Key Results)

### 4.1 故障解耦诊断 (`outputs/plot_fault_*.png`)

  * **冷鼓胀 (Cold Swelling)**: 物理模型 (Mech) 紧跟故障阶跃，软传感器 (Soft) 保持平稳，物理残差显著。
  * **传感器漂移 (Sensor Drift)**: 测量值 (Meas) 独自漂移，模型保持一致，测量残差显著。

### 4.2 寿命状态估算 (`outputs/plot_sox_estimation.png`)

  * **SOH Convergence**: 蓝线 (Est SOH) 从初始 100% 快速、平滑地**收敛至真值 90%**，验证了算法对 $dQ/dSOC$ 变化的敏感性。

### 4.3 实时监控看板 (`realtime_monitor.py` 界面)

  * **Severity 仪表**: 实时显示归一化严重度。正常工况下 $<0.5$，故障发生时迅速飙升至 $>1.0$ 并触发红色警报。
  * **动态响应**: 可直观观测到 SOH 在老化阶段的动态调整过程。

-----

## 5\. 版本演进 (Version History)

### [v14.3] - Real-Time HIL Ready (Current)

  * **Online Engine**: 封装了 `OnlineBMSEngine` 类，实现了算法的流式化重构。
  * **HIL Dashboard**: 新增 `realtime_monitor.py`，提供实时数据回放与可视化监控。

### [v13.0] - Ultimate Refactoring

  * **SOH Physics Loop**: 重构了老化数据的物理闭环生成逻辑，确保 SOH 算法在数学上可观测。
  * **Robust Soft Sensor**: 引入混合训练与去偏平滑，解决了 AI 温度失真问题。

### [v9.0 - v11.0] - Engineering Refactoring

  * **Severity Metric**: 引入归一化严重度指标与鲁棒阈值标定。

-----

## 6\. 实机部署路线图 (MCU Porting Guide)

**`src/online_engine.py` 已按照嵌入式逻辑设计，下一步只需进行语法转译即可上车。**

### 6.1 核心算法 C 化 (Core Algorithm Porting)

  * **目标**：将 `OnlineBMSEngine` 类转换为 `bms_core.c`。
  * **步骤**：
    1.  **数据结构**：定义 `BMS_Input_t` (电压、电流等) 和 `BMS_Output_t` (SOC、SOH、报警位)。
    2.  **去对象化**：将 Python 类方法 `step()` 转换为 C 函数 `BMS_Step(const BMS_Input_t* in, BMS_Output_t* out)`。
    3.  **移除库依赖**：使用手写 Ring Buffer 替代 `collections.deque`，使用 `math.h` 替代 `numpy`。

### 6.2 AI 模型轻量化 (Model Distillation)

  * **挑战**：MCU 无法直接运行 XGBoost 库。
  * **方案**：
      * **方法 A (Treelite)**: 使用 Treelite 将 XGBoost 模型编译为无依赖的 C 代码（if-else 森林）。
      * **方法 B (查表法)**: 将软传感器离散化为 3D LUT (Look-Up Table)，输入 V/I/T，查表得到 P/T\_core。

### 6.3 参数配置固化 (Configuration Freezing)

  * **现状**：参数分散在 Python 变量中。
  * **目标**：生成 `bms_config.h`。
  * **实现**：编写脚本将离线标定的阈值、多项式系数自动生成为 C 头文件宏定义：
    ```c
    // bms_config.h (Auto-generated)
    #define TH_PHYS_KPA  12.5f
    #define SOH_LEARN_RATE 0.2f
    static const float MECH_PARAMS[3] = {0.5f, 0.01f, 100.0f}; // k1, k3, p0
    ```

-----

**© 2025 Battery Digital Twin Project**

```
```
