"""
final_dashboard_v2.py
[BMS Digital Twin] - 电池内外部状态实时可视化终端 (修正版)
功能：
1. 实时对比显示：内部气压 (Pressure) 与 核心温度 (Temperature)。
2. 内置 MCU 级诊断算法 (EMA滤波 + 状态机)，严格对齐物理时间尺度。
3. 修正了索引错误、标定数据清洗逻辑，符合工程标准。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.thermal_model_rc import RCThermalParams
from src.mech_strain_pressure import MechParams, MechanicalPressureModel
from src.soft_sensor import ModelConfig, DataDrivenEstimator
from src.synthetic_data import NoiseConfig, SyntheticScenarioConfig, generate_synthetic_dataset

# ==========================================
# 1. 嵌入式算法模拟 (C Code Logic in Python)
# ==========================================
class MCUDiagnostic:
    """
    模拟 MCU 中的去抖动诊断逻辑
    参数严格对应 C 代码宏定义
    """
    def __init__(self, alpha=0.2, th_on=20.0, th_off=15.0, time_confirm=3.0):
        self.alpha = alpha              # 滤波系数 (对应 DIAG_FILTER_ALPHA)
        self.th_on = th_on              # 触发阈值 (对应 THRESH_SWELLING_SET)
        self.th_off = th_off            # 恢复阈值 (对应 THRESH_SWELLING_CLR)
        self.time_confirm = time_confirm # 确认时间(秒) (对应 FAULT_CONFIRM_TICKS * dt)
        
        self.resid_flt = 0.0    # 滤波后的残差
        self.fault_time_s = 0.0 # 故障计时器 (秒)
        self.is_fault = False   # 故障标志
        
    def update(self, p_mech, p_soft, dt_s):
        # 1. 计算瞬时残差 (绝对值)
        resid_inst = abs(p_mech - p_soft)
        
        # 2. EMA 低通滤波 (IIR)
        # y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
        self.resid_flt = self.alpha * resid_inst + (1 - self.alpha) * self.resid_flt
        
        # 3. 状态机 (基于物理时间 dt_s 累加)
        if not self.is_fault:
            if self.resid_flt > self.th_on: # 触发阈值
                self.fault_time_s += dt_s
                if self.fault_time_s >= self.time_confirm: 
                    self.is_fault = True
            else:
                self.fault_time_s = 0.0 # 漏桶清零
        else:
            if self.resid_flt < self.th_off: # 恢复阈值
                self.is_fault = False
                self.fault_time_s = 0.0
                
        return self.resid_flt, self.is_fault

# ==========================================
# 2. 系统初始化与模型训练 (Offline Process)
# ==========================================
def init_system():
    print(">> [System] Initializing Digital Twin...")
    
    # --- A. 配置参数 ---
    # 物理参数
    rc_p = RCThermalParams(800, 300, 1000, 2.0, 1.0, 0.5, 0.5)
    mech_p = MechParams(2e5, 1e8, 100, 23e-6)
    
    # 噪声配置
    noise_train = NoiseConfig(outlier_prob=0.0, sigma_Pgas=0.5, sigma_eps=1e-6)
    noise_test  = NoiseConfig(outlier_prob=0.0, sigma_Pgas=1.0, sigma_eps=2e-6)

    # --- B. 生成训练数据 (Golden Sample) ---
    print(">> [Train] Generating Calibration Data...")
    scens_train = [SyntheticScenarioConfig("calib_golden", 1800, is_calib=True)]
    df_train_raw = generate_synthetic_dataset(scens_train, rc_p, mech_p, "cubic", 10.0, noise_train)
    
    # [Fix]: 显式过滤掉任何非正常工况数据 (虽然 calib_golden 本身无故障，但这是工程规范)
    df_train = df_train_raw[df_train_raw["fault_type"] == "none"].copy()
    
    # --- C. 训练模型 ---
    # 1. 机械模型 (D2)
    print(">> [Train] Fitting Mech Model...")
    mech_model = MechanicalPressureModel("cubic", alpha_thermal=23e-6)
    mech_model.fit(df_train)
    
    # 2. 软传感器 (D3) - 同时预测 P 和 T
    print(">> [Train] Training Soft Sensor (Pressure + Temp)...")
    cfg = ModelConfig(target_cols=['P_gas_phys_kPa', 'T_core_phys_degC'])
    est = DataDrivenEstimator(cfg)
    est.fit(df_train)
    
    return mech_model, est, rc_p, mech_p, noise_test

# ==========================================
# 3. 实时仿真主程序 (Online Animation)
# ==========================================
def main():
    # 1. 初始化模型
    mech_model, soft_sensor, rc_p, mech_p, noise_test = init_system()
    
    # 2. 生成演示工况 (故障注入：严重过压)
    print(">> [Sim] Generating Fault Scenario Stream...")
    target_scen = SyntheticScenarioConfig("fleet_fault_op", 1800, is_calib=False, 
                                          anomaly_type="overpressure", anomaly_level="severe")
    # 生成带噪声的测试流
    df_demo = generate_synthetic_dataset([target_scen], rc_p, mech_p, "cubic", 10.0, noise_test)
    
    # [Fix]: 统一使用 index 作为时间轴，并转为 numpy 数组加速
    t_arr = df_demo.index.to_numpy()
    
    # 3. 预推理 (模拟 MCU 实时计算)
    # D2 推理 (应变 -> 压力)
    df_demo["P_mech"] = mech_model.predict(df_demo)["P_gas_mech_est"]
    # D3 推理 (V/I/T -> 压力 + 温度)
    df_pred = soft_sensor.predict(df_demo)
    df_demo["P_soft"] = df_pred["P_gas_phys_kPa_pred"]
    df_demo["T_core_soft"] = df_pred["T_core_phys_degC_pred"]
    
    # [Fix]: 提取所有需要绘图的列为 numpy 数组 (模拟 MCU Buffer)
    p_true_arr = df_demo["P_gas_phys_kPa"].to_numpy()
    p_mech_arr = df_demo["P_mech"].to_numpy()
    p_soft_arr = df_demo["P_soft"].to_numpy()
    
    t_core_true_arr = df_demo["T_core_phys_degC"].to_numpy()
    t_core_est_arr  = df_demo["T_core_soft"].to_numpy()
    t_surf_arr      = df_demo["T_surf_degC"].to_numpy()
    
    data_len = len(t_arr)
    
    # 实例化诊断算法 (参数与 C 代码一致)
    # 20kPa 触发, 15kPa 恢复, 3.0s 确认
    diag_algo = MCUDiagnostic(alpha=0.2, th_on=20.0, th_off=15.0, time_confirm=3.0)

    # --- 绘图配置 ---
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # 子图1: 压力 (Pressure)
    ln_p_true, = ax1.plot([], [], 'k-', lw=2.5, alpha=0.8, label="True Physics (P)")
    ln_p_mech, = ax1.plot([], [], 'g--', lw=2.0, label="Mech Est (Strain)")
    ln_p_soft, = ax1.plot([], [], 'r:',  lw=2.5, label="Soft Est (V/I/T)")
    
    ax1.set_ylabel("Pressure (kPa)", fontsize=12, fontweight='bold')
    ax1.set_title("Real-time Pressure Monitor & Diagnostics", fontsize=14)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(80, 320)

    # 子图2: 温度 (Temperature)
    ln_t_true, = ax2.plot([], [], 'r-', lw=2.0, alpha=0.6, label="Core Temp True")
    ln_t_est,  = ax2.plot([], [], 'm--', lw=2.0, label="Core Temp AI Est")
    ln_t_surf, = ax2.plot([], [], 'b-', lw=1.5, alpha=0.8, label="Surf Temp Meas")

    ax2.set_ylabel("Temperature (°C)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_title("Internal vs External Temperature", fontsize=12)
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(20, 55)

    # 状态显示框
    box_props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    text_status = ax1.text(0.02, 0.5, "System Init...", transform=ax1.transAxes, 
                           fontsize=11, verticalalignment='center', bbox=box_props, fontfamily='monospace')

    # 动画参数
    play_speed = 5 # 每帧跳过 5 个物理采样点 (加速显示)
    dt_phys = 1.0  # 物理采样周期 1.0s (原始数据 dt)
    
    def update(frame):
        # 当前播放进度索引
        curr_idx = min(frame * play_speed, data_len - 1)
        prev_idx = max(0, (frame - 1) * play_speed)
        
        # [Fix]: 逐点喂入诊断算法 (严格对齐 MCU 逻辑)
        # 虽然动画是跳帧画的，但诊断状态必须连续演变
        resid_flt = 0.0
        is_fault = False
        
        # 补齐被跳过的点，确保时间积分正确
        # 注意：这里我们假设上一帧结束到这一帧结束中间所有点都按 dt_phys 间隔输入
        for k in range(prev_idx, curr_idx + 1):
             resid_flt, is_fault = diag_algo.update(p_mech_arr[k], p_soft_arr[k], dt_phys)
        
        # 获取绘图数据切片
        view_slice = slice(0, curr_idx + 1)
        current_t = t_arr[view_slice]
        
        # 更新压力曲线
        ln_p_true.set_data(current_t, p_true_arr[view_slice])
        ln_p_mech.set_data(current_t, p_mech_arr[view_slice])
        ln_p_soft.set_data(current_t, p_soft_arr[view_slice])
        
        # 更新温度曲线
        ln_t_true.set_data(current_t, t_core_true_arr[view_slice])
        ln_t_est.set_data(current_t, t_core_est_arr[view_slice])
        ln_t_surf.set_data(current_t, t_surf_arr[view_slice])
        
        # 动态调整 X 轴
        ax1.set_xlim(0, max(100, t_arr[curr_idx] + 10))
        
        # 更新状态框
        status_str = "Status: [NORMAL]"
        color_code = "#d9f7be" # 浅绿
        
        if is_fault:
            status_str = "Status: [FAULT: SWELLING]"
            color_code = "#ffccc7" # 浅红
        
        info_text = (
            f"Time: {t_arr[curr_idx]:.1f} s\n"
            f"{'-'*20}\n"
            f"{status_str}\n"
            f"Flt Resid: {resid_flt:.1f} kPa\n"
            f"Conf Time: {diag_algo.fault_time_s:.1f} / {diag_algo.time_confirm:.1f} s\n"
            f"{'-'*20}\n"
            f"P_Mech: {p_mech_arr[curr_idx]:.1f} kPa\n"
            f"P_Soft: {p_soft_arr[curr_idx]:.1f} kPa\n"
            f"{'-'*20}\n"
            f"T_Core: {t_core_est_arr[curr_idx]:.1f} °C\n"
            f"T_Surf: {t_surf_arr[curr_idx]:.1f} °C"
        )
        text_status.set_text(info_text)
        text_status.set_bbox(dict(boxstyle='round', facecolor=color_code, alpha=0.9))
        
        return ln_p_true, ln_p_mech, ln_p_soft, ln_t_true, ln_t_est, ln_t_surf, text_status

    print(">> [Display] Starting Dashboard...")
    # interval=20ms, play_speed=5 -> 实际播放速度 250 FPS (5s物理时间/秒)
    ani = animation.FuncAnimation(fig, update, frames=data_len // play_speed, interval=20, blit=False)
    plt.show()

if __name__ == "__main__":
    main()