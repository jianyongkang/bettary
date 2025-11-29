"""
plot_2d_scatter.py (V1.3 Final Paper Version)
绘制诊断指标的二维散点图，展示故障解耦。
[Update]: 
1. 术语统一: Resid -> Residual
2. 图例简化: 去除冗余的 (Calib)/(Fleet) 后缀
3. 标注优化: 文本对齐
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# [CONFIG] Visualization Thresholds
X_TH_CONSISTENCY = 10.0
Y_TH_MEASURE = 10.0

def main() -> None:
    print("=== Plotting 2D Diagnosis Scatter (Final Paper Version) ===")

    # 1. 读取数据
    csv_filename = "diagnosis_indicators_table.csv"
    if os.path.exists(csv_filename):
        csv_path = csv_filename
    elif os.path.exists(os.path.join("outputs", csv_filename)):
        csv_path = os.path.join("outputs", csv_filename)
    else:
        raise FileNotFoundError(f"Cannot find {csv_filename}. Run 'analyze_metrics.py' first.")

    df = pd.read_csv(csv_path)

    # 2. 列名校验
    required_cols = ["Scenario", "Type", "Resid_Consistency (kPa)", "Resid_Meas (kPa)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing: raise ValueError(f"CSV missing columns: {missing}")

    # 3. 处理数据
    df["has_sensor"] = df["Resid_Meas (kPa)"].notna()
    df["Resid_Meas_Plot"] = df["Resid_Meas (kPa)"].fillna(0.0)

    plt.figure(figsize=(10, 7))

    # 4. 颜色与标签 (V1.3 优化图例文字)
    color_map = {'none': 'blue', 'overpressure': 'red', 'sensor_fault': 'orange'}
    
    # 简化图例：形状已区分Calib/Fleet，文字只描述工况
    label_map = {
        'none': 'Normal',
        'overpressure': 'Overpressure (Physical)',
        'sensor_fault': 'Sensor Drift (Sensor)'
    }

    # 5. 分组绘制散点
    for c_type in df["Type"].unique():
        d_sub = df[df["Type"] == c_type]
        
        # A. 有传感器 (Calib) -> 圆点 'o'
        d_calib = d_sub[d_sub["has_sensor"]]
        if not d_calib.empty:
            plt.scatter(
                d_calib["Resid_Consistency (kPa)"], d_calib["Resid_Meas_Plot"],
                c=color_map.get(c_type, "gray"), marker="o", s=150, alpha=0.8, edgecolors="k",
                label=label_map.get(c_type)  # 简化 Label
            )
            
        # B. 无传感器 (Fleet) -> 叉号 'X'
        d_fleet = d_sub[~d_sub["has_sensor"]]
        if not d_fleet.empty:
            # 这里的 Label 逻辑：如果同类型已有 Calib 画过(如Normal)，为了避免图例重复，
            # 这一行可以设 label=None，或者在图例中接受重复。
            # 为了严谨，保留 label，matplotlib 会自动处理重复图例或者显示两个。
            # 这里的策略是：保留 label，让读者看到 Circle 和 X 代表同一种工况的不同来源。
            plt.scatter(
                d_fleet["Resid_Consistency (kPa)"], d_fleet["Resid_Meas_Plot"],
                c=color_map.get(c_type, "gray"), marker="X", s=150, alpha=0.8,
                label=label_map.get(c_type) 
            )

    # 6. 绘制边界与区域 (V1.3 术语统一 Resid -> Residual)
    plt.axvline(x=X_TH_CONSISTENCY, color="red", linestyle="--", alpha=0.3)
    plt.text(X_TH_CONSISTENCY + 2, Y_TH_MEASURE + 8, "Physical Fault Zone\n(High Consistency Residual)", color="red", fontsize=10, fontweight='bold')
    
    plt.axhline(y=Y_TH_MEASURE, color="orange", linestyle="--", alpha=0.3)
    plt.text(1, Y_TH_MEASURE + 2, "Sensor Fault Zone\n(High Meas Residual)", color="orange", fontsize=10, fontweight='bold')
    
    plt.text(1, 1, "Normal Zone", color="blue", fontsize=10, fontweight='bold')

    # 7. 关键点标注 (Annotation)
    annotate_targets = [
        ("fleet_fault_op", "Overpressure\n(High Phys Residual)"),
        ("calib_fault_sensor", "Sensor Drift\n(High Meas Residual)")
    ]
    
    for scen_name, text in annotate_targets:
        row = df[df["Scenario"] == scen_name]
        if not row.empty:
            x = row.iloc[0]["Resid_Consistency (kPa)"]
            y = row.iloc[0]["Resid_Meas_Plot"]
            
            # 动态调整文本位置
            xytext = (x - 15, y + 5) if x > 20 else (x + 5, y + 5)
            
            plt.annotate(
                text, xy=(x, y), xytext=xytext,
                arrowprops=dict(facecolor='black', arrowstyle="->", alpha=0.6),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )

    # 8. 装饰 & 保存
    plt.xlabel("Physics Consistency Residual (Mech - Soft) [kPa]", fontsize=12)
    plt.ylabel("Measurement Residual (Meas - Soft) [kPa]", fontsize=12)
    plt.title("Fault Decoupling Map: Physics vs Measurement Residuals", fontsize=14)
    
    # 自动去重图例 (处理同名 Label 但不同 Marker 的情况)
    # Matplotlib 默认行为通常是可以的，这里显式处理一下 handles
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # 这会去重，可能会丢失 Marker 信息
    # 既然我们要展示形状区别，最好保留所有 unique 的 (label, marker) 组合
    # 简单起见，直接调用默认 legend，通常效果最好
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    
    plt.grid(True, alpha=0.3)

    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join("outputs", "plot_metrics_scatter.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Paper-ready plot saved to {out_file}")

if __name__ == "__main__":
    main()