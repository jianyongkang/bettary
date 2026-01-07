"""
plot_2d_scatter.py (V1.4)
绘制诊断指标的二维散点图（故障解耦），阈值自动基于健康数据标定。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fallback thresholds，当健康数据不足时使用
X_TH_DEFAULT = 10.0
Y_TH_DEFAULT = 10.0


def main() -> None:
    print("=== Plotting 2D Diagnosis Scatter (Auto-Calibrated) ===")

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
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # 3. 处理数据
    df["has_sensor"] = df["Resid_Meas (kPa)"].notna()
    df["Resid_Meas_Plot"] = df["Resid_Meas (kPa)"].fillna(0.0)

    # 健康样本（非校准、非老化、Type=none）自动标定阈值
    def _calc_th(col: str, fallback: float) -> float:
        healthy = df[
            (df["Type"] == "none")
            & (~df["Scenario"].str.contains("calib", case=False, na=False))
            & (~df["Scenario"].str.contains("aging", case=False, na=False))
        ]
        if healthy.empty or col not in df:
            return fallback
        vals = healthy[col].dropna()
        if vals.empty:
            return fallback
        mu = float(np.median(vals))
        mad = float(np.median(np.abs(vals - mu)) + 1e-6)
        th = mu + 6.0 * 1.4826 * mad
        return float(np.clip(th, 8.0, 25.0))

    x_th = _calc_th("Resid_Consistency (kPa)", X_TH_DEFAULT)
    y_th = _calc_th("Resid_Meas (kPa)", Y_TH_DEFAULT)

    plt.figure(figsize=(10, 7))

    # 4. 颜色与标签
    color_map = {'none': 'blue', 'overpressure': 'red', 'sensor_fault': 'orange'}
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
                label=label_map.get(c_type)
            )

        # B. 无传感器 (Fleet) -> 叉号 'X'
        d_fleet = d_sub[~d_sub["has_sensor"]]
        if not d_fleet.empty:
            plt.scatter(
                d_fleet["Resid_Consistency (kPa)"], d_fleet["Resid_Meas_Plot"],
                c=color_map.get(c_type, "gray"), marker="X", s=150, alpha=0.8,
                label=label_map.get(c_type)
            )

    # 6. 边界与区域
    plt.axvline(x_th, color="red", linestyle="--", alpha=0.3)
    plt.text(x_th + 1, y_th + 4, "Physical Fault Zone\n(High Consistency Residual)",
             color="red", fontsize=10, fontweight='bold')

    plt.axhline(y_th, color="orange", linestyle="--", alpha=0.3)
    plt.text(1, y_th + 2, "Sensor Fault Zone\n(High Meas Residual)",
             color="orange", fontsize=10, fontweight='bold')

    plt.text(1, 1, "Normal Zone", color="blue", fontsize=10, fontweight='bold')

    # 7. 标注关键点
    annotate_targets = [
        ("fleet_fault_op", "Overpressure\n(High Phys Residual)"),
        ("calib_fault_sensor", "Sensor Drift\n(High Meas Residual)")
    ]

    for scen_name, text in annotate_targets:
        row = df[df["Scenario"] == scen_name]
        if not row.empty:
            x = row.iloc[0]["Resid_Consistency (kPa)"]
            y = row.iloc[0]["Resid_Meas_Plot"]
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
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.grid(True, alpha=0.3)

    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join("outputs", "plot_metrics_scatter.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    main()
