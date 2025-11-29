"""
analyze_metrics.py
[论文专用] 生成关键诊断指标对比表。
目标：量化展示不同故障模式下，"物理一致性残差" 与 "测量残差" 的数量级差异。
"""
import os
import numpy as np
import pandas as pd

from src.thermal_model_rc import RCThermalParams
from src.mech_strain_pressure import MechParams, MechanicalPressureModel
from src.synthetic_data import NoiseConfig, SyntheticScenarioConfig, generate_synthetic_dataset
from src.soft_sensor import ModelConfig, DataDrivenEstimator
from src.diagnostics import IndicatorConfig, compute_cycle_indicators

def main() -> None:
    print("=== Metric Analysis for Thesis Table ===")

    # 1. 配置 (保持与 main_pipeline.py 一致)
    rc_p = RCThermalParams(800, 300, 1000, 2.0, 1.0, 0.5, 0.5)
    mech_p = MechParams(2e5, 1e8, 100, 23e-6)
    noise = NoiseConfig(outlier_prob=0.001, missing_prob=0.001)

    scens = []
    # Calib Scenarios
    for i in range(3): scens.append(SyntheticScenarioConfig(f"calib_{i}", 1800, is_calib=True))
    scens.append(SyntheticScenarioConfig("calib_fault_sensor", 1800, is_calib=True, anomaly_type="sensor_fault", anomaly_level="severe"))
    
    # Fleet Scenarios
    for i in range(5): scens.append(SyntheticScenarioConfig(f"fleet_norm_{i}", 1800, is_calib=False))
    scens.append(SyntheticScenarioConfig("fleet_fault_op", 1800, is_calib=False, anomaly_type="overpressure", anomaly_level="severe"))

    # 2. 生成数据
    print("Generating data...")
    df_all = generate_synthetic_dataset(scens, rc_p, mech_p, "cubic", 10.0, noise)

    # 3. 物理模型 (D2)
    print("Running D2 Mech Model...")
    df_calib_norm = df_all[(df_all["is_calib"]) & (df_all["fault_type"] == "none")].copy()
    mech_model = MechanicalPressureModel("cubic", alpha_thermal=23e-6)
    mech_model.fit(df_calib_norm)
    
    df_mech = mech_model.predict(df_all, out_col="P_gas_mech_est")
    df_all["P_gas_mech_est"] = df_mech["P_gas_mech_est"]

    # 4. 软传感器 (D3)
    print("Running D3 Soft Sensor...")
    est = DataDrivenEstimator(ModelConfig(target_cols=["P_gas_phys_kPa"]))
    est.fit(df_calib_norm)
    
    df_soft = est.predict(df_all)
    df_all["P_gas_soft_est"] = df_soft["P_gas_phys_kPa_pred"]

    # 5. 计算指标
    print("Calculating indicators...")
    cfg = IndicatorConfig()
    rows = []
    for (scen, cyc), g in df_all.groupby(["scenario", "cycle_index"]):
        ind = compute_cycle_indicators(g, cfg)
        rows.append({
            "Scenario": scen,
            "Type": g["fault_type"].iloc[0],
            # 物理一致性残差 (Mech - Soft) -> 越大越说明物理结构变了(鼓包)
            "Resid_Consistency (kPa)": ind.get("P_resid_consistency", np.nan),
            # 测量残差 (Meas - Soft) -> 越大越说明传感器变了(漂移)
            "Resid_Meas (kPa)": ind.get("P_resid_meas", np.nan),
        })

    # 排序以方便查看
    df_res = pd.DataFrame(rows).sort_values(["Type", "Scenario"])

    # 6. 打印 & 保存
    print("\n" + "=" * 80)
    print("   Critical Indicators for Diagnosis (Thesis Data)")
    print("=" * 80)
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.2f}" if pd.notnull(x) else "-"))
    print("=" * 80)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "diagnosis_indicators_table.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()