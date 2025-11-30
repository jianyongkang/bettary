from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diagnostics import IndicatorConfig, DiagnosticModel, build_indicator_table
from src.mech_strain_pressure import MechParams, MechanicalPressureModel
from src.soft_sensor import DataDrivenEstimator, ModelConfig, split_by_group
from src.synthetic_data import NoiseConfig, SyntheticScenarioConfig, generate_synthetic_dataset
from src.thermal_model_rc import RCThermalParams

OUTPUT_DIR = "outputs"
DEFAULT_SEED = 42


def build_default_scenarios() -> List[SyntheticScenarioConfig]:
    """Construct a repeatable set of calibration and fleet scenarios."""

    scenarios: List[SyntheticScenarioConfig] = []

    # Calibration cells (healthy + sensor fault)
    for i in range(3):
        scenarios.append(SyntheticScenarioConfig(f"calib_{i}", 1800, is_calib=True))
    scenarios.append(
        SyntheticScenarioConfig(
            "calib_fault_sensor",
            1800,
            is_calib=True,
            anomaly_type="sensor_fault",
            anomaly_level="severe",
        )
    )

    # Fleet cells (healthy + true overpressure)
    for i in range(5):
        scenarios.append(SyntheticScenarioConfig(f"fleet_norm_{i}", 1800, is_calib=False))
    scenarios.append(
        SyntheticScenarioConfig(
            "fleet_fault_op",
            1800,
            is_calib=False,
            anomaly_type="overpressure",
            anomaly_level="severe",
        )
    )
    return scenarios


def main() -> None:
    """End-to-end pipeline for synthetic data generation, training, and diagnostics."""

    # Enforce reproducibility for any implicit NumPy operations.
    np.random.seed(DEFAULT_SEED)  # Updated: centralize seeding to keep runs repeatable.

    print("=== [V4.3.3 Final Rigor] Pipeline Start ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Config
    rc_p = RCThermalParams(800, 300, 1000, 2.0, 1.0, 0.5, 0.5)
    mech_p = MechParams(2e5, 1e8, 100, 23e-6)
    noise = NoiseConfig(outlier_prob=0.001, missing_prob=0.001)

    # 2. Scenarios
    scenarios = build_default_scenarios()

    # 3. Generate Data
    print(">>> Generating Data...")
    df_all = generate_synthetic_dataset(scenarios, rc_p, mech_p, "cubic", 10.0, noise, base_seed=DEFAULT_SEED)
    print(f"    Shape: {df_all.shape}")

    # 4. Mech Model Fitting (D2)
    print(">>> D2: Mech Model Fitting...")
    df_calib_norm = df_all[(df_all["is_calib"]) & (df_all["fault_type"] == "none")].copy()
    if df_calib_norm.empty:
        raise RuntimeError("No healthy calibration data available for mechanical fitting.")

    mech_model = MechanicalPressureModel("cubic", alpha_thermal=23e-6)
    mech_model.fit(df_calib_norm)

    df_mech = mech_model.predict(df_all, out_col="P_gas_mech_est")
    df_all["P_gas_mech_est"] = df_mech["P_gas_mech_est"]

    # 5. Soft Sensor Training (D3)
    print(">>> D3: Soft Sensor Training...")
    est = DataDrivenEstimator(ModelConfig(target_cols=["P_gas_phys_kPa"]))

    # [Fix]: shuffle=False for reproducibility in papers
    df_calib_train, df_calib_test = split_by_group(
        df_calib_norm, group_col="scenario", test_ratio=0.3, seed=DEFAULT_SEED, shuffle=False
    )

    est.fit(df_calib_train)

    metrics = est.evaluate(df_calib_test)
    print(f"    [Eval] Soft Sensor Metrics: {metrics}")

    r2_val = metrics.get("P_gas_phys_kPa_R2", 0)
    if r2_val < 0.9:
        print(f"    [WARN] Soft Sensor R2 ({r2_val:.3f}) < 0.9. Check features or params.")

    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, "soft_sensor_metrics.csv"), index=False)

    df_soft = est.predict(df_all)
    df_all["P_gas_soft_est"] = df_soft["P_gas_phys_kPa_pred"]

    # 6. Diagnostics (D4)
    print(">>> D4: Diagnostics...")
    df_ind = build_indicator_table(df_all, IndicatorConfig())

    diag = DiagnosticModel()
    diag.fit_unsupervised(df_ind)
    if not diag._valid:
        print("    [WARN] Diagnostics model not valid; scores will be marked unknown.")

    scores = []
    risks = []
    for _, row in df_ind.iterrows():
        res = diag.score_cycle(row)
        scores.append(res.get("iso_score", 0))
        risks.append(res.get("risk", "unknown"))

    df_ind["score"] = scores
    df_ind["pred_risk"] = risks

    # 7. Visualization
    print(">>> Plotting...")
    plot_comparison(
        df_all,
        "fleet_fault_op",
        os.path.join(OUTPUT_DIR, "plot_fault_overpressure.png"),
        "True Overpressure: Mech Est follows True P, Soft Est underestimates the rise.",
    )

    plot_comparison(
        df_all,
        "calib_fault_sensor",
        os.path.join(OUTPUT_DIR, "plot_fault_sensor.png"),
        "Sensor Drift: Only Meas P deviates. True P, Mech & Soft Est consistent.",
    )

    plot_risk_bar(df_ind, os.path.join(OUTPUT_DIR, "risk_score.png"))

    print(f"=== Finished. Check '{OUTPUT_DIR}/' ===")


def plot_risk_bar(df_ind: pd.DataFrame, path: str) -> None:
    """Plot Isolation Forest scores with alarm coloring."""

    plt.figure(figsize=(10, 5))
    colors = ["r" if r == "alarm" else "b" for r in df_ind["pred_risk"]]
    plt.bar(df_ind["scenario"], df_ind["score"], color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Anomaly Score (IF)")
    plt.title("Fleet Health Status (Colored by Model Prediction)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_comparison(df: pd.DataFrame, scen_name: str, path: str, note: str = "") -> None:
    """Plot a per-scenario pressure comparison."""

    d = df[df["scenario"] == scen_name]
    if d.empty:
        print(f"    [WARN] Scenario '{scen_name}' not found; skipping plot.")
        return

    t = d.index

    plt.figure(figsize=(10, 6))
    plt.plot(t, d["P_gas_phys_kPa"], "k-", lw=2, label="True P")

    if d["is_calib"].iloc[0]:
        plt.plot(t, d["P_gas_kPa"], "c.", ms=3, alpha=0.5, label="Meas P")

    plt.plot(t, d["P_gas_mech_est"], "g--", lw=1.5, label="Mech Est")
    plt.plot(t, d["P_gas_soft_est"], "r:", lw=1.5, label="Soft Est")

    plt.title(f"Scenario: {scen_name}\n{note}", fontsize=10)
    plt.xlabel("Time (s)")
    plt.ylabel("Internal Pressure (kPa)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
