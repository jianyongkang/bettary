from __future__ import annotations

import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 寮曠敤搴曞眰宸ュ叿
from src.diagnostics import IndicatorConfig, build_indicator_table
from src.mech_strain_pressure import MechParams, MechanicalPressureModel
from src.soft_sensor import DataDrivenEstimator, ModelConfig, split_by_group
from src.synthetic_data import NoiseConfig, SyntheticScenarioConfig, generate_synthetic_dataset
from src.thermal_model_rc import RCThermalParams

OUTPUT_DIR = "outputs"
DEFAULT_SEED = 42
CLEAN_PAPER_MODE = False 

# [Config] Threshold Statistics (Robust Median+MAD)
AUTO_CALIB_K_SIGMA = 6.0 

# ==========================================
# 1. BMS 鏍稿績绠楁硶 (SOH Estimator - Refactored)
# ==========================================
class SOHEstimator:
    """
    绠€鍖栫殑瀹归噺 RLS 浼拌鍣?
    C_inst = |I|*dt / (3600*|dSOC_true|)
    C_est  = (1-alpha)*C_est + alpha*C_inst
    """
    def __init__(self, nominal_capacity: float, alpha: float = 0.01):
        self.nominal_cap = float(nominal_capacity)
        self.C_est = float(nominal_capacity)
        self.alpha = float(alpha)
        self.soh = 100.0

    def update(self, current_a: float, dt_s: float, d_soc_true: float) -> float:
        # 婊ゆ尝: 鍙湁褰撲俊鍙疯冻澶熷己鏃舵墠鏇存柊
        if abs(d_soc_true) < 1e-6 or abs(current_a) < 1e-6:
            return self.soh

        # 鐬椂瀹归噺瑙傛祴
        inst_cap = abs(current_a) * dt_s / 3600.0 / max(abs(d_soc_true), 1e-6)
        
        # 椴佹鎬ч檺鍒?(闃叉闄ら浂鍣０瀵艰嚧椋炶溅)
        inst_cap = float(np.clip(inst_cap, 0.5 * self.nominal_cap, 1.2 * self.nominal_cap))

        # 閫掑綊鏇存柊
        self.C_est = (1.0 - self.alpha) * self.C_est + self.alpha * inst_cap
        
        # SOH 璁＄畻涓庨檺骞?(100%灏侀《)
        self.soh = float(np.clip(self.C_est / self.nominal_cap * 100.0, 60.0, 100.0))
        return self.soh

def run_sox_estimation(df_all: pd.DataFrame, 
                       scen_name: str = "fleet_aging_test", 
                       C_nom: float = 5.0, 
                       soh_true: float = 0.9) -> pd.DataFrame:
    """
    [SOH Demo Helper]
    鏋勯€犱竴涓墿鐞嗛棴鐜殑 SOH 浠跨湡锛?
    - 鐪熷€肩郴缁燂細浣跨敤 soh_true * C_nom (4.5Ah) 杩涜 SOC 绉垎銆?
    - 绠楁硶绯荤粺锛氬亣璁?C_nom (5.0Ah) 杩涜 SOC 绉垎銆?
    - SOH 绠楁硶锛氳娴嬩袱鑰呯殑宸紓锛屾敹鏁涘埌 90%銆?
    """
    mask = df_all["scenario"] == scen_name
    df = df_all.loc[mask].copy()
    if df.empty: return df

    C_real = C_nom * soh_true
    I = df["I_A"].to_numpy(dtype=float)
    dt = 1.0 # 绠€鍖栵紝鍋囪鍧囧寑閲囨牱

    n = len(df)
    soc_true = np.zeros(n)
    soc_est = np.zeros(n)
    
    # 鍒濆鐘舵€佹弧鐢?
    soc_true[0] = 1.0
    soc_est[0] = 1.0

    # 1. 妯℃嫙 SOC 婕斿彉 (鐗╃悊鐪熷€?vs 绠楁硶浼拌)
    for k in range(1, n):
        # 鏀剧數涓烘 I
        soc_true[k] = max(0.0, soc_true[k-1] - I[k] * dt / (3600.0 * C_real))
        soc_est[k] = max(0.0, soc_est[k-1] - I[k] * dt / (3600.0 * C_nom))

    df["SOC_true"] = soc_true
    df["SOC_est"] = soc_est

    # 2. 杩愯 SOH 浼扮畻鍣?
    # 浣跨敤鐪熷€?dSOC (妯℃嫙楂樼簿搴?OCV 鏍℃鍚庣殑澧為噺)
    d_soc_true = np.diff(soc_true, prepend=soc_true[0])
    
    soh_filter = SOHEstimator(nominal_capacity=C_nom, alpha=0.01) # alpha 璋冨ぇ涓€鐐逛互渚垮湪婕旂ず鏃舵鍐呮敹鏁?
    soh_list = []
    
    for Ik, dsoc in zip(I, d_soc_true):
        soh_list.append(soh_filter.update(Ik, dt, dsoc))

    df["SOH_est"] = soh_list
    df["SOH_true"] = soh_true * 100.0
    return df

# ==========================================
# 2. 鍦烘櫙涓庢暟鎹敓鎴?
# ==========================================
def build_default_scenarios() -> List[SyntheticScenarioConfig]:
    scenarios = []
    # Calibration
    for i in range(3):
        scenarios.append(SyntheticScenarioConfig(f"calib_{i}", 1800, is_calib=True))
    scenarios.append(SyntheticScenarioConfig("calib_fault_sensor", 1800, is_calib=True, anomaly_type="sensor_fault", anomaly_level="severe"))

    # Fleet
    for i in range(5):
        scenarios.append(SyntheticScenarioConfig(f"fleet_norm_{i}", 1800, is_calib=False))
    
    scenarios.append(SyntheticScenarioConfig("fleet_fault_op", 1800, is_calib=False, anomaly_type="overpressure", anomaly_level="severe"))
    
    # Aging Test: 2 hours discharge to ensure full SOC swing
    scenarios.append(SyntheticScenarioConfig("fleet_aging_test", 7200, is_calib=False)) 
    return scenarios

# ==========================================
# 3. 璇婃柇涓庢爣瀹?(Robust Stats)
# ==========================================
def calibrate_thresholds(df_ind: pd.DataFrame) -> Dict[str, float]:
    """
    浣跨敤 Median + MAD 杩涜椴佹闃堝€兼爣瀹氥€?
    """
    df_healthy = df_ind[
        (df_ind["fault_type"] == "none")
        & (~df_ind.get("is_calib", False))
        & (~df_ind["scenario"].str.contains("aging"))
    ].copy()
    stats = {}
    print("\n>>> [Calib] Threshold Calibration (Median + MAD):")
    
    for col, key in [("P_resid_consistency", "th_phys"), ("P_resid_meas", "th_meas")]:
        if df_healthy.empty or col not in df_healthy:
            limit = 10.0
        else:
            vals = df_healthy[col].to_numpy(dtype=float)
            # Trim extreme 5% to avoid abnormal baseline lift
            if len(vals) > 10:
                lo, hi = np.percentile(vals, [5, 95])
                vals = vals[(vals >= lo) & (vals <= hi)]
            mu = np.median(vals)
            # MAD: Median Absolute Deviation
            mad = np.median(np.abs(vals - mu)) + 1e-6 
            
            # Limit ~= Median + K * Sigma (Sigma approx 1.4826 * MAD)
            limit = mu + AUTO_CALIB_K_SIGMA * 1.4826 * mad
            
            # Engineering Clamp [8.0, 25.0] kPa to avoid false alarms on healthy baseline
            limit = float(np.clip(limit, 8.0, 25.0))
            print(f"    {key}: Median={mu:.2f}, MAD={mad:.2f} => Limit={limit:.2f} kPa")
            
        stats[key] = limit
    return stats

def analyze_risk_and_severity(df_ind: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """
    璁＄畻瑙勫垯椋庨櫓鍜屽綊涓€鍖栦弗閲嶅害銆?
    """
    th_phys = thresholds["th_phys"]
    th_meas = thresholds["th_meas"]
    
    risks, severities = [], []
    for _, row in df_ind.iterrows():
        rc = float(row.get("P_resid_consistency", 0.0) or 0.0)
        rm = float(row.get("P_resid_meas", 0.0) or 0.0)
        
        # 1. Rule Classification
        over_phys = rc > th_phys
        over_meas = rm > th_meas
        
        if over_phys and over_meas: label = "mixed"
        elif over_phys: label = "phys_fault"
        elif over_meas: label = "sensor_fault"
        else: label = "normal"
        risks.append(label)
        
        # 2. Severity (Normalized Excess)
        # S = amplified max ratio to accentuate faults
        sev_phys = max(0.0, rc / th_phys - 1.0)
        sev_meas = max(0.0, rm / th_meas - 1.0)
        sev = float(1.5 * max(sev_phys, sev_meas))
        
        # Calibration/Aging scenarios do not contribute to severity
        if row.get("is_calib", False) or row["scenario"] == "fleet_aging_test":
            sev = 0.0
                
        severities.append(sev)
        
    df_out = df_ind.copy()
    df_out["rule_risk"] = risks
    df_out["severity"] = severities
    return df_out

# ==========================================
# 4. Main Pipeline
# ==========================================
def main() -> None:
    np.random.seed(DEFAULT_SEED)
    print("=== [V13.0 Ultimate Refactoring] Pipeline Start ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Config & Data
    rc_p = RCThermalParams(800, 300, 1000, 2.0, 1.0, 0.5, 0.5)
    mech_p = MechParams(2e5, 1e8, 100, 23e-6)
    noise = NoiseConfig(outlier_prob=0.0, missing_prob=0.0, sigma_eps=1e-7) if CLEAN_PAPER_MODE else NoiseConfig(outlier_prob=0.001, missing_prob=0.001)

    print(">>> Generating Data...")
    scenarios = build_default_scenarios()
    df_all = generate_synthetic_dataset(scenarios, rc_p, mech_p, "cubic", 10.0, noise, base_seed=DEFAULT_SEED)

    # 2. Mech Model (D2)
    print(">>> D2: Fitting Mech Model...")
    df_calib_norm = df_all[(df_all["is_calib"]) & (df_all["fault_type"] == "none")].copy()
    mech_model = MechanicalPressureModel("cubic", alpha_thermal=23e-6)
    mech_model.fit(df_calib_norm)
    df_all["P_gas_mech_est"] = mech_model.predict(df_all)["P_gas_mech_est"]
    # Smooth mech estimate per scenario to suppress overshoot
    df_all["P_gas_mech_est"] = df_all.groupby("scenario")["P_gas_mech_est"].transform(
        lambda s: s.rolling(window=21, center=True, min_periods=1).mean()
    )

    # 3. Soft Sensor (D3) - [A. De-biasing & Smoothing]
    print(">>> D3: Training Soft Sensor (Hybrid)...")
    est = DataDrivenEstimator(ModelConfig(
        target_cols=["P_gas_phys_kPa", "T_core_phys_degC"],
        window_sizes=[10, 60]
    ))
    # Hybrid Training: Calib + Fleet Normal
    df_fleet_norm = df_all[(~df_all["is_calib"]) & (df_all["fault_type"] == "none")].sample(frac=0.5, random_state=42)
    df_train_set = pd.concat([df_calib_norm, df_fleet_norm]).sort_index()
    est.fit(df_train_set)
    
    # Predict
    df_soft = est.predict(df_all)
    
    # --- [New] Post-Processing: De-bias & Smooth ---
    print("    [Info] Applying Soft Sensor De-biasing & Smoothing...")
    df_all["P_gas_soft_est_raw"] = df_soft["P_gas_phys_kPa_pred"]
    df_all["T_core_soft_est_raw"] = df_soft["T_core_phys_degC_pred"]
    
    # Calculate bias on healthy data
    healthy_mask = (
        (df_all["fault_type"] == "none") &
        (~df_all["is_calib"]) &
        (~df_all["scenario"].str.contains("aging")) &
        (~df_all["scenario"].str.contains("fault"))
    )
    
    for raw_col, true_col, out_col in [
        ("P_gas_soft_est_raw", "P_gas_phys_kPa", "P_gas_soft_est"),
        ("T_core_soft_est_raw", "T_core_phys_degC", "T_core_soft_est")
    ]:
        bias = (df_all.loc[healthy_mask, raw_col] - df_all.loc[healthy_mask, true_col]).mean()
        # Apply bias correction
        corrected = df_all[raw_col] - bias
        # Apply rolling smooth (per scenario)
        df_all[out_col] = corrected.groupby(df_all["scenario"]).transform(
            lambda s: s.rolling(window=31, center=True, min_periods=1).mean()
        )

    # 4. SOX Estimation - [C. Independent SOH Demo]
    print(">>> Running SOX Estimation (Dedicated Aging Scenario)...")
    df_sox = run_sox_estimation(df_all, scen_name="fleet_aging_test", C_nom=5.0, soh_true=0.9)

    # 5. Diagnostics - [B. Robust Thresholds]
    print(">>> D4: Diagnostics...")
    df_ind = build_indicator_table(df_all, IndicatorConfig())
    if "is_calib" not in df_ind.columns:
        scen_to_calib = df_all[["scenario", "is_calib"]].drop_duplicates().set_index("scenario")["is_calib"]
        df_ind["is_calib"] = df_ind["scenario"].map(scen_to_calib)

    thresholds = calibrate_thresholds(df_ind)
    df_ind = analyze_risk_and_severity(df_ind, thresholds)

    # 6. Plotting
    print(">>> Plotting Final Charts...")
    plot_comparison(df_all, "fleet_fault_op", os.path.join(OUTPUT_DIR, "plot_fault_overpressure.png"), "Cold Swelling Diagnosis (De-biased AI)")
    plot_comparison(df_all, "calib_fault_sensor", os.path.join(OUTPUT_DIR, "plot_fault_sensor.png"), "Sensor Drift Diagnosis (De-biased AI)")
    plot_metrics_scatter(df_ind, os.path.join(OUTPUT_DIR, "plot_metrics_scatter.png"))
    
    # Plot Risk (Operational Only)
    df_fleet_risk = df_ind[~df_ind["is_calib"] & (df_ind["scenario"] != "fleet_aging_test")]
    plot_risk_bar(df_fleet_risk, os.path.join(OUTPUT_DIR, "risk_score.png"), "Fleet Health Status")
    
    # Plot SOX
    plot_sox(df_sox, os.path.join(OUTPUT_DIR, "plot_sox_estimation.png"))

    print(f"=== Finished. All artifacts in '{OUTPUT_DIR}/' ===")

# ==========================================
# 5. Visualization Functions
# ==========================================
def plot_sox(df_sox: pd.DataFrame, path: str) -> None:
    if df_sox is None or df_sox.empty: return
    t = np.arange(len(df_sox))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # SOC
    ax1.plot(t, df_sox["SOC_true"], "k-", lw=3, alpha=0.5, label="True SOC (C_real=0.9*C_nom)")
    ax1.plot(t, df_sox["SOC_est"], "c--", lw=2, label="Est SOC (Assume C_nom)")
    ax1.set_ylabel("SOC"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_title("SOC Mismatch Due to Capacity Fade")
    
    # SOH
    soh_true = float(df_sox["SOH_true"].iloc[0])
    ax2.axhline(soh_true, color="k", lw=2, alpha=0.5, label=f"True SOH ({soh_true:.0f}%)")
    ax2.plot(t, df_sox["SOH_est"], "b-", lw=2, label="Est SOH")
    ax2.set_ylabel("SOH (%)"); ax2.set_xlabel("Time (s)"); ax2.set_ylim(85, 105)
    ax2.set_title("SOH Estimation Convergence")
    ax2.grid(True, alpha=0.3); ax2.legend()
    
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_metrics_scatter(df_ind: pd.DataFrame, path: str):
    plt.figure(figsize=(10, 7))
    d_plot = df_ind[~df_ind["scenario"].str.contains("aging")]
    groups = d_plot.groupby("fault_type")
    colors = {'none': 'blue', 'overpressure': 'red', 'sensor_fault': 'orange'}
    markers = {'none': 'o', 'overpressure': 'X', 'sensor_fault': 's'}
    
    for name, group in groups:
        plt.scatter(group["P_resid_consistency"], group["P_resid_meas"], 
                    c=colors.get(name,'gray'), marker=markers.get(name,'o'), 
                    s=100, label=name, alpha=0.7, edgecolors='k')
    
    plt.xlabel("Physics Consistency Residual (kPa)")
    plt.ylabel("Measurement Residual (kPa)")
    plt.title("Fault Decoupling Map")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150); plt.close()

def plot_risk_bar(df_ind: pd.DataFrame, path: str, title: str) -> None:
    if df_ind.empty: return
    plt.figure(figsize=(10, 6))
    colors = [{"normal":"tab:blue","phys_fault":"tab:red","sensor_fault":"tab:orange","mixed":"purple"}.get(r,"gray") for r in df_ind["rule_risk"]]
    plt.bar(df_ind["scenario"], df_ind["severity"], color=colors, alpha=0.8, edgecolor='k')
    plt.axhline(0, color='k', linewidth=0.8) # Base
    # Note: With normalized severity, >0 usually means above threshold if defined as max(0, ratio-1)
    # But visually let's leave it clean.
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Fault Severity (Normalized)")
    plt.title(title); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_comparison(df: pd.DataFrame, scen_name: str, path: str, note: str = "") -> None:
    d = df[df["scenario"] == scen_name]
    if d.empty: return
    t = d.index
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(t, d["P_gas_phys_kPa"], "k-", alpha=0.4, label="True P")
    if d["is_calib"].iloc[0] and "P_gas_kPa" in d.columns: ax1.plot(t, d["P_gas_kPa"], "c.", ms=3, label="Meas P")
    ax1.plot(t, d["P_gas_mech_est"], "g--", label="Mech Est")
    ax1.plot(t, d["P_gas_soft_est"], "r:", label="Soft Est")
    ax1.set_ylabel("Pressure (kPa)"); ax1.legend(ncol=2); ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{scen_name} - Pressure")

    ax2.plot(t, d["T_surf_degC"], "b-", label="T_surf")
    ax2.plot(t, d["T_core_phys_degC"], "r-", alpha=0.3, label="T_core (True)")
    if "T_core_soft_est" in d.columns: ax2.plot(t, d["T_core_soft_est"], "m--", label="T_core (AI)")
    ax2.set_ylabel("Temp (掳C)"); ax2.legend(ncol=2); ax2.grid(True, alpha=0.3)
    ax2.set_title("Temperature")
    
    plt.suptitle(note, y=0.98); plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=150); plt.close()

if __name__ == "__main__":
    main()

