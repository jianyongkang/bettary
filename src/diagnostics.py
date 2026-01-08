from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import warnings

@dataclass
class IndicatorConfig:
    dQ_min_for_slope: float = 0.1
    use_core_temp: bool = True

def compute_cycle_indicators(df_cycle: pd.DataFrame, cfg: IndicatorConfig) -> pd.Series:
    s = df_cycle.sort_index()
    get = lambda k: s.get(k, pd.Series(np.nan, index=s.index)).values
    
    P_mech = s.get("P_gas_mech_est", s.get("P_gas_phys_kPa", pd.Series(np.nan, index=s.index))).values
    P_soft = s.get("P_gas_soft_est", s.get("P_gas_phys_kPa_pred", pd.Series(np.nan, index=s.index))).values
    P_meas = get("P_gas_kPa")
    P_true = s.get("P_gas_phys_kPa", pd.Series(np.nan, index=s.index)).values
    
    ind = {}
    dQ = s["Q_Ah"].iloc[-1] - s["Q_Ah"].iloc[0] if len(s) > 1 else 0
    
    if abs(dQ) < cfg.dQ_min_for_slope:
        ind["deps_dQ"] = np.nan
    else:
        eps = get("eps_eq")
        ind["deps_dQ"] = (eps[-1]-eps[0])/dQ if len(eps)>1 else np.nan

    ind["eps_max"] = np.nanmax(get("eps_eq")) if len(s) else np.nan
    
    # 1. Physics Consistency (Mech vs Soft)
    if len(P_mech) == len(P_soft) and len(P_mech) > 0:
        ind["P_resid_consistency"] = np.sqrt(np.nanmean((P_mech - P_soft)**2))
        ind["P_mech_max"] = np.nanmax(P_mech)
        ind["P_soft_max"] = np.nanmax(P_soft)
    else:
        ind["P_resid_consistency"] = np.nan
        
    # 2. Measurement Consistency (Meas vs Soft)
    if not np.all(np.isnan(P_meas)) and len(P_soft) == len(P_meas):
        ind["P_resid_meas"] = np.sqrt(np.nanmean((P_meas - P_soft)**2))
    else:
        ind["P_resid_meas"] = np.nan

    # Truth Residual (Truth vs Soft) for visualization fallback when P_meas is missing
    if not np.all(np.isnan(P_true)) and len(P_soft) == len(P_true):
        ind["P_resid_truth"] = np.sqrt(np.nanmean((P_true - P_soft)**2))
    else:
        ind["P_resid_truth"] = np.nan

    ind["has_P_meas"] = 1.0 if not np.all(np.isnan(P_meas)) else 0.0

    return pd.Series(ind)

def build_indicator_table(df_all, cfg):
    rows = []
    for (cid, cyc), g in df_all.groupby(["cell_id", "cycle_index"]):
        r = compute_cycle_indicators(g, cfg)
        r["cell_id"], r["cycle_index"] = cid, cyc
        r["scenario"] = g["scenario"].iloc[0]
        r["is_anomaly"] = g["is_anomaly"].iloc[0]
        r["fault_type"] = g["fault_type"].iloc[0]
        rows.append(r)
    return pd.DataFrame(rows)

class DiagnosticModel:
    def __init__(self):
        self.iso = None
        self.clf = None
        self.feats = []
        self._valid = False 

    def fit_unsupervised(self, df_ind):
        df_norm = df_ind[~df_ind["is_anomaly"]].copy()
        if len(df_norm) < 3: 
            warnings.warn("Not enough normal samples. Diagnostics invalid.")
            self._valid = False
            return
        
        # [V4.5 Fix]: Manually select physics-relevant features only.
        # This reduces noise for small datasets compared to selecting all numeric columns.
        target_feats = ["P_resid_consistency", "P_resid_meas", "eps_max"]
        
        # Only use features that actually exist in the dataframe
        self.feats = [f for f in target_feats if f in df_norm.columns]
        
        if not self.feats:
            warnings.warn("No valid diagnostic features found.")
            self._valid = False
            return

        # Fill NaN with 0 (Crucial for P_resid_meas which is NaN for fleet cells)
        X = df_norm[self.feats].fillna(0.0).values
        
        self.iso = IsolationForest(contamination=0.01, random_state=42)
        self.iso.fit(X)
        scores = -self.iso.decision_function(X)
        
        # [V4.5 Fix]: Use 90% quantile to be more sensitive to anomalies in small batches
        self.th_warn = np.quantile(scores, 0.90) 
        self._valid = True

    def fit_supervised(self, df_ind):
        if not self._valid: return
        df_l = df_ind.dropna(subset=["is_anomaly"])
        if df_l.empty: return
        X = df_l[self.feats].fillna(0.0).values
        y = df_l["is_anomaly"].astype(int).values
        self.clf = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.clf.fit(X, y)

    def score_cycle(self, row_ind):
        if not self._valid: return {"risk": "unknown"}
        x = row_ind.reindex(self.feats).astype(float).fillna(0.0).values.reshape(1, -1)
        s = -self.iso.decision_function(x)[0]
        risk = "alarm" if s > self.th_warn else "normal"
        res = {"iso_score": s, "risk": risk}
        if self.clf: res["clf_prob"] = self.clf.predict_proba(x)[0, 1]
        return res
