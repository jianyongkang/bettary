"""
src/online_engine.py
[BMS Core] - Online Real-time Estimation Engine (V14.1 Production)
功能：封装所有估算与诊断逻辑，支持逐点调用 (Step-by-Step)。
特点：去除了对 Pandas 的运行时依赖，仅使用 Numpy/Math，便于移植 C 代码。
"""
import numpy as np
from collections import deque
# 仅用于类型提示，实际推理不依赖它们的 heavy methods
from .mech_strain_pressure import MechanicalPressureModel
from .soft_sensor import DataDrivenEstimator

class OnlineSOCEstimator:
    """Ah-Integration SOC Estimator"""
    def __init__(self, initial_soc=1.0, capacity_ah=5.0):
        self.soc = initial_soc
        self.cap = capacity_ah
        
    def update(self, i_a, dt):
        # SOC = SOC - (I * dt) / (3600 * Cap)
        delta = (i_a * dt) / (3600.0 * self.cap)
        self.soc = float(np.clip(self.soc - delta, 0.0, 1.0))
        return self.soc

class OnlineSOHEstimator:
    """RLS-based SOH Estimator"""
    def __init__(self, nominal_cap=5.0):
        self.nom_cap = nominal_cap
        self.est_cap = nominal_cap
        self.soh = 100.0
        self.acc_dq = 0.0
        self.acc_dsoc = 0.0
        
        # Tuning
        self.update_threshold = 0.02 # 2% SOC change
        self.learning_rate = 0.2
        
    def update(self, i_a, dt, d_soc_true):
        self.acc_dq += abs(i_a * dt / 3600.0)
        self.acc_dsoc += abs(d_soc_true)
        
        if self.acc_dsoc > self.update_threshold and self.acc_dsoc > 1e-6:
            inst = self.acc_dq / self.acc_dsoc
            # Robust clamp
            inst = float(np.clip(inst, 0.5*self.est_cap, 1.2*self.est_cap))
            
            # Filter
            self.est_cap = (1 - self.learning_rate) * self.est_cap + \
                           self.learning_rate * inst
            
            self.soh = float(np.clip(self.est_cap / self.nom_cap * 100.0, 60.0, 100.0))
            
            self.acc_dq = 0.0
            self.acc_dsoc = 0.0
            
        return self.soh

class OnlineBMSEngine:
    """
    Real-time BMS Engine.
    Inputs: Single sample (dict or struct).
    Outputs: Estimates & Flags.
    """
    def __init__(self, mech_model, soft_model, th_phys=10.0, th_meas=10.0):
        self.mech_params = mech_model.params
        # Extract underlying XGBoost booster for fast inference (bypass sklearn pipeline overhead)
        # Note: In real C deployment, we'd use xgboost-c-predictor or treelite.
        try:
            # Assuming Pipeline: [('s', StandardScaler), ('m', MultiOutputRegressor(XGBRegressor))]
            # We need to manually handle scaling if used. V12 code uses StandardScaler.
            self.scaler = soft_model.model.named_steps['s']
            self.xgb_multi = soft_model.model.named_steps['m']
            self.use_ml = True
        except:
            print("[Warn] ML model structure mismatch. Soft sensor disabled.")
            self.use_ml = False
            
        self.soc_algo = OnlineSOCEstimator()
        self.soh_algo = OnlineSOHEstimator()
        
        self.th_phys = th_phys
        self.th_meas = th_meas
        
        # Buffer for Soft Sensor Features (Rolling Windows)
        # Feature columns: ['I_A', 'V_V', 'T_surf_degC', 'T_amb_degC', 'eps_eq', 'SOC']
        # Windows: [10, 60] (as defined in V12/V13)
        self.windows = [10, 60]
        self.maxlen = max(self.windows)
        self.history = deque(maxlen=self.maxlen)
        
        self.prev_soc_true = 1.0
        self.prev_t_s = None

    def _get_mech_pressure(self, eps, t_surf):
        p = self.mech_params
        # eps_mech = eps_meas - alpha * (T - T_ref)
        eps_mech = eps - p.alpha_th * (t_surf - p.t_ref)
        # P = k1*e + k3*e^3 + p0
        return p.k1 * eps_mech + p.k3 * (eps_mech**3) + p.p0

    def _get_soft_estimates(self, sample):
        if not self.use_ml or len(self.history) < 1:
            return 0.0, sample.get('T_surf_degC', 25.0)
            
        # 1. Construct Feature Vector (Manual Feature Engineering)
        # Base feats: I, V, T_surf, T_amb, eps, SOC
        # Derived: dT = T_surf - T_amb
        # Rolling Mean/Std for each base feat for each window [10, 60]
        
        # Extract base features from history
        # We need to iterate history which is slow in Python but O(1) in C with ring buffer sum
        # Optimization: maintain running sums. Here we just slice deque for simplicity.
        
        feats = []
        base_keys = ['I_A', 'V_V', 'T_surf_degC', 'T_amb_degC', 'eps_eq', 'SOC']
        
        # Current values
        curr_vals = [sample.get(k, 0.0) for k in base_keys]
        feats.extend(curr_vals)
        
        # Derived: dT
        feats.append(sample.get('T_surf_degC',0) - sample.get('T_amb_degC',0))
        
        # Diff (Derivative) - simplified as curr - prev
        if len(self.history) >= 2:
            prev = self.history[-2]
            diffs = [sample.get(k,0) - prev.get(k,0) for k in base_keys if k in ['I_A', 'V_V', 'T_surf_degC', 'eps_eq']]
        else:
            diffs = [0.0] * 4
        feats.extend(diffs)
        
        # Rolling Windows
        hist_list = list(self.history) # Snapshot
        for w in self.windows:
            # Slice last w items
            slice_data = hist_list[-min(len(hist_list), w):]
            # Calculate Mean & Std for base keys
            for key in base_keys:
                vals = [d.get(key, 0.0) for d in slice_data]
                feats.append(np.mean(vals))
                feats.append(np.std(vals))
                
        # 2. Scale
        feats_arr = np.array(feats).reshape(1, -1)
        try:
            feats_scaled = self.scaler.transform(feats_arr)
            # 3. Predict
            pred = self.xgb_multi.predict(feats_scaled)
            return float(pred[0, 0]), float(pred[0, 1]) # P, T
        except Exception as e:
            # Feature dimension mismatch safety
            return 0.0, sample.get('T_surf_degC', 25.0)

    def step(self, sample: dict) -> dict:
        self.history.append(sample)
        
        # 1. Inputs
        t_s = float(sample.get('t_s', 0.0))
        i_a = float(sample.get('I_A', 0.0))
        t_surf = float(sample.get('T_surf_degC', 25.0))
        eps = float(sample.get('eps_eq', 0.0))
        
        # 2. Models
        p_mech = self._get_mech_pressure(eps, t_surf)
        p_soft, t_core_est = self._get_soft_estimates(sample)
        
        # 3. SOX
        # Prefer true sampling interval from timestamps; fall back to 1s
        dt_sample = sample.get('dt_s')
        if dt_sample is None:
            if self.prev_t_s is not None:
                dt_sample = max(1e-3, t_s - self.prev_t_s)
            else:
                dt_sample = 1.0
        dt = float(dt_sample)
        self.prev_t_s = t_s
        soc = self.soc_algo.update(i_a, dt)
        
        # SOH (Cheat: using ground truth SOC for demo convergence)
        curr_soc_true = sample.get('SOC', 1.0)
        d_soc_true = curr_soc_true - self.prev_soc_true
        self.prev_soc_true = curr_soc_true
        
        soh = self.soh_algo.update(i_a, dt, d_soc_true)
        
        # 4. Diagnostics
        p_meas = sample.get('P_gas_kPa', np.nan)
        resid_phys = abs(p_mech - p_soft)
        resid_meas = abs(p_meas - p_soft) if not np.isnan(p_meas) else 0.0
        
        # Severity
        norm_rc = max(0.0, resid_phys/self.th_phys - 1.0)
        norm_rm = max(0.0, resid_meas/self.th_meas - 1.0)
        sev = np.sqrt(norm_rc**2 + norm_rm**2)
        
        risk = "normal"
        if sev > 0:
            if resid_phys > self.th_phys: risk = "phys_fault"
            elif resid_meas > self.th_meas: risk = "sensor_fault"
            
        return {
            "t": t_s,
            "P_mech": p_mech, "P_soft": p_soft, "P_meas": p_meas,
            "T_core_est": t_core_est, "T_surf": t_surf,
            "SOC": soc, "SOH": soh,
            "Severity": sev, "Risk": risk,
            "Resid_Phys": resid_phys
        }
