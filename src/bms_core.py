"""
src/bms_core.py
[BMS Core Algorithms]
纯数学逻辑层，不依赖 pandas/sklearn，仅依赖 numpy/math。
可直接翻译为 C 语言 (bms_core.c)。
"""
import numpy as np

class SOCEstimator:
    """安时积分法 (Ampere-Hour Integration)"""
    def __init__(self, initial_soc=1.0, capacity_ah=5.0):
        self.soc = initial_soc
        self.cap = capacity_ah
    
    def update(self, i_a, dt):
        # SOC_k = SOC_{k-1} - (I * dt) / (3600 * Cap)
        # 物理限制: 0.0 <= SOC <= 1.0
        delta = (i_a * dt) / (3600.0 * self.cap)
        self.soc = float(np.clip(self.soc - delta, 0.0, 1.0))
        return self.soc

class SOHEstimator:
    """
    基于 OCV 观测的 SOH 估算器 (RLS 思想)
    SOH = C_curr / C_nom
    Update Rule: C_new = (1-k)*C_old + k*(dQ / dSOC_ocv)
    """
    def __init__(self, nominal_capacity=5.0):
        self.nominal_cap = nominal_capacity
        self.est_capacity = nominal_capacity
        self.soh = 100.0
        
        # 积分累加器
        self.acc_dq = 0.0
        self.acc_dsoc = 0.0
        
        # [Config] 标定参数
        self.min_valid_dsoc = 0.02  # 至少 2% SOC 变化才更新
        self.learning_rate = 0.2    # 滤波系数
        self.min_soh = 60.0
        self.max_soh = 100.0

    def update(self, i_a, dt, d_soc_obs):
        """
        d_soc_obs: 观测到的 SOC 变化量 (来自 OCV 查表，而非真值)
        """
        # 累积安时 (Ah)
        self.acc_dq += abs(i_a * dt / 3600.0)
        # 累积观测到的 SOC 变化
        self.acc_dsoc += abs(d_soc_obs)
        
        # 触发更新条件
        if self.acc_dsoc > self.min_valid_dsoc:
            # 防除零保护
            denom = max(self.acc_dsoc, 1e-6)
            inst_cap = self.acc_dq / denom
            
            # 鲁棒性钳位: 单次观测不应偏离当前估计太远 (e.g. +/- 50%)
            inst_cap = float(np.clip(inst_cap, 0.5 * self.est_capacity, 1.5 * self.est_capacity))
            
            # 递归更新
            self.est_capacity = (1 - self.learning_rate) * self.est_capacity + \
                                self.learning_rate * inst_cap
            
            # 计算 SOH 并限幅
            raw_soh = (self.est_capacity / self.nominal_cap) * 100.0
            self.soh = float(np.clip(raw_soh, self.min_soh, self.max_soh))
            
            # 重置累积器
            self.acc_dq = 0.0
            self.acc_dsoc = 0.0
            
        return self.soh