from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

@dataclass
class ModelConfig:
    window_sizes: List[int] = field(default_factory=lambda: [10, 30])
    use_derivatives: bool = True
    target_cols: List[str] = field(default_factory=lambda: ['T_core_phys_degC', 'P_gas_phys_kPa'])
    feature_cols: List[str] = field(default_factory=lambda: [
        'I_A', 'V_V', 'T_surf_degC', 'T_amb_degC', 'eps_eq', 'SOC'
    ])
    model_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    })

def split_by_group(
    df: pd.DataFrame,
    group_col: str = "scenario",
    test_ratio: float = 0.2,
    seed: int = 42,
    test_groups: Optional[List[str]] = None,
    shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if group_col not in df.columns:
        if 'cycle_index' in df.columns: group_col = 'cycle_index'
        else:
            n = int(len(df) * (1 - test_ratio))
            return df.iloc[:n].copy(), df.iloc[n:].copy()
            
    all_groups = np.sort(df[group_col].unique())
    
    if test_groups is not None:
        test_mask = df[group_col].isin(test_groups)
        return df[~test_mask].copy(), df[test_mask].copy()
    
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(all_groups)
    
    n_test = max(1, int(round(len(all_groups) * test_ratio)))
    test_group_names = all_groups[:n_test]
    
    test_mask = df[group_col].isin(test_group_names)
    return df[~test_mask].copy(), df[test_mask].copy()

class DataDrivenEstimator:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config if config else ModelConfig()
        self.model = None
        self._is_fitted = False
        self.feature_names_in_: List[str] = []
        if not _HAS_ML: warnings.warn("sklearn/xgboost not found.")

    def _feat_eng_group_logic(self, sub_df: pd.DataFrame) -> pd.DataFrame:
        df_feat = sub_df[self.config.feature_cols].copy()
        if 'T_surf_degC' in df_feat and 'T_amb_degC' in df_feat:
            df_feat['dT_surf_amb'] = df_feat['T_surf_degC'] - df_feat['T_amb_degC']
        
        cols = [c for c in ['I_A', 'V_V', 'T_surf_degC', 'eps_eq'] if c in df_feat]
        if self.config.use_derivatives:
            for c in cols: df_feat[f'd{c}'] = df_feat[c].diff()
        for w in self.config.window_sizes:
            for c in cols:
                r = df_feat[c].rolling(window=w, min_periods=1)
                df_feat[f'{c}_mean_{w}'] = r.mean()
                df_feat[f'{c}_std_{w}'] = r.std()
        return df_feat

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = [k for k in ['cell_id', 'cycle_index'] if k in df.columns]
        if keys: 
            # [Fix V5.1]: Explicitly set include_groups=False for Pandas 2.2+ compatibility
            try:
                X = df.groupby(keys, group_keys=False).apply(self._feat_eng_group_logic, include_groups=False)
            except TypeError:
                # Fallback for older Pandas versions that don't support include_groups
                X = df.groupby(keys, group_keys=False).apply(self._feat_eng_group_logic)
        else: 
            X = self._feat_eng_group_logic(df)
        
        # [Fix V5.0]: 科学的缺失值处理
        # 1. 线性插值处理中间断点 (保持物理连续性)
        X = X.interpolate(method='linear', limit_direction='both')
        
        # 2. 均值填充处理首尾缺失 (避免0值物理谬误)
        if X.isnull().values.any():
            X = X.fillna(X.mean())
            # 3. 兜底: 如果整列都是NaN (mean也是NaN), 只能填0
            X = X.fillna(0.0)
            
        return X.select_dtypes(include=[np.number])

    def fit(self, df_train: pd.DataFrame, min_samples_per_target: int = 30):
        if not _HAS_ML: raise ImportError("ML libraries missing.")
        
        targets = self.config.target_cols
        miss = [t for t in targets if t not in df_train.columns]
        if miss: raise ValueError(f"Targets missing: {miss}")

        df_valid = df_train.dropna(subset=targets).copy()
        if len(df_valid) < 50: raise ValueError(f"Not enough valid data ({len(df_valid)}).")
        
        counts = {t: df_valid[t].notna().sum() for t in targets}
        insufficient = {t: n for t, n in counts.items() if n < min_samples_per_target}
        if insufficient: raise ValueError(f"Insufficient samples per target: {insufficient}")

        X = self._feature_engineering(df_valid)
        
        variances = X.var()
        zero_cols = variances[variances < 1e-12].index.tolist()
        if zero_cols:
            print(f"[Info] Dropping zero-variance: {zero_cols}")
            X = X.drop(columns=zero_cols)
            
        if X.shape[1] == 0:
            raise ValueError("All features have zero variance. Check inputs.")

        self.feature_names_in_ = X.columns.tolist()
        y = df_valid[targets]
        
        est = XGBRegressor(**self.config.model_params)
        self.model = Pipeline([('s', StandardScaler()), ('m', MultiOutputRegressor(est))])
        
        print(f"Training SoftSensor on {len(X)} samples, {len(self.feature_names_in_)} features...")
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted: raise RuntimeError("Model not fitted.")
        X = self._feature_engineering(df)
        X = X.reindex(columns=self.feature_names_in_, fill_value=0.0) # 这里的0.0仅用于全新的缺失列
        y_pred = self.model.predict(X)
        df_out = df.copy()
        for i, c in enumerate(self.config.target_cols): df_out[f"{c}_pred"] = y_pred[:, i]
        return df_out

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        targets = self.config.target_cols
        if not all(c in df_test.columns for c in targets): return {}
        
        df_valid = df_test.dropna(subset=targets)
        if len(df_valid) == 0: return {}
        
        df_pred = self.predict(df_valid)
        metrics = {}
        for c in targets:
            y_t, y_p = df_valid[c], df_pred[f"{c}_pred"]
            metrics[f"{c}_MAE"] = mean_absolute_error(y_t, y_p)
            metrics[f"{c}_RMSE"] = np.sqrt(mean_squared_error(y_t, y_p))
            metrics[f"{c}_R2"] = r2_score(y_t, y_p)
        return metrics