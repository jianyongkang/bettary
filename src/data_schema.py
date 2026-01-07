from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class ColumnMeta:
    name: str
    symbol: str
    desc_zh: str
    unit: str
    is_measured: bool


COLUMN_SCHEMA: Dict[str, ColumnMeta] = {
    "t_s": ColumnMeta("t_s", "t", "time", "s", False),
    "I_A": ColumnMeta("I_A", "I(t)", "current (discharge>0)", "A", True),
    "V_V": ColumnMeta("V_V", "V(t)", "terminal voltage", "V", True),
    "Q_Ah": ColumnMeta("Q_Ah", "Q(t)", "cumulative charge", "Ah", False),
    "SOC": ColumnMeta("SOC", "SOC(t)", "state of charge", "-", False),
    "T_amb_degC": ColumnMeta("T_amb_degC", "T_amb(t)", "ambient temperature", "degC", True),
    "T_surf_degC": ColumnMeta("T_surf_degC", "T_surf(t)", "surface temperature", "degC", True),
    "eps_eq": ColumnMeta("eps_eq", "eps_eq(t)", "equivalent strain", "-", True),
    "T_in_degC": ColumnMeta("T_in_degC", "T_in_meas(t)", "internal temp (measured/noisy)", "degC", True),
    "P_gas_kPa": ColumnMeta("P_gas_kPa", "P_gas_meas(t)", "gas pressure (measured/noisy)", "kPa", True),
    "T_core_phys_degC": ColumnMeta("T_core_phys_degC", "T_core_true(t)", "core temp (true)", "degC", False),
    "P_gas_phys_kPa": ColumnMeta("P_gas_phys_kPa", "P_gas_true(t)", "gas pressure (true)", "kPa", False),
    "cell_id": ColumnMeta("cell_id", "-", "cell id", "-", False),
    "cycle_index": ColumnMeta("cycle_index", "-", "cycle index", "-", False),
    "step_index": ColumnMeta("step_index", "-", "step index", "-", False),
    "is_calib": ColumnMeta("is_calib", "-", "is calibration cell", "-", False),
    "state_tag": ColumnMeta("state_tag", "-", "state tag", "-", False),
    "fault_type": ColumnMeta("fault_type", "-", "fault type", "-", False),
    "scenario": ColumnMeta("scenario", "-", "scenario name", "-", False),
    "is_anomaly": ColumnMeta("is_anomaly", "-", "is anomaly", "-", False),
}
