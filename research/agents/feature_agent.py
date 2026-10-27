"""
Agent 1 -- Feature Extraction Agent

Role: turn sparse, irregularly-sampled raw BMS readings (multiple rows per
charge cycle) into one engineered feature vector per cycle, per vehicle.
This is the interface between raw telemetry and every downstream agent.
"""
import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "avg_current", "max_current", "avg_voltage", "min_voltage",
    "avg_temp", "max_temp", "fast_charge_flag", "ah_throughput",
    "soc_swing", "n_readings",
]


def extract_cycle_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse sparse per-reading rows into one row per (Vehicle_ID, Cycle_Number),
    with the true SoH label carried through as the training target.
    """
    grouped = raw_df.groupby(["Vehicle_ID", "Cycle_Number"])

    records = []
    for (vehicle_id, cycle_num), g in grouped:
        records.append({
            "Vehicle_ID": vehicle_id,
            "Cycle_Number": cycle_num,
            "avg_current": g["Current_measured"].abs().mean(),
            "max_current": g["Current_measured"].abs().max(),
            "avg_voltage": g["Voltage_measured"].mean(),
            "min_voltage": g["Voltage_measured"].min(),
            "avg_temp": g["Ambient_Temperature"].mean(),
            "max_temp": g["Ambient_Temperature"].max(),
            "fast_charge_flag": int((g["Cycle_Type"] == "fast_charge").any()),
            "ah_throughput": g["Ah"].abs().sum(),
            "soc_swing": g["soc"].max() - g["soc"].min(),
            "n_readings": len(g),
            "SoH_true": g["SoH_true"].iloc[-1],
        })

    feat_df = pd.DataFrame(records).sort_values(["Vehicle_ID", "Cycle_Number"]).reset_index(drop=True)

    # rolling/cumulative signals -- degradation is a trend, not a single reading
    feat_df["cum_ah_throughput"] = feat_df.groupby("Vehicle_ID")["ah_throughput"].cumsum()
    feat_df["rolling_avg_temp_5"] = (
        feat_df.groupby("Vehicle_ID")["avg_temp"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    feat_df["fast_charge_ratio_10"] = (
        feat_df.groupby("Vehicle_ID")["fast_charge_flag"]
        .transform(lambda s: s.rolling(10, min_periods=1).mean())
    )
    return feat_df


FULL_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "cum_ah_throughput", "rolling_avg_temp_5", "fast_charge_ratio_10"
]
