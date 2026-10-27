"""
Real-data loader

Turns a raw charge/discharge log (the schema of
charging_and_discharging_dataset.csv: Cycle_Number, Cycle_Type, Time,
Ambient_Temperature, Delta t, Current_measured, Voltage_measured, Current_Mod,
cummulative Ah, Ah, soc) into one labelled row per discharge cycle.

Label
  capacity_ah = charge delivered over the full discharge (last "cummulative Ah")
  SoH_true    = capacity_ah / capacity of the cell's first discharge

Features -- deliberately things a BMS can measure WITHOUT a full discharge,
otherwise the model would just be reading the label back:
  * from the charge that precedes the discharge (constant-current /
    constant-voltage protocol): CC/CV phase durations and a partial voltage
    window (3.9 V -> 4.1 V) that a car passes through on almost any top-up
  * from the first 300 s of the discharge only: loaded-voltage sag, which
    tracks internal-resistance growth
  * usage history from earlier cycles: equivalent full cycles, cumulative Ah
"""
from pathlib import Path

import numpy as np
import pandas as pd

CC_CURRENT_MIN = 1.0      # A, charge current above this = constant-current phase
CV_VOLTAGE_MIN = 4.19     # V, charge voltage above this = constant-voltage phase
CV_END_CURRENT = 0.05     # A, charge considered finished below this
WINDOW_V = (3.9, 4.1)     # partial charging window
EARLY_DISCHARGE_S = 300   # seconds of discharge visible to the model
MIN_CHARGE_ROWS = 50      # shorter charge logs are truncated / aborted

REAL_FEATURE_COLUMNS = [
    "cc_duration_s",
    "cv_duration_s",
    "window_time_s",
    "window_ah",
    "charge_start_voltage",
    "early_discharge_v_drop",
    "early_discharge_v_mean",
    "equiv_full_cycles",
    "cum_ah_throughput",
]


def _norm_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _charge_features(g):
    t = g["Time"].to_numpy()
    i = g["Current_measured"].to_numpy()
    v = g["Voltage_measured"].to_numpy()
    dt = g["Delta t"].to_numpy()

    cc = i > CC_CURRENT_MIN
    cv = (v >= CV_VOLTAGE_MIN) & (i > CV_END_CURRENT)
    in_window = cc & (v >= WINDOW_V[0]) & (v <= WINDOW_V[1])

    return {
        "cc_duration_s": float(dt[cc].sum()),
        "cv_duration_s": float(dt[cv].sum()),
        "window_time_s": float(dt[in_window].sum()),
        "window_ah": float((i[in_window] * dt[in_window]).sum() / 3600),
        "charge_start_voltage": float(v[:3].mean()),
        "charge_ah": float((np.clip(i, 0, None) * dt).sum() / 3600),
        # the charge must start below the window, otherwise the window is cut short
        "charge_ok": bool(len(g) >= MIN_CHARGE_ROWS and in_window.any() and v[:3].mean() < WINDOW_V[0]),
        "charge_end_time": float(t[-1]),
    }


def _discharge_features(g):
    t = g["Time"].to_numpy()
    v = g["Voltage_measured"].to_numpy()
    i = g["Current_measured"].to_numpy()

    loaded = i < -CC_CURRENT_MIN
    early = loaded & (t <= t[loaded][0] + EARLY_DISCHARGE_S) if loaded.any() else loaded
    v_early = v[early]
    return {
        "early_discharge_v_drop": float(v_early[0] - v_early[-1]) if len(v_early) > 1 else np.nan,
        "early_discharge_v_mean": float(v_early.mean()) if len(v_early) else np.nan,
        "capacity_ah": float(g["cummulative Ah"].iloc[-1]),
    }


def load_real_cell(csv_path, cell_id=None):
    """Return one labelled feature row per usable discharge cycle of one cell."""
    csv_path = Path(csv_path)
    cell_id = cell_id or csv_path.stem
    raw = _norm_columns(pd.read_csv(csv_path))
    raw["Cycle_Type"] = raw["Cycle_Type"].str.strip().str.lower()

    records = []
    last_charge = None
    cum_ah = 0.0
    rated = None  # SoH reference = capacity of the very first discharge
    for cycle_num, g in raw.groupby("Cycle_Number", sort=True):
        ctype = g["Cycle_Type"].iloc[0]
        if ctype == "charge":
            last_charge = _charge_features(g)
            cum_ah += last_charge["charge_ah"]
        elif ctype == "discharge":
            d = _discharge_features(g)
            if rated is None:
                rated = d["capacity_ah"]
            if last_charge is not None and last_charge["charge_ok"]:
                records.append({
                    "Vehicle_ID": cell_id,
                    "Cycle_Number": int(cycle_num),
                    **{k: last_charge[k] for k in REAL_FEATURE_COLUMNS[:5]},
                    "early_discharge_v_drop": d["early_discharge_v_drop"],
                    "early_discharge_v_mean": d["early_discharge_v_mean"],
                    "cum_ah_throughput": cum_ah,       # history BEFORE this discharge
                    "capacity_ah": d["capacity_ah"],
                })
            cum_ah += d["capacity_ah"]
            last_charge = None  # each charge pairs with at most one discharge

    df = pd.DataFrame(records).dropna().reset_index(drop=True)
    df["equiv_full_cycles"] = df["cum_ah_throughput"] / (2 * rated)
    df["SoH_true"] = df["capacity_ah"] / rated
    return df


def load_real_cells(csv_paths):
    """Each CSV is one battery = one federated client."""
    ids = ["real_cell" if Path(p).stem == "charging_and_discharging_dataset" else None
           for p in csv_paths]
    return pd.concat([load_real_cell(p, i) for p, i in zip(csv_paths, ids)], ignore_index=True)


if __name__ == "__main__":
    import sys
    df = load_real_cell(sys.argv[1] if len(sys.argv) > 1 else
                        Path(__file__).resolve().parent.parent / "charging_and_discharging_dataset.csv")
    pd.set_option("display.width", 200)
    print(df.iloc[::15].to_string())
    print(len(df), "usable discharge cycles")
    print(df[REAL_FEATURE_COLUMNS + ["SoH_true"]].corr()["SoH_true"].round(3))
