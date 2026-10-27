"""
Convert NASA PCoE Li-ion aging data (B0005, B0006, B0007, B0018, ...) into
the raw-log schema of charging_and_discharging_dataset.csv, one CSV per cell,
so each cell can be passed to orchestrator.py --csv as one FL client.

Source: NASA Prognostics Center of Excellence "Battery Data Set"
  https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip  (~210 MB)
  B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics
  Data Repository, NASA Ames Research Center, Moffett Field, CA.

Usage:
  python data/nasa_to_csv.py --zip "5. Battery Data Set.zip" --out data/nasa
  python data/nasa_to_csv.py --mat B0005.mat B0006.mat --out data/nasa

Output columns (same as charging_and_discharging_dataset.csv):
  Cycle_Number        1, 2, 3, ... over charge and discharge cycles in test order
                      (impedance cycles are dropped and not numbered)
  Cycle_Type          "charge" / "discharge"
  Time                seconds since the start of the cycle (NASA "Time")
  Ambient_Temperature deg C
  "Delta t "          Time[k] - Time[k-1] (0 for the first sample of a cycle);
                      note the trailing space, kept to match the original file
  Current_measured    A, + while charging, - while discharging
  Voltage_measured    V
  Current_Mod         -Current_measured on discharge rows, empty on charge rows
  cummulative Ah      running charge in / out within the cycle
  Ah                  charge moved in this sample: I * Delta t / 3600
                      (charge: Current_measured, discharge: Current_Mod)
  soc                 % of a 2.0 Ah nominal cell: 100*cumAh/2 when charging,
                      100 - 100*cumAh/2 when discharging

Samples with a missing time, current or voltage reading are dropped.

Note: charging_and_discharging_dataset.csv in this repo is NASA B0006 (same
readings, but its first sample of every cycle and last ~6 samples of every
discharge are missing, so its capacities are ~3 % low). Don't pass it together
with data/nasa/B0006.csv: that would be one cell counted as two clients.
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

NOMINAL_AH = 2.0
DEFAULT_CELLS = ["B0005", "B0006", "B0007", "B0018"]
COLUMNS = ["Cycle_Number", "Cycle_Type", "Time", "Ambient_Temperature", "Delta t ",
           "Current_measured", "Voltage_measured", "Current_Mod", "cummulative Ah", "Ah", "soc"]


def _cycle_frame(cycle_number, ctype, ambient, data):
    t = np.asarray(data.Time, dtype=float).ravel()
    i = np.asarray(data.Current_measured, dtype=float).ravel()
    v = np.asarray(data.Voltage_measured, dtype=float).ravel()
    # a few samples are sensor dropouts (NaN); left in, they poison every
    # running sum downstream (B0018 charge 91 would wipe out all later cycles)
    ok = np.isfinite(t) & np.isfinite(i) & np.isfinite(v)
    t, i, v = t[ok], i[ok], v[ok]
    dt =np.diff(t, prepend=t[0])

    if ctype == "charge":
        current_mod = np.full_like(i, np.nan)
        ah = i * dt / 3600
        cum = np.cumsum(ah)
        soc = 100 * cum / NOMINAL_AH
    else:
        current_mod = -i
        ah = current_mod * dt / 3600
        cum = np.cumsum(ah)
        soc = 100 - 100 * cum / NOMINAL_AH

    return pd.DataFrame({
        "Cycle_Number": cycle_number,
        "Cycle_Type": ctype,
        "Time": t,
        "Ambient_Temperature": int(ambient),
        "Delta t ": dt,
        "Current_measured": i,
        "Voltage_measured": v,
        "Current_Mod": current_mod,
        "cummulative Ah": cum,
        "Ah": ah,
        "soc": soc,
    })


def convert_mat(mat_source, cell_id):
    """mat_source: path or file-like object holding <cell_id>.mat. Returns (df, nasa_capacity)."""
    mat = loadmat(mat_source, squeeze_me=True, struct_as_record=False)
    cycles = np.atleast_1d(mat[cell_id].cycle)

    frames, nasa_capacity = [], []
    n = 0
    for c in cycles:
        ctype = str(c.type).strip().lower()
        if ctype not in ("charge", "discharge"):
            continue  # impedance (EIS) cycles carry no time series of this kind
        n += 1
        frames.append(_cycle_frame(n, ctype, c.ambient_temperature, c.data))
        if ctype == "discharge":
            nasa_capacity.append((n, float(c.data.Capacity)))
    return pd.concat(frames, ignore_index=True)[COLUMNS], nasa_capacity


def _mats_from_zip(zip_path, cells):
    """Find <cell>.mat inside the NASA outer zip (which nests one zip per test batch)."""
    found = {}

    def scan(zf):
        for name in zf.namelist():
            if len(found) == len(cells):
                return
            stem = Path(name).stem
            if name.lower().endswith(".mat") and stem in cells and stem not in found:
                found[stem] = io.BytesIO(zf.read(name))
            elif name.lower().endswith(".zip"):
                scan(zipfile.ZipFile(io.BytesIO(zf.read(name))))

    scan(zipfile.ZipFile(zip_path))
    missing = set(cells) - set(found)
    if missing:
        raise SystemExit(f"not found in {zip_path}: {sorted(missing)}")
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="NASA 'Battery Data Set' zip")
    src.add_argument("--mat", nargs="+", help="one or more <cell>.mat files")
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS, help="cells to take from --zip")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "nasa"))
    args = ap.parse_args(argv)

    if args.zip:
        sources = _mats_from_zip(args.zip, args.cells)
    else:
        sources = {Path(p).stem: p for p in args.mat}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cell_id in sorted(sources):
        df, cap = convert_mat(sources[cell_id], cell_id)
        path = out / f"{cell_id}.csv"
        df.to_csv(path, index=False)
        n_dis = (df.groupby("Cycle_Number")["Cycle_Type"].first() == "discharge").sum()
        print(f"{cell_id}: {df['Cycle_Number'].nunique()} cycles ({n_dis} discharge), "
              f"{len(df)} rows, NASA capacity {cap[0][1]:.3f} -> {cap[-1][1]:.3f} Ah -> {path}")


if __name__ == "__main__":
    main()
