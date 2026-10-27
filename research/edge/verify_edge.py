"""
Check that the deployable inference code gives the same SoH as the pipeline.

For one exported twin it compares, on every cycle of that cell:
  reference  NumPy forward pass with the dequantized int8 weights
  python     edge/soh_infer.py (no third-party imports, MicroPython-ready)
  c          edge/soh_infer.c compiled with gcc (skipped if gcc is missing)

Usage:
  python edge/verify_edge.py outputs/edge_model_real_cell.json outputs/features.csv
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

EDGE = Path(__file__).resolve().parent
sys.path.insert(0, str(EDGE))
import soh_infer                      # noqa: E402
from build_c_model import build_header  # noqa: E402


def reference_predict(model, raw):
    w = {n: np.array(l["q"], dtype=np.float64) * l["scale"] + l["min"] for n, l in model["layers"].items()}
    hidden = model["layers"]["b1"]["shape"][0]
    x = (raw - np.array(model["scaler_mean"])) / np.array(model["scaler_std"])
    W1 = w["W1"].reshape(-1, hidden)
    return x @ w["Ws"] + np.tanh(x @ W1 + w["b1"]) @ w["W2"] + w["b2"][0]


def c_predict(model, raw):
    gcc = shutil.which("gcc")
    if gcc is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # copy the source next to the generated header: #include "..." searches
        # the source file's own folder first, so edge/soh_model.h would win
        shutil.copy(EDGE / "soh_infer.c", tmp / "soh_infer.c")
        (tmp / "soh_model.h").write_text(build_header(model))
        exe = tmp / "soh_infer"
        subprocess.run([gcc, "-O2", "-std=c99", "-DSOH_TEST_MAIN",
                        str(tmp / "soh_infer.c"), "-lm", "-o", str(exe)], check=True)
        stdin = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in raw) + "\n"
        out = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, check=True).stdout
    return np.array([float(v) for v in out.split()])


def verify(model_path, features_path):
    model = json.loads(Path(model_path).read_text())
    cell = Path(model_path).stem.removeprefix("edge_model_")
    df = pd.read_csv(features_path)
    df = df[df["Vehicle_ID"].astype(str) == cell]
    if df.empty:
        raise SystemExit(f"no rows for {cell} in {features_path}")
    raw = df[model["features"]].to_numpy(dtype=float)

    ref = reference_predict(model, raw)
    m = soh_infer.load_model(model_path)
    results = {"n_cycles": len(df),
               "mae_vs_measured_soh": float(np.mean(np.abs(ref - df["SoH_true"].to_numpy())))}
    results["python_max_abs_diff"] = float(np.max(np.abs(np.array([soh_infer.predict(m, r) for r in raw]) - ref)))
    c = c_predict(model, raw)
    results["c_max_abs_diff"] = None if c is None else float(np.max(np.abs(c - ref)))
    return cell, results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    cell, r = verify(sys.argv[1], sys.argv[2])
    print(f"{cell}: {r['n_cycles']} cycles, MAE vs measured SoH (all cycles, incl. training) {r['mae_vs_measured_soh']:.4f}")
    print(f"  pure Python max |diff| vs reference: {r['python_max_abs_diff']:.2e}")
    print("  C (gcc) max |diff| vs reference:     " +
          ("skipped, gcc not found" if r["c_max_abs_diff"] is None else f"{r['c_max_abs_diff']:.2e}"))
