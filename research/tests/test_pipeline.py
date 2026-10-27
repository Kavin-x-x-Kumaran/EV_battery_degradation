"""
Tests for the pipeline. Standard library unittest, no extra dependencies:
  python -m unittest discover -s tests -v
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents"))
sys.path.insert(0, str(ROOT / "edge"))

from mlp import SoHRegressor, fed_avg                      # noqa: E402
from data_loader import load_real_cell, REAL_FEATURE_COLUMNS  # noqa: E402
from fl_digital_twin_agent import run_federated_training   # noqa: E402
from edge_agent import quantize_weights, export_quantized_model  # noqa: E402
import orchestrator                                       # noqa: E402
import verify_edge                                        # noqa: E402


def _charge(cycle, capacity, start_v=3.4):
    """CC at 1.5 A up to 4.2 V, then CV with decaying current. Higher capacity = longer CC."""
    rows, t = [], 0.0
    cc_s = 3600 * capacity / 1.5 * 0.8
    for k in range(80):
        t += cc_s / 80
        rows.append((t, 1.5, start_v + (4.2 - start_v) * (k + 1) / 80))
    for k in range(40):
        t += 60
        rows.append((t, 1.5 * np.exp(-k / 8), 4.2))
    return _frame(cycle, "charge", rows)


def _discharge(cycle, capacity):
    """2 A until capacity is used; sag grows as capacity fades."""
    rows, t = [(0.0, 0.0, 4.18)], 0.0
    n = 100
    for k in range(n):
        t += 3600 * capacity / 2.0 / n
        rows.append((t, -2.0, 4.0 - (0.15 + (2.0 - capacity) * 0.2) - 1.2 * k / n))
    return _frame(cycle, "discharge", rows)


def _frame(cycle, ctype, rows):
    t, i, v = map(np.array, zip(*rows))
    dt = np.diff(t, prepend=0.0)
    ah = np.abs(i) * dt / 3600
    return pd.DataFrame({
        "Cycle_Number": cycle, "Cycle_Type": ctype, "Time": t, "Ambient_Temperature": 24,
        "Delta t ": dt, "Current_measured": i, "Voltage_measured": v,
        "Current_Mod": np.where(i < 0, -i, np.nan), "cummulative Ah": np.cumsum(ah),
        "Ah": ah, "soc": 50.0,
    })


def write_fake_cell(path, capacities, bad_charge_at=None):
    frames, cycle = [], 1
    for k, cap in enumerate(capacities):
        frames.append(_charge(cycle, cap, start_v=4.0 if k == bad_charge_at else 3.4))
        frames.append(_discharge(cycle + 1, cap))
        cycle += 2
    pd.concat(frames).to_csv(path, index=False)


class TestMLP(unittest.TestCase):
    def test_fed_avg_is_sample_weighted(self):
        a = [np.full((2, 2), 1.0), np.zeros(2)]
        b = [np.full((2, 2), 4.0), np.ones(2)]
        avg = fed_avg([a, b], [1, 2])
        np.testing.assert_allclose(avg[0], 3.0)
        np.testing.assert_allclose(avg[1], 2 / 3)

    def test_weights_round_trip_and_training_reduces_loss(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, 4))
        y = X @ np.array([0.1, -0.2, 0.05, 0.0]) + 0.8
        m = SoHRegressor(4, hidden=8)
        clone = SoHRegressor(4, hidden=8, seed=99)
        clone.set_weights(m.get_weights())
        np.testing.assert_allclose(clone.predict(X), m.predict(X))
        losses = m.fit(X, y, epochs=300, lr=0.1)
        self.assertLess(losses[-1], losses[0] / 10)

    def test_skip_path_extrapolates_linear_trend(self):
        X = np.linspace(-1, 1, 50).reshape(-1, 1)
        m = SoHRegressor(1, hidden=4, weight_decay=0.0)
        m.fit(X, 1 - 0.2 * X.ravel(), epochs=3000, lr=0.1)
        # well outside the training range the prediction keeps following the line
        self.assertAlmostEqual(float(m.predict(np.array([[3.0]]))[0]), 0.4, delta=0.1)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_labels_and_features(self):
        caps = [2.0, 1.9, 1.8, 1.7, 1.6]
        write_fake_cell(self.tmp / "cell.csv", caps)
        df = load_real_cell(self.tmp / "cell.csv")
        self.assertEqual(len(df), 5)
        np.testing.assert_allclose(df["SoH_true"], np.array(caps) / 2.0, atol=1e-6)
        self.assertTrue(set(REAL_FEATURE_COLUMNS) <= set(df.columns))
        self.assertFalse(df[REAL_FEATURE_COLUMNS].isna().any().any())
        # throughput is history BEFORE each discharge, so it only grows
        self.assertTrue(np.all(np.diff(df["cum_ah_throughput"]) > 0))
        # a faded cell spends less time in the CC window and sags more
        self.assertGreater(df["window_time_s"].iloc[0], df["window_time_s"].iloc[-1])
        self.assertLess(df["early_discharge_v_drop"].iloc[0], df["early_discharge_v_drop"].iloc[-1])

    def test_charge_starting_above_window_is_dropped(self):
        write_fake_cell(self.tmp / "cell.csv", [2.0, 1.9, 1.8], bad_charge_at=1)
        df = load_real_cell(self.tmp / "cell.csv")
        self.assertEqual(list(df["Cycle_Number"]), [2, 6])
        # SoH reference is still the very first discharge
        self.assertAlmostEqual(df["SoH_true"].iloc[1], 0.9, places=6)


class TestSplitAndLeakage(unittest.TestCase):
    def _feat_df(self):
        rows = []
        for vid in ["A", "B"]:
            for c in range(10):
                rows.append({"Vehicle_ID": vid, "Cycle_Number": c, "f1": float(c), "f2": 1.0,
                             "cum_ah_throughput": float(c), "SoH_true": 1 - 0.01 * c})
        return pd.DataFrame(rows)

    def test_chronological_split_per_client(self):
        df = orchestrator.chronological_split(self._feat_df(), 0.7)
        for _, g in df.groupby("Vehicle_ID"):
            self.assertEqual((g["split"] == "train").sum(), 7)
            self.assertLess(g[g.split == "train"]["Cycle_Number"].max(),
                            g[g.split == "test"]["Cycle_Number"].min())

    def test_test_rows_never_reach_the_scaler(self):
        df = orchestrator.chronological_split(self._feat_df(), 0.7)
        df.loc[df.split == "test", "f1"] = 1e9
        _, scaler, _, _ = run_federated_training(df, ["f1", "f2"], n_rounds=1, local_epochs=1)
        self.assertAlmostEqual(scaler.mean_[0], 3.0)  # mean of cycles 0..6


class TestEdge(unittest.TestCase):
    def test_quantization_error_is_bounded(self):
        w = [np.random.default_rng(1).normal(size=(9, 16))]
        q, orig, quant, packed = quantize_weights(w)
        self.assertEqual((orig, quant), (9 * 16 * 4, 9 * 16))
        self.assertLessEqual(np.max(np.abs(q[0] - w[0])), packed[0]["scale"] / 2 + 1e-7)

    def test_exported_model_matches_python_and_c(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(40, len(REAL_FEATURE_COLUMNS)))
        df = pd.DataFrame(X, columns=REAL_FEATURE_COLUMNS).assign(
            Vehicle_ID="t", SoH_true=0.9)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        model = SoHRegressor(X.shape[1])
        model.fit(scaler.transform(X), np.full(40, 0.9), epochs=50)
        with tempfile.TemporaryDirectory() as tmp:
            export_quantized_model(model, scaler, REAL_FEATURE_COLUMNS, Path(tmp) / "edge_model_t.json")
            df.to_csv(Path(tmp) / "features.csv", index=False)
            _, r = verify_edge.verify(Path(tmp) / "edge_model_t.json", Path(tmp) / "features.csv")
        self.assertLess(r["python_max_abs_diff"], 1e-9)
        if r["c_max_abs_diff"] is None:
            self.skipTest("gcc not found; C check skipped")
        self.assertLess(r["c_max_abs_diff"], 1e-4)


class TestEndToEnd(unittest.TestCase):
    def test_real_mode_runs_on_a_small_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_fake_cell(tmp / "cell.csv", list(np.linspace(2.0, 1.4, 20)))
            report = orchestrator.main(["--csv", str(tmp / "cell.csv"), "--rounds", "5",
                                        "--out", str(tmp / "out")])
            self.assertEqual(report["data"]["n_labelled_cycles"], 20)
            self.assertEqual(report["data"]["n_test_cycles"], 6)
            for f in ["pipeline_report.json", "twin_predictions.png", "features.csv",
                      "edge_model_cell.json", "sample_explanations.txt"]:
                self.assertTrue((tmp / "out" / f).exists(), f)
            json.loads((tmp / "out" / "pipeline_report.json").read_text())


if __name__ == "__main__":
    unittest.main()
