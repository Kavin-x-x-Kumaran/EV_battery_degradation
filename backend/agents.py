from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FEATURES = [
    "mileage_km",
    "soc",
    "voltage_delta_mv",
    "temp_delta_c",
    "mean_temp_c",
    "pack_voltage_v",
    "abs_current_a",
    "speed_kmph",
    "chemistry_lfp",
    "capacity_proxy_kwh",
    "twin_stress",
]


@dataclass
class ModelBundle:
    global_weights: list[float]
    client_weights: dict[str, list[float]]
    feature_mean: dict[str, float]
    feature_std: dict[str, float]
    metadata: dict[str, Any]


class DataCleaningAgent:
    columns = [
        "vin",
        "Timestamp",
        "VehicleStatus",
        "ChargingStatus",
        "Speed",
        "Mileage",
        "TotalVoltage",
        "TotalCurrent",
        "SOC",
        "MaxCellVoltage",
        "MinCellVoltage",
        "MaxTemp",
        "MinTemp",
        "TotalCells",
    ]

    def clean(self, frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
        df = frame.copy()
        df["source_file"] = source_file
        df["chemistry_lfp"] = (df["TotalCells"] >= 150).astype(float)
        df["timestamp"] = pd.to_datetime(
            df["Timestamp"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
        )
        df = df.dropna(subset=["timestamp", "vin", "Mileage", "SOC"])
        df = df[(df["SOC"].between(1, 100)) & (df["TotalVoltage"] > 150)]
        df = df[(df["MaxCellVoltage"] > 1500) & (df["MinCellVoltage"] > 1500)]
        df = df[(df["MaxTemp"].between(-30, 80)) & (df["MinTemp"].between(-30, 80))]
        df = df[df["VehicleStatus"] != 254]
        df["voltage_delta_mv"] = df["MaxCellVoltage"] - df["MinCellVoltage"]
        df["temp_delta_c"] = df["MaxTemp"] - df["MinTemp"]
        df["abs_current_a"] = df["TotalCurrent"].abs()
        return df.sort_values(["vin", "timestamp"])


class FeatureEngineeringAgent:
    def build_training_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for vin, group in df.groupby("vin", sort=False):
            g = group.sort_values("timestamp").copy()
            chemistry_lfp = float(g["chemistry_lfp"].median())
            capacity_series = self._charging_capacity_proxy(g)
            early_capacity = capacity_series.dropna().head(8).median()
            if not np.isfinite(early_capacity) or early_capacity <= 0:
                early_capacity = self._nominal_capacity_proxy(g)

            g["date"] = g["timestamp"].dt.date
            daily = g.groupby("date").agg(
                mileage_km=("Mileage", "max"),
                soc=("SOC", "mean"),
                voltage_delta_mv=("voltage_delta_mv", "median"),
                temp_delta_c=("temp_delta_c", "median"),
                mean_temp_c=("MaxTemp", "mean"),
                pack_voltage_v=("TotalVoltage", "mean"),
                abs_current_a=("abs_current_a", "mean"),
                speed_kmph=("Speed", "mean"),
            )
            daily["chemistry_lfp"] = chemistry_lfp
            daily["capacity_proxy_kwh"] = np.interp(
                np.arange(len(daily)),
                np.linspace(0, len(daily) - 1, max(1, len(capacity_series))),
                capacity_series.ffill().fillna(early_capacity).to_numpy(),
            )
            daily["twin_stress"] = DigitalTwinAgent().stress_score(daily)
            mileage_span = max(float(daily["mileage_km"].max() - daily["mileage_km"].min()), 1.0)
            mileage_progress = (daily["mileage_km"] - daily["mileage_km"].min()) / mileage_span
            capacity_ratio = (daily["capacity_proxy_kwh"] / early_capacity).replace([np.inf, -np.inf], np.nan).fillna(1)
            capacity_signal = (100 * capacity_ratio).clip(82, 103)
            stress_penalty = daily["twin_stress"].expanding().mean().fillna(0) * 3.0
            imbalance_penalty = 0.006 * daily["voltage_delta_mv"].clip(lower=0)
            mileage_signal = 100 - 18 * mileage_progress
            daily["soh_pct"] = (
                0.68 * mileage_signal
                + 0.32 * capacity_signal
                - stress_penalty
                - imbalance_penalty
            ).clip(72, 105)
            daily["vin"] = vin
            rows.extend(daily.reset_index(drop=True).to_dict("records"))
        return pd.DataFrame(rows)

    def live_rows(self, df: pd.DataFrame, max_rows_per_vehicle: int = 300) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for vin, group in df.groupby("vin", sort=False):
            g = group.sort_values("timestamp")
            if len(g) > max_rows_per_vehicle:
                g = g.iloc[np.linspace(0, len(g) - 1, max_rows_per_vehicle).astype(int)]
            for _, row in g.iterrows():
                rows.append(
                    {
                        "vin": row["vin"],
                        "timestamp": row["timestamp"].isoformat(),
                        "mileage_km": float(row["Mileage"]),
                        "soc": float(row["SOC"]),
                        "voltage_delta_mv": float(row["voltage_delta_mv"]),
                        "temp_delta_c": float(row["temp_delta_c"]),
                        "mean_temp_c": float(row["MaxTemp"]),
                        "pack_voltage_v": float(row["TotalVoltage"]),
                        "abs_current_a": float(abs(row["TotalCurrent"])),
                        "speed_kmph": float(row["Speed"]),
                        "chemistry_lfp": float(row["chemistry_lfp"]),
                        "capacity_proxy_kwh": float(self._nominal_capacity_proxy(group)),
                    }
                )
        return rows

    def _charging_capacity_proxy(self, g: pd.DataFrame) -> pd.Series:
        charging = g[(g["ChargingStatus"] == 1) | (g["TotalCurrent"] < -5)].copy()
        if charging.empty:
            return pd.Series(dtype=float)
        gap = charging["timestamp"].diff().dt.total_seconds().fillna(0)
        charging["session"] = (gap > 1800).cumsum()
        caps: list[float] = []
        for _, s in charging.groupby("session"):
            s = s.sort_values("timestamp")
            dsoc = float(s["SOC"].max() - s["SOC"].min())
            if dsoc < 8:
                continue
            dt_hours = s["timestamp"].diff().dt.total_seconds().fillna(0).clip(0, 120) / 3600
            energy = float((s["TotalVoltage"] * s["TotalCurrent"].abs() * dt_hours).sum() / 1000)
            if energy > 0:
                caps.append(energy / (dsoc / 100))
        if not caps:
            return pd.Series(dtype=float)
        caps = pd.Series(caps).replace([np.inf, -np.inf], np.nan).dropna()
        return caps[(caps > caps.quantile(0.05)) & (caps < caps.quantile(0.95))]

    def _nominal_capacity_proxy(self, g: pd.DataFrame) -> float:
        cells = float(g["TotalCells"].median())
        voltage = float(g["TotalVoltage"].median())
        return max(18.0, min(85.0, voltage * cells / 12000))


class DigitalTwinAgent:
    def stress_score(self, frame: pd.DataFrame) -> pd.Series:
        temp = ((frame["mean_temp_c"] - 25).clip(lower=0) / 20).fillna(0)
        imbalance = (frame["voltage_delta_mv"].clip(lower=0) / 80).fillna(0)
        current = (frame["abs_current_a"].clip(lower=0) / 150).fillna(0)
        return (0.45 * temp + 0.35 * imbalance + 0.20 * current).clip(0, 3)

    def feedback_adjustment(self, features: dict[str, float]) -> float:
        stress = float(features.get("twin_stress", 0))
        temp_delta = float(features.get("temp_delta_c", 0))
        voltage_delta = float(features.get("voltage_delta_mv", 0))
        return -(0.35 * stress + 0.01 * max(voltage_delta - 35, 0) + 0.04 * max(temp_delta - 5, 0))


class FederatedLearningAgent:
    def train(self, train_df: pd.DataFrame) -> ModelBundle:
        means = train_df[FEATURES].mean()
        stds = train_df[FEATURES].std().replace(0, 1).fillna(1)
        clients: dict[str, list[float]] = {}
        sizes: dict[str, int] = {}
        for vin, g in train_df.groupby("vin"):
            x = self._design((g[FEATURES] - means) / stds)
            y = g["soh_pct"].to_numpy(dtype=float)
            clients[vin] = self._ridge(x, y).tolist()
            sizes[vin] = len(g)
        total = sum(sizes.values()) or 1
        global_w = sum(np.array(clients[k]) * (sizes[k] / total) for k in clients)
        meta = {
            "vehicles": sorted(clients),
            "rows": int(len(train_df)),
            "features": FEATURES,
            "client_rows": sizes,
            "threshold_soh": 80,
        }
        return ModelBundle(
            global_weights=global_w.tolist(),
            client_weights=clients,
            feature_mean=means.to_dict(),
            feature_std=stds.to_dict(),
            metadata=meta,
        )

    def dynamic_weights(self, bundle: ModelBundle, features: dict[str, float]) -> dict[str, float]:
        sims: dict[str, float] = {}
        target_lfp = features.get("chemistry_lfp", 0)
        target_mileage = features.get("mileage_km", 0)
        target_voltage = features.get("voltage_delta_mv", 0)
        target_temp = features.get("mean_temp_c", 0)
        for vin, profile in bundle.metadata.get("client_profiles", {}).items():
            chem_dist = abs(float(profile["chemistry_lfp"]) - target_lfp)
            mileage_dist = abs(float(profile["mileage_km"]) - target_mileage) / 250000
            voltage_dist = abs(float(profile["voltage_delta_mv"]) - target_voltage) / 90
            temp_dist = abs(float(profile["mean_temp_c"]) - target_temp) / 35
            sims[vin] = math.exp(-(2.4 * chem_dist + 1.15 * mileage_dist + 0.7 * voltage_dist + 0.45 * temp_dist))
        total = sum(sims.values()) or 1
        return {k: v / total for k, v in sims.items()}

    def normalized_override(self, bundle: ModelBundle, override: dict[str, float] | None) -> dict[str, float] | None:
        if not override:
            return None
        cleaned = {
            vin: max(0.0, float(override.get(vin, 0.0)))
            for vin in bundle.client_weights
        }
        total = sum(cleaned.values())
        if total <= 0:
            return None
        return {vin: value / total for vin, value in cleaned.items()}

    def predict(
        self,
        bundle: ModelBundle,
        features: dict[str, float],
        weight_override: dict[str, float] | None = None,
    ) -> tuple[float, dict[str, float]]:
        x = np.array([(features[f] - bundle.feature_mean[f]) / bundle.feature_std[f] for f in FEATURES])
        xd = np.r_[1.0, x]
        weights = self.normalized_override(bundle, weight_override) or self.dynamic_weights(bundle, features)
        if weights:
            w = sum(np.array(bundle.client_weights[k]) * weights[k] for k in weights)
        else:
            w = np.array(bundle.global_weights)
        return float(np.dot(w, xd)), weights

    def predict_client(self, bundle: ModelBundle, features: dict[str, float], vin: str) -> float:
        x = np.array([(features[f] - bundle.feature_mean[f]) / bundle.feature_std[f] for f in FEATURES])
        xd = np.r_[1.0, x]
        return float(np.dot(np.array(bundle.client_weights[vin]), xd))

    def _design(self, x: pd.DataFrame) -> np.ndarray:
        return np.c_[np.ones(len(x)), x.to_numpy(dtype=float)]

    def _ridge(self, x: np.ndarray, y: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        reg = np.eye(x.shape[1]) * alpha
        reg[0, 0] = 0
        return np.linalg.pinv(x.T @ x + reg) @ x.T @ y


class XaiAgent:
    def explain(self, bundle: ModelBundle, features: dict[str, float], client_mix: dict[str, float]) -> list[dict[str, float | str]]:
        if client_mix:
            w = sum(np.array(bundle.client_weights[k]) * client_mix[k] for k in client_mix)
        else:
            w = np.array(bundle.global_weights)
        rows = []
        for i, name in enumerate(FEATURES, start=1):
            z = (features[name] - bundle.feature_mean[name]) / bundle.feature_std[name]
            rows.append({"feature": name, "contribution": float(w[i] * z), "value": float(features[name])})
        return sorted(rows, key=lambda r: abs(float(r["contribution"])), reverse=True)

    def explain_client(self, bundle: ModelBundle, features: dict[str, float], vin: str) -> list[dict[str, float | str]]:
        return self.explain(bundle, features, {vin: 1.0})


class Orchestrator:
    def __init__(self, root: Path):
        self.root = root
        self.zip_path = root / "Vehicle data (1).zip"
        self.model_path = root / "models" / "model_bundle.json"
        self.telemetry_path = root / "models" / "live_telemetry.json"
        self.cleaner = DataCleaningAgent()
        self.features = FeatureEngineeringAgent()
        self.twin = DigitalTwinAgent()
        self.fl = FederatedLearningAgent()
        self.xai = XaiAgent()

    def train_or_load(self) -> ModelBundle:
        if self.model_path.exists() and self.telemetry_path.exists():
            return self._load()
        return self.train()

    def train(self) -> ModelBundle:
        frames = []
        with zipfile.ZipFile(self.zip_path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".parquet")]
            for name in names:
                data = archive.read(name)
                table = pq.ParquetFile(io.BytesIO(data)).read(columns=self.cleaner.columns)
                clean = self.cleaner.clean(table.to_pandas(), name)
                if len(clean) > 35000:
                    clean = clean.iloc[np.linspace(0, len(clean) - 1, 35000).astype(int)]
                frames.append(clean)
        telemetry = pd.concat(frames, ignore_index=True)
        train_df = self.features.build_training_rows(telemetry)
        bundle = self.fl.train(train_df)
        profiles = train_df.groupby("vin")[FEATURES].median().to_dict("index")
        bundle.metadata["client_profiles"] = profiles
        bundle.metadata["soh_min"] = float(train_df["soh_pct"].min())
        bundle.metadata["soh_max"] = float(train_df["soh_pct"].max())
        live = self.features.live_rows(telemetry)
        capacity_by_vin = train_df.groupby("vin")["capacity_proxy_kwh"].median().to_dict()
        for row in live:
            row["capacity_proxy_kwh"] = float(capacity_by_vin.get(row["vin"], row["capacity_proxy_kwh"]))
        self.model_path.parent.mkdir(exist_ok=True)
        self.model_path.write_text(json.dumps(bundle.__dict__, indent=2), encoding="utf-8")
        self.telemetry_path.write_text(json.dumps(live[:1200], indent=2), encoding="utf-8")
        return bundle

    def predict(self, telemetry: dict[str, float]) -> dict[str, Any]:
        bundle = self.train_or_load()
        features = {f: float(telemetry[f]) for f in FEATURES if f in telemetry}
        override = telemetry.get("weight_override")
        features["twin_stress"] = float(self.twin.stress_score(pd.DataFrame([features])).iloc[0])
        raw_soh, mix = self.fl.predict(bundle, features, override)
        soh = max(40.0, min(105.0, raw_soh + self.twin.feedback_adjustment(features)))
        capacity = self._soh_to_capacity(features, soh)
        mileage = features["mileage_km"]
        threshold = 80.0
        trajectory = self._capacity_trajectory(bundle, features, mix)
        remaining_km = self._remaining_from_trajectory(trajectory, mileage, threshold)
        monthly_km = 1800
        client_predictions = self._client_predictions(bundle, features)
        return {
            "soh_pct": round(soh, 2),
            "max_capacity_kwh": round(capacity, 2),
            "remaining_km_to_80": round(max(0, remaining_km), 1),
            "remaining_months_to_80": round(max(0, remaining_km) / monthly_km, 1),
            "threshold_soh": threshold,
            "digital_twin": {
                "stress_score": round(features["twin_stress"], 3),
                "feedback_adjustment": round(self.twin.feedback_adjustment(features), 3),
            },
            "weight_mode": "manual_override" if override else "automatic_similarity",
            "client_mix": {k: round(v, 3) for k, v in mix.items()},
            "weight_reasoning": self._weight_reasoning(bundle, features, mix, bool(override)),
            "client_predictions": client_predictions,
            "trajectory": trajectory,
            "xai": self.xai.explain(bundle, features, mix)[:8],
            "client_xai": {
                vin: self.xai.explain_client(bundle, features, vin)[:6]
                for vin in bundle.client_weights
            },
            "features": features,
            "current_mileage_km": round(mileage, 1),
        }

    def _client_predictions(self, bundle: ModelBundle, features: dict[str, float]) -> dict[str, Any]:
        rows = {}
        feedback = self.twin.feedback_adjustment(features)
        for vin in bundle.client_weights:
            soh = max(40.0, min(105.0, self.fl.predict_client(bundle, features, vin) + feedback))
            rows[vin] = {
                "soh_pct": round(soh, 2),
                "max_capacity_kwh": round(self._soh_to_capacity(features, soh), 2),
            }
        return rows

    def _capacity_trajectory(
        self,
        bundle: ModelBundle,
        features: dict[str, float],
        mix: dict[str, float],
    ) -> list[dict[str, float]]:
        now = features["mileage_km"]
        probe = dict(features)
        points = []
        for month, km in enumerate(np.linspace(now, now + 240000, 25)):
            probe["mileage_km"] = float(km)
            pred, _ = self.fl.predict(bundle, probe, mix)
            soh = max(40.0, min(105.0, pred + self.twin.feedback_adjustment(probe)))
            points.append(
                {
                    "mileage_km": round(float(km), 1),
                    "month": round(float(month * 240000 / 24 / 1800), 1),
                    "soh_pct": round(soh, 2),
                    "max_capacity_kwh": round(self._soh_to_capacity(features, soh), 2),
                }
            )
        return points

    def _remaining_from_trajectory(self, trajectory: list[dict[str, float]], now: float, threshold: float) -> float:
        for point in trajectory:
            if point["soh_pct"] <= threshold:
                return float(point["mileage_km"] - now)
        return float(trajectory[-1]["mileage_km"] - now) if trajectory else 0.0

    def _soh_to_capacity(self, features: dict[str, float], soh: float) -> float:
        return max(0.0, float(features["capacity_proxy_kwh"]) * soh / 100)

    def _weight_reasoning(
        self,
        bundle: ModelBundle,
        features: dict[str, float],
        mix: dict[str, float],
        manual: bool,
    ) -> dict[str, str]:
        if manual:
            return {
                vin: f"{vin} uses the manually selected override weight of {mix[vin] * 100:.1f}%."
                for vin in mix
            }
        target_chem = "LFP" if features["chemistry_lfp"] >= 0.5 else "NMC"
        rows = {}
        for vin, profile in bundle.metadata.get("client_profiles", {}).items():
            profile_chem = "LFP" if float(profile["chemistry_lfp"]) >= 0.5 else "NMC"
            parts = []
            if profile_chem == target_chem:
                parts.append(f"chemistry matches {target_chem}")
            else:
                parts.append(f"chemistry differs ({profile_chem} vs {target_chem})")
            mileage_gap = abs(float(profile["mileage_km"]) - features["mileage_km"])
            voltage_gap = abs(float(profile["voltage_delta_mv"]) - features["voltage_delta_mv"])
            temp_gap = abs(float(profile["mean_temp_c"]) - features["mean_temp_c"])
            parts.append(f"mileage gap is {mileage_gap:,.0f} km")
            parts.append(f"voltage-spread gap is {voltage_gap:.1f} mV")
            parts.append(f"temperature gap is {temp_gap:.1f} C")
            rows[vin] = f"{vin} receives {mix.get(vin, 0) * 100:.1f}% because " + ", ".join(parts) + "."
        return rows

    def telemetry(self) -> list[dict[str, Any]]:
        self.train_or_load()
        return json.loads(self.telemetry_path.read_text(encoding="utf-8"))

    def _load(self) -> ModelBundle:
        return ModelBundle(**json.loads(self.model_path.read_text(encoding="utf-8")))
