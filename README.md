# Multi-Agent Explainable AI Framework for EV Battery Degradation

**Project title:** Multi-Agent Explainable AI Framework for Early Life Prediction of Long-Term EV Battery Degradation Using Sparse Charging Cycles, Features and Digital Twin Feedback

This repository contains a localhost demonstration of an interactive EV battery digital twin. It combines modular agents, vehicle-wise federated models, dynamic client weighting, explainable AI, and a React dashboard for SOH and remaining-life prediction.

## Features

- Manual battery-status input and plausible dataset-derived auto-generation.
- Per-client SOH and maximum-capacity predictions.
- Automatic similarity-based federated client weighting with manual override.
- Plain-language reasoning for client selection.
- Digital-twin stress feedback from temperature, current, and cell-voltage imbalance.
- Feature-contribution explanations for the final weighted model and each client model.
- Predicted maximum-capacity trajectory over mileage or months.
- Remaining mileage and time estimate to the 80% SOH threshold.

## Repository Layout

```text
backend/       Python agents and HTTP API server
frontend/      React UI served as static files
run.ps1        Windows launcher
README.md      Setup and usage guide
```

The large dataset archive and generated model cache are intentionally excluded from Git by `.gitignore`. Obtain the dataset separately and place `Vehicle data (1).zip` in the repository root before running the application.

## Requirements

- Windows PowerShell
- Python 3.10 or newer
- Python packages: `numpy`, `pandas`, and `pyarrow`
- Internet access on first page load if the browser must fetch the React UMD scripts from unpkg

Install the Python dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas pyarrow
```

## Run Locally

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
.\run.ps1
```

Open `http://127.0.0.1:8000` in a browser. Stop the server with `Ctrl+C`.

If PowerShell blocks the launcher for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1
```

The first request builds a generated cache under `models/`; later starts reuse it. Delete `models/` when a full retraining run is required.

## API

- `GET /api/status` reports whether the model cache is ready.
- `GET /api/train` builds or loads the model cache.
- `GET /api/telemetry` returns simulated telemetry for demonstrations.
- `POST /api/predict` accepts battery features and returns SOH, capacity, remaining life, client predictions, weights, reasoning, trajectory, digital-twin feedback, and XAI contributions.

Example request:

```json
{
  "mileage_km": 145000,
  "soc": 68,
  "voltage_delta_mv": 18,
  "temp_delta_c": 3,
  "mean_temp_c": 29,
  "pack_voltage_v": 365,
  "abs_current_a": 22,
  "speed_kmph": 18,
  "chemistry_lfp": 0,
  "capacity_proxy_kwh": 52
}
```

## Important Limitation

The supplied dataset has no direct measured SOH label. The implementation therefore derives a proxy target from sparse charging-cycle capacity estimates, mileage progression, cell-voltage imbalance, and thermal/current stress. Results are intended for academic demonstration and are not a safety-certified battery-management system.

## Research Pipeline

`research/` holds the offline evaluation pipeline. It trains on real
cycling data (one cell plus four NASA PCoE cells) with true capacity-based SOH
labels and a held-out chronological split. It compares against ridge and
usage-only baselines, explains predictions with SHAP, and exports an int8
model as verified C code for a BMS microcontroller. See
[research/README.md](research/README.md) for results and how to run it.

## License and Data

No license has been selected for this academic project. Add a license before publishing if required by your institution. Check the dataset's own terms before redistributing the archive; the archive is excluded from this repository by default.
