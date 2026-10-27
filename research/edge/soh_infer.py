"""
SoH inference with no third-party imports, so it also runs on MicroPython
(copy this file and an outputs/edge_model_<cell>.json to the board).

  model = load_model("edge_model_real_cell.json")
  soh = predict(model, [cc_duration_s, cv_duration_s, ...])  # order: model["features"]
"""
import json
import math


def load_model(path):
    with open(path) as f:
        m = json.load(f)
    layers = {}
    for name, layer in m["layers"].items():
        layers[name] = [q * layer["scale"] + layer["min"] for q in layer["q"]]
    return {
        "features": m["features"],
        "mean": m["scaler_mean"],
        "std": m["scaler_std"],
        "hidden": m["layers"]["b1"]["shape"][0],
        "w": layers,
    }


def predict(model, raw):
    w, hidden = model["w"], model["hidden"]
    x = [(r - mu) / sd for r, mu, sd in zip(raw, model["mean"], model["std"])]
    y = w["b2"][0] + sum(xi * wi for xi, wi in zip(x, w["Ws"]))
    for j in range(hidden):
        z = w["b1"][j]
        for i, xi in enumerate(x):
            z += xi * w["W1"][i * hidden + j]
        y += math.tanh(z) * w["W2"][j]
    return y
