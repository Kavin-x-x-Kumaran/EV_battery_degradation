"""
Agent 4 -- Edge Deployment Agent

Role: demonstrate the trained digital-twin model is light enough to run on
constrained BMS hardware. We don't have physical microcontroller access for
a demo, so we do what's standard in edge-AI papers when hardware isn't
available: (1) quantize the model (float32 -> int8-simulated), (2) measure
model size before/after, (3) benchmark pure-Python/NumPy inference latency
as a stand-in for embedded inference time, since the model has no framework
dependency and could be ported to C/embedded NumPy-equivalent trivially.
"""
import time
import numpy as np


def quantize_weights(weights, bits=8):
    """
    Simple linear (affine) quantization per-array, simulating int8 deployment.
    Returns: quantized (as float, for fidelity testing), plus size stats.
    """
    quantized, packed = [], []
    orig_bytes = 0
    quant_bytes = 0
    for w in weights:
        orig_bytes += w.size * 4  # deployed baseline is float32, not numpy's float64
        w_min, w_max = w.min(), w.max()
        scale = (w_max - w_min) / (2 ** bits - 1) if w_max > w_min else 1.0
        q = np.round((w - w_min) / scale).astype(np.uint8)
        quant_bytes += q.nbytes
        dq = q.astype(np.float32) * scale + w_min  # dequantized, for testing accuracy loss
        quantized.append(dq)
        packed.append({"shape": list(w.shape), "min": float(w_min), "scale": float(scale),
                       "q": q.ravel().tolist()})
    return quantized, orig_bytes, quant_bytes, packed


def benchmark_inference(model, X, n_repeats=200):
    """Average single-sample inference latency, simulating on-device prediction."""
    single_sample = X[:1]
    # warm-up
    for _ in range(5):
        model.predict(single_sample)

    start = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(single_sample)
    elapsed = time.perf_counter() - start
    return (elapsed / n_repeats) * 1000  # ms per inference


def evaluate_quantized_accuracy(model, X, y, weights_backup):
    """Compare MAE before/after quantizing weights in-place, then restore."""
    y_pred_fp = model.predict(X)
    mae_fp = np.mean(np.abs(y_pred_fp - y))

    q_weights, orig_bytes, quant_bytes, _ = quantize_weights(model.get_weights())
    model.set_weights(q_weights)
    y_pred_q = model.predict(X)
    mae_q = np.mean(np.abs(y_pred_q - y))

    model.set_weights(weights_backup)  # restore original fp weights
    return {
        "mae_fp32": float(mae_fp),
        "mae_quantized_int8": float(mae_q),
        "accuracy_delta": float(mae_q - mae_fp),
        "orig_size_bytes": int(orig_bytes),
        "quantized_size_bytes": int(quant_bytes),
        "size_reduction_pct": round(100 * (1 - quant_bytes / orig_bytes), 1),
    }


def run_edge_report(model, X, y):
    weights_backup = model.get_weights()
    latency_ms = benchmark_inference(model, X)
    acc_report = evaluate_quantized_accuracy(model, X, y, weights_backup)
    report = {
        "n_params": model.n_params(),
        "avg_inference_latency_ms": round(latency_ms, 4),
        **acc_report,
    }
    return report


def export_quantized_model(model, scaler, feature_cols, path):
    """Write the int8 twin + its input scaler as JSON -- everything a BMS needs."""
    import json
    _, _, _, packed = quantize_weights(model.get_weights())
    with open(path, "w") as f:
        json.dump({
            "architecture": "x = (raw-mean)/std;  y = x@Ws + tanh(x@W1+b1)@W2 + b2",
            "features": list(feature_cols),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std": scaler.scale_.tolist(),
            "layers": dict(zip(["W1", "b1", "W2", "b2", "Ws"], packed)),
            "dequantize": "w = q * scale + min",
        }, f)
