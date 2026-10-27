"""
Multi-Agent Orchestrator

Runs the full pipeline end to end:
  Data -> Agent 1 (Feature Extraction) -> Agent 2 (Federated Aggregator +
  Digital Twins) -> Agent 3 (Explainability) -> Agent 4 (Edge Deployment)

Usage:
  python orchestrator.py                          # real data (default CSV)
  python orchestrator.py --csv B0005.csv B0006.csv  # several cells = several FL clients
  python orchestrator.py --data synthetic         # simulated 5-vehicle fleet

With 2+ real clients a leave-one-cell-out evaluation also runs (FedAvg on
N-1 cells, predicted on the unseen cell); --no-loco skips it.

Every client's cycles are split chronologically: the first --train-frac are
used for training, the rest are held out and only predicted. All accuracy
numbers in the report are on those held-out, later cycles.

Produces in --out (default outputs/):
  - fl_training_curve.png, twin_predictions.png, shap_importance.png
  - pipeline_report.json  (machine-readable summary for the report/paper)
  - sample_explanations.txt
  - edge_model_<client>.json  (int8 twin weights + scaler, ready for a BMS)
  - features.csv  (the labelled per-cycle feature table this run used)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from data.generate_data import generate_fleet
from data_loader import load_real_cells, REAL_FEATURE_COLUMNS
from feature_agent import extract_cycle_features, FULL_FEATURE_COLUMNS
from fl_digital_twin_agent import run_federated_training, build_digital_twins
from xai_agent import train_surrogate_and_explain, explain_single_cycle
from edge_agent import run_edge_report, export_quantized_model

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "charging_and_discharging_dataset.csv"


def _display_path(p):
    """Repo-relative when possible, so reports don't embed one machine's layout."""
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def metrics(y, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
        "r2": round(float(r2_score(y, y_pred)), 4) if len(y) > 1 else None,
    }


def chronological_split(feat_df, train_frac):
    feat_df = feat_df.sort_values(["Vehicle_ID", "Cycle_Number"]).reset_index(drop=True)
    rank = feat_df.groupby("Vehicle_ID").cumcount()
    size = feat_df.groupby("Vehicle_ID")["Cycle_Number"].transform("size")
    feat_df["split"] = np.where(rank < np.floor(size * train_frac), "train", "test")
    return feat_df


def baselines(feat_df, feature_cols):
    """Simple centralised models the MLP has to beat to be worth its complexity."""
    tr, te = feat_df[feat_df.split == "train"], feat_df[feat_df.split == "test"]
    out = {}
    trend = LinearRegression().fit(tr[["cum_ah_throughput"]], tr["SoH_true"])
    out["linear_trend_on_throughput"] = metrics(te["SoH_true"], trend.predict(te[["cum_ah_throughput"]]))
    ridge = Ridge(alpha=1.0).fit(tr[feature_cols], tr["SoH_true"])
    out["ridge_all_features"] = metrics(te["SoH_true"], ridge.predict(te[feature_cols]))
    if feat_df["Vehicle_ID"].nunique() > 1:
        # one ridge per client on its own training cycles: the linear analogue of a twin
        y, p = [], []
        for vid, g in te.groupby("Vehicle_ID"):
            g_tr = tr[tr["Vehicle_ID"] == vid]
            y.append(g["SoH_true"].values)
            p.append(Ridge(alpha=1.0).fit(g_tr[feature_cols], g_tr["SoH_true"]).predict(g[feature_cols]))
        out["ridge_per_client"] = metrics(np.concatenate(y), np.concatenate(p))
    return out


def leave_one_cell_out(feat_df, feature_cols, n_rounds, lr, seed=0):
    """
    For each client k: FedAvg over the other N-1 clients (all their cycles),
    then predict every cycle of client k, which no model has seen. Ridge and
    the throughput trend are fitted on the same N-1 clients for comparison.
    There is no twin here: the unseen cell has no training cycles to fine-tune on.
    """
    client_ids = sorted(feat_df["Vehicle_ID"].unique())
    per_cell, pooled = {}, {"y": [], "fedavg_global_mlp": [], "ridge_all_features": [],
                            "linear_trend_on_throughput": []}
    for held in client_ids:
        tr = feat_df[feat_df["Vehicle_ID"] != held].assign(split="train")
        te = feat_df[feat_df["Vehicle_ID"] == held]
        model, scaler, _, _ = run_federated_training(tr, feature_cols, n_rounds=n_rounds,
                                                     local_epochs=25, lr=lr, seed=seed)
        preds = {
            "fedavg_global_mlp": model.predict(scaler.transform(te[feature_cols].values.astype(float))),
            "ridge_all_features": Ridge(alpha=1.0).fit(tr[feature_cols], tr["SoH_true"]).predict(te[feature_cols]),
            "linear_trend_on_throughput": LinearRegression().fit(tr[["cum_ah_throughput"]], tr["SoH_true"])
                                                            .predict(te[["cum_ah_throughput"]]),
        }
        y = te["SoH_true"].values
        per_cell[held] = {"n_test_cycles": int(len(te)), "trained_on": [c for c in client_ids if c != held],
                          **{name: metrics(y, p) for name, p in preds.items()}}
        pooled["y"].append(y)
        for name, p in preds.items():
            pooled[name].append(p)
        print(f"  -> unseen {held}: " + "  ".join(
            f"{name.split('_')[0]} MAE={per_cell[held][name]['mae']:.4f}" for name in preds))
    y_all = np.concatenate(pooled.pop("y"))
    return {"per_cell": per_cell,
            "pooled": {name: metrics(y_all, np.concatenate(p)) for name, p in pooled.items()}}


def load_data(args, out_dir):
    if args.data == "synthetic":
        print("\n[Data Layer] Generating SYNTHETIC multi-vehicle fleet (demo of FL mechanics only)...")
        raw_df = generate_fleet(n_vehicles=5, n_cycles=120, out_path=str(ROOT / "data" / "fleet_raw.csv"))
        print(f"  -> {len(raw_df)} raw readings across {raw_df['Vehicle_ID'].nunique()} vehicles")
        print("\n[Agent 1: Feature Extraction] Aggregating cycles into features...")
        feat_df = extract_cycle_features(raw_df)
        feature_cols = FULL_FEATURE_COLUMNS
    else:
        csvs = args.csv or [DEFAULT_CSV]
        missing = [str(c) for c in csvs if not Path(c).exists()]
        if missing:
            raise SystemExit(f"dataset not found: {', '.join(missing)}\n"
                             "See 'Data' in research/README.md for where to get it, "
                             "or run with --data synthetic.")
        print(f"\n[Data Layer + Agent 1] Loading REAL cycling data from {len(csvs)} file(s); "
              "pairing each discharge with its preceding charge...")
        feat_df = load_real_cells(csvs)
        feature_cols = REAL_FEATURE_COLUMNS
    # per-run, so a NASA run can't overwrite the single-cell features
    feat_df.to_csv(out_dir / "features.csv", index=False)
    print(f"  -> {len(feat_df)} labelled cycles, {feat_df['Vehicle_ID'].nunique()} client(s), "
          f"{len(feature_cols)} features/cycle")
    return feat_df, feature_cols


def plot_all(out_dir, history, twins, client_ids, importance, train_frac):
    plt.figure(figsize=(7, 4.5))
    plt.plot([h["round"] for h in history], [h["avg_client_loss"] for h in history],
             marker="o", color="#2563eb")
    plt.xlabel("Federated Round")
    plt.ylabel("Avg Client Training Loss (MSE)")
    plt.title(f"Federated Learning Convergence ({len(client_ids)} client(s))")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fl_training_curve.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, len(client_ids), figsize=(max(6, 4 * len(client_ids)), 4.2),
                             sharey=True, squeeze=False)
    for ax, vid in zip(axes[0], client_ids):
        t = twins[vid]
        ax.plot(t["cycle_numbers"], t["y"], label="Measured SoH", color="black", lw=1.5)
        ax.plot(t["cycle_numbers"], t["y_pred_global"], label="Global model", color="#93c5fd", ls="--")
        ax.plot(t["cycle_numbers"], t["y_pred_twin"], label="Personalized twin", color="#dc2626")
        split_at = t["cycle_numbers"][~t["is_train"]][0] if (~t["is_train"]).any() else None
        if split_at is not None:
            ax.axvline(split_at, color="grey", ls=":", lw=1)
            ax.axvspan(split_at, t["cycle_numbers"][-1], color="grey", alpha=0.08, label="Held-out test")
        ax.set_title(vid if len(vid) < 25 else vid[:22] + "...")
        ax.set_xlabel("Cycle")
    axes[0][0].set_ylabel("State of Health (SoH)")
    axes[0][0].legend(fontsize=8)
    plt.suptitle(f"SoH prediction: trained on first {int(train_frac * 100)}% of cycles, "
                 "tested on the rest")
    plt.tight_layout()
    plt.savefig(out_dir / "twin_predictions.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    importance.sort_values().plot(kind="barh", color="#059669")
    plt.xlabel("Mean |SHAP value| (impact on predicted SoH)")
    plt.title("Global Feature Importance (Explainability Agent)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=150)
    plt.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--csv", nargs="+", help="one CSV per battery cell (real mode)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--rounds", type=int, default=200, help="FedAvg rounds (200 = training loss converged)")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0, help="MLP weight-init seed")
    ap.add_argument("--no-loco", action="store_true",
                    help="skip the leave-one-cell-out evaluation (runs by default with 2+ clients)")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTI-AGENT XAI FRAMEWORK FOR EV BATTERY DEGRADATION PREDICTION")
    print("=" * 70)

    feat_df, feature_cols = load_data(args, out_dir)
    feat_df = chronological_split(feat_df, args.train_frac)
    test_mask = (feat_df["split"] == "test").values

    # ---------- Agent 2: Federated Aggregator + Digital Twins ----------
    print("\n[Agent 2: Federated Aggregator] Training across clients "
          "(no raw data leaves any client)...")
    global_model, scaler, history, client_ids = run_federated_training(
        feat_df, feature_cols, n_rounds=args.rounds, local_epochs=25, lr=args.lr, seed=args.seed)
    print(f"  -> {len(client_ids)} FL client(s), final avg client loss = {history[-1]['avg_client_loss']:.6f}")
    if len(client_ids) == 1:
        print("  !! only one client: FedAvg reduces to ordinary local training. "
              "Pass more cells with --csv for a real federated setting.")

    print("[Agent 2: Digital Twin] Personalizing global model per client...")
    twins = build_digital_twins(global_model, scaler, feat_df, feature_cols, client_ids)
    twin_summary = {}
    for vid, t in twins.items():
        te = ~t["is_train"]
        twin_summary[vid] = {
            "n_train_cycles": int(t["is_train"].sum()), "n_test_cycles": int(te.sum()),
            "global_model_test": metrics(t["y"][te], t["y_pred_global"][te]),
            "personalized_twin_test": metrics(t["y"][te], t["y_pred_twin"][te]),
        }
        print(f"  -> {vid}: held-out MAE global={twin_summary[vid]['global_model_test']['mae']:.4f}  "
              f"twin={twin_summary[vid]['personalized_twin_test']['mae']:.4f}")

    y_test = np.concatenate([twins[v]["y"][~twins[v]["is_train"]] for v in client_ids])
    twin_test = np.concatenate([twins[v]["y_pred_twin"][~twins[v]["is_train"]] for v in client_ids])
    overall = {"mlp_digital_twin": metrics(y_test, twin_test), **baselines(feat_df, feature_cols)}
    print("\n  Held-out test comparison (all clients pooled):")
    for name, m in overall.items():
        print(f"     {name:28s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']}")

    loco = None
    if len(client_ids) > 1 and not args.no_loco:
        print("\n[Agent 2: Leave-one-cell-out] FedAvg on N-1 clients, tested on the unseen one...")
        loco = leave_one_cell_out(feat_df, feature_cols, args.rounds, args.lr, args.seed)
        print("  Leave-one-cell-out comparison (all unseen cells pooled):")
        for name, m in loco["pooled"].items():
            print(f"     {name:28s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']}")

    # ---------- Agent 3: Explainability ----------
    print("\n[Agent 3: Explainability] Fitting SHAP surrogate & generating explanations...")
    xai = train_surrogate_and_explain(feat_df, feature_cols,
                                      {vid: twins[vid]["y_pred_twin"] for vid in client_ids})
    importance = xai["importance"]
    print(f"  -> surrogate fidelity R2 vs twin predictions = {xai['fidelity_r2']:.3f}")
    print("  -> Top drivers of predicted SoH (global):")
    for feat_name, val in importance.head(5).items():
        print(f"     {feat_name}: {val:.4f}")

    rng = np.random.default_rng(0)
    sample_idxs = sorted(rng.choice(len(xai["feat_df"]), size=6, replace=False))
    sample_explanations = [explain_single_cycle(xai, feature_cols, i) for i in sample_idxs]
    (out_dir / "sample_explanations.txt").write_text("\n".join(sample_explanations) + "\n")
    print(f"  -> wrote {len(sample_explanations)} sample explanations to sample_explanations.txt")

    # ---------- Agent 4: Edge Deployment ----------
    print("\n[Agent 4: Edge Deployment] Benchmarking quantized inference per twin...")
    edge_reports = {}
    for vid, t in twins.items():
        te = ~t["is_train"]
        edge_reports[vid] = run_edge_report(t["model"], t["X"][te], t["y"][te])
        export_quantized_model(t["model"], scaler, feature_cols, out_dir / f"edge_model_{vid}.json")
        print(f"  -> {vid}: {t['model'].n_params()} params, "
              f"{edge_reports[vid]['avg_inference_latency_ms']} ms/inference, "
              f"{edge_reports[vid]['orig_size_bytes']} B fp32 -> {edge_reports[vid]['quantized_size_bytes']} B int8, "
              f"test MAE change {edge_reports[vid]['accuracy_delta']:+.5f}")

    # ---------- Visuals ----------
    print("\n[Reporting] Generating plots...")
    plot_all(out_dir, history, twins, client_ids, importance, args.train_frac)

    pipeline_report = {
        "data": {
            "mode": args.data,
            "sources": [_display_path(p) for p in (args.csv or [DEFAULT_CSV])] if args.data == "real" else "synthetic",
            "n_clients": len(client_ids),
            "n_labelled_cycles": int(len(feat_df)),
            "n_test_cycles": int(test_mask.sum()),
            "mlp_seed": args.seed,
            "split": f"chronological, first {args.train_frac:.0%} of each client's cycles for training",
            "features_per_cycle": list(feature_cols),
        },
        "held_out_test_metrics": overall,
        "leave_one_cell_out": loco,
        "federated_learning": {
            "n_rounds": len(history),
            "final_avg_client_loss": history[-1]["avg_client_loss"],
            "loss_history": history,
        },
        "digital_twins": twin_summary,
        "explainability": {
            "surrogate_fidelity_r2": xai["fidelity_r2"],
            "top_global_features": importance.head(5).round(5).to_dict(),
            "sample_explanations": sample_explanations,
        },
        "edge_deployment": edge_reports,
    }
    with open(out_dir / "pipeline_report.json", "w") as f:
        json.dump(pipeline_report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE. Full report: {out_dir / 'pipeline_report.json'}")
    print("=" * 70)
    return pipeline_report


if __name__ == "__main__":
    main()
