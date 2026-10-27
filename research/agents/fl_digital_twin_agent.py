"""
Agent 2 -- Federated Aggregator Agent  +  Digital Twin (per-vehicle) logic

Role:
  - Each vehicle/cell is an FL "client": trains locally on its own cycles only,
    never uploads raw data -- only model weights.
  - The aggregator agent runs FedAvg across clients each round to build a
    shared global degradation model.
  - After FL converges, each vehicle keeps a personalized "digital twin":
    the global model fine-tuned briefly on that vehicle's own cycles, so the
    twin tracks that specific battery's drift from the fleet average.

Only rows with split == "train" are ever used for fitting (scaler, FedAvg,
twin fine-tuning). Rows with split == "test" are the later cycles of each
client and are only predicted, never trained on.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.append(os.path.dirname(__file__))
from mlp import SoHRegressor, fed_avg


def _train_rows(df):
    return df[df["split"] == "train"]


def run_federated_training(feat_df, feature_cols, n_rounds=15, local_epochs=25,
                           lr=0.05, hidden=16, seed=0):
    """
    Simulates federated learning across all clients present in feat_df.
    Returns: global_model, scaler, history (per-round avg loss), client_ids
    """
    client_ids = sorted(feat_df["Vehicle_ID"].unique())
    train_df = _train_rows(feat_df)

    # Normalization stats from training rows only (realistic FL setups agree
    # on normalization stats up front; raw data itself is never pooled)
    scaler = StandardScaler().fit(train_df[feature_cols].values.astype(float))

    global_model = SoHRegressor(len(feature_cols), hidden=hidden, seed=seed)

    history = []
    for rnd in range(1, n_rounds + 1):
        client_weights, client_counts, client_losses = [], [], []
        for vid in client_ids:
            client_df = train_df[train_df["Vehicle_ID"] == vid]
            X = scaler.transform(client_df[feature_cols].values.astype(float))
            y = client_df["SoH_true"].values.astype(float)

            local_model = SoHRegressor(len(feature_cols), hidden=hidden, seed=seed)
            local_model.set_weights(global_model.get_weights())  # start from global
            losses = local_model.fit(X, y, epochs=local_epochs, lr=lr)

            client_weights.append(local_model.get_weights())
            client_counts.append(len(X))
            client_losses.append(losses[-1])

        # aggregator agent: FedAvg
        global_model.set_weights(fed_avg(client_weights, client_counts))
        history.append({"round": rnd, "avg_client_loss": float(np.mean(client_losses))})

    return global_model, scaler, history, client_ids


def build_digital_twins(global_model, scaler, feat_df, feature_cols, client_ids,
                        finetune_epochs=40, lr=0.03):
    """
    Each vehicle's digital twin = global model + brief fine-tuning on that
    vehicle's training cycles. Predictions are returned for ALL its cycles.
    Returns dict: vehicle_id -> {model, X, y, split, y_pred_global, y_pred_twin, ...}
    """
    twins = {}
    for vid in client_ids:
        client_df = feat_df[feat_df["Vehicle_ID"] == vid]
        is_train = (client_df["split"] == "train").values
        X = scaler.transform(client_df[feature_cols].values.astype(float))
        y = client_df["SoH_true"].values.astype(float)

        twin = SoHRegressor(len(feature_cols), hidden=global_model.hidden, seed=1)
        twin.set_weights(global_model.get_weights())
        y_pred_global = twin.predict(X).copy()
        twin.fit(X[is_train], y[is_train], epochs=finetune_epochs, lr=lr)

        twins[vid] = {
            "model": twin,
            "X": X, "y": y, "is_train": is_train,
            "y_pred_global": y_pred_global,
            "y_pred_twin": twin.predict(X),
            "cycle_numbers": client_df["Cycle_Number"].values,
        }
    return twins
