"""
Agent 3 -- Explainability Agent

Role: attach a "reason" to every degradation prediction. SHAP's TreeExplainer
needs a tree model, but our production/edge model is the tiny NumPy MLP
(chosen for FedAvg-friendliness and edge footprint). So this agent trains a
RandomForest *surrogate* on the same features + the twin's own predictions as
target -- a standard technique for explaining black-box models that aren't
natively tree-based -- and runs SHAP on that surrogate.

The surrogate's fidelity (R^2 against the twin's predictions) is reported:
an explanation is only as trustworthy as the surrogate is faithful.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def train_surrogate_and_explain(feat_df, feature_cols, twin_predictions_by_vehicle):
    """
    twin_predictions_by_vehicle: dict vid -> array of predicted SoH (from digital twin)
    aligned with feat_df rows for that vehicle, in Cycle_Number order.
    """
    feat_df = feat_df.sort_values(["Vehicle_ID", "Cycle_Number"]).reset_index(drop=True)
    y_surrogate_target = np.concatenate(
        [twin_predictions_by_vehicle[vid] for vid in sorted(feat_df["Vehicle_ID"].unique())]
    )

    X = feat_df[feature_cols].values
    surrogate = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=0)
    surrogate.fit(X, y_surrogate_target)
    fidelity_r2 = float(r2_score(y_surrogate_target, surrogate.predict(X)))

    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)

    return {
        "surrogate": surrogate, "explainer": explainer, "shap_values": shap_values,
        "importance": importance, "fidelity_r2": fidelity_r2,
        "feat_df": feat_df, "predictions": y_surrogate_target,
    }


def explain_single_cycle(xai, feature_cols, row_idx):
    """Human-readable explanation string for one specific cycle's prediction."""
    feat_df, shap_values = xai["feat_df"], xai["shap_values"]
    row = feat_df.iloc[row_idx]
    base = float(np.ravel(xai["explainer"].expected_value)[0])
    contribs = pd.Series(shap_values[row_idx], index=feature_cols)
    top = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(3)

    total_abs = contribs.abs().sum()
    parts = []
    for feat_name, val in top.items():
        pct = 100 * abs(val) / total_abs if total_abs > 0 else 0
        direction = "raises" if val > 0 else "lowers"
        parts.append(f"{feat_name}={row[feat_name]:.3g} {direction} SoH by {abs(val):.3f} "
                     f"({pct:.0f}% of the shift)")

    return (f"[{row['Vehicle_ID']} / cycle {int(row['Cycle_Number'])}] "
            f"predicted SoH {xai['predictions'][row_idx]:.3f} (true {row['SoH_true']:.3f}, "
            f"average prediction {base:.3f}): " + "; ".join(parts) + ".")
