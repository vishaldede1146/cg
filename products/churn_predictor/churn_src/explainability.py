"""
SHAP explainability for the sequence LSTM model.

We use shap.GradientExplainer, which supports Keras models with 3D inputs
directly (unlike KernelExplainer, which would need flattening and is far
slower). Attributions come back with shape (n_samples, timesteps, n_features);
we aggregate across timesteps (sum of absolute contribution) to get a single
importance value per feature, which is what's shown to business users, plus
we keep the raw per-timestep matrix for a "recent months mattered most" view.
"""

import os
import numpy as np

from churn_src import config


def get_explainer(model, background: np.ndarray):
    import shap
    return shap.GradientExplainer(model, background)


def explain_instance(model, background: np.ndarray, X_instance: np.ndarray, feature_names: list):
    """
    X_instance: shape (1, timesteps, n_features) — a single scaled customer sequence.
    Returns dict with:
        feature_importance: {feature_name: aggregated_abs_shap_value}
        per_timestep: raw (timesteps, n_features) shap matrix
        top_features: sorted list of (feature_name, value) most driving churn risk
    """
    explainer = get_explainer(model, background)
    shap_values = explainer.shap_values(X_instance)

    # shap_values shape can be (1, timesteps, n_features, 1) or (1, timesteps, n_features)
    sv = np.array(shap_values)
    sv = sv.reshape(X_instance.shape[0], X_instance.shape[1], X_instance.shape[2])
    sv_instance = sv[0]  # (timesteps, n_features)

    agg_importance = np.abs(sv_instance).sum(axis=0)  # sum |contribution| across time
    signed_importance = sv_instance.sum(axis=0)  # net direction across time

    feature_importance = {
        feature_names[i]: float(agg_importance[i]) for i in range(len(feature_names))
    }
    signed = {
        feature_names[i]: float(signed_importance[i]) for i in range(len(feature_names))
    }

    top_features = sorted(
        [(f, signed[f]) for f in feature_names],
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:8]

    return {
        "feature_importance": feature_importance,
        "signed_importance": signed,
        "per_timestep": sv_instance.tolist(),
        "top_features": top_features,
    }


def load_background():
    return np.load(config.SHAP_BACKGROUND_PATH)


if __name__ == "__main__":
    import json
    from tensorflow import keras

    model = keras.models.load_model(config.MODEL_PATH)
    background = load_background()

    test = np.load(os.path.join(config.ARTIFACTS_DIR, "test_split.npz"))
    X_test = test["X_test"]

    with open(config.FEATURE_LIST_PATH) as f:
        feature_names = json.load(f)

    result = explain_instance(model, background, X_test[:1], feature_names)
    print("Top features driving this prediction (feature, signed shap value):")
    for feat, val in result["top_features"]:
        direction = "increases" if val > 0 else "decreases"
        print(f"  {feat}: {val:.4f} ({direction} churn risk)")
