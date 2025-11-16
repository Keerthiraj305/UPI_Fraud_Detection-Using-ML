from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib


def predict(model_path: str | Path, X: pd.DataFrame):
    clf = joblib.load(model_path)
    return clf.predict(X.fillna(0))


def explain_shap(feat_series: pd.Series, model, feature_names: List[str] | None = None) -> pd.Series:
    """Return SHAP-like contributions per feature for a single row.

    - If SHAP is installed, uses `shap.Explainer` (or `TreeExplainer`) and returns class-1 attributions.
    - Else, falls back to `feature_importances_ * value` heuristic (normalized).
    """
    # ensure feature order
    if feature_names is None:
        feature_names = list(feat_series.index)
    x = pd.DataFrame([[feat_series.get(f, 0.0) for f in feature_names]], columns=feature_names)

    try:
        import shap  # type: ignore

        explainer = None
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer(x)
        except Exception:
            explainer = shap.Explainer(model)
            sv = explainer(x)
        values = sv.values
        # values shape handling: (1, nfeat) or (1, nclasses, nfeat)
        if values.ndim == 3:
            # pick positive class 1 if available
            if values.shape[1] > 1:
                contrib = values[0, 1, :]
            else:
                contrib = values[0, 0, :]
        else:
            contrib = values[0]
        return pd.Series(contrib, index=feature_names)
    except Exception:
        # simple fallback using feature importances if available
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is None:
                raise AttributeError
            contrib = np.array(importances) * x.values[0]
            # scale to sum to 1 in abs sense for display
            s = np.sum(np.abs(contrib)) or 1.0
            contrib = contrib / s
            return pd.Series(contrib, index=feature_names)
        except Exception:
            # last resort: zeros
            return pd.Series(np.zeros(len(feature_names)), index=feature_names)


if __name__ == "__main__":
    print("predict module ready — functions: predict, explain_shap")
