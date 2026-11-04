"""shap_analysis.py — объяснение dynamic Markov через SHAP."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _default_proba_fn(model, state: int, X: np.ndarray) -> np.ndarray:
    """P(default | state, x) для каждой строки X."""
    out = np.zeros(len(X), dtype=float)
    for i, x in enumerate(X):
        probs = model.predict_proba_row(int(state), x)
        out[i] = float(probs[2])  # default_state = 2
    return out


def shap_for_state(
    model,
    X: pd.DataFrame,
    state: int,
    feature_cols: list[str],
    max_samples: int = 500,
    random_state: int = 42,
) -> dict:
    """
    SHAP для фиксированного исходного состояния (0 или 1).
    Объясняем вероятность перехода в default.
    """
    if not HAS_SHAP:
        raise ImportError("pip install shap")

    X = X[feature_cols].copy()
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)

    X_np = X.to_numpy(dtype=float)

    # Для LogReg удобен LinearExplainer; fallback — KernelExplainer на подвыборке
    def f(data: np.ndarray) -> np.ndarray:
        return _default_proba_fn(model, state, data)

    # background для Kernel/Linear
    background = shap.sample(X, min(100, len(X)), random_state=random_state)

    try:
        # если внутри state-модели есть sklearn LogisticRegression
        clf = model.models_.get(int(state))
        scaler = model.scalers_.get(int(state))
        if clf is not None and hasattr(clf, "coef_"):
            # объясняем уже scaled-пространство через wrapper
            def f_scaled(data: np.ndarray) -> np.ndarray:
                xs = scaler.transform(data)
                # proba класса default, если класс есть
                proba = clf.predict_proba(xs)
                classes = list(clf.classes_)
                if 2 in classes:
                    return proba[:, classes.index(2)]
                return np.zeros(len(data))

            explainer = shap.LinearExplainer(clf, scaler.transform(background.to_numpy(float)))
            shap_values = explainer.shap_values(scaler.transform(X_np))
            # multi-class: берём output для default
            if isinstance(shap_values, list):
                classes = list(clf.classes_)
                shap_values = shap_values[classes.index(2)] if 2 in classes else shap_values[-1]
            expected = explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                classes = list(clf.classes_)
                expected = expected[classes.index(2)] if 2 in classes else expected[-1]
        else:
            explainer = shap.KernelExplainer(f, background.to_numpy(float))
            shap_values = explainer.shap_values(X_np, nsamples=100)
            expected = explainer.expected_value
    except Exception:
        explainer = shap.KernelExplainer(f, background.to_numpy(float))
        shap_values = explainer.shap_values(X_np, nsamples=100)
        expected = explainer.expected_value

    shap_values = np.asarray(shap_values, dtype=float)
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    importance = (
        pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "state": int(state),
        "expected_value": float(np.asarray(expected).reshape(-1)[0]),
        "importance": importance,
        "shap_values": shap_values,
        "X": X,
    }


def global_shap_report(
    model,
    features_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: str = "reports/shap",
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Global SHAP отдельно для state=0 и state=1.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for state in (0, 1):
        sub = features_df[features_df["state"] == state]
        if sub.empty:
            continue
        res = shap_for_state(model, sub, state, feature_cols, max_samples=max_samples)
        imp = res["importance"].copy()
        imp["state"] = state
        imp.to_csv(out / f"shap_importance_state_{state}.csv", index=False)
        rows.append(imp)

        # summary plot (если есть matplotlib)
        try:
            import matplotlib.pyplot as plt
            shap.summary_plot(
                res["shap_values"],
                res["X"],
                feature_names=feature_cols,
                show=False,
                max_display=15,
            )
            plt.tight_layout()
            plt.savefig(out / f"shap_summary_state_{state}.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[shap] plot skipped for state={state}: {e}")

    report = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not report.empty:
        report.to_csv(out / "shap_importance_all.csv", index=False)
    print(f"[shap] saved to {out}")
    return report


def local_shap_row(
    model,
    x: np.ndarray,
    state: int,
    feature_cols: list[str],
    background_X: pd.DataFrame,
) -> pd.DataFrame:
    """Локальное объяснение одного клиента."""
    if not HAS_SHAP:
        raise ImportError("pip install shap")

    x = np.asarray(x, dtype=float).reshape(1, -1)
    bg = background_X[feature_cols].sample(min(50, len(background_X)), random_state=42)

    def f(data: np.ndarray) -> np.ndarray:
        return _default_proba_fn(model, state, data)

    explainer = shap.KernelExplainer(f, bg.to_numpy(float))
    sv = np.asarray(explainer.shap_values(x, nsamples=100)).reshape(-1)

    return (
        pd.DataFrame({
            "feature": feature_cols,
            "value": x.reshape(-1),
            "shap": sv,
        })
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .drop(columns=["abs_shap"])
        .reset_index(drop=True)
    )