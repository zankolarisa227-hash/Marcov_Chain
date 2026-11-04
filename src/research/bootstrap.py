"""
bootstrap_pd.py — client-level bootstrap CI for PD / metrics on OOT.

Не переобучает модель (fixed-model bootstrap):
  - фиксируем dynamic model
  - много раз сэмплируем client_id из OOT с возвращением
  - считаем mean PD, Brier, Logloss, total ECL
  - берём percentile CI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def _safe_proba(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def _predict_pd1m(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    default_state: int = 2,
) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=float)
    states = df["state"].to_numpy(dtype=int)
    proba = model.predict_proba(states, X)
    if proba.ndim == 2:
        return proba[:, default_state]
    return np.asarray(proba, dtype=float)


def bootstrap_oot_metrics(
    model,
    oot_df: pd.DataFrame,
    feature_cols: list[str],
    default_state: int = 2,
    n_boot: int = 300,
    alpha: float = 0.05,
    ead: float = 50_000.0,
    lgd: float = 0.45,
    client_col: str = "client_id",
    target_col: str = "next_state",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Fixed-model client bootstrap on OOT.

    Returns point estimates + percentile CI.
    """
    need = {client_col, "state", target_col, *feature_cols}
    missing = [c for c in need if c not in oot_df.columns]
    if missing:
        raise ValueError(f"OOT df missing columns: {missing}")

    df = oot_df.dropna(subset=[target_col]).copy()
    if df.empty:
        raise ValueError("OOT is empty after dropna(next_state)")

    clients = df[client_col].unique()
    if len(clients) < 10:
        raise ValueError(f"Too few OOT clients for bootstrap: {len(clients)}")

    rng = np.random.default_rng(random_state)

    # point estimate on full OOT
    y_full = (df[target_col].astype(int) == default_state).to_numpy(dtype=int)
    pd_full = _predict_pd1m(model, df, feature_cols, default_state)
    pd_full = _safe_proba(pd_full)

    point = {
        "mean_pd": float(pd_full.mean()),
        "brier": float(brier_score_loss(y_full, pd_full)),
        "logloss": float(log_loss(y_full, pd_full)),
        "total_ecl": float((pd_full * ead * lgd).sum()),
        "actual_rate": float(y_full.mean()),
        "n_rows": int(len(df)),
        "n_clients": int(len(clients)),
    }

    # bootstrap storage
    keys = ["mean_pd", "brier", "logloss", "total_ecl", "actual_rate"]
    boot = {k: np.zeros(n_boot, dtype=float) for k in keys}

    for b in range(n_boot):
        sample_clients = rng.choice(clients, size=len(clients), replace=True)
        # gather rows of sampled clients (all months of those clients)
        parts = [df[df[client_col] == c] for c in sample_clients]
        # faster alternative without huge concat of many frames:
        # use membership on unique sample with multiplicity approx via merge
        # For clarity keep group-based selection:
        mask = df[client_col].isin(set(sample_clients))
        # To approximate with replacement multiplicity, expand by counts
        vc = pd.Series(sample_clients).value_counts()
        pieces = []
        for c, cnt in vc.items():
            sub = df[df[client_col] == c]
            if len(sub) == 0:
                continue
            if cnt == 1:
                pieces.append(sub)
            else:
                pieces.append(pd.concat([sub] * int(cnt), ignore_index=True))
        sample = pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0]

        if sample.empty:
            for k in keys:
                boot[k][b] = np.nan
            continue

        y = (sample[target_col].astype(int) == default_state).to_numpy(dtype=int)
        p = _safe_proba(_predict_pd1m(model, sample, feature_cols, default_state))

        boot["mean_pd"][b] = float(p.mean())
        boot["brier"][b] = float(brier_score_loss(y, p)) if y.size else np.nan
        boot["logloss"][b] = float(log_loss(y, p)) if y.size and y.min() != y.max() else np.nan
        # if only one class in y, log_loss may fail — guard:
        try:
            boot["logloss"][b] = float(log_loss(y, p, labels=[0, 1]))
        except Exception:
            boot["logloss"][b] = np.nan
        boot["total_ecl"][b] = float((p * ead * lgd).sum())
        boot["actual_rate"][b] = float(y.mean())

    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    ci = {}
    for k in keys:
        arr = boot[k]
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            ci[k] = {"low": None, "high": None, "std": None}
            continue
        ci[k] = {
            "low": float(np.percentile(arr, lo_q)),
            "high": float(np.percentile(arr, hi_q)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    return {
        "method": "fixed_model_client_bootstrap",
        "n_boot": n_boot,
        "alpha": alpha,
        "point": point,
        "ci": ci,
    }


def bootstrap_to_frame(result: dict) -> pd.DataFrame:
    rows = []
    for metric, point_val in result["point"].items():
        if metric in {"n_rows", "n_clients"}:
            rows.append({"metric": metric, "point": point_val, "ci_low": None, "ci_high": None, "std": None})
            continue
        c = result["ci"].get(metric, {})
        rows.append({
            "metric": metric,
            "point": point_val,
            "ci_low": c.get("low"),
            "ci_high": c.get("high"),
            "std": c.get("std"),
        })
    return pd.DataFrame(rows)


def save_bootstrap_report(
    result: dict,
    out_dir: str | Path = "reports/research/bootstrap",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = bootstrap_to_frame(result)
    path = out / "bootstrap_pd_ci.csv"
    frame.to_csv(path, index=False)

    meta = {
        "method": result["method"],
        "n_boot": result["n_boot"],
        "alpha": result["alpha"],
    }
    pd.Series(meta).to_csv(out / "bootstrap_meta.csv", header=False)
    print(f"[bootstrap] saved {path}")
    return path


if __name__ == "__main__":
    print("Use bootstrap_oot_metrics(model, oot_df, feature_cols) from train/research runner.")