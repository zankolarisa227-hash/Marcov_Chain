"""
batch_score.py — batch scoring processed features

Production-like flow:
  processed features + trained dynamic model
    → pd_1m, ecl_1m
    → scores parquet + summary json

Research-friendly options:
  - oot_only=True  → score only future fold
  - summary stats for report
  - explicit limitations in meta
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.processed_reader import load_processed_features
from src.models.dynamic_model import DynamicMarkovModel


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pd1m_from_model(
    model: DynamicMarkovModel,
    features: pd.DataFrame,
    feature_cols: list[str],
    default_state: int,
) -> np.ndarray:
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = features[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    states = features["state"].to_numpy(dtype=int)

    proba = model.predict_proba(states, X)
    if proba.ndim != 2 or proba.shape[1] <= default_state:
        raise ValueError(f"Unexpected predict_proba shape: {getattr(proba, 'shape', None)}")
    return proba[:, default_state].astype(float)


def batch_score(
    config_path: str = "configs/default.yaml",
    oot_only: bool = False,
) -> pd.DataFrame:
    """
    Score processed feature rows.

    oot_only:
      False → all rows in processed features
      True  → only month >= split_month (for evaluation/report)
    """
    cfg = load_config(config_path)

    features_dir = cfg["paths"]["features_dir"]
    models_dir = Path(cfg["paths"]["models_dir"])
    scores_dir = Path(cfg["paths"]["scores_dir"])
    scores_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "dynamic_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run training first."
        )

    features = load_processed_features(features_dir)
    if features.empty:
        raise ValueError("Processed features are empty")

    if oot_only:
        split_month = cfg["split"]["split_month"]
        features = features[features["month"] >= split_month].copy()
        if features.empty:
            raise ValueError(
                f"No OOT rows for month >= {split_month}. Check split_month/data."
            )

    model = DynamicMarkovModel.load(str(model_path))
    feature_cols = list(model.feature_cols)

    default_state = int(cfg["model"]["default_state"])
    lgd = float(cfg["risk"]["lgd"])
    ead = float(cfg["risk"]["ead_default"])

    pd_1m = _pd1m_from_model(
        model=model,
        features=features,
        feature_cols=feature_cols,
        default_state=default_state,
    )

    out = features[["client_id", "month", "state"]].copy()
    out["pd_1m"] = pd_1m
    out["ecl_1m"] = out["pd_1m"] * ead * lgd

    stem = "scores_oot" if oot_only else "scores"
    out_path = scores_dir / f"{stem}.parquet"
    out.to_parquet(out_path, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(out)),
        "oot_only": bool(oot_only),
        "split_month": cfg["split"].get("split_month"),
        "default_state": default_state,
        "lgd": lgd,
        "ead_default": ead,
        "model_path": str(model_path),
        "backend": getattr(model, "backend_", "unknown"),
        "feature_cols": feature_cols,
        "pd_1m_mean": float(np.mean(pd_1m)) if len(pd_1m) else None,
        "pd_1m_std": float(np.std(pd_1m)) if len(pd_1m) else None,
        "pd_1m_p50": float(np.median(pd_1m)) if len(pd_1m) else None,
        "pd_1m_p95": float(np.quantile(pd_1m, 0.95)) if len(pd_1m) else None,
        "pd_1m_max": float(np.max(pd_1m)) if len(pd_1m) else None,
        "ecl_1m_total": float(out["ecl_1m"].sum()) if len(out) else 0.0,
        "ecl_1m_mean": float(out["ecl_1m"].mean()) if len(out) else 0.0,
        "limitations": [
            "pd_1m is one-step transition probability into default_state",
            "ecl_1m uses exogenous LGD and EAD assumptions",
            "this is portfolio scoring output, not a full IFRS9 staging engine",
        ],
    }

    summary_path = scores_dir / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[batch_score] saved {out_path} rows={len(out):,}")
    if summary["pd_1m_mean"] is not None:
        print(
            f"[batch_score] pd_mean={summary['pd_1m_mean']:.4f} "
            f"pd_p95={summary['pd_1m_p95']:.4f} "
            f"ecl_total={summary['ecl_1m_total']:.2f}"
        )
    print(f"[batch_score] summary: {summary_path}")
    return out


if __name__ == "__main__":
    # full portfolio
    batch_score(oot_only=False)
    # OOT-only for evaluation/report
    try:
        batch_score(oot_only=True)
    except ValueError as e:
        print(f"[batch_score] OOT scoring skipped: {e}")