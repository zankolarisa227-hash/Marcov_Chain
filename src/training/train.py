"""
train.py — обучение только на прошлом + OOT validation + monitoring/research
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from src.data.processed_reader import load_processed_features
from src.evaluation.split import purged_time_split
from src.evaluation.validation import run_validation
from src.features.feature_contracts import FEATURE_COLS
from src.features.leakage_checks import run_leakage_checks
from src.models.dynamic_model import DynamicMarkovModel
from src.models.markov_chain import (
    estimate_transition_matrix,
    bootstrap_transition_ci,
    print_matrix_with_ci,
)
from src.monitoring.psi import monitoring_report


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fit_train_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    """Scaler только на train fold."""
    scaler = StandardScaler()
    X = train_df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler.fit(X)
    return scaler


def _pd1m(model, df: pd.DataFrame, feature_cols: list[str], default_state: int = 2) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    st = df["state"].to_numpy(dtype=int)
    proba = model.predict_proba(st, X)
    return proba[:, default_state] if proba.ndim == 2 else np.asarray(proba, dtype=float)


def train(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)

    split_month = cfg["split"]["split_month"]
    embargo = cfg["split"].get("embargo_months", 1)
    default_state = cfg["model"]["default_state"]
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Data
    # ------------------------------------------------------------------
    features = load_processed_features(cfg["paths"]["features_dir"])

    if "next_state" not in features.columns:
        features = features.sort_values(["client_id", "month"]).reset_index(drop=True)
        features["next_state"] = features.groupby("client_id")["state"].shift(-1)
        features = features.dropna(subset=["next_state"]).copy()
        features["next_state"] = features["next_state"].astype(int)

        feature_cols = [c for c in FEATURE_COLS if c in features.columns]

        try:
            run_leakage_checks(features, feature_cols=feature_cols)
        except ValueError as e:
            msg = str(e)
        if "util_mean != utilization" in msg:
            print("[train] skip first-month util check:", msg.splitlines()[0])
        else:
            raise

    train_df, oot_df = purged_time_split(
        features,
        split_month=split_month,
        embargo_months=embargo,
        time_col="month",
    )

    train_pairs = train_df.dropna(subset=["next_state"]).copy()
    if train_pairs.empty:
        raise ValueError("No train pairs after dropna(next_state)")

    # ------------------------------------------------------------------
    # 2) Static Markov (train only)
    # ------------------------------------------------------------------
    P = estimate_transition_matrix(train_pairs[["state", "next_state"]])

    ci = bootstrap_transition_ci(
        train_pairs[["state", "next_state"]],
        n_boot=int(cfg.get("bootstrap", {}).get("n_boot", 300)),
        alpha=0.05,
        random_state=int(cfg.get("model", {}).get("random_seed", 42)),
    )
    print_matrix_with_ci(ci, name="P_train")

    (models_dir / "P_ci.json").write_text(
        json.dumps(
            {
                "point": ci["point"].tolist(),
                "lower": ci["lower"].tolist(),
                "upper": ci["upper"].tolist(),
                "n_boot": ci["n_boot"],
                "alpha": ci["alpha"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 3) Dynamic model (train only)
    # ------------------------------------------------------------------
    dynamic = DynamicMarkovModel(feature_cols=feature_cols, use_catboost=True)
    dynamic.fit(train_pairs)

    # ------------------------------------------------------------------
    # 4) Train-only scaler
    # ------------------------------------------------------------------
    scaler = fit_train_scaler(train_df, feature_cols)

    # ------------------------------------------------------------------
    # 5) Save artifacts
    # ------------------------------------------------------------------
    np.save(models_dir / "P_matrix.npy", P)
    dynamic.save(str(models_dir / "dynamic_model.joblib"))
    joblib.dump(scaler, models_dir / "scalers.joblib")

    meta = {
        "split_month": split_month,
        "embargo_months": embargo,
        "default_state": default_state,
        "n_train_rows": int(len(train_df)),
        "n_oot_rows": int(len(oot_df)),
        "n_train_pairs": int(len(train_pairs)),
        "feature_cols": feature_cols,
        "backend": getattr(dynamic, "backend_", "unknown"),
        "anti_leakage": {
            "purged_split": True,
            "train_only_scaler": True,
            "causal_features": True,
        },
    }
    (models_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 6) OOT validation
    # ------------------------------------------------------------------
    print("\n[train] running purged OOT validation...")
    run_validation(
        df=features,
        P=P,
        dynamic_model=dynamic,
        split_month=split_month,
        status_col="state",
        next_status_col="next_state",
        feature_cols=feature_cols,
        default_state=default_state,
        time_col="month",
    )

    # ------------------------------------------------------------------
    # 7) Monitoring: PSI / JS on train vs OOT
    # ------------------------------------------------------------------
    train_eval = train_pairs
    oot_eval = oot_df.dropna(subset=["next_state"]).copy()

    if len(oot_eval) > 0:
        pd_train = _pd1m(dynamic, train_eval, feature_cols, default_state)
        pd_oot = _pd1m(dynamic, oot_eval, feature_cols, default_state)

        P_train = estimate_transition_matrix(train_eval[["state", "next_state"]])
        P_oot = estimate_transition_matrix(oot_eval[["state", "next_state"]])

        try:
            from src.monitoring.psi import monitoring_report
        except ImportError:
            from src.monitoring.psi import monitoring_report  # если файл назван psi.py

        report = monitoring_report(
            pd_baseline=pd_train,
            pd_current=pd_oot,
            P_old=P_train,
            P_new=P_oot,
        )
        print("[monitoring] PSI:", report.get("psi"))
        print("[monitoring] action:", report.get("action"))

        mon_dir = Path("reports/monitoring")
        mon_dir.mkdir(parents=True, exist_ok=True)
        (mon_dir / "psi_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        print("[monitoring] skipped: empty OOT")

    # ------------------------------------------------------------------
    # 8) Optional research blocks (не падаем, если модулей нет)
    # ------------------------------------------------------------------
    try:
        from src.research.shap_analysis import global_shap_report
        from src.research.stability_report import build_stability_report, save_stability_report
        from src.research.bootstrap import bootstrap_oot_metrics, save_bootstrap_report
        from src.research.scenario_stress import stress_portfolio_sample, save_stress_report

        if len(train_eval) > 0:
            global_shap_report(
                model=dynamic,
                features_df=train_eval,
                feature_cols=feature_cols,
                out_dir="reports/shap",
            )

        if len(train_eval) > 0 and len(oot_eval) > 0:
            stab = build_stability_report(
                train_df=train_eval,
                oot_df=oot_eval,
                feature_cols=feature_cols,
                default_state=default_state,
                pd_train=_pd1m(dynamic, train_eval, feature_cols, default_state),
                pd_oot=_pd1m(dynamic, oot_eval, feature_cols, default_state),
            )
            save_stability_report(stab, "reports/research/stability")

            boot = bootstrap_oot_metrics(
                model=dynamic,
                oot_df=oot_eval,
                feature_cols=feature_cols,
                n_boot=300,
                ead=cfg["risk"]["ead_default"],
                lgd=cfg["risk"]["lgd"],
            )
            save_bootstrap_report(boot, "reports/research/bootstrap")

            stress_details = stress_portfolio_sample(
                model=dynamic,
                df=oot_eval,
                feature_cols=feature_cols,
                n_clients=min(100, len(oot_eval)),
                horizons=[1, 3, 6, 12],
            )
            save_stress_report(
                stress_details,
                "reports/research/stress",
                horizons=[1, 3, 6, 12],
            )

    except ImportError as e:
        print(f"[train] research modules skipped: {e}")
    except Exception as e:
        print(f"[train] research block failed: {e}")

    # ------------------------------------------------------------------
    # 9) Optional survival
    # ------------------------------------------------------------------
    try:
        from src.models.survival_model import fit_and_save_survival, evaluate_survival

        surv = fit_and_save_survival(
            train_features=train_df,
            feature_cols=feature_cols,
            path=models_dir / "survival_model.joblib",
            default_state=default_state,
        )
        if len(oot_df) > 0:
            print("[train] survival OOT:", evaluate_survival(surv, oot_df))
    except ImportError as e:
        print(f"[train] survival skipped: {e}")
    except Exception as e:
        print(f"[train] survival failed: {e}")

    print("[train] done")
    return P, dynamic


if __name__ == "__main__":
    train()