"""
stability_report.py — train vs OOT stability for bank monitoring style.

Проверяет:
  - default rate shift
  - PSI по признакам
  - JS drift transition matrix
  - optional metric shift if predictions provided
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")

    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, qs))
    if len(cuts) < 3:
        return 0.0

    e_hist, _ = np.histogram(expected, bins=cuts)
    a_hist, _ = np.histogram(actual, bins=cuts)
    e = e_hist / max(e_hist.sum(), 1)
    a = a_hist / max(a_hist.sum(), 1)
    e = np.clip(e, eps, None); e = e / e.sum()
    a = np.clip(a, eps, None); a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 0, None) + eps
    q = np.clip(q, 0, None) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def transition_counts(
    df: pd.DataFrame,
    state_col: str = "state",
    next_col: str = "next_state",
    n_states: int = 3,
) -> np.ndarray:
    tmp = df.dropna(subset=[next_col]).copy()
    N = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        sub = tmp[tmp[state_col] == i]
        if sub.empty:
            continue
        vc = sub[next_col].astype(int).value_counts()
        for j, cnt in vc.items():
            if 0 <= int(j) < n_states:
                N[i, int(j)] = float(cnt)
    # absorbing default
    N[n_states - 1, :] = 0.0
    N[n_states - 1, n_states - 1] = max(N[n_states - 1, n_states - 1], 1.0)
    return N


def counts_to_matrix(N: np.ndarray) -> np.ndarray:
    P = np.zeros_like(N, dtype=float)
    for i in range(N.shape[0] - 1):
        s = N[i].sum()
        P[i] = N[i] / s if s > 0 else np.eye(N.shape[0])[i]
    P[-1, :] = 0.0
    P[-1, -1] = 1.0
    return P


def matrix_js_report(P_old: np.ndarray, P_new: np.ndarray) -> dict[str, Any]:
    names = {0: "current", 1: "delinquent", 2: "default"}
    per_state = {}
    for i in range(P_old.shape[0]):
        per_state[names.get(i, str(i))] = round(js_divergence(P_old[i], P_new[i]), 6)
    worst = max(per_state, key=per_state.get)
    return {
        "per_state_js": per_state,
        "max_js": per_state[worst],
        "worst_state": worst,
    }


def default_rate(df: pd.DataFrame, next_col: str = "next_state", default_state: int = 2) -> float:
    tmp = df.dropna(subset=[next_col])
    if tmp.empty:
        return float("nan")
    return float((tmp[next_col].astype(int) == default_state).mean())


def feature_psi_table(
    train_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        if c not in train_df.columns or c not in oot_df.columns:
            continue
        val = psi(train_df[c].to_numpy(), oot_df[c].to_numpy(), bins=bins)
        rows.append({"feature": c, "psi": val})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("psi", ascending=False).reset_index(drop=True)
        out["psi_flag"] = pd.cut(
            out["psi"],
            bins=[-np.inf, 0.10, 0.25, np.inf],
            labels=["stable", "moderate", "severe"],
        )
    return out


def decide_action(
    psi_table: pd.DataFrame,
    default_rate_train: float,
    default_rate_oot: float,
    matrix_js: dict[str, Any],
    psi_severe: float = 0.25,
    rate_shift_alert: float = 0.02,
    js_alert: float = 0.02,
) -> dict[str, Any]:
    severe_psi_n = 0
    moderate_psi_n = 0
    if not psi_table.empty and "psi" in psi_table.columns:
        severe_psi_n = int((psi_table["psi"] >= psi_severe).sum())
        moderate_psi_n = int(((psi_table["psi"] >= 0.10) & (psi_table["psi"] < psi_severe)).sum())

    rate_shift = abs(default_rate_oot - default_rate_train) if np.isfinite(default_rate_oot) and np.isfinite(default_rate_train) else 0.0
    js_alert_flag = bool(matrix_js.get("max_js", 0.0) > js_alert)

    if severe_psi_n > 0 or rate_shift >= rate_shift_alert or js_alert_flag:
        action = "retrain_recommended"
        level = "severe"
    elif moderate_psi_n > 0:
        action = "investigate"
        level = "moderate"
    else:
        action = "no_action"
        level = "stable"

    return {
        "level": level,
        "action": action,
        "severe_psi_features": severe_psi_n,
        "moderate_psi_features": moderate_psi_n,
        "default_rate_shift_abs": float(rate_shift),
        "matrix_js_alert": js_alert_flag,
    }

def build_stability_report(
    train_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    feature_cols: list[str],
    default_state: int = 2,
    bins: int = 10,
    # новое:
    pd_train: np.ndarray | None = None,
    pd_oot: np.ndarray | None = None,
) -> dict:
    dr_tr = default_rate(train_df, default_state=default_state)
    dr_oot = default_rate(oot_df, default_state=default_state)

    N_tr = transition_counts(train_df)
    N_oot = transition_counts(oot_df)
    P_tr = counts_to_matrix(N_tr)
    P_oot = counts_to_matrix(N_oot)
    js_info = matrix_js_report(P_tr, P_oot)

    psi_table = feature_psi_table(train_df, oot_df, feature_cols, bins=bins)

    # --- PD score drift ---
    pd_psi = None
    monthly_oot = None
    if pd_train is not None and pd_oot is not None:
        pd_psi = psi_score(pd_train, pd_oot, bins=bins)
        # monthly только на OOT (и train при желании)
        oot_aligned = oot_df.dropna(subset=["next_state"]).copy()
        if len(pd_oot) == len(oot_aligned):
            monthly_oot = monthly_stability(
                oot_aligned, pd_oot, default_state=default_state
            )

    decision = decide_action_v2(
        psi_table=psi_table,
        default_rate_train=dr_tr,
        default_rate_oot=dr_oot,
        matrix_js=js_info,
        pd_psi=pd_psi,
    )

    return {
        "default_rate_train": dr_tr,
        "default_rate_oot": dr_oot,
        "psi_table": psi_table,
        "matrix_js": js_info,
        "P_train": P_tr,
        "P_oot": P_oot,
        "N_train": N_tr,
        "N_oot": N_oot,
        "pd_score_psi": pd_psi,
        "monthly_oot": monthly_oot,
        "decision": decision,
    }

def save_stability_report(report: dict, out_dir: str | Path = "reports/research/stability") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report["psi_table"].to_csv(out / "psi_features.csv", index=False)
    pd.DataFrame(report["P_train"]).to_csv(out / "P_train.csv", index=False)
    pd.DataFrame(report["P_oot"]).to_csv(out / "P_oot.csv", index=False)

    if report.get("monthly_oot") is not None:
        report["monthly_oot"].to_csv(out / "monthly_oot_stability.csv")

    summary = {
        "default_rate_train": report["default_rate_train"],
        "default_rate_oot": report["default_rate_oot"],
        "max_matrix_js": report["matrix_js"]["max_js"],
        "worst_state_js": report["matrix_js"]["worst_state"],
        "pd_score_psi": report.get("pd_score_psi"),
        **report["decision"],
    }
    pd.Series(summary).to_csv(out / "stability_summary.csv", header=False)
    print(f"[stability] action={summary['action']} pd_psi={summary.get('pd_score_psi')}")
    return out




def psi_score(
    pd_ref: np.ndarray,
    pd_cur: np.ndarray,
    bins: int = 10,
) -> float:
    """PSI по распределению PD-score."""
    return psi(pd_ref, pd_cur, bins=bins)


def monthly_stability(
    df: pd.DataFrame,
    pd_scores: np.ndarray,
    next_col: str = "next_state",
    default_state: int = 2,
    time_col: str = "month",
) -> pd.DataFrame:
    """
    По месяцам: actual default rate и mean PD.
    df и pd_scores должны быть выровнены по строкам.
    """
    tmp = df.copy()
    tmp = tmp.dropna(subset=[next_col]).copy()
    # если pd_scores считались на df до dropna — лучше сразу передавать уже очищенный df
    if len(pd_scores) != len(tmp):
        raise ValueError("pd_scores length must match df after alignment")

    tmp = tmp.copy()
    tmp["pd"] = np.asarray(pd_scores, dtype=float)
    tmp["y"] = (tmp[next_col].astype(int) == default_state).astype(int)

    out = tmp.groupby(time_col).agg(
        n=("y", "size"),
        actual_rate=("y", "mean"),
        mean_pd=("pd", "mean"),
    )
    out["gap"] = out["mean_pd"] - out["actual_rate"]
    return out.sort_index()


def decide_action_v2(
    psi_table: pd.DataFrame,
    default_rate_train: float,
    default_rate_oot: float,
    matrix_js: dict,
    pd_psi: float | None = None,
    psi_severe: float = 0.25,
    rate_shift_alert: float = 0.02,
    js_alert: float = 0.02,
    pd_psi_alert: float = 0.25,
) -> dict:
    severe_psi_n = 0
    moderate_psi_n = 0
    if psi_table is not None and not psi_table.empty and "psi" in psi_table.columns:
        severe_psi_n = int((psi_table["psi"] >= psi_severe).sum())
        moderate_psi_n = int(((psi_table["psi"] >= 0.10) & (psi_table["psi"] < psi_severe)).sum())

    rate_shift = 0.0
    if np.isfinite(default_rate_oot) and np.isfinite(default_rate_train):
        rate_shift = abs(float(default_rate_oot) - float(default_rate_train))

    js_alert_flag = bool(matrix_js.get("max_js", 0.0) > js_alert)
    pd_psi_flag = bool(pd_psi is not None and np.isfinite(pd_psi) and pd_psi >= pd_psi_alert)

    if severe_psi_n > 0 or rate_shift >= rate_shift_alert or js_alert_flag or pd_psi_flag:
        action, level = "retrain_recommended", "severe"
    elif moderate_psi_n > 0 or (pd_psi is not None and np.isfinite(pd_psi) and pd_psi >= 0.10):
        action, level = "investigate", "moderate"
    else:
        action, level = "no_action", "stable"

    return {
        "level": level,
        "action": action,
        "severe_psi_features": severe_psi_n,
        "moderate_psi_features": moderate_psi_n,
        "default_rate_shift_abs": float(rate_shift),
        "matrix_js_alert": js_alert_flag,
        "pd_score_psi": None if pd_psi is None else float(pd_psi),
        "pd_score_psi_alert": pd_psi_flag,
    }


if __name__ == "__main__":
    print("Use build_stability_report(train_df, oot_df, feature_cols) from research runner.")


