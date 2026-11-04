"""
calibration.py — OOT probability calibration diagnostics.

Primary metrics:
  - ECE
  - calibration slope / intercept
  - reliability table
  - Brier skill vs baseline

Hosmer–Lemeshow is secondary (bin-sensitive; on large n often "significant").
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _as_binary(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob length mismatch")
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must be binary {0,1}")
    return y_true, y_prob


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """
    ECE = sum_k (n_k / n) * |acc_k - conf_k|
    strategy:
      - uniform: equal width bins in [0,1]
      - quantile: equal count bins (often stabler)
    """
    y_true, y_prob = _as_binary(y_true, y_prob)
    n = len(y_true)
    if n == 0:
        return np.nan

    if strategy == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(y_prob, qs))
        if len(edges) < 3:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            m = (y_prob >= left) & (y_prob < right)
        else:
            m = (y_prob >= left) & (y_prob <= right)
        nk = int(m.sum())
        if nk == 0:
            continue
        acc = float(y_true[m].mean())
        conf = float(y_prob[m].mean())
        ece += (nk / n) * abs(acc - conf)
    return float(ece)


def calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Logistic recalibration:
        P(Y=1) = sigmoid(a + b * logit(p_hat))
    Well-calibrated ≈ intercept a=0, slope b=1.
    """
    y_true, p = _as_binary(y_true, y_prob)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(logit, y_true)
    return {
        "intercept": float(clf.intercept_[0]),
        "slope": float(clf.coef_[0, 0]),
    }


def reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    y_true, y_prob = _as_binary(y_true, y_prob)
    sk_strategy = "quantile" if strategy == "quantile" else "uniform"
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy=sk_strategy
        )
    except ValueError:
        # fallback if too few unique probs
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=min(n_bins, 5), strategy="uniform"
        )
    return pd.DataFrame({
        "mean_predicted": mean_pred,
        "fraction_positive": frac_pos,
        "gap": np.abs(mean_pred - frac_pos),
    })


def hosmer_lemeshow(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Secondary diagnostic only.
    On large samples HL often rejects even acceptable calibration.
    """
    y_true, y_prob = _as_binary(y_true, y_prob)
    order = np.argsort(y_prob)
    y_true, y_prob = y_true[order], y_prob[order]
    idx_bins = np.array_split(np.arange(len(y_true)), n_bins)

    stat = 0.0
    rows = []
    for b in idx_bins:
        if len(b) == 0:
            continue
        obs = float(y_true[b].sum())
        exp = float(y_prob[b].sum())
        n = float(len(b))
        exp = min(max(exp, 1e-6), n - 1e-6)
        stat += (obs - exp) ** 2 / (exp * (1.0 - exp / n))
        rows.append({
            "n": int(n),
            "obs_events": obs,
            "exp_events": exp,
            "mean_p": float(y_prob[b].mean()),
        })

    df = max(len(rows) - 2, 1)
    p_value = float(1.0 - chi2.cdf(stat, df))
    return {
        "stat": float(stat),
        "df": int(df),
        "p_value": p_value,
        "bins": pd.DataFrame(rows),
        "note": "Secondary test; interpret with caution on large n",
        "conclusion": (
            "HL rejects perfect calibration (p<0.05)"
            if p_value < 0.05
            else "HL does not reject perfect calibration"
        ),
    }


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    baseline_rate: Optional[float] = None,
) -> dict:
    y_true, y_prob = _as_binary(y_true, y_prob)
    br = brier_score_loss(y_true, y_prob)
    if baseline_rate is None:
        baseline_rate = float(y_true.mean())
    base = np.full_like(y_prob, float(baseline_rate), dtype=float)
    br_base = brier_score_loss(y_true, base)
    skill = 1.0 - br / br_base if br_base > 0 else np.nan
    return {
        "brier": float(br),
        "brier_baseline": float(br_base),
        "brier_skill": float(skill),
        "baseline_rate": float(baseline_rate),
    }


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
    baseline_rate: Optional[float] = None,
    by_state: Optional[np.ndarray] = None,
) -> dict:
    """
    Main entrypoint for OOT calibration.
    If by_state is provided (0/1), also reports per-state calibration.
    """
    y_true, y_prob = _as_binary(y_true, y_prob)
    slope = calibration_slope_intercept(y_true, y_prob)
    hl = hosmer_lemeshow(y_true, y_prob, n_bins=n_bins)
    skill = brier_skill_score(y_true, y_prob, baseline_rate=baseline_rate)

    report = {
        "n": int(len(y_true)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy),
        "slope": slope["slope"],
        "intercept": slope["intercept"],
        "brier": skill["brier"],
        "brier_skill": skill["brier_skill"],
        "hosmer_lemeshow": {
            "stat": hl["stat"],
            "df": hl["df"],
            "p_value": hl["p_value"],
            "conclusion": hl["conclusion"],
            "note": hl["note"],
        },
        "reliability": reliability_table(y_true, y_prob, n_bins=n_bins, strategy=strategy),
    }

    if by_state is not None:
        by_state = np.asarray(by_state).ravel()
        per = {}
        for s in sorted(np.unique(by_state)):
            m = by_state == s
            if m.sum() < max(30, n_bins):
                continue
            per[int(s)] = {
                "n": int(m.sum()),
                "ece": expected_calibration_error(y_true[m], y_prob[m], n_bins=min(n_bins, 5)),
                **calibration_slope_intercept(y_true[m], y_prob[m]),
                "brier": float(brier_score_loss(y_true[m], y_prob[m])),
            }
        report["by_state"] = per

    return report