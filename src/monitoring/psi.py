"""
psi.py — мониторинг дрейфа модели (engineering + research-ready)

1) PSI по распределению PD-скоров
2) Bootstrap CI для PSI
3) Sensitivity PSI к числу бинов
4) Jensen–Shannon по строкам матрицы переходов
5) KS по PD
"""


from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# 1. PSI
def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if expected.size == 0 or actual.size == 0:
        raise ValueError("expected/actual must be non-empty")

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, quantiles))
    if edges.size < 2:
        return 0.0

    edges = edges.astype(float)
    edges[0] = min(edges[0], actual.min()) - 1e-12
    edges[-1] = max(edges[-1], actual.max()) + 1e-12

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)

    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_interpret(psi_value: float) -> str:
    if psi_value < 0.10:
        return "стабильно (PSI < 0.10)"
    if psi_value < 0.25:
        return "умеренный дрейф (0.10 <= PSI < 0.25)"
    return "сильный дрейф (PSI >= 0.25)"


def psi_sensitivity(
    expected: np.ndarray,
    actual: np.ndarray,
    bins_grid: list[int] | None = None,
) -> dict[int, float]:
    """PSI при разных bins — проверка устойчивости вывода."""
    bins_grid = bins_grid or [5, 10, 20]
    return {int(b): round(psi(expected, actual, bins=int(b)), 6) for b in bins_grid}


def bootstrap_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_boot: int = 300,
    bins: int = 10,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap CI для PSI.
    Resample expected/actual independently with replacement.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if expected.size == 0 or actual.size == 0:
        raise ValueError("expected/actual must be non-empty")

    rng = np.random.default_rng(random_state)
    point = psi(expected, actual, bins=bins)

    stats = np.empty(n_boot, dtype=float)
    n_e, n_a = expected.size, actual.size
    for i in range(n_boot):
        e_s = expected[rng.integers(0, n_e, size=n_e)]
        a_s = actual[rng.integers(0, n_a, size=n_a)]
        stats[i] = psi(e_s, a_s, bins=bins)

    lower = float(np.quantile(stats, alpha / 2))
    upper = float(np.quantile(stats, 1 - alpha / 2))
    return {
        "point": round(float(point), 6),
        "mean": round(float(stats.mean()), 6),
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "std": round(float(stats.std(ddof=1)), 6),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "bins": int(bins),
    }



# 2. Jensen–Shannon

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """JS divergence, range ≈ [0, ln(2)]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")

    p = np.clip(p, 0, None) + eps
    q = np.clip(q, 0, None) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _assert_square_matrices(P_old: np.ndarray, P_new: np.ndarray) -> None:
    P_old = np.asarray(P_old, dtype=float)
    P_new = np.asarray(P_new, dtype=float)
    if P_old.ndim != 2 or P_new.ndim != 2:
        raise ValueError("P_old and P_new must be 2D")
    if P_old.shape != P_new.shape:
        raise ValueError(f"matrix shape mismatch: {P_old.shape} vs {P_new.shape}")
    if P_old.shape[0] != P_old.shape[1]:
        raise ValueError("transition matrices must be square")


#находим состояние с максимальным дрейфом
def transition_matrix_drift(
    P_old: np.ndarray,
    P_new: np.ndarray,
    state_names: Optional[Dict[int, str]] = None,
    alert_threshold: float = 0.02,
) -> dict:
    _assert_square_matrices(P_old, P_new)
    if state_names is None:
        state_names = {0: "current", 1: "delinquent", 2: "default"}

    n_states = P_old.shape[0]
    per_state = {}
    for i in range(n_states):
        name = state_names.get(i, str(i))
        per_state[name] = round(js_divergence(P_old[i], P_new[i]), 6)

    worst_state = max(per_state, key=per_state.get)
    max_js = per_state[worst_state]
    alert = bool(max_js > alert_threshold)

    return {
        "per_state_js": per_state,
        "max_js": round(float(max_js), 6),
        "worst_state": worst_state,
        "alert": alert,
        "interpretation": (
            f"ALERT: max drift in '{worst_state}' (JS={max_js:.4f} > {alert_threshold})"
            if alert
            else f"matrix stable (max JS={max_js:.4f})"
        ),
    }


# 3. KS
def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.sort(np.asarray(expected, dtype=float))
    actual = np.sort(np.asarray(actual, dtype=float))
    if expected.size == 0 or actual.size == 0:
        raise ValueError("expected/actual must be non-empty")

    grid = np.unique(np.concatenate([expected, actual]))
    cdf_e = np.searchsorted(expected, grid, side="right") / expected.size
    cdf_a = np.searchsorted(actual, grid, side="right") / actual.size
    return float(np.max(np.abs(cdf_e - cdf_a)))


# 4. Sanity-check helpers (for research notebook/tests)
def psi_sanity_checks(random_state: int = 42) -> dict:
    
    rng = np.random.default_rng(random_state)
    base = rng.beta(2, 20, size=3000)
    same = base.copy()
    shifted = np.clip(base + 0.05, 0, 1)

    psi_same = psi(base, same)
    psi_shift = psi(base, shifted)
    return {
        "psi_identical": round(float(psi_same), 6),
        "psi_shifted": round(float(psi_shift), 6),
        "ok_identical_small": bool(psi_same < 0.05),
        "ok_shift_larger": bool(psi_shift > psi_same),
    }


# 5. Full report
def monitoring_report(
    pd_baseline: np.ndarray,
    pd_current: np.ndarray,
    P_old: np.ndarray,
    P_new: np.ndarray,
    psi_severe: float = 0.25,
    psi_moderate: float = 0.10,
    js_threshold: float = 0.02,
    bins: int = 10,
    n_boot: int = 300,
    min_samples: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Сводный мониторинг с research-полями:
      - PSI point + bootstrap CI
      - PSI sensitivity by bins
      - JS matrix drift
      - KS
    """
    pd_baseline = np.asarray(pd_baseline, dtype=float)
    pd_current = np.asarray(pd_current, dtype=float)
    pd_baseline = pd_baseline[np.isfinite(pd_baseline)]
    pd_current = pd_current[np.isfinite(pd_current)]

    warnings = []
    if pd_baseline.size < min_samples or pd_current.size < min_samples:
        warnings.append(
            f"small sample: baseline={pd_baseline.size}, current={pd_current.size}, "
            f"min_samples={min_samples}"
        )

    psi_point = psi(pd_baseline, pd_current, bins=bins)
    psi_boot = bootstrap_psi(
        pd_baseline,
        pd_current,
        n_boot=n_boot,
        bins=bins,
        random_state=random_state,
    )
    psi_sens = psi_sensitivity(pd_baseline, pd_current)

    drift_info = transition_matrix_drift(P_old, P_new, alert_threshold=js_threshold)
    ks_val = ks_statistic(pd_baseline, pd_current)

    severe = psi_point >= psi_severe or drift_info["alert"]
    moderate = (psi_moderate <= psi_point < psi_severe) and not severe

    if severe:
        action = "retrain_recommended"
    elif moderate:
        action = "investigate"
    else:
        action = "no_action"

    return {
        "meta": {
            "n_baseline": int(pd_baseline.size),
            "n_current": int(pd_current.size),
            "bins": int(bins),
            "n_boot": int(n_boot),
            "random_state": int(random_state),
            "warnings": warnings,
        },
        "psi": {
            "value": round(float(psi_point), 4),
            "interpretation": psi_interpret(psi_point),
            "bootstrap": psi_boot,
            "sensitivity_bins": psi_sens,
        },
        "ks": {
            "value": round(float(ks_val), 4),
        },
        "matrix_drift": drift_info,
        "action_needed": bool(severe or moderate),
        "action": action,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    base = rng.beta(2, 20, size=2000)
    curr = rng.beta(2.5, 18, size=2000)

    P0 = np.array([
        [0.90, 0.08, 0.02],
        [0.20, 0.60, 0.20],
        [0.00, 0.00, 1.00],
    ])
    P1 = np.array([
        [0.88, 0.09, 0.03],
        [0.18, 0.57, 0.25],
        [0.00, 0.00, 1.00],
    ])

    print("sanity:", psi_sanity_checks())
    print("report:", monitoring_report(base, curr, P0, P1))