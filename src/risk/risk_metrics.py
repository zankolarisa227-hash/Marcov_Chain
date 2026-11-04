"""
risk_metrics.py — PD term-structure, ECL, limit policy helpers

Что делает модуль:
  1) PD term-structure
     - static: P^h[start_state, default]
     - (dynamic PD считается в dynamic_model.pd_term_structure)
  2) ECL / EL approximations
  3) Сравнение term-structure и sensitivity
  4) Упрощённый limit decision (boundary optimum)

Limitations:
  - ECL ≈ PD * LGD * EAD — упрощённый Stage-1 style estimate
  - LGD/EAD задаются извне, не оцениваются из Alfa-CT
  - нет полного IFRS9 SICR / Stage2-3 движка
  - static PD предполагает time-homogeneous P
  - dynamic PD зависит от выбранного covariate path
  - PD в limit-policy считается независимым от величины лимита
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


LIMITATIONS = [
    "ECL here is a simplified one-period approximation: EL ≈ PD * LGD * EAD.",
    "LGD and EAD are exogenous assumptions, not estimated from the dataset.",
    "No explicit IFRS9 SICR / Stage2-3 transfer logic is implemented.",
    "Static PD term-structure assumes time-homogeneous transitions.",
    "Dynamic PD term-structure depends on the chosen covariate path.",
    "Limit policy assumes PD independent of limit (no feedback loop).",
]


def _validate_pd(pd: float) -> None:
    if not 0.0 <= float(pd) <= 1.0:
        raise ValueError(f"PD must be in [0, 1], got {pd}")


def _validate_lgd(lgd: float) -> None:
    if not 0.0 <= float(lgd) <= 1.0:
        raise ValueError(f"LGD must be in [0, 1], got {lgd}")


# ---------------------------------------------------------------------------
# 1. PD term-structure
# ---------------------------------------------------------------------------

def pd_term_structure_static(
    P: np.ndarray,
    start_state: int,
    horizons: List[int],
    default_state: int = 2,
) -> Dict[int, float]:
    """Кумулятивный PD на горизонтах через P^h."""
    P = np.asarray(P, dtype=float)
    result: Dict[int, float] = {}
    for h in horizons:
        if h < 0:
            raise ValueError("horizon must be >= 0")
        if h == 0:
            result[0] = 1.0 if start_state == default_state else 0.0
            continue
        P_h = np.linalg.matrix_power(P, h)
        result[int(h)] = float(P_h[start_state, default_state])
    return result


def assert_pd_term_monotonic(pd_term: Dict[int, float], tol: float = 1e-9) -> bool:
    """
    Для absorbing-default кумулятивный PD(h) должен быть неубывающим по h.
    """
    hs = sorted(pd_term.keys())
    vals = [float(pd_term[h]) for h in hs]
    for a, b in zip(vals, vals[1:]):
        if b + tol < a:
            return False
    return True


def compare_pd_term_structures(
    pd_static: Dict[int, float],
    pd_dynamic: Dict[int, float],
) -> List[Dict[str, Any]]:
    """Сравнение static vs dynamic PD term-structure."""
    horizons = sorted(set(pd_static) | set(pd_dynamic))
    rows: List[Dict[str, Any]] = []
    for h in horizons:
        s = pd_static.get(h, np.nan)
        d = pd_dynamic.get(h, np.nan)
        rows.append(
            {
                "horizon": int(h),
                "pd_static": s,
                "pd_dynamic": d,
                "diff_dynamic_minus_static": (
                    float(d - s) if np.isfinite(s) and np.isfinite(d) else np.nan
                ),
                "abs_diff": (
                    float(abs(d - s)) if np.isfinite(s) and np.isfinite(d) else np.nan
                ),
            }
        )
    return rows


def pd_term_structure_from_matrix_ci(
    P_point: np.ndarray,
    P_lower: np.ndarray,
    P_upper: np.ndarray,
    start_state: int,
    horizons: List[int],
    default_state: int = 2,
) -> Dict[str, Any]:
    """
    Point/lower/upper PD(h) через matrix power.

    Важно: это scenario-like bounds (power от lower/upper матриц),
    а не точный simultaneous confidence band.
    """
    return {
        "point": pd_term_structure_static(P_point, start_state, horizons, default_state),
        "lower": pd_term_structure_static(P_lower, start_state, horizons, default_state),
        "upper": pd_term_structure_static(P_upper, start_state, horizons, default_state),
        "note": "Bounds are scenario-like matrix-power bounds, not exact analytical CIs.",
        "monotonic_point": assert_pd_term_monotonic(
            pd_term_structure_static(P_point, start_state, horizons, default_state)
        ),
    }


# ---------------------------------------------------------------------------
# 2. ECL / EL
# ---------------------------------------------------------------------------

def ecl(pd: float, ead: float, lgd: float = 0.45) -> float:
    """
    Simplified expected loss:
        EL = PD * LGD * EAD

    Not a full IFRS 9 engine.
    """
    _validate_pd(pd)
    _validate_lgd(lgd)
    if ead < 0:
        raise ValueError("EAD must be >= 0")
    return float(pd * ead * lgd)


def ecl_lifetime_approx(
    pd_term: Dict[int, float],
    ead: float,
    lgd: float = 0.45,
    discount_rate: float = 0.05,
) -> float:
    """
    Грубая lifetime-style сумма:
    маржинальные приросты кумулятивного PD с дисконтированием.
    """
    _validate_lgd(lgd)
    if ead < 0:
        raise ValueError("EAD must be >= 0")

    horizons = sorted(pd_term.keys())
    if not horizons:
        return 0.0

    lifetime = 0.0
    prev_pd = 0.0
    for h in horizons:
        marginal = max(float(pd_term[h]) - prev_pd, 0.0)
        df = 1.0 / ((1.0 + discount_rate) ** (h / 12.0))
        lifetime += marginal * ead * lgd * df
        prev_pd = float(pd_term[h])
    return float(lifetime)


def ecl_portfolio(
    pd_scores: List[float] | np.ndarray,
    eads: List[float] | np.ndarray,
    lgd: float = 0.45,
) -> dict:
    _validate_lgd(lgd)
    pd_arr = np.asarray(pd_scores, dtype=float)
    ead_arr = np.asarray(eads, dtype=float)

    if len(pd_arr) != len(ead_arr):
        raise ValueError("pd_scores and eads must have same length")

    ecl_arr = pd_arr * ead_arr * lgd
    total_ead = float(ead_arr.sum()) if len(ead_arr) else 0.0

    return {
        "total_ecl": float(ecl_arr.sum()) if len(ecl_arr) else 0.0,
        "mean_pd": float(pd_arr.mean()) if len(pd_arr) else 0.0,
        "mean_ecl": float(ecl_arr.mean()) if len(ecl_arr) else 0.0,
        "ecl_ratio": float(ecl_arr.sum() / total_ead) if total_ead > 0 else 0.0,
        "n_clients": int(len(pd_arr)),
        "limitations": LIMITATIONS,
    }


def ecl_sensitivity(
    pd: float,
    ead: float,
    lgd_grid: Optional[List[float]] = None,
    ead_scale_grid: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Чувствительность EL к LGD и масштабу EAD при фиксированном PD."""
    _validate_pd(pd)
    lgd_grid = lgd_grid or [0.35, 0.45, 0.55]
    ead_scale_grid = ead_scale_grid or [0.8, 1.0, 1.2]

    rows: List[Dict[str, Any]] = []
    for lgd in lgd_grid:
        for scale in ead_scale_grid:
            ead_i = float(ead) * float(scale)
            rows.append(
                {
                    "pd": float(pd),
                    "lgd": float(lgd),
                    "ead": ead_i,
                    "ead_scale": float(scale),
                    "ecl": ecl(pd, ead_i, lgd),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 3. Limit policy (boundary optimum under fixed PD)
# ---------------------------------------------------------------------------

def optimal_limit(
    pd_12m: float,
    margin: float = 0.18,
    lgd: float = 0.45,
    max_pd: float = 0.20,
    max_el_ratio: float = 0.05,
    limit_bounds: tuple = (1_000.0, 100_000.0),
) -> dict:
    """
    Policy decision under fixed PD.

    profit(L) = L * ((1-PD)*margin - PD*LGD)
    => linear in L => optimum at boundary of feasible set.
    """
    _validate_pd(pd_12m)
    _validate_lgd(lgd)

    min_limit, max_limit = limit_bounds
    if min_limit <= 0 or max_limit < min_limit:
        raise ValueError("Invalid limit_bounds")

    el_ratio = float(pd_12m) * float(lgd)
    unit = (1.0 - float(pd_12m)) * float(margin) - float(pd_12m) * float(lgd)

    if pd_12m > max_pd:
        return {
            "approved": False,
            "reason": f"PD={pd_12m:.4f} > max_pd={max_pd:.4f}",
            "optimal_limit": float(min_limit),
            "expected_profit": 0.0,
            "unit_profit": float(unit),
            "pd_12m": float(pd_12m),
            "el_ratio": float(el_ratio),
            "limitations": LIMITATIONS,
        }

    if el_ratio > max_el_ratio:
        return {
            "approved": False,
            "reason": f"EL ratio={el_ratio:.4f} > max_el_ratio={max_el_ratio:.4f}",
            "optimal_limit": float(min_limit),
            "expected_profit": 0.0,
            "unit_profit": float(unit),
            "pd_12m": float(pd_12m),
            "el_ratio": float(el_ratio),
            "limitations": LIMITATIONS,
        }

    best_limit = float(max_limit) if unit > 0 else float(min_limit)
    expected_profit = best_limit * unit

    return {
        "approved": True,
        "reason": "unit_profit>0 → max_limit" if unit > 0 else "unit_profit<=0 → min_limit",
        "optimal_limit": round(best_limit, 2),
        "expected_profit": round(float(expected_profit), 2),
        "unit_profit": round(float(unit), 6),
        "pd_12m": round(float(pd_12m), 4),
        "el_ratio": round(float(el_ratio), 4),
        "margin": float(margin),
        "lgd": float(lgd),
        "note": "Optimum is boundary because profit is linear in limit under fixed PD",
        "limitations": LIMITATIONS,
    }


if __name__ == "__main__":
    P = np.array(
        [
            [0.90, 0.08, 0.02],
            [0.20, 0.60, 0.20],
            [0.00, 0.00, 1.00],
        ],
        dtype=float,
    )
    
    pd_ts = pd_term_structure_static(P, start_state=0, horizons=[1, 3, 6, 12])
    print("PD term:", pd_ts)
    print("monotonic:", assert_pd_term_monotonic(pd_ts))
    print("ECL:", ecl(pd_ts[12], ead=50_000, lgd=0.45))
    print("limit:", optimal_limit(pd_ts[12]))