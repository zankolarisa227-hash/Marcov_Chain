

from __future__ import annotations

from typing import Any

import numpy as np


LIMITATIONS = [
    "PD is treated as independent of limit (no feedback loop limit → utilization → PD).",
    "Under fixed PD, expected profit is linear in limit; optimum is at a boundary.",
    "Risk gates (risk_appetite, max_el_ratio) are policy thresholds, not estimated quantities.",
    "This module is a decision layer on top of PD, not a causal credit-demand model.",
]


def _validate_inputs(pd: float, lgd: float, margin: float) -> None:
    if not 0.0 <= pd <= 1.0:
        raise ValueError(f"pd must be in [0, 1], got {pd}")
    if not 0.0 <= lgd <= 1.0:
        raise ValueError(f"lgd must be in [0, 1], got {lgd}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")


def unit_profit(pd: float, margin: float, lgd: float) -> float:
    """Unit expected profit per 1 currency unit of limit."""
    return (1.0 - pd) * margin - pd * lgd


def optimize_limit(
    pd: float,
    risk_appetite: float = 0.05,
    margin: float = 0.05,
    lgd: float = 0.45,
    min_limit: float = 1_000.0,
    max_limit: float = 50_000.0,
    max_el_ratio: float = 0.05,
) -> dict[str, Any]:
    """
    Возвращает dict с решением и диагностикой.

    Ограничения:
      1) PD <= risk_appetite
      2) PD * LGD <= max_el_ratio
      3) limit in [min_limit, max_limit]
    """
    _validate_inputs(pd, lgd, margin)

    if min_limit <= 0 or max_limit < min_limit:
        raise ValueError("Invalid limit bounds")

    el_ratio = pd * lgd
    up = unit_profit(pd, margin, lgd)

    # Risk gates
    if pd > risk_appetite:
        return {
            "approved": False,
            "reason": f"PD={pd:.4f} > risk_appetite={risk_appetite:.4f}",
            "optimal_limit": float(min_limit),
            "expected_profit": 0.0,
            "unit_profit": float(up),
            "el_ratio": float(el_ratio),
            "pd": float(pd),
            "policy": {
                "risk_appetite": risk_appetite,
                "max_el_ratio": max_el_ratio,
                "margin": margin,
                "lgd": lgd,
                "limit_bounds": [min_limit, max_limit],
            },
            "limitations": LIMITATIONS,
        }

    if el_ratio > max_el_ratio:
        return {
            "approved": False,
            "reason": f"EL ratio={el_ratio:.4f} > max_el_ratio={max_el_ratio:.4f}",
            "optimal_limit": float(min_limit),
            "expected_profit": 0.0,
            "unit_profit": float(up),
            "el_ratio": float(el_ratio),
            "pd": float(pd),
            "policy": {
                "risk_appetite": risk_appetite,
                "max_el_ratio": max_el_ratio,
                "margin": margin,
                "lgd": lgd,
                "limit_bounds": [min_limit, max_limit],
            },
            "limitations": LIMITATIONS,
        }

    optimal = float(max_limit) if up > 0 else float(min_limit)
    expected = optimal * up

    return {
        "approved": True,
        "reason": "unit_profit>0 → max_limit" if up > 0 else "unit_profit<=0 → min_limit",
        "optimal_limit": optimal,
        "expected_profit": float(expected),
        "unit_profit": float(up),
        "el_ratio": float(el_ratio),
        "pd": float(pd),
        "policy": {
            "risk_appetite": risk_appetite,
            "max_el_ratio": max_el_ratio,
            "margin": margin,
            "lgd": lgd,
            "limit_bounds": [min_limit, max_limit],
        },
        "limitations": LIMITATIONS,
        "note": "Optimum is boundary because profit is linear in limit under fixed PD",
    }


def optimize_limit_batch(
    pd_array: np.ndarray,
    risk_appetite: float = 0.05,
    margin: float = 0.05,
    lgd: float = 0.45,
    min_limit: float = 1_000.0,
    max_limit: float = 50_000.0,
    max_el_ratio: float = 0.05,
) -> np.ndarray:
    pd_array = np.asarray(pd_array, dtype=float)
    limits = np.empty(len(pd_array), dtype=float)
    for i, pd in enumerate(pd_array):
        res = optimize_limit(
            pd=float(pd),
            risk_appetite=risk_appetite,
            margin=margin,
            lgd=lgd,
            min_limit=min_limit,
            max_limit=max_limit,
            max_el_ratio=max_el_ratio,
        )
        limits[i] = res["optimal_limit"]
    return limits


def sensitivity_analysis(
    pd: float,
    margin_grid: list[float] | None = None,
    lgd_grid: list[float] | None = None,
    risk_appetite_grid: list[float] | None = None,
    max_el_ratio_grid: list[float] | None = None,
    min_limit: float = 1_000.0,
    max_limit: float = 50_000.0,
) -> list[dict[str, Any]]:
    """
    Research helper: как policy-решение зависит от предпосылок.

    Фиксируем PD, варьируем policy/economics параметры.
    """
    margin_grid = margin_grid or [0.03, 0.05, 0.08, 0.12]
    lgd_grid = lgd_grid or [0.35, 0.45, 0.55]
    risk_appetite_grid = risk_appetite_grid or [0.02, 0.05, 0.10]
    max_el_ratio_grid = max_el_ratio_grid or [0.03, 0.05, 0.08]

    rows: list[dict[str, Any]] = []
    for margin in margin_grid:
        for lgd in lgd_grid:
            for risk_appetite in risk_appetite_grid:
                for max_el_ratio in max_el_ratio_grid:
                    res = optimize_limit(
                        pd=pd,
                        risk_appetite=risk_appetite,
                        margin=margin,
                        lgd=lgd,
                        min_limit=min_limit,
                        max_limit=max_limit,
                        max_el_ratio=max_el_ratio,
                    )
                    rows.append(
                        {
                            "pd": pd,
                            "margin": margin,
                            "lgd": lgd,
                            "risk_appetite": risk_appetite,
                            "max_el_ratio": max_el_ratio,
                            "approved": res["approved"],
                            "optimal_limit": res["optimal_limit"],
                            "expected_profit": res["expected_profit"],
                            "unit_profit": res["unit_profit"],
                            "el_ratio": res["el_ratio"],
                            "reason": res["reason"],
                        }
                    )
    return rows


def sensitivity_to_dataframe(rows: list[dict[str, Any]]):
    """Optional pandas view for notebooks/reports."""
    import pandas as pd

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd_ok = 0.015
    res = optimize_limit(pd=pd_ok, risk_appetite=0.05, margin=0.05, lgd=0.45)
    print("decision:", res)

    rows = sensitivity_analysis(pd=pd_ok)
    print(f"sensitivity rows: {len(rows)}")
    print("limitations:")
    for x in LIMITATIONS:
        print("-", x)