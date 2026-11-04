"""
scenario_stress.py — base / stress PD term-structure for risk what-if.

Для клиента (state, x):
  base path: x без изменений
  util stress / macro stress / combined
  считаем PD(h) через model.pd_term_structure
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DEFAULT_SCENARIOS = {
    "base": {},
    "util_stress": {
        "utilization": 0.15,
        "util_mean": 0.10,
        "util_trend": 0.05,
        "util_mean_last3": 0.10,
        "high_util_ratio": 0.15,
        "util_rising_3m": 1.0,
    },
    "macro_stress": {
        "macro_stress": 0.80,
    },
    "combined_stress": {
        "utilization": 0.15,
        "util_mean": 0.10,
        "util_trend": 0.05,
        "util_mean_last3": 0.10,
        "high_util_ratio": 0.15,
        "util_rising_3m": 1.0,
        "macro_stress": 0.80,
    },
}


def _apply_scenario(
    x: np.ndarray,
    feature_cols: list[str],
    scenario_delta: dict[str, float],
    mode: str = "add",
) -> np.ndarray:
    """
    mode='add'  : x_new = clip(x + delta)
    mode='set'  : overwrite selected features by absolute values in scenario_delta
    """
    x_new = np.array(x, dtype=float, copy=True)
    name_to_idx = {c: i for i, c in enumerate(feature_cols)}

    for feat, val in scenario_delta.items():
        if feat not in name_to_idx:
            continue
        i = name_to_idx[feat]
        if mode == "set":
            x_new[i] = float(val)
        else:
            x_new[i] = float(x_new[i] + val)

    # soft clips for known bounded features
    bounded_01 = {
        "utilization", "util_mean", "util_max", "util_min",
        "util_mean_last3", "util_mean_prev", "frac_current",
        "frac_delinquent", "frac_default", "high_util_ratio",
        "state_entropy",
    }
    for feat in bounded_01:
        if feat in name_to_idx:
            i = name_to_idx[feat]
            x_new[i] = float(np.clip(x_new[i], 0.0, 1.0))

    if "macro_stress" in name_to_idx:
        i = name_to_idx["macro_stress"]
        x_new[i] = float(np.clip(x_new[i], -1.5, 1.5))

    if "util_x_macro" in name_to_idx and "utilization" in name_to_idx and "macro_stress" in name_to_idx:
        x_new[name_to_idx["util_x_macro"]] = (
            x_new[name_to_idx["utilization"]] * x_new[name_to_idx["macro_stress"]]
        )

    if "ever_delinquent" in name_to_idx:
        x_new[name_to_idx["ever_delinquent"]] = float(
            np.clip(x_new[name_to_idx["ever_delinquent"]], 0.0, 1.0)
        )
    if "util_rising_3m" in name_to_idx:
        x_new[name_to_idx["util_rising_3m"]] = float(
            np.clip(round(x_new[name_to_idx["util_rising_3m"]]), 0.0, 1.0)
        )

    return x_new


def pd_under_scenario(
    model,
    state: int,
    x: np.ndarray,
    feature_cols: list[str],
    scenario_delta: dict[str, float],
    horizons: list[int] = (1, 3, 6, 12),
    mode: str = "add",
) -> dict[int, float]:
    x_s = _apply_scenario(x, feature_cols, scenario_delta, mode=mode)
    path = np.tile(x_s, (max(horizons), 1))
    return model.pd_term_structure(int(state), path, list(horizons))


def stress_one_client(
    model,
    state: int,
    x: np.ndarray,
    feature_cols: list[str],
    scenarios: Optional[dict[str, dict[str, float]]] = None,
    horizons: list[int] = (1, 3, 6, 12),
    mode: str = "add",
) -> pd.DataFrame:
    scenarios = scenarios or DEFAULT_SCENARIOS
    rows = []
    base_pd = None

    for name, delta in scenarios.items():
        pd_ts = pd_under_scenario(
            model, state, x, feature_cols, delta, horizons=horizons, mode=mode
        )
        if name == "base":
            base_pd = pd_ts
        row = {"scenario": name, "state": int(state)}
        for h, v in pd_ts.items():
            row[f"pd_{h}m"] = float(v)
            if base_pd is not None and name != "base":
                row[f"delta_pd_{h}m"] = float(v - base_pd[h])
            else:
                row[f"delta_pd_{h}m"] = 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def stress_portfolio_sample(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clients: int = 100,
    horizons: list[int] = (1, 3, 6, 12),
    scenarios: Optional[dict[str, dict[str, float]]] = None,
    random_state: int = 42,
    mode: str = "add",
) -> pd.DataFrame:
    """
    Берём случайных клиентов (последний доступный month на клиента)
    и считаем stress PD.
    """
    scenarios = scenarios or DEFAULT_SCENARIOS
    tmp = df.sort_values(["client_id", "month"]).groupby("client_id", as_index=False).tail(1)
    tmp = tmp[tmp["state"].isin([0, 1])].copy()
    if tmp.empty:
        raise ValueError("No non-default rows for stress sample")

    tmp = tmp.sample(n=min(n_clients, len(tmp)), random_state=random_state)

    all_rows = []
    for _, row in tmp.iterrows():
        x = row[feature_cols].to_numpy(dtype=float)
        state = int(row["state"])
        one = stress_one_client(
            model, state, x, feature_cols,
            scenarios=scenarios, horizons=list(horizons), mode=mode
        )
        one.insert(0, "client_id", row["client_id"])
        one.insert(1, "month", row["month"])
        all_rows.append(one)

    return pd.concat(all_rows, ignore_index=True)


def stress_summary(details: pd.DataFrame, horizons: list[int] = (1, 3, 6, 12)) -> pd.DataFrame:
    rows = []
    for scen, g in details.groupby("scenario"):
        item = {"scenario": scen, "n": len(g)}
        for h in horizons:
            col = f"pd_{h}m"
            dcol = f"delta_pd_{h}m"
            item[f"mean_pd_{h}m"] = float(g[col].mean())
            item[f"mean_delta_pd_{h}m"] = float(g[dcol].mean())
        rows.append(item)
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


def save_stress_report(
    details: pd.DataFrame,
    out_dir: str | Path = "reports/research/stress",
    horizons: list[int] = (1, 3, 6, 12),
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    details_path = out / "stress_client_details.csv"
    summary_path = out / "stress_summary.csv"
    details.to_csv(details_path, index=False)
    summary = stress_summary(details, horizons=list(horizons))
    summary.to_csv(summary_path, index=False)
    print(f"[stress] saved {details_path}")
    print(f"[stress] saved {summary_path}")
    return {"details": details_path, "summary": summary_path}


if __name__ == "__main__":
    print("Use stress_portfolio_sample(model, df, feature_cols) from research runner.")