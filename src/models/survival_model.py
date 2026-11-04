"""
survival_model.py — time-to-default для кредитного риска.

Модели:
  - Kaplan–Meier (non-parametric baseline survival curve)
  - Cox PH (semi-parametric, covariate-adjusted)

Событие:
  first transition to default_state.

Время:
  month (порядковый месяц в panel) на момент события/цензурирования.

Интеграция:
  вход = features_df с client_id, month, state + FEATURE_COLS
  выход = PD(h) = 1 - S(h | x), сравнимо с Markov PD term-structure
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError as e:
    LIFELINES_AVAILABLE = False
    _LIFELINES_IMPORT_ERROR = e


DEFAULT_STATE = 2


def _require_lifelines() -> None:
    if not LIFELINES_AVAILABLE:
        raise ImportError(
            "lifelines is required for survival_model. "
            "Install: pip install lifelines\n"
            f"Original error: {_LIFELINES_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# 1. Построение survival-таблицы (один клиент — одна строка)
# ---------------------------------------------------------------------------

def build_survival_table(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    default_state: int = DEFAULT_STATE,
    client_col: str = "client_id",
    time_col: str = "month",
    state_col: str = "state",
    covariate_at: str = "first",
) -> pd.DataFrame:
    """
    Преобразует panel/features в survival format.

    duration:
        month первого default, иначе последний доступный month (цензура)
    event:
        1 если default наблюдался, иначе 0

    covariate_at:
        "first" — ковариаты с первого наблюдения клиента
        "last_pre_event" — последние ковариаты до default/цензуры
    """
    required = {client_col, time_col, state_col}
    missing = required - set(features_df.columns)
    if missing:
        raise ValueError(f"features_df missing columns: {missing}")

    cols = [c for c in feature_cols if c in features_df.columns]
    df = features_df.sort_values([client_col, time_col]).copy()

    rows: list[dict[str, Any]] = []
    for cid, g in df.groupby(client_col, sort=False):
        g = g.sort_values(time_col)
        default_hits = g[g[state_col] == default_state]

        if len(default_hits) > 0:
            event_row = default_hits.iloc[0]
            duration = int(event_row[time_col])
            event = 1
            pre = g[g[time_col] <= duration]
        else:
            event_row = g.iloc[-1]
            duration = int(event_row[time_col])
            event = 0
            pre = g

        if covariate_at == "first":
            cov_source = g.iloc[0]
        elif covariate_at == "last_pre_event":
            cov_source = pre.iloc[-1]
        else:
            raise ValueError("covariate_at must be 'first' or 'last_pre_event'")

        item = {
            client_col: cid,
            "duration": max(duration, 1),
            "event": int(event),
        }
        for c in cols:
            item[c] = cov_source[c]
        rows.append(item)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("survival table is empty")
    return out


# ---------------------------------------------------------------------------
# 2. Модель
# ---------------------------------------------------------------------------

@dataclass
class SurvivalModel:
    km: Any
    cox: Any
    feature_cols: list[str]
    default_state: int = DEFAULT_STATE
    penalizer: float = 0.1
    covariate_at: str = "first"

    def save(self, path: str | Path = "models/survival_model.joblib") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path = "models/survival_model.joblib") -> "SurvivalModel":
        return joblib.load(path)


def fit_survival_model(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    default_state: int = DEFAULT_STATE,
    penalizer: float = 0.1,
    covariate_at: str = "first",
) -> SurvivalModel:
    """
    Обучает KM + Cox PH.
    Важно: передавайте только TRAIN / past fold (не весь датасет).
    """
    _require_lifelines()

    table = build_survival_table(
        features_df=features_df,
        feature_cols=feature_cols,
        default_state=default_state,
        covariate_at=covariate_at,
    )
    cols = [c for c in feature_cols if c in table.columns]
    if not cols:
        raise ValueError("No usable feature columns for Cox model")

    # Kaplan–Meier
    km = KaplanMeierFitter()
    km.fit(durations=table["duration"], event_observed=table["event"], label="KM")

    # Cox PH
    cox_df = table[["duration", "event"] + cols].copy()
    # lifelines чувствителен к константным столбцам
    nunique = cox_df[cols].nunique(dropna=False)
    cols = [c for c in cols if nunique.get(c, 0) > 1]
    if not cols:
        raise ValueError("All covariates are constant; cannot fit Cox")

    cox_df = table[["duration", "event"] + cols].dropna()
    cox = CoxPHFitter(penalizer=penalizer)
    cox.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)

    return SurvivalModel(
        km=km,
        cox=cox,
        feature_cols=cols,
        default_state=default_state,
        penalizer=penalizer,
        covariate_at=covariate_at,
    )


# ---------------------------------------------------------------------------
# 3. PD term-structure из survival
# ---------------------------------------------------------------------------

def survival_function_at(
    model: SurvivalModel,
    x: dict | pd.Series | np.ndarray,
    times: Optional[list[int]] = None,
) -> pd.Series:
    """
    S(t | x) для одного клиента.
    """
    _require_lifelines()

    if isinstance(x, np.ndarray):
        if len(x) != len(model.feature_cols):
            raise ValueError("x length must match model.feature_cols")
        row = {c: float(v) for c, v in zip(model.feature_cols, x)}
    elif isinstance(x, pd.Series):
        row = {c: float(x[c]) for c in model.feature_cols}
    else:
        row = {c: float(x[c]) for c in model.feature_cols}

    pred = model.cox.predict_survival_function(pd.DataFrame([row]))
    # pred index = times, one column
    s = pred.iloc[:, 0]
    if times is None:
        return s

    # интерполяция/ближайшее значение слева
    out = {}
    idx = s.index.to_numpy(dtype=float)
    vals = s.to_numpy(dtype=float)
    for t in times:
        usable = idx[idx <= t]
        if len(usable) == 0:
            out[t] = 1.0
        else:
            out[t] = float(vals[np.argmax(idx == usable.max())] if usable.max() in idx else vals[idx <= t][-1])
            # simpler:
            out[t] = float(vals[idx <= t][-1])
    return pd.Series(out, name="S")


def pd_term_structure_survival(
    model: SurvivalModel,
    x: dict | pd.Series | np.ndarray,
    horizons: list[int],
) -> dict[int, float]:
    """
    PD(h) = 1 - S(h | x)
    """
    s = survival_function_at(model, x, times=horizons)
    return {int(h): float(1.0 - s.loc[h]) for h in horizons}


def pd_term_structure_km(
    model: SurvivalModel,
    horizons: list[int],
) -> dict[int, float]:
    """Unconditional KM-based PD(h) = 1 - S_km(h)."""
    out = {}
    sf = model.km.survival_function_
    # columns usually ['KM']
    series = sf.iloc[:, 0]
    idx = series.index.to_numpy(dtype=float)
    vals = series.to_numpy(dtype=float)
    for h in horizons:
        mask = idx <= h
        s = float(vals[mask][-1]) if mask.any() else 1.0
        out[int(h)] = float(1.0 - s)
    return out


# ---------------------------------------------------------------------------
# 4. Оценка качества
# ---------------------------------------------------------------------------

def evaluate_survival(
    model: SurvivalModel,
    features_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Concordance index на переданном fold (обычно OOT).
    """
    _require_lifelines()
    table = build_survival_table(
        features_df=features_df,
        feature_cols=model.feature_cols,
        default_state=model.default_state,
        covariate_at=model.covariate_at,
    )
    cols = model.feature_cols
    x = table[cols]
    # partial hazard as risk score
    risk = model.cox.predict_partial_hazard(x)
    c_index = float(
        concordance_index(table["duration"], -risk, table["event"])
    )
    event_rate = float(table["event"].mean())
    return {
        "c_index": round(c_index, 4),
        "event_rate": round(event_rate, 4),
        "n_clients": int(len(table)),
        "n_events": int(table["event"].sum()),
    }


# ---------------------------------------------------------------------------
# 5. Сравнение с Markov PD
# ---------------------------------------------------------------------------

def compare_pd_curves(
    markov_pd: dict[int, float],
    survival_pd: dict[int, float],
) -> pd.DataFrame:
    horizons = sorted(set(markov_pd) | set(survival_pd))
    rows = []
    for h in horizons:
        m = markov_pd.get(h, np.nan)
        s = survival_pd.get(h, np.nan)
        rows.append(
            {
                "horizon": h,
                "pd_markov": m,
                "pd_survival": s,
                "abs_diff": abs(m - s) if pd.notna(m) and pd.notna(s) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def report_survival_vs_markov(
    model: SurvivalModel,
    x: dict | pd.Series | np.ndarray,
    markov_pd: dict[int, float],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    horizons = horizons or sorted(markov_pd.keys())
    surv_pd = pd_term_structure_survival(model, x, horizons)
    return compare_pd_curves(markov_pd, surv_pd)


# ---------------------------------------------------------------------------
# 6. Утилита для train pipeline
# ---------------------------------------------------------------------------

def fit_and_save_survival(
    train_features: pd.DataFrame,
    feature_cols: list[str],
    path: str | Path = "models/survival_model.joblib",
    default_state: int = DEFAULT_STATE,
    penalizer: float = 0.1,
) -> SurvivalModel:
    model = fit_survival_model(
        features_df=train_features,
        feature_cols=feature_cols,
        default_state=default_state,
        penalizer=penalizer,
        covariate_at="first",
    )
    model.save(path)
    return model


if __name__ == "__main__":
    # smoke test on synthetic-like table
    _require_lifelines()
    rng = np.random.default_rng(42)
    n = 300
    demo = pd.DataFrame(
        {
            "client_id": np.repeat(np.arange(100), 3),
            "month": np.tile([1, 2, 3], 100),
            "state": rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]),
            "utilization": rng.random(n),
            "macro_stress": rng.normal(0, 0.5, n),
            "util_mean": rng.random(n),
        }
    )
    feats = ["utilization", "macro_stress", "util_mean"]
    model = fit_survival_model(demo, feats)
    metrics = evaluate_survival(model, demo)
    print("metrics:", metrics)
    pd_ts = pd_term_structure_survival(
        model,
        x={"utilization": 0.4, "macro_stress": 0.0, "util_mean": 0.35},
        horizons=[1, 2, 3],
    )
    print("pd_term:", pd_ts)