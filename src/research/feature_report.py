"""
feature_report.py — сводка по готовой feature-витрине.

Запуск:
    python -m src.research.feature_report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.processed_reader import load_processed_features
from src.features.feature_contracts import FEATURE_COLS

DEFAULT_STATE = 2
OUT_DIR = Path("reports/features")


def _ensure_next_state(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантирует next_state: берёт готовую колонку или делает shift(-1)."""
    if "next_state" in df.columns:
        return df.dropna(subset=["next_state"]).copy()
    tmp = df.sort_values(["client_id", "month"]).copy()
    tmp["next_state"] = tmp.groupby("client_id")["state"].shift(-1)
    return tmp.dropna(subset=["next_state"]).copy()


def summarize_size(df: pd.DataFrame) -> pd.Series:
    y = (df["next_state"].astype(int) == DEFAULT_STATE)
    return pd.Series(
        {
            "строк": int(len(df)),
            "клиентов": int(df["client_id"].nunique()),
            "признаков": int(sum(c in df.columns for c in FEATURE_COLS)),
            "месяц_мин": int(df["month"].min()) if len(df) else np.nan,
            "месяц_макс": int(df["month"].max()) if len(df) else np.nan,
            "дефолтов_в_next_state": int(y.sum()),
            "доля_дефолта_в_next_state": float(y.mean()) if len(df) else np.nan,
        }
    )


def missing_and_constants(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURE_COLS:
        if c not in df.columns:
            rows.append({
                "признак": c,
                "пропусков": np.nan,
                "доля_пропусков": np.nan,
                "уникальных": 0,
                "константа": True,
                "статус": "нет_в_витрине",
            })
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "признак": c,
            "пропусков": int(s.isna().sum()),
            "доля_пропусков": float(s.isna().mean()),
            "уникальных": int(s.nunique(dropna=True)),
            "константа": bool(s.nunique(dropna=True) <= 1),
            "статус": "ок",
        })
    return pd.DataFrame(rows)


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].describe().T

def corr_with_default(df: pd.DataFrame) -> pd.DataFrame:
    y = (df["next_state"].astype(int) == DEFAULT_STATE).astype(float)
    if y.nunique() < 2:
        return pd.DataFrame(columns=["признак", "корр_с_будущим_дефолтом"])

    rows = []
    for c in FEATURE_COLS:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.nunique(dropna=True) < 2:
            continue
        corr = x.corr(y)
        if pd.isna(corr):
            continue
        rows.append({"признак": c, "корр_с_будущим_дефолтом": float(corr)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("корр_с_будущим_дефолтом", key=lambda s: s.abs(), ascending=False)


def collinear_pairs(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    cols = [
        c for c in cols
        if pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) >= 2
    ]
    if len(cols) < 2:
        return pd.DataFrame()

    corr = df[cols].corr()
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(float(v)) >= threshold:
                rows.append({"признак_1": a, "признак_2": b, "корреляция": float(v)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("корреляция", key=lambda s: s.abs(), ascending=False)


def default_rate_by_bins(df: pd.DataFrame, col: str, q: int = 4) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    tmp = df[[col, "next_state"]].copy()
    tmp["y"] = (tmp["next_state"].astype(int) == DEFAULT_STATE).astype(int)
    x = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.loc[x.notna()].copy()
    if tmp["y"].nunique() < 2 or tmp[col].nunique() < 2:
        return pd.DataFrame()
    tmp["_bin"] = pd.qcut(x.loc[tmp.index], q=q, duplicates="drop")
    return (
        tmp.groupby("_bin", observed=True)
        .agg(n=("y", "size"), default_rate=("y", "mean"))
        .reset_index()
        .rename(columns={"_bin": col})
    )

def first_month_util_check(df: pd.DataFrame, tol: float = 0.01) -> pd.Series:
    if "util_mean" not in df.columns or "utilization" not in df.columns:
        return pd.Series({"проверка": False})
    first = df["month"] == df.groupby("client_id")["month"].transform("min")
    diff = (df.loc[first, "util_mean"] - df.loc[first, "utilization"]).abs()
    return pd.Series(
        {
            "проверка": True,
            "строк_первого_месяца": int(first.sum()),
            "расхождений": int((diff > tol).sum()),
            "макс_разница": float(diff.max()) if len(diff) else 0.0,
        }
    )

def run_feature_report(
    features_dir: str | Path = "data/processed/features",
    out_dir: str | Path = OUT_DIR,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_features(features_dir)
    df = _ensure_next_state(df)

    size = summarize_size(df)
    miss = missing_and_constants(df)
    desc = describe_features(df)
    corr = corr_with_default(df)
    pairs = collinear_pairs(df)
    pit = first_month_util_check(df)

    size.to_csv(out_dir / "summary.csv")
    miss.to_csv(out_dir / "missing_constants.csv", index=False)
    desc.to_csv(out_dir / "describe.csv")
    corr.to_csv(out_dir / "corr_with_default.csv", index=False)
    pairs.to_csv(out_dir / "collinear_pairs.csv", index=False)
    pit.to_csv(out_dir / "first_month_check.csv")

    for col in ("utilization", "util_trend", "frac_delinquent"):
        bins = default_rate_by_bins(df, col)
        if not bins.empty:
            bins.to_csv(out_dir / f"default_rate_by_{col}.csv", index=False)

    print("=== ОТЧЁТ ПО ПРИЗНАКАМ ===")
    print(size.to_string())
    print("\nРаспределение next_state:")
    print(df["next_state"].value_counts(normalize=True).sort_index().to_string())
    print("\nСписок признаков:")
    print(FEATURE_COLS)
    print("\nТоп корреляций с будущим дефолтом:")
    print(corr.head(10).to_string(index=False) if len(corr) else "нет (часто target константа)")
    print("\nДубли |корр|>=0.90:")
    print(pairs.head(10).to_string(index=False) if len(pairs) else "нет")
    print("\nPIT-проверка первого месяца:")
    print(pit.to_string())
    print(f"\nСохранено в {out_dir.resolve()}")

if __name__ == "__main__":
    run_feature_report()