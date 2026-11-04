"""
eda_report.py — EDA панели и признаков для Markov PD.

python -m src.research.eda_report
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.processed_reader import load_processed_features, load_processed_panel
from src.features.feature_contracts import FEATURE_COLS

try:
    from src.models.markov_chain import estimate_transition_matrix
except ImportError:
    estimate_transition_matrix = None

DEFAULT_STATE = 2
STATE_NAMES = {0: "current", 1: "delinquent", 2: "default"}
OUT_DIR = Path("reports/eda")


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150)
    plt.close(fig)


def _pairs(df: pd.DataFrame) -> pd.DataFrame:
    x = df.dropna(subset=["next_state"]).copy()
    x["next_state"] = x["next_state"].astype(int)
    x["state"] = x["state"].astype(int)
    return x


def level1_overview(panel: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    rows = {
        "panel_строк": len(panel),
        "panel_клиентов": panel["client_id"].nunique(),
        "feat_строк": len(feat),
        "feat_клиентов": feat["client_id"].nunique(),
        "месяц_мин": int(feat["month"].min()),
        "месяц_макс": int(feat["month"].max()),
        "пропусков_panel": int(panel.isna().sum().sum()),
        "пропусков_feat": int(feat.isna().sum().sum()),
    }
    if "next_state" in feat.columns:
        y = (feat["next_state"] == DEFAULT_STATE)
        rows["доля_next_default"] = float(y.mean())
    pd.Series(rows).to_csv(OUT_DIR / "01_overview.csv")
    return pd.Series(rows)


def level2_target(panel: pd.DataFrame, feat: pd.DataFrame) -> None:
    # доли state
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = panel["state"].value_counts(normalize=True).sort_index()
    ax.bar([STATE_NAMES.get(i, i) for i in vc.index], vc.values, color="#4C78A8")
    ax.set_ylabel("Доля строк")
    ax.set_title("Распределение состояний в panel")
    _save(fig, "02_state_share.png")

    # next_state
    if "next_state" in feat.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = feat["next_state"].value_counts(normalize=True).sort_index()
        ax.bar([STATE_NAMES.get(int(i), i) for i in vc.index], vc.values, color="#E45756")
        ax.set_ylabel("Доля")
        ax.set_title("Распределение next_state (target)")
        _save(fig, "02_next_state_share.png")

    # динамика state по месяцам
    mix = (
        panel.groupby("month")["state"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
        .pivot(index="month", columns="state", values="share")
        .fillna(0)
    )
    mix.to_csv(OUT_DIR / "02_state_mix_by_month.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, name in STATE_NAMES.items():
        if s in mix.columns:
            ax.plot(mix.index, mix[s], marker="o", label=name)
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Доля")
    ax.set_title("Смесь состояний во времени")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "02_state_mix_lines.png")


def level3_features(feat: pd.DataFrame) -> None:
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    y = (feat["next_state"] == DEFAULT_STATE).astype(float)

    # гистограммы ключевых, не всех 26
    key = [c for c in ["utilization", "util_mean", "util_trend", "frac_delinquent"] if c in cols]
    if key:
        fig, axes = plt.subplots(1, len(key), figsize=(4 * len(key), 3))
        if len(key) == 1:
            axes = [axes]
        for ax, c in zip(axes, key):
            ax.hist(pd.to_numeric(feat[c], errors="coerce").dropna(), bins=40, color="#4C78A8")
            ax.set_title(c)
        _save(fig, "03_key_hists.png")

    # corr с default, без констант
    rows = []
    for c in cols:
        x = pd.to_numeric(feat[c], errors="coerce")
        if x.nunique(dropna=True) < 2 or y.nunique() < 2:
            continue
        r = x.corr(y)
        if pd.notna(r):
            rows.append({"признак": c, "корр": float(r)})
    corr = pd.DataFrame(rows).sort_values("корр", key=lambda s: s.abs(), ascending=False)
    corr.to_csv(OUT_DIR / "03_corr_default.csv", index=False)

    if len(corr):
        top = corr.head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#E45756" if v > 0 else "#4C78A8" for v in top["корр"]]
        ax.barh(top["признак"][::-1], top["корр"][::-1], color=colors[::-1])
        ax.set_xlabel("corr с next_state==default")
        ax.set_title("Связь признаков с будущим дефолтом")
        _save(fig, "03_corr_bars.png")

    # мультиколлинеарность
    use = [c for c in cols if pd.to_numeric(feat[c], errors="coerce").nunique(dropna=True) >= 2]
    if len(use) >= 2:
        cm = feat[use].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1)
        ax.set_title("Корреляции признаков")
        _save(fig, "03_feature_corr_heatmap.png")


def level4_markov(feat: pd.DataFrame) -> None:
    pairs = _pairs(feat)
    if pairs.empty:
        return

    # счётчик переходов
    N = pd.crosstab(pairs["state"], pairs["next_state"], dropna=False)
    P = N.div(N.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    N.to_csv(OUT_DIR / "04_transition_counts.csv")
    P.to_csv(OUT_DIR / "04_transition_P.csv")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    labels = [STATE_NAMES.get(i, str(i)) for i in P.index]
    sns.heatmap(
        P.values,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=[STATE_NAMES.get(i, str(i)) for i in P.columns],
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Куда (t+1)")
    ax.set_ylabel("Откуда (t)")
    ax.set_title("Статическая матрица P")
    _save(fig, "04_P_heatmap.png")

    # P по месяцам — для однородности
    months = sorted(pairs["month"].unique())
    fig, axes = plt.subplots(1, len(months), figsize=(4 * max(len(months), 1), 3.5))
    if len(months) == 1:
        axes = [axes]
    for ax, m in zip(axes, months):
        sl = pairs[pairs["month"] == m]
        Pm = pd.crosstab(sl["state"], sl["next_state"], normalize="index")
        sns.heatmap(Pm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1, cbar=False)
        ax.set_title(f"месяц {m}")
    _save(fig, "04_P_by_month.png")

    # длина истории
    hist = feat.groupby("client_id")["month"].nunique()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(hist, bins=range(1, int(hist.max()) + 2), color="#4C78A8", align="left")
    ax.set_xlabel("Число месяцев в витрине")
    ax.set_ylabel("Клиенты")
    ax.set_title("Длина истории")
    _save(fig, "04_history_len.png")

    # default rate по utilization
    if "utilization" in pairs.columns:
        tmp = pairs.copy()
        tmp["y"] = (tmp["next_state"] == DEFAULT_STATE).astype(float)
        tmp["util_bucket"] = pd.cut(
            pd.to_numeric(tmp["utilization"], errors="coerce"),
            bins=[-0.01, 0.3, 0.6, 0.8, 1.01],
            labels=["0-0.3", "0.3-0.6", "0.6-0.8", "0.8-1"],
        )
        tab = tmp.groupby("util_bucket", observed=True)["y"].mean()
        tab.to_csv(OUT_DIR / "04_default_by_util.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(tab.index.astype(str), tab.values, color="#E45756")
        ax.set_ylabel("P(next_state=default)")
        ax.set_xlabel("Utilization")
        ax.set_title("Дефолт по корзинам утилизации")
        _save(fig, "04_default_by_util.png")


def run_eda() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    panel = load_processed_panel("data/processed/panel_synth")
    feat = load_processed_features("data/processed/features_synth")
    if "next_state" not in feat.columns:
        feat = feat.sort_values(["client_id", "month"])
        feat["next_state"] = feat.groupby("client_id")["state"].shift(-1)
        feat = feat.dropna(subset=["next_state"])

    print("=== EDA ===")
    print(level1_overview(panel, feat).to_string())
    level2_target(panel, feat)
    level3_features(feat)
    level4_markov(feat)
    print(f"графики: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    run_eda()