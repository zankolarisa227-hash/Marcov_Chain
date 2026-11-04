"""
Синтетическая panel → features_synth.

Запуск:
    python -m src.data.build_synth_features
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.processed_reader import load_processed_panel
from src.features.build_features import add_next_state, build_feature_matrix

PANEL_DIR = "data/processed/panel_synth"
OUT_DIR = Path("data/processed/features_synth")
MIN_HISTORY = 3


def main() -> None:
    print("[synth_features] читаем панель:", PANEL_DIR)
    panel = load_processed_panel(PANEL_DIR)
    print(
        "[synth_features] panel:",
        "строк=", len(panel),
        "клиентов=", panel["client_id"].nunique(),
        "колонки=", list(panel.columns),
    )
    print("[synth_features] доли state:")
    print(panel["state"].value_counts(normalize=True).sort_index())

    print("[synth_features] считаем признаки, min_history=", MIN_HISTORY)
    feat = build_feature_matrix(panel, min_history=MIN_HISTORY)
    print("[synth_features] после FE строк=", len(feat))

    print("[synth_features] добавляем next_state")
    feat = add_next_state(feat)

    print("[synth_features] таблица переходов state → next_state:")
    print(pd.crosstab(feat["state"], feat["next_state"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "part-000.parquet"
    feat.to_parquet(out_path, index=False)
    print("[synth_features] сохранено:", out_path.resolve())
    print("[synth_features] features строк=", len(feat), "колонок=", feat.shape[1])
    print("[synth_features] готово")


if __name__ == "__main__":
    main()