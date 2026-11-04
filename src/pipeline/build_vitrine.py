"""
Сборка витрины features из готовой panel.
Запуск:
    python -m src.pipeline.build_vitrine
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


PANEL_DIRS = [
    Path("data/processed/panel_synth"),
    Path("data/processed/panel"),
]
OUT_DIR = Path("data/processed/features_synth")
MIN_HISTORY = 3
ROWS_PER_PART = 200_000


def load_panel() -> pd.DataFrame:
    for d in PANEL_DIRS:
        files = sorted(d.glob("*.parquet"))
        if files:
            print(f"[vitrine] panel: {d} files={len(files)}")
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            return df.sort_values(["client_id", "month"]).reset_index(drop=True)
    raise FileNotFoundError("нет panel в data/processed/panel_synth или panel")


def load_feature_builder():
    try:
        from src.features.build_features import FEATURE_COLS, build_feature_matrix
        print("[vitrine] берём build_feature_matrix из src.features.build_features")
        return build_feature_matrix, FEATURE_COLS
    except Exception as e:
        print("[vitrine] не удалось импортировать build_features:", e)
        raise


def write_parts(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    n = len(df)
    n_parts = max(1, (n + ROWS_PER_PART - 1) // ROWS_PER_PART)
    for i in range(n_parts):
        sl = df.iloc[i * ROWS_PER_PART : (i + 1) * ROWS_PER_PART]
        path = out_dir / f"part-{i:03d}.parquet"
        sl.to_parquet(path, index=False)
        print(f"[vitrine] wrote {path.name} rows={len(sl)}")


def main() -> None:
    panel = load_panel()
    print(
        "[vitrine] panel rows=",
        len(panel),
        "clients=",
        panel["client_id"].nunique(),
        "cols=",
        list(panel.columns),
    )
    print("[vitrine] state shares:")
    print(panel["state"].value_counts(normalize=True).sort_index())

    build_feature_matrix, feature_cols = load_feature_builder()
    print("[vitrine] считаем признаки, min_history=", MIN_HISTORY)
    features = build_feature_matrix(panel, min_history=MIN_HISTORY)

    needed = ["client_id", "month", "state", *feature_cols]
    missing = [c for c in needed if c not in features.columns]
    if missing:
        raise ValueError(f"в features нет колонок: {missing}")
    features = features[needed]

    print(
        "[vitrine] features rows=",
        len(features),
        "clients=",
        features["client_id"].nunique(),
        "n_features=",
        len(feature_cols),
    )
    write_parts(features, OUT_DIR)
    print("[vitrine] done ->", OUT_DIR.resolve())


if __name__ == "__main__":
    main()