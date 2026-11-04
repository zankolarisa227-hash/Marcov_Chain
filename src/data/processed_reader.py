"""
processed_reader.py — чтение собранной панели и признаков.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _read_parts(
    dir_path: str | Path,
    pattern: str = "part-*.parquet",
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Директория не найдена: {path.resolve()}")

    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Нет файлов по шаблону '{pattern}' в {path.resolve()}")

    parts = []
    for f in files:
        print(f"[reader] читаю {f.name}")
        parts.append(pd.read_parquet(f, columns=columns))

    df = pd.concat(parts, ignore_index=True)
    return df


def load_processed_panel(
    panel_dir: str = "data/processed/pannel",
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Читает все part-*.parquet из папки панели.
    Подходит и для синтетики, и для part-sample_2009.parquet и т.д.
    """
    df = _read_parts(panel_dir, pattern="part-*.parquet", columns=columns)
    return df.sort_values(["client_id", "month"]).reset_index(drop=True)


def load_processed_features(
    features_dir: str = "data/processed/features",
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    df = _read_parts(features_dir, pattern="part-*.parquet", columns=columns)
    return df.sort_values(["client_id", "month"]).reset_index(drop=True)


if __name__ == "__main__":
    panel = load_processed_panel()
    print(panel.shape)
    print(panel.columns.tolist())
    print(panel.head())