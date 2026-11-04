"""
loader.py — последовательная загрузка raw parquet.

Зачем:
    batch_build читает один файл → panel/features → пишет part-*.parquet → забывает.
    Так raw не обязан целиком помещаться в RAM.

Контракт raw (Alfa-CT):
    app_id, transactions, flag
    transactions: список/массив операций; [:, 0] ≈ amount, [:, 16] ≈ days_before

NOTE:
    Только локальный data/raw. Схема должна совпадать с preprocess_panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


def iter_raw_parquet(data_dir: str | Path = "data/raw") -> Iterator[Path]:

    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Raw dir not found: {path.resolve()}")

    files = sorted(path.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {path.resolve()}")

    yield from files


def load_raw_file(
    file_path: str | Path,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Читает один parquet.

    Parameters
    ----------
    columns :
        Column pruning — меньше I/O и RAM. Для panel обычно
        ["app_id", "transactions", "flag"].

    TODO
    ----
    - assert обязательных колонок после чтения
    - опциональный лог: path, nrows, memory_usage
    """
    file_path = Path(file_path)
    return pd.read_parquet(file_path, columns=columns)


def load_raw_concat(
    data_dir: str | Path = "data/raw",
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Склеить весь raw (только для отладки / маленького sample).

    WARNING: на полном датасете может не влезть в память.
    В batch_build предпочтителен iter_raw_parquet + обработка по файлам.
    """
    parts = [load_raw_file(p, columns=columns) for p in iter_raw_parquet(data_dir)]
    return pd.concat(parts, ignore_index=True)