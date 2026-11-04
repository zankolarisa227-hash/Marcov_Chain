"""
preprocess_panel.py — raw Alfa-CT → panel для Markov PD.

Выход (минимум):
    client_id, month, state, utilization, macro_stress

state:
    0 = current
    1 = delinquent
    2 = default (absorbing)

PIT / leakage
-------------
Panel на месяце t не должен знать будущее после t.
next_state = S_{t+1} считается позже: groupby(client_id).shift(-1)
на этапе features — не здесь.

FIXME (критично для academic honesty)
-------------------------------------
1. state — эвристика от flag + наличия транзакций, НЕ bureau DPD / официальный default.
2. utilization — proxy (avg amount / 10000), не util = balance/limit.
3. macro_stress = 0.0 — заглушка.
4. iterrows медленный на большом raw → векторлизация / батчи по файлам.
5. month — относительный индекс 1..max_months, не календарный month_id.

Это допустимо для курсовой/research при явном Limitations-разделе.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_RAW_COLS = ("app_id", "transactions", "flag")

COL_AMNT = 0
COL_ECOM = 6
COL_INCOME = 8
COL_MCC = 9
COL_DOW = 13
COL_HOUR = 14
COL_DAYS = 16

def _validate_raw(df_raw: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RAW_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Raw missing columns: {missing}")


def aggregate_transactions_to_panel(
    df_raw: pd.DataFrame,
    max_months: int = 6,
    default_state: int = 2,
    amount_scale: float = 10_000.0,
) -> pd.DataFrame:
    """
    app_id, transactions, flag → client × month panel.

    Parameters
    ----------
    max_months :
        Длина истории в месяцах (padding до этой длины).
    default_state :
        Код поглощающего default (обычно 2).
    amount_scale :
        Нормировка proxy-utilization: min(mean(amount) / amount_scale, 1).

    Returns
    -------
    DataFrame
        client_id, month, state, utilization, macro_stress
        отсортирован по (client_id, month).

    TODO
    ----
    - явные правила state из платёжной дисциплины (если появятся поля)
    - календарный month_id
    - ненулевой macro по внешним рядам (ключ.ставка / CPI / unemployment)
    - unit-test: 1 client → ровно max_months строк; state ∈ {0,1,2}
    - убрать iterrows (explode + groupby)
    """
    _validate_raw(df_raw)
    logger.info("Building panel from raw: n_rows=%s", len(df_raw))

    records: list[dict[str, Any]] = []

    for _, row in df_raw.iterrows():
        client_id = row["app_id"]
        trans = row["transactions"]
        flag = int(row["flag"]) if pd.notna(row["flag"]) else 0

        # --- пустая история: нейтральный current ---
        if trans is None or len(trans) == 0:
            for month in range(1, max_months + 1):
                records.append(
                    {
                        "client_id": client_id,
                        "month": month,
                        "state": 0,
                        "utilization": 0.0,
                        "macro_stress": 0.0,  # FIXME: stub
                    }
                )
            continue

        arr = np.asarray(trans)
        # Контракт Alfa-CT: col0=amount, col16=days_before (проверить на своём raw!)
        n = len(arr)
        days_before = arr[:, 16] if arr.ndim == 2 and arr.shape[1] > 16 else np.zeros(n)
        amounts = arr[:, 0] if arr.ndim == 2 and arr.shape[1] > 0 else np.zeros(n)

        months_from_data = (np.asarray(days_before) // 30).astype(int)
        unique_months = np.unique(months_from_data)
        if len(unique_months) > max_months:
            unique_months = unique_months[-max_months:]

        for m in unique_months:
            mask = months_from_data == m
            month_amounts = amounts[mask]
            avg_amount = float(np.mean(month_amounts)) if len(month_amounts) else 0.0

            # FIXME: proxy utilization, не product util
            utilization = float(min(avg_amount / amount_scale, 1.0))

            # FIXME: heuristic state — не DPD ladder
            if len(month_amounts) > 0:
                state = 0
            else:
                state = 0

            month_index = int(m - unique_months[0] + 1)
            records.append(
                {
                    "client_id": client_id,
                    "month": month_index,
                    "state": int(state),
                    "utilization": utilization,
                    "macro_stress": 0.0,  # FIXME: stub
                }
            )

        # padding короткой истории до max_months
        existing = {r["month"] for r in records if r["client_id"] == client_id}
        for m in range(1, max_months + 1):
            if m not in existing:
                records.append(
                    {
                        "client_id": client_id,
                        "month": m,
                        "state": 0,
                        "utilization": 0.0,
                        "macro_stress": 0.0,
                    }
                )

            if flag == 1:
                idx = [i for i, r in enumerate(records) if r["client_id"] == client_id]
                if len(idx) >= 2:
                    last_i = max(idx, key=lambda i: records[i]["month"])
                    prev_i = max(
                        (i for i in idx if records[i]["month"] < records[last_i]["month"]),
                        key=lambda i: records[i]["month"],
                    )
                records[prev_i]["state"] = 0
                records[last_i]["state"] = default_state
            elif idx:
                records[idx[0]]["state"] = default_state

    panel = pd.DataFrame.from_records(records)

    panel = pd.DataFrame.from_records(records)
    panel = panel.sort_values(["client_id", "month"]).reset_index(drop=True)

    # лёгкие инварианты
    assert set(panel["state"].unique()).issubset({0, 1, 2})
    assert panel.groupby("client_id")["month"].count().min() >= 1

    logger.info("Panel built: rows=%s cols=%s", panel.shape[0], panel.shape[1])
    return panel


if __name__ == "__main__":
    # Не load_data — его нет в loader.py
    from src.data.loader import iter_raw_parquet, load_raw_file

    cols = ["app_id", "transactions", "flag"]
    parts = [load_raw_file(p, columns=cols) for p in iter_raw_parquet("data/raw")]
    df_raw = pd.concat(parts, ignore_index=True)

    panel = aggregate_transactions_to_panel(df_raw, max_months=6)
    print(panel.head())
    print("shape:", panel.shape)
    print("cols:", panel.columns.tolist())
    print(panel["state"].value_counts(normalize=True).sort_index())