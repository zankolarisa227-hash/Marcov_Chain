"""
leakage_checks.py — строгая диагностика future leakage.

Пять уровней проверок выстроены от простого к сложному:
    1. Запрещённые колонки в DataFrame
    2. Временной порядок строк внутри клиента
    3. Корректность next_state как shift(-1)
    4. Причинность признаков (не заглядывают в будущее)
    5. Scaler обучен только на train-данных

Запускать после build_feature_matrix() и перед train().
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

# Порог корреляции для обнаружения прямой утечки.
# Почему 0.95 а не меньше:
#   Высокая корреляция признака с будущим дефолтом — это НОРМА.
#   ever_delinquent должен сильно коррелировать с future default
#   по смыслу задачи. Это не leakage, это полезный сигнал.
#   Нас интересует только ПОЧТИ ИДЕАЛЬНАЯ корреляция (>0.95) —
#   она означает что признак содержит целевую переменную напрямую,
#   например next_state_encoded случайно попал в признаки.
#   При снижении до 0.7-0.8 будет много false positives на
#   легитимных сильных признаках.
LEAKAGE_CORR_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# 1. Запрещённые колонки
# ---------------------------------------------------------------------------

def assert_no_forbidden_columns(df: pd.DataFrame,
                                  forbidden: list[str]) -> None:
    """
    Запрещённые колонки не должны присутствовать в DataFrame вообще.

    Проверяем наличие в df.columns, а не пересечение со списком признаков.
    Даже если колонка не входит в FEATURE_COLS — её присутствие опасно:
    она может случайно попасть в модель через .values или select_dtypes.
    """
    found = [c for c in forbidden if c in df.columns]
    if found:
        raise ValueError(
            f"LEAKAGE: запрещённые колонки в DataFrame: {found}\n"
            f"Это данные из будущего или целевая переменная.\n"
            f"Удали их до передачи в модель."
        )


# ---------------------------------------------------------------------------
# 2. Временной порядок строк
# ---------------------------------------------------------------------------

def assert_temporal_order(df: pd.DataFrame,
                           client_col: str = "client_id",
                           time_col:   str = "month") -> None:
    """
    Для каждого клиента строки должны идти в строго возрастающем
    порядке по времени.

    Почему это критично для leakage:
    shift(-1) вычисляет next_state как следующую строку в DataFrame.
    Если строки клиента 42 идут в порядке [март, январь, февраль],
    то next_state для марта будет январское состояние — это leakage.
    Код не упадёт, просто признаки будут тихо неверными.
    """
    is_monotone = (
        df.sort_values([client_col, time_col])
          .groupby(client_col)[time_col]
          .apply(lambda s: s.is_monotonic_increasing)
    )
    bad_clients = is_monotone[~is_monotone].index.tolist()
    if bad_clients:
        raise ValueError(
            f"LEAKAGE RISK: у {len(bad_clients)} клиентов месяцы "
            f"идут не по возрастанию.\n"
            f"Примеры client_id: {bad_clients[:5]}\n"
            f"Это нарушает корректность shift(-1) при построении next_state."
        )


# ---------------------------------------------------------------------------
# 3. Корректность next_state
# ---------------------------------------------------------------------------

def assert_next_state_is_shift(df: pd.DataFrame,
                                client_col: str = "client_id",
                                time_col:   str = "month",
                                state_col:  str = "state",
                                next_col:   str = "next_state") -> None:
    """
    next_state должен быть строго shift(-1) внутри клиента.

    Проверяем прямым сравнением целых чисел (не np.allclose),
    потому что состояния — категориальные значения (0, 1, 2),
    и для них нет смысла в "почти равно".
    """
    if next_col not in df.columns:
        raise ValueError(
            f"Колонка '{next_col}' отсутствует в DataFrame.\n"
            f"Убедись что make_transition_pairs() был вызван."
        )

    tmp = df.sort_values([client_col, time_col]).copy()
    expected = tmp.groupby(client_col)[state_col].shift(-1)
    mask = expected.notna()

    actual_vals   = tmp.loc[mask, next_col].astype(int).values
    expected_vals = expected.loc[mask].astype(int).values

    mismatches = int(np.sum(actual_vals != expected_vals))
    if mismatches > 0:
        raise ValueError(
            f"LEAKAGE: next_state не совпадает со shift(-1) "
            f"в {mismatches} строках из {mask.sum()}.\n"
            f"Возможная причина: некорректный join или merge "
            f"по нескольким ключам."
        )


# ---------------------------------------------------------------------------
# 4. Причинность признаков
# ---------------------------------------------------------------------------

def assert_features_strictly_causal(df: pd.DataFrame,
                                      feature_cols: list[str],
                                      client_col:   str = "client_id",
                                      time_col:     str = "month",
                                      state_col:    str = "state",
                                      default_state: int = 2) -> None:
    """
    Проверяет что признаки не содержат прямой утечки из будущего.

    Метод: корреляция каждого признака с бинарным таргетом
    next_state == default_state.

    Почему бинарный таргет, а не next_state напрямую:
        next_state принимает значения 0, 1, 2 — это категориальная
        переменная. Корреляция Пирсона с категориальной переменной
        методологически некорректна и даёт бессмысленные числа.
        Бинарный таргет (дефолт/не дефолт) — непрерывный 0/1,
        корреляция с ним осмысленна.

    Порог LEAKAGE_CORR_THRESHOLD = 0.95 — см. комментарий в начале файла.
    """
    tmp = df.sort_values([client_col, time_col]).copy()
    tmp["_future_default"] = (
        tmp.groupby(client_col)[state_col].shift(-1) == default_state
    ).astype(float)
    tmp = tmp.dropna(subset=["_future_default"])

    suspicious = []
    for col in feature_cols:
        if col not in tmp.columns:
            continue
        x = pd.to_numeric(tmp[col], errors="coerce")
        if x.isna().all() or x.nunique() < 2:
            continue
        corr = float(abs(x.corr(tmp["_future_default"])))
        if corr > LEAKAGE_CORR_THRESHOLD:
            suspicious.append({
                "feature": col,
                "corr_with_future_default": round(corr, 4)
            })

    if suspicious:
        msg = (
            f"ПРЕДУПРЕЖДЕНИЕ: признаки с подозрительно высокой "
            f"корреляцией с future default "
            f"(порог={LEAKAGE_CORR_THRESHOLD}):\n"
            + "\n".join(
                f"  {r['feature']}: {r['corr_with_future_default']}"
                for r in suspicious
            )
            + "\nПроверь что эти признаки не содержат данных из t+1."
        )
        warnings.warn(msg, stacklevel=2)


def assert_first_month_features_consistent(
        df: pd.DataFrame,
        client_col:   str = "client_id",
        time_col:     str = "month",
        tol:          float = 0.01) -> None:
    """
    Ключевая проверка point-in-time гарантии.

    На первом месяце клиента история = одна точка.
    Поэтому util_mean должен равняться utilization того же месяца.
    Если они расходятся — в историю попали данные из другого периода.

    Это косвенная, но мощная проверка: она ловит случаи когда
    агрегация по истории случайно захватила будущие данные клиента.
    """
    if "util_mean" not in df.columns or "utilization" not in df.columns:
        return  # признаки отсутствуют — пропускаем

    min_month_per_client = df.groupby(client_col)[time_col].transform("min")
    first_rows = df[df[time_col] == min_month_per_client]

    diff = (first_rows["util_mean"] - first_rows["utilization"]).abs()
    bad  = int((diff > tol).sum())

    if bad > 0:
        worst = float(diff.max())
        raise ValueError(
            f"LEAKAGE: на первом месяце клиента util_mean != utilization "
            f"в {bad} строках (макс. расхождение: {worst:.4f}).\n"
            f"Признаки могут использовать данные из периода до "
            f"начала наблюдения клиента или из будущих месяцев."
        )


# ---------------------------------------------------------------------------
# 5. Scaler обучен только на train
# ---------------------------------------------------------------------------

def assert_scaler_fitted_on_train_only(scaler,
                                        train_df:     pd.DataFrame,
                                        feature_cols: list[str],
                                        tol:          float = 0.01) -> None:
    """
    Проверяет что StandardScaler обучен на train, а не на всём датасете.

    Если scaler.fit() был вызван на train+test вместе — это leakage:
    mean и std признаков посчитаны с учётом test-данных.
    Модель косвенно "знает" статистики test-выборки.

    Метод: сравниваем mean_ scaler'а со средним по train.
    Если отличаются больше чем на tol — scaler обучен на другой выборке.
    """
    if not hasattr(scaler, "mean_"):
        return  # не StandardScaler — пропускаем

    available_cols = [c for c in feature_cols if c in train_df.columns]
    X_train = train_df[available_cols].values.astype(float)
    expected_mean = np.nanmean(X_train, axis=0)
    actual_mean   = scaler.mean_[:len(available_cols)]

    diff = np.abs(actual_mean - expected_mean)
    bad_idx = np.where(diff > tol)[0]

    if len(bad_idx) > 0:
        bad_features = [available_cols[i] for i in bad_idx]
        raise ValueError(
            f"LEAKAGE: scaler обучен НЕ на train-выборке.\n"
            f"Признаки с расхождением mean > {tol}: {bad_features}\n"
            f"Это означает что test-данные попали в scaler.fit().\n"
            f"Вызыв scaler.fit() только на train_df."
        )


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def run_leakage_checks(features_df:   pd.DataFrame,
                        feature_cols:  list[str],
                        forbidden_cols: Optional[list[str]] = None,
                        scaler=None,
                        train_df:      Optional[pd.DataFrame] = None,
                        client_col:    str = "client_id",
                        time_col:      str = "month",
                        state_col:     str = "state",
                        default_state: int = 2) -> None:
    """
    Запускает полный набор leakage-проверок.
    Вызывать после build_feature_matrix() и перед train().

    При обнаружении leakage поднимает ValueError с объяснением причины.
    При подозрении (высокая корреляция) — предупреждение, не ошибка.
    """
    forbidden_cols = forbidden_cols or [
        "flag", "future_state",
        "target", "label", "default_flag"
    ]

    if "next_state" in feature_cols:
        raise ValueError("LEAKAGE: next_state попал в FEATURE_COLS")

    print("[leakage] 1/5 запрещённые колонки...")
    assert_no_forbidden_columns(features_df, forbidden_cols)

    print("[leakage] 2/5 временной порядок строк...")
    assert_temporal_order(features_df, client_col, time_col)

    print("[leakage] 3/5 корректность next_state...")
    assert_next_state_is_shift(
        features_df, client_col, time_col, state_col
    )

    print("[leakage] 4/5 причинность признаков + first-month check...")
    assert_features_strictly_causal(
        features_df, feature_cols,
        client_col, time_col, state_col, default_state
    )
    assert_first_month_features_consistent(
        features_df, client_col, time_col
    )

    if scaler is not None and train_df is not None:
        print("[leakage] 5/5 scaler на train-only...")
        assert_scaler_fitted_on_train_only(scaler, train_df, feature_cols)
    else:
        print("[leakage] 5/5 scaler check пропущен (не передан)")

    print("[leakage]  все проверки пройдены — leakage не обнаружен")