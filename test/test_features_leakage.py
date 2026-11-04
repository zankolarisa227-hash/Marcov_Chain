from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features      import add_next_state, build_feature_matrix
from src.features.feature_contracts   import FEATURE_COLS, FORBIDDEN_COLS
from src.features.leakage_checks      import (
    assert_next_state_is_shift,          # правильное имя функции
    assert_temporal_order,
    assert_first_month_features_consistent,
)


# ---------------------------------------------------------------------------
# 1. Контракт признаков
# ---------------------------------------------------------------------------

def test_next_state_is_not_a_feature():
    """
    next_state — целевая переменная, не признак.
    Проверяем оба условия независимо — каждое несёт свой смысл.
    """
    assert "next_state" not in FEATURE_COLS, \
        "next_state не должен быть в FEATURE_COLS — это таргет"
    assert "next_state" in FORBIDDEN_COLS, \
        "next_state должен быть явно запрещён в FORBIDDEN_COLS"


def test_no_forbidden_col_in_feature_cols():
    """
    Ни один запрещённый столбец не должен попасть в список признаков.
    Проверяем весь FORBIDDEN_COLS, а не только next_state.
    """
    overlap = set(FORBIDDEN_COLS) & set(FEATURE_COLS)
    assert len(overlap) == 0, \
        f"Запрещённые колонки попали в FEATURE_COLS: {overlap}"


# ---------------------------------------------------------------------------
# 2. Point-in-time гарантия — признаки не смотрят в будущее
# ---------------------------------------------------------------------------

def test_future_month_cannot_leak_into_past_features():
    """
    Клиент 1: резкий скачок utilization только в month=3.
    Признаки на month=2 не должны знать про 0.99 из будущего.

    Проверяем util_max и util_mean — оба должны отражать
    только историю до month=2 включительно.
    """
    panel = pd.DataFrame({
        "client_id":    [1, 1, 1],
        "month":        [1, 2, 3],
        "state":        [0, 0, 0],
        "utilization":  [0.10, 0.10, 0.99],
        "macro_stress": [0.0, 0.0, 0.0],
    })
    feats = build_feature_matrix(panel, min_history=1)
    row2  = feats[(feats["client_id"] == 1) & (feats["month"] == 2)].iloc[0]

    assert row2["util_max"] <= 0.10 + 1e-9, \
        f"util_max на month=2 = {row2['util_max']:.3f}, " \
        f"но данные из month=3 (0.99) не должны быть известны"
    assert row2["util_mean"] <= 0.10 + 1e-9, \
        "util_mean на month=2 включает данные из month=3 — leakage"
    assert row2["utilization"] == pytest.approx(0.10), \
        "текущая utilization на month=2 должна быть 0.10"


def test_first_month_util_mean_equals_utilization():
    """
    На первом месяце клиента история = 1 точка.
    util_mean должен точно совпадать с utilization.
    Расхождение означает что в историю попали посторонние данные.
    """
    panel = pd.DataFrame({
        "client_id":    [1, 1, 1],
        "month":        [0, 1, 2],
        "state":        [0, 0, 1],
        "utilization":  [0.30, 0.40, 0.50],
        "macro_stress": [0.0, 0.0, 0.0],
    })
    feats    = build_feature_matrix(panel, min_history=1)
    first    = feats[(feats["client_id"] == 1) & (feats["month"] == 0)].iloc[0]

    assert first["util_mean"] == pytest.approx(first["utilization"], abs=1e-6), \
        "На первом месяце util_mean != utilization — возможен leakage"


# ---------------------------------------------------------------------------
# 3. next_state не пересекает клиентов
# ---------------------------------------------------------------------------

def test_next_state_does_not_cross_clients():
    """
    Последняя строка клиента A должна получить next_state=NaN,
    а не состояние первой строки клиента B.
    """
    panel = pd.DataFrame({
        "client_id":    [1, 1, 2, 2],
        "month":        [1, 2, 1, 2],
        "state":        [0, 1, 2, 0],
        "utilization":  [0.1, 0.2, 0.3, 0.4],
        "macro_stress": [0.0, 0.0, 0.0, 0.0],
    })
    out = add_next_state(panel)

    # Клиент 1, month=1: единственная обучающая строка → next_state=1
    c1_train = out[(out["client_id"] == 1)]
    assert len(c1_train) == 1, \
        "После add_next_state клиент 1 должен дать 1 обучающую строку"
    assert int(c1_train.iloc[0]["next_state"]) == 1, \
        "next_state для client=1, month=1 должен быть 1"

    # Главное: next_state != 2 (это state клиента 2, а не продолжение клиента 1)
    assert int(c1_train.iloc[0]["next_state"]) != 2, \
        "next_state клиента 1 не должен брать состояние клиента 2"


# ---------------------------------------------------------------------------
# 4. Leakage checks ловят реальные проблемы
# ---------------------------------------------------------------------------

def test_leakage_check_catches_shuffled_target():
    """
    Если next_state перемешан случайно — это нарушает
    инвариант shift(-1). Проверка должна это поймать.
    """
    panel = pd.DataFrame({
        "client_id":    [1, 1, 1],
        "month":        [1, 2, 3],
        "state":        [0, 1, 0],
        "utilization":  [0.1, 0.2, 0.3],
        "macro_stress": [0.0, 0.0, 0.0],
    })
    feats = add_next_state(panel)
    bad   = feats.copy()
    # Перемешиваем next_state — нарушаем shift(-1) инвариант
    bad["next_state"] = (
        bad["next_state"].sample(frac=1.0, random_state=0).to_numpy()
    )
    with pytest.raises(ValueError, match="next_state"):
        assert_next_state_is_shift(bad)


def test_temporal_order_check_catches_shuffled_rows():
    """
    Если строки клиента перемешаны по времени —
    проверка должна это поймать до того как shift(-1) даст неверный результат.
    """
    panel = pd.DataFrame({
        "client_id":    [1, 1, 1],
        "month":        [3, 1, 2],   # перемешанный порядок
        "state":        [0, 1, 0],
        "utilization":  [0.3, 0.1, 0.2],
        "macro_stress": [0.0, 0.0, 0.0],
    })
    with pytest.raises(ValueError, match="порядк"):
        assert_temporal_order(panel)


def test_first_month_check_catches_leakage():
    """
    Симулируем баг: util_mean на первом месяце посчитан
    по всей истории клиента вместо только текущего момента.
    assert_first_month_features_consistent должна это поймать.
    """
    panel = pd.DataFrame({
        "client_id":    [1, 1, 1],
        "month":        [0, 1, 2],
        "state":        [0, 0, 0],
        "utilization":  [0.10, 0.10, 0.90],
        "macro_stress": [0.0, 0.0, 0.0],
    })
    feats = build_feature_matrix(panel, min_history=1)

    # Симулируем баг: подменяем util_mean на первом месяце
    # на значение посчитанное по всей истории (включая 0.90)
    feats_bad = feats.copy()
    first_mask = (
        (feats_bad["client_id"] == 1) & (feats_bad["month"] == 0)
    )
    feats_bad.loc[first_mask, "util_mean"] = 0.367  # (0.10+0.10+0.90)/3

    with pytest.raises(ValueError, match="util_mean"):
        assert_first_month_features_consistent(feats_bad)