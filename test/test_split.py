from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.split import purged_time_split


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture()
def panel_month() -> pd.DataFrame:
    return pd.DataFrame({
        "client_id": [1]*10, 
        "month": list(range(1,11)), 
        "state": [0,0,1,1,0,0,1,1,2,2], 
        "utilization": [0.1 * i for i in range(1,11)],
    })
    
@pytest.fixture()
def multi_client_panel() -> pd.DataFrame: 
    rows = []
    for cid, months in [(1, range(1,9)), (2, range(1,7)), (3, range(3,11))]:
        for m in months:
            rows.append({
                "client_id": cid,
                "month": int(m),
                "state": 0 if m < 5 else 1,
                "utilization": 0.05 * m,
            })
    return pd.DataFramr(rows)



# границы split/embargo
def test_train_only_before_split(panel_month):
    #распаковка кортежа
    train, test = purged_time_split(
        panel_month, split_month=5, embargo_month=1
    )
    assert len(train)>0
    assert int(train["month"].max())<5


def test_only_for_embargo(panel_months):
    train, test = purged_time_split(
        panel_months, split_month=5, embargo_month=1
    )
    assert len(test)>0
    assert int (test["month"]. min()) >=6


def test_embargo_excluded_from_train_and_test(panel_months):
    split, embargo = 5, 2
    train, test = purged_time_split(
        panel_months, split_month=split, embargo_months=embargo
    )
    embargo_months = set(range(split, split + embargo))  # {5, 6}
    used = set(train["month"].tolist()) | set(test["month"].tolist())
    assert used.isdisjoint(embargo_months)

def test_train_test_months_disjoint(panel_months):
    train, test = purged_time_split(
        panel_months, split_month=4, embargo_months=1
    )
    assert set(train["month"]).isdisjoint(set(test["month"]))

def test_zero_embargo_allowed(panel_months):
    train, test = purged_time_split(
        panel_months, split_month=6, embargo_months=0
    )
    assert int(train["month"].max()) <6
    assert int(test["month"]).min() >=6

def test_negative_embargo_raises(panel_months):
    with pytest.raises(ValueError):
        purged_time_split(panel_month, split_month=5, embargo_months=-1)

def test_non_embargo_rows_fully_assugned(panel_months):
    split, embargo = 5,1
    train, test = purged_time_split(
        panel_months, split_month=split, embargo_months=embargo
    )
    embargo_set = set(range(split, split + embargo))
    expected = panel_months[~panel_months["month"].isin(embargo_set)]
    got = pd.concat([train, test], ignore_index=True)

    assert len(got) == len(expected)
    exp_keys = set(zip(expected["client_id"], expected["month"]))
    got_keys = set(zip(got["client_id"], got["month"]))
    assert exp_keys == got_keys

def test_split_does_not_mutate_input(panel_months):
    original = panel_months.copy(deep=True)
    purged_time_split(panel_month, split_months=5, embargo_months=1)
    pd.testing.assert_frame_equal(panel_months, original)

def test_split_preserves_columns(multi_client_panel):
    train, test = purged_time_split(
        multi_client_panel, split_month=5, embargo_months=1
    )
    assert list(train.columns) == list(multi_client_panel.columns)
    assert list(test.columns) == list(multi_client_panel.columns)


def test_time_split_not_by_client_id(multi_client_panel):
    """
    Один и тот же client_id может быть и в train, и в test
    на разных month — это time-based split, не client holdout.
    """
    train, test = purged_time_split(
        multi_client_panel, split_month=5, embargo_months=1
    )
    assert set(train["month"]).isdisjoint(set(test["month"]))
    # при наличии месяцев по обе стороны split — клиенты могут пересекаться
    if len(train) and len(test):
        assert int(train["month"].max()) < 5
        assert int(test["month"].min()) >= 6