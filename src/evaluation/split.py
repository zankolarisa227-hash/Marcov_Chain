from __future__ import annotations 

import pandas as pd 

def purged_time_split(
        df: pd.DataFrame, 
        split_month: int, 
        embargo_months: int=1, 
        time_col: str = "month",
) ->  tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train: month < split_month
    Embargo: [split_month, split_month + embargo)
    Test: month >= split_month + embargo

    Это уменьшает leakage около границы split.
    """
    if embargo_months < 0:
        raise ValueError("embargo_months must be >= 0")

    train = df[df[time_col] < split_month].copy()
    test = df[df[time_col] >= (split_month + embargo_months)].copy()

    print(
        f"Purged OOT | train month<{split_month}: {len(train):,} | "
        f"embargo={embargo_months} | "
        f"test month>={split_month + embargo_months}: {len(test):,}"
    )
    return train, test