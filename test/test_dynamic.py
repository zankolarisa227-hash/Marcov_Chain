from __future__ import annotations


import numpy as np
import pandas as pd 
import pytest 


from src.features.feature_contracts import FEATURE_COLS
from src.models.dynamic_model import DynamicMarkovModel

DEFAULT = 2

@pytest.fixture
def small_model() -> DynamicMarkovModel:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(100):
        u = float(rng.uniform(0, 1))
        r = {c: 0.0 for c in FEATURE_COLS}
        r["utilization"] = u
        r["util_mean"] = u
        r["state"] = 0
        r["next_state"] = DEFAULT if u > 0.55 else 0
        rows.append(r)
    return DynamicMarkovModel(feature_cols=list(FEATURE_COLS)).fit(pd.DataFrame(rows))


#ероятности перехода во все возможные состояния должны в сумме давать 1.
def test_proba_sum_to_one(small_model):
    x=np.zeros(len(FEATURE_COLS))
    p=small_model.predict_proba_row(0,x)
    assert np.allclose(p.sum(), 1.0, atol=1e-5)
    assert np.all(p>=-1e-9)


#вероятности перехода во все возможные состояния должны в сумме быть 1.
def test_default_origin_locked(small_model):
    x=np.allclose(small_model.predict_proba_row(DEFAULT,x), [0,0,1])

#после загрузки те же предсказания, что и до сохранения
def save_load_same_proba(small_model, tmp_path):
    x=np.zeros(len(FEATURE_COLS))
    x[FEATURE_COLS.index("utilization")] =0.4
    path = tmp_path / "m.joblib"
    small_model.save(str(path))
    loaded = DynamicMarkovModel.load(str(path))
    assert np.allclose(
        small_model.predict_proba_row(0, x),
        loaded.predict_proba_row(0, x),
        atol=1e-8,
    )

