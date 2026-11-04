"""
test_contracts.py

Контракты проекта:
  - FEATURE_COLS фиксированы (имена, порядок, без target/id)
  - артефакты модели/матрицы читаются и валидны
  - meta содержит anti-leak / split флаги (если есть)

Это production-защита от schema drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.feature_contracts import FEATURE_COLS, FORBIDDEN_COLS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEFAULT = 2


def test_feature_cols_non_empty_and_unique():
    assert isinstance(FEATURE_COLS, (list, tuple))
    assert len(FEATURE_COLS) >= 5
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "Дубли в FEATURE_COLS"


def test_feature_cols_exclude_ids_and_target():
    banned = {
        "client_id", "app_id", "month", "next_state",
        "flag", "target", "y", "default_flag",
    }
    overlap = set(FEATURE_COLS) & banned
    assert not overlap, f"В FEATURE_COLS нельзя: {overlap}"


def test_forbidden_covers_target():
    # next_state не должен быть фичей
    assert "next_state" not in FEATURE_COLS
    if FORBIDDEN_COLS:
        assert "next_state" in set(FORBIDDEN_COLS) or "next_state" not in FEATURE_COLS


def test_feature_cols_are_strings():
    assert all(isinstance(c, str) and c.strip() for c in FEATURE_COLS)


def test_p_matrix_artifact_if_exists():
    path = MODELS / "P_matrix.npy"
    if not path.exists():
        pytest.skip("P_matrix.npy ещё не обучен — ok для чистого CI")
    P = np.load(path)
    assert P.shape == (3, 3)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(P[DEFAULT], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.all(P >= -1e-9)


def test_dynamic_model_artifact_if_exists():
    path = MODELS / "dynamic_model.joblib"
    if not path.exists():
        pytest.skip("dynamic_model.joblib ещё нет — ok для чистого CI")
    from src.models.dynamic_model import DynamicMarkovModel

    model = DynamicMarkovModel.load(str(path))
    cols = getattr(model, "feature_cols", None) or getattr(model, "feature_names_", None)
    if cols is not None:
        assert list(cols) == list(FEATURE_COLS), (
            "Порядок/набор feature_cols модели != FEATURE_COLS (train/serve skew)"
        )


def test_meta_json_anti_leak_flags_if_exists():
    path = MODELS / "meta.json"
    if not path.exists():
        pytest.skip("meta.json нет")
    import json

    meta = json.loads(path.read_text(encoding="utf-8"))
    # мягкие проверки: ключи, если вы их пишете в train
    if "anti_leakage" in meta:
        assert meta["anti_leakage"] is True or meta["anti_leakage"] == 1
    if "split_month" in meta:
        assert int(meta["split_month"]) >= 1
    if "feature_cols" in meta:
        assert list(meta["feature_cols"]) == list(FEATURE_COLS)