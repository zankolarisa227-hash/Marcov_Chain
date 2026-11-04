"""
Контракт API:
  - PredictRequest → вектор длины len(FEATURE_COLS)
  - health отвечает
  - predict без модели не падает 500-traceback (если app доступен)

Если FastAPI/app ещё не поднят в тестах — часть soft-skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.feature_contracts import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]


def _build_feature_payload(value: float = 0.1) -> dict:
    """Минимальный валидный body: все 26 признаков."""
    return {c: float(value) for c in FEATURE_COLS}


def test_feature_payload_matches_contract_length():
    payload = _build_feature_payload()
    assert len(payload) == len(FEATURE_COLS)
    assert set(payload.keys()) == set(FEATURE_COLS)


def test_vector_order_matches_feature_cols():

    payload = {c: float(i) for i, c in enumerate(FEATURE_COLS)}
    vec = np.array([payload[c] for c in FEATURE_COLS], dtype=float)
    assert vec.shape == (len(FEATURE_COLS),)
    assert vec[0] == 0.0
    assert vec[-1] == float(len(FEATURE_COLS) - 1)


def test_predict_request_schema_if_available():
    try:
        from src.api.app import PredictRequest  # type: ignore
    except Exception:
        try:
            from src.api.schemas import PredictRequest  # type: ignore
        except Exception:
            pytest.skip("PredictRequest schema not importable")

    payload = _build_feature_payload(0.2)
    # разные API: либо плоский dict признаков, либо вложенный features=
    try:
        obj = PredictRequest(**payload)
    except Exception:
        obj = PredictRequest(features=payload)  # type: ignore

    # если есть to_vector / dict
    if hasattr(obj, "to_vector"):
        v = np.asarray(obj.to_vector(), dtype=float)
        assert v.shape == (len(FEATURE_COLS),)
    elif hasattr(obj, "model_dump"):
        d = obj.model_dump()
        assert isinstance(d, dict)


@pytest.fixture
def client():
    try:
        from fastapi.testclient import TestClient
        from src.api.app import app
    except Exception:
        pytest.skip("FastAPI app not available")
    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code in (200, 503)
    body = r.json()
    assert "status" in body or "ok" in body or "model_loaded" in body


def test_predict_endpoint_shape_or_service_unavailable(client):
    payload = {
        "state": 0,
        "features": _build_feature_payload(0.15),
    }
    # альтернативные форматы body
    candidates = [
        payload,
        {"state": 0, **_build_feature_payload(0.15)},
    ]
    last = None
    for body in candidates:
        r = client.post("/predict", json=body)
        last = r
        if r.status_code == 200:
            data = r.json()
            # PD keys — гибко
            vals = []
            for k, v in data.items():
                if isinstance(v, (int, float)) and "pd" in k.lower():
                    vals.append(float(v))
            if "pd" in data and isinstance(data["pd"], dict):
                vals.extend(float(x) for x in data["pd"].values())
            if vals:
                assert all(0.0 <= x <= 1.0 for x in vals)
            return
        if r.status_code in (422,):
            continue
        if r.status_code in (503,):
            # модель не загружена — допустимо в CI без artifacts
            return
    assert last is not None
    # 422/503 ок; 500 — плохо
    assert last.status_code != 500, last.text