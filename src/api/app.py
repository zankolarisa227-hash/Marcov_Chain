"""
src/api/main.py — FastAPI-обёртка над dynamic Markov PD/ECL.

Эндпоинты:
    GET  /health
    GET  /ready
    GET  /model/info
    POST /predict
    POST /matrix

Запуск из корня проекта:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

NOTE:
    API тонкий: валидация → score → JSON.
    Математика PD/ECL живёт в risk/scoring-модулях, не здесь.
    Demo-only: CORS *, без auth.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Paths / logging
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]  # .../project_root
MODEL_PATH = ROOT / "models" / "dynamic_model.joblib"
META_PATH = ROOT / "models" / "meta.joblib"  # optional: feature medians, version

logger = logging.getLogger("credit_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Feature contract (единый порядок с train)
# При наличии src.data.contracts — импортируйте оттуда.
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    # A scale
    "util_mean",
    "util_std",
    "util_max",
    "util_min",
    "util_range",
    # B dynamics
    "util_trend",
    "util_mean_last3",
    "util_mean_prev",
    "util_accel",
    "util_growth_ratio",
    # C state history
    "n_state_changes",
    "state_change_rate",
    "frac_current",
    "frac_delinquent",
    "frac_default",
    "max_consec_delinquent",
    "months_since_last_del",
    # D irregularity
    "state_entropy",
    "util_cv",
    # E flags
    "high_util_ratio",
    "ever_delinquent",
    "util_rising_3m",
    "util_vs_mean_dev",
    # F contemporaneous
    "utilization",
    "macro_stress",
    "util_x_macro",
]

STATE_NAMES = ["current", "delinquent", "default"]


# ---------------------------------------------------------------------------
# Optional project imports (подставьте свои пути при необходимости)
# ---------------------------------------------------------------------------

try:
    from src.risk.risk_metrics import ecl, optimal_limit, pd_term_structure
except ImportError:
    try:
        from risk_metrics import ecl, optimal_limit, pd_term_structure  # type: ignore
    except ImportError:
        # fallback-заглушки, чтобы файл читался даже без risk-модуля
        def pd_term_structure(model, start_state, covariate_path, horizons):
            raise RuntimeError("pd_term_structure not available — подключите risk_metrics")

        def ecl(pd_h, ead=50_000.0, lgd=0.45):
            return float(pd_h) * float(ead) * float(lgd)

        def optimal_limit(pd_12m, lgd=0.45):
            return {"action": "review", "pd_12m": float(pd_12m), "lgd": float(lgd)}


# App
app = FastAPI(
    title="Dynamic Markov Credit Engine",
    description=(
        "Covariate-dependent Markov migration → PD term-structure, ECL, limit hint. "
        "Research/demo API, not production banking stack."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_meta: Dict[str, Any] = {}


def _load_meta() -> Dict[str, Any]:
    if META_PATH.exists():
        meta = joblib.load(META_PATH)
        return meta if isinstance(meta, dict) else {"raw": meta}
    return {
        "model_version": "unknown",
        "feature_cols": FEATURE_COLS,
        "reference_profile": {c: 0.0 for c in FEATURE_COLS},
    }


def get_model():
    """Singleton: модель в память один раз."""
    global _model, _meta
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. Train pipeline first."
            )
        _model = joblib.load(MODEL_PATH)
        _meta = _load_meta()
        logger.info("Loaded model=%s meta_keys=%s", MODEL_PATH, list(_meta.keys()))
    return _model


def get_meta() -> Dict[str, Any]:
    get_model()
    return _meta


def reference_feature_vector() -> np.ndarray:
    """
    Профиль «типичного» клиента для /matrix.
    Берём medians/reference из meta, иначе нули (хуже, но безопасно падает).
    """
    meta = get_meta()
    ref = meta.get("reference_profile") or meta.get("feature_medians") or {}
    x = np.zeros(len(FEATURE_COLS), dtype=float)
    for i, col in enumerate(FEATURE_COLS):
        if col in ref:
            x[i] = float(ref[col])
    return x


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Полный PIT-вектор + business fields."""

    current_state: int = Field(..., ge=0, le=2, description="0=current, 1=delinq, 2=default")

    # A
    util_mean: float = Field(..., ge=0, le=1)
    util_std: float = Field(..., ge=0)
    util_max: float = Field(..., ge=0, le=1)
    util_min: float = Field(..., ge=0, le=1)
    util_range: float = Field(..., ge=0)

    # B
    util_trend: float
    util_mean_last3: float = Field(..., ge=0, le=1)
    util_mean_prev: float = Field(..., ge=0, le=1)
    util_accel: float
    util_growth_ratio: float = Field(..., ge=0)

    # C
    n_state_changes: int = Field(..., ge=0)
    state_change_rate: float = Field(..., ge=0)
    frac_current: float = Field(..., ge=0, le=1)
    frac_delinquent: float = Field(..., ge=0, le=1)
    frac_default: float = Field(..., ge=0, le=1)
    max_consec_delinquent: int = Field(..., ge=0)
    months_since_last_del: int = Field(..., ge=0)

    # D
    state_entropy: float = Field(..., ge=0)
    util_cv: float = Field(..., ge=0)

    # E
    high_util_ratio: float = Field(..., ge=0, le=1)
    ever_delinquent: int = Field(..., ge=0, le=1)
    util_rising_3m: int = Field(..., ge=0, le=1)
    util_vs_mean_dev: float

    # F
    utilization: float = Field(..., ge=0, le=1)
    macro_stress: float = Field(..., ge=-3.0, le=3.0)
    util_x_macro: float

    # business
    ead: float = Field(50_000.0, gt=0)
    lgd: float = Field(0.45, ge=0, le=1)
    horizons: List[int] = Field(default_factory=lambda: [1, 3, 6, 12])
    scenario: str = Field(default="frozen", description="frozen path of x on horizon")

    def to_feature_vector(self) -> np.ndarray:
        missing = [c for c in FEATURE_COLS if not hasattr(self, c)]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return np.array([float(getattr(self, c)) for c in FEATURE_COLS], dtype=float)


class PredictResponse(BaseModel):
    current_state: int
    scenario: str
    model_version: str
    pd_term_structure: Dict[str, float]
    ecl_12m: float
    limit_recommendation: Dict[str, Any]


class MatrixRequest(BaseModel):
    """
    Либо полный набор фичей (как PredictRequest без business),
    либо частично — недостающее заполняется reference_profile из meta.
    """

    current_state: int = Field(0, ge=0, le=2)
    utilization: Optional[float] = Field(None, ge=0, le=1)
    macro_stress: Optional[float] = Field(None, ge=-3.0, le=3.0)
    util_mean: Optional[float] = None
    util_trend: Optional[float] = None
    ever_delinquent: Optional[int] = Field(None, ge=0, le=1)

    # опционально полный override
    features: Optional[Dict[str, float]] = None

    def to_feature_vector(self) -> np.ndarray:
        x = reference_feature_vector()
        overrides: Dict[str, float] = {}
        if self.features:
            overrides.update({k: float(v) for k, v in self.features.items()})
        for key in (
            "utilization",
            "macro_stress",
            "util_mean",
            "util_trend",
            "ever_delinquent",
        ):
            val = getattr(self, key)
            if val is not None:
                overrides[key] = float(val)
        if "utilization" in overrides and "macro_stress" in overrides:
            overrides.setdefault(
                "util_x_macro",
                overrides["utilization"] * overrides["macro_stress"],
            )
        for col, val in overrides.items():
            if col in FEATURE_COLS:
                x[FEATURE_COLS.index(col)] = val
        return x


# ---------------------------------------------------------------------------
# Scoring helper (тонкий; при рефакторинге вынести в src/scoring/score_one.py)
# ---------------------------------------------------------------------------

def score_one(
    model: Any,
    state: int,
    x: np.ndarray,
    horizons: List[int],
    ead: float,
    lgd: float,
    scenario: str = "frozen",
) -> Dict[str, Any]:
    """state + x → PD(h), ECL, limit."""
    if state == 2:
        # absorbing default
        pd_ts = {h: 1.0 for h in horizons}
        ecl_12 = ecl(1.0, ead=ead, lgd=lgd)
        limit_rec = optimal_limit(pd_12m=1.0, lgd=lgd)
        return {
            "pd_term_structure": pd_ts,
            "ecl_12m": ecl_12,
            "limit_recommendation": limit_rec,
            "scenario": scenario,
        }

    h_max = max(horizons)
    # frozen path: x_t постоянен на горизонте
    cov_path = np.tile(x.reshape(1, -1), (h_max, 1))

    pd_ts = pd_term_structure(
        model,
        start_state=state,
        covariate_path=cov_path,
        horizons=horizons,
    )
    # pd_ts: dict[int, float] ожидается
    pd_12 = float(pd_ts.get(12, pd_ts[max(pd_ts.keys())]))
    ecl_12 = ecl(pd_12, ead=ead, lgd=lgd)
    limit_rec = optimal_limit(pd_12m=pd_12, lgd=lgd)

    return {
        "pd_term_structure": {int(k): float(v) for k, v in pd_ts.items()},
        "ecl_12m": float(ecl_12),
        "limit_recommendation": limit_rec,
        "scenario": scenario,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health():
    """Liveness: процесс отвечает."""
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
def ready():
    """Readiness: модель на диске и загружается."""
    try:
        get_model()
        return {
            "status": "ready",
            "model_path": str(MODEL_PATH),
            "model_version": str(get_meta().get("model_version", "unknown")),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("ready failed")
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")


@app.get("/model/info", tags=["system"])
def model_info():
    try:
        meta = get_meta()
        return {
            "model_path": str(MODEL_PATH),
            "model_version": meta.get("model_version", "unknown"),
            "n_features": len(FEATURE_COLS),
            "feature_cols": FEATURE_COLS,
            "states": STATE_NAMES,
            "default_horizons": [1, 3, 6, 12],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(req: PredictRequest):
    try:
        model = get_model()
        meta = get_meta()
        x = req.to_feature_vector()

        out = score_one(
            model=model,
            state=req.current_state,
            x=x,
            horizons=req.horizons,
            ead=req.ead,
            lgd=req.lgd,
            scenario=req.scenario,
        )

        pd_fmt = {f"{h}m": round(float(v), 6) for h, v in out["pd_term_structure"].items()}

        return PredictResponse(
            current_state=req.current_state,
            scenario=out["scenario"],
            model_version=str(meta.get("model_version", "unknown")),
            pd_term_structure=pd_fmt,
            ecl_12m=round(float(out["ecl_12m"]), 2),
            limit_recommendation=out["limit_recommendation"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.post("/matrix", tags=["scoring"])
def transition_matrix(req: MatrixRequest):
    """
    P(x) в точке x.
    Недостающие фичи ← reference_profile из meta (не нули «с потолка»).
    """
    try:
        model = get_model()
        x = req.to_feature_vector()

        if not hasattr(model, "transition_matrix_at"):
            raise HTTPException(
                status_code=501,
                detail="Model has no transition_matrix_at(x)",
            )

        P = np.asarray(model.transition_matrix_at(x), dtype=float)
        if P.shape != (3, 3):
            raise HTTPException(
                status_code=500,
                detail=f"Expected P shape (3,3), got {P.shape}",
            )

        # absorbing enforce on display
        P = P.copy()
        P[2, :] = 0.0
        P[2, 2] = 1.0

        return {
            "model_version": str(get_meta().get("model_version", "unknown")),
            "current_state": req.current_state,
            "note": "Missing features filled from meta reference_profile",
            "matrix": {
                STATE_NAMES[i]: {
                    STATE_NAMES[j]: round(float(P[i, j]), 6) for j in range(3)
                }
                for i in range(3)
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("matrix failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")