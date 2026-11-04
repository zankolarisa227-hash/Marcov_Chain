
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV



N_STATES = 3
DEFAULT_STATE = 2

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from src.features.build_features import FEATURE_COLS as DEFAULT_FEATURE_COLS
except ImportError: 
    try:
        from src.features.feature_contracts import FEATURE_COLS as DEFAULT_FEATURE_COLS
    except ImportError:
        DEFAULT_FEATURE_COLS = [
            "util_mean", "util_std", "util_max", "util_min", "util_range",
            "util_trend", "util_mean_last3", "util_mean_prev", "util_accel", "util_growth_ratio",
            "n_state_changes", "state_change_rate",
            "frac_current", "frac_delinquent", "frac_default",
            "max_consec_delinquent", "months_since_last_del",
            "state_entropy", "util_cv",
            "high_util_ratio", "ever_delinquent", "util_rising_3m", "util_vs_mean_dev",
            "utilization", "macro_stress", "util_x_macro",
        ]


class DynamicMarkovModel:
    #Для каждого исходного состояния i != default обучаем отдельный классификатор:  P(s_{t+1} | s_t = i, x_t)
    def __init__(
        self,
        feature_cols: Optional[list[str]] = None,
        use_catboost: bool = True,
        catboost_params: Optional[dict] = None,
        random_state: int = 42,
    ):
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(DEFAULT_FEATURE_COLS)
        self.use_catboost = bool(use_catboost and CATBOOST_AVAILABLE)
        self.random_state = random_state
        
        self.catboost_params = catboost_params or {
            "loss_function": "MultiClass",
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": random_state,
            "verbose": False,
            "allow_writing_files": False,
        }

        self.models_: dict[int, object] = {}
        self.scalers_: dict[int, Optional[StandardScaler]] = {}
        self.classes_: dict[int, np.ndarray] = {}
        self.backend_: str = "catboost" if self.use_catboost else "logreg"
        self.is_fitted_: bool = False

    def fit(self, pairs: pd.DataFrame) -> "DynamicMarkovModel":
        required = {"state", "next_state"}
        if not required.issubset(pairs.columns):
            raise ValueError(f"pairs must contain {required}")

        missing = [c for c in self.feature_cols if c not in pairs.columns]
        if missing:
            raise ValueError(f"Missing feature columns in pairs: {missing}")

        self.models_.clear()
        self.scalers_.clear()
        self.classes_.clear()

        for state in sorted(pairs["state"].unique()):
            state = int(state)
            if state == DEFAULT_STATE:
                continue

            subset = pairs[pairs["state"] == state]
            X = subset[self.feature_cols].to_numpy(dtype=float)
            y = subset["next_state"].to_numpy(dtype=int)

            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                # Один возможный переход — константная модель
                self.models_[state] = None
                self.scalers_[state] = None
                self.classes_[state] = unique_classes
                continue

            if self.use_catboost:
                clf = CatBoostClassifier(**self.catboost_params)
                clf.fit(X, y)
                scaler = None  # CatBoost не требует StandardScaler
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
              
              
                base = LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=self.random_state,
                )
              
              
                clf = CalibratedClassifierCV(
                    estimator=base,
                    method="isotonic",
                    cv=3,
                )
                clf.fit(X_scaled, y)
                self.model_ = clf


            self.models_[state] = clf
            self.scalers_[state] = scaler
            self.classes_[state] = (
                clf.classes_ if hasattr(clf, "classes_") else unique_classes
            )

        self.is_fitted_ = True
        return self

    def _predict_proba_raw(self, state: int, x: np.ndarray) -> np.ndarray:
        """Вероятности только по классам, которые видела модель state."""
        clf = self.models_[state]
        scaler = self.scalers_.get(state)

        x = np.asarray(x, dtype=float).reshape(1, -1)
        if x.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Expected {len(self.feature_cols)} features, got {x.shape[1]}"
            )

        if scaler is not None:
            x = scaler.transform(x)

        return clf.predict_proba(x)[0]

    def predict_proba_row(self, state: int, x: np.ndarray) -> np.ndarray:

        probs = np.zeros(N_STATES, dtype=float)
        state = int(state)

        if state == DEFAULT_STATE:
            probs[DEFAULT_STATE] = 1.0
            return probs

        clf = self.models_.get(state)
        classes = self.classes_.get(state)

        # Состояние не встречалось в train
        if classes is None and clf is None:
            probs[state] = 1.0  # остаёмся на месте как safe fallback
            return probs

        # Константный переход
        if clf is None:
            only = int(classes[0])
            probs[only] = 1.0
            return probs

        raw = self._predict_proba_raw(state, x)
        for cls, p in zip(classes, raw):
            probs[int(cls)] = float(p)

        # численная защита
        s = probs.sum()
        if s <= 0:
            probs[state] = 1.0
        else:
            probs = probs / s
        return probs

    def predict_proba(self, status_t: np.ndarray, X: np.ndarray) -> np.ndarray:
        status_t = np.asarray(status_t, dtype=int)
        X = np.asarray(X, dtype=float)
        if len(status_t) != len(X):
            raise ValueError("status_t and X must have same length")

        out = np.zeros((len(status_t), N_STATES), dtype=float)
        for i, (s, x) in enumerate(zip(status_t, X)):
            out[i] = self.predict_proba_row(int(s), x)
        return out

    def transition_matrix_at(self, x: np.ndarray) -> np.ndarray:
        P = np.zeros((N_STATES, N_STATES), dtype=float)
        for s in range(N_STATES):
            P[s] = self.predict_proba_row(s, x)
        return P

    def pd_term_structure(
        self,
        start_state: int,
        covariate_path: np.ndarray,
        horizons: list[int],
    ) -> dict[int, float]:
        
        if not horizons:
            return {}

        max_h = max(horizons)
        path = np.asarray(covariate_path, dtype=float)
        if path.ndim != 2:
            raise ValueError("covariate_path must be 2D: (T, n_features)")

        dist = np.zeros(N_STATES, dtype=float)
        dist[int(start_state)] = 1.0
        result: dict[int, float] = {}

        for t in range(1, max_h + 1):
            x_t = path[min(t - 1, len(path) - 1)]
            P_t = self.transition_matrix_at(x_t)
            dist = dist @ P_t
            if t in horizons:
                result[t] = float(dist[DEFAULT_STATE])
        return result

    def feature_importances(self) -> pd.DataFrame:
        rows = []
        for state, clf in self.models_.items():
            if clf is None:
                continue
            if hasattr(clf, "get_feature_importance"):
                imp = np.asarray(clf.get_feature_importance(), dtype=float)
            elif hasattr(clf, "feature_importances_"):
                imp = np.asarray(clf.feature_importances_, dtype=float)
            else:
                continue
            for name, val in zip(self.feature_cols, imp):
                rows.append({"state": state, "feature": name, "importance": float(val)})
        if not rows:
            return pd.DataFrame(columns=["state", "feature", "importance"])
        return pd.DataFrame(rows)

    def save(self, path: str = "models/dynamic_model.joblib") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = "models/dynamic_model.joblib") -> "DynamicMarkovModel":
        return joblib.load(path)