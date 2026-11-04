"""
validation.py — валидация моделей.

Содержит:
  - run_validation  — сравнение baseline/static/dynamic на один шаг вперёд
  - backtest        — multi-horizon бэктест (3, 6, 12 месяцев)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


# ======================================================================
# Вспомогательные функции
# ======================================================================

def split_oot(df: pd.DataFrame, split_month: int, time_col: str = "month"):
    """Out-of-time сплит."""
    train = df[df[time_col] < split_month].copy()
    test = df[df[time_col] >= split_month].copy() 
    print(f"OOT split | Train: {len(train):,} | Test: {len(test):,}")
    return train, test


def _safe_proba(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Клиппинг вероятностей, чтобы log_loss не падал."""
    return np.clip(y_pred, 1e-6, 1 - 1e-6)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Метрики для бинарного события (дефолт = 1).
    """
    y_pred = _safe_proba(y_true, y_pred)
    return {
        "brier": float(brier_score_loss(y_true, y_pred)),
        "logloss": float(log_loss(y_true, y_pred)),
        "mean_pd": float(np.mean(y_pred)),
        "actual_rate": float(np.mean(y_true)),
    }


def baseline_predict(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """
    Baseline: всегда предсказываем среднюю частоту дефолта из train.
    """
    base_rate = float(np.mean(y_train))
    return np.full(n_test, base_rate)


def static_predict(P: np.ndarray, status_t: np.ndarray, default_state: int = 2) -> np.ndarray:
    """
    Статическая модель: берём P[s_t, default] из матрицы переходов.
    """
    return np.array([P[int(s), default_state] for s in status_t], dtype=float)


def dynamic_predict(model, status_t: np.ndarray, X_test: np.ndarray, default_state: int = 2) -> np.ndarray:
    """
    Динамическая модель: P(default | s_t, x_t).
    Ожидает, что model.predict_proba возвращает (n_samples, n_states)
    или (n_samples,) если уже вероятность дефолта.
    """
    proba = model.predict_proba(status_t, X_test)

    if proba.ndim == 2:
        return proba[:, default_state]
    return proba.astype(float)


def compare_models(
    y_true: np.ndarray,
    status_t: np.ndarray,
    X_test: np.ndarray,
    P: np.ndarray,
    dynamic_model,
    y_train_for_baseline: np.ndarray,
    default_state: int = 2,
) -> pd.DataFrame:
    """
    Полное сравнение: baseline → static → dynamic.
    """
    # 1. Baseline
    baseline_pred = baseline_predict(y_train_for_baseline, len(y_true))

    # 2. Static Markov
    static_pred = static_predict(P, status_t, default_state)

    # 3. Dynamic
    dynamic_pred = dynamic_predict(dynamic_model, status_t, X_test, default_state)

    # Метрики
    results = {
        "baseline": evaluate_binary(y_true, baseline_pred),
        "static": evaluate_binary(y_true, static_pred),
        "dynamic": evaluate_binary(y_true, dynamic_pred),
    }

    table = pd.DataFrame(results).T
    table = table[["brier", "logloss", "mean_pd", "actual_rate"]]
    table = table.round(4)

    return table


def print_comparison(table: pd.DataFrame):
    """Красивый вывод в стиле архитектуры проекта."""
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (OOT)")
    print("=" * 60)
    print(table.to_string())
    print("-" * 60)

    brier_base = table.loc["baseline", "brier"]
    brier_static = table.loc["static", "brier"]
    brier_dynamic = table.loc["dynamic", "brier"]

    print(f"Brier Score:  {brier_base:.3f} → {brier_static:.3f} → {brier_dynamic:.3f}")
    print(f"Log-loss:     {table.loc['baseline','logloss']:.3f} → "
          f"{table.loc['static','logloss']:.3f} → {table.loc['dynamic','logloss']:.3f}")
    print(f"Mean PD:      {table.loc['baseline','mean_pd']:.3f} → "
          f"{table.loc['static','mean_pd']:.3f} → {table.loc['dynamic','mean_pd']:.3f}")
    print("=" * 60)


# ======================================================================
# Основные функции валидации
# ======================================================================

def run_validation(
    df: pd.DataFrame,
    P: np.ndarray,
    dynamic_model,
    split_month: int = 18,
    status_col: str = "status",
    next_status_col: str = "status_next",
    feature_cols: list = None,
    default_state: int = 2,
    time_col: str = "month",
):
    """
    Сравнение моделей на один шаг вперёд (t → t+1).
    """
    train, test = split_oot(df, split_month, time_col)

    y_train = (train[next_status_col] == default_state).astype(int).values
    y_test = (test[next_status_col] == default_state).astype(int).values
    status_t = test[status_col].values

    if feature_cols is None:
        exclude = {status_col, next_status_col, time_col, "client_id", "app_id"}
        feature_cols = [c for c in test.columns if c not in exclude and np.issubdtype(test[c].dtype, np.number)]

    X_test = test[feature_cols].values

    table = compare_models(
        y_true=y_test,
        status_t=status_t,
        X_test=X_test,
        P=P,
        dynamic_model=dynamic_model,
        y_train_for_baseline=y_train,
        default_state=default_state,
    )

    print_comparison(table)
    return table

def backtest(
    panel: pd.DataFrame,
    features_df: pd.DataFrame,
    P: np.ndarray,
    dynamic_model,
    split_month: int = 16,
    horizons: list = [3, 6, 12],
    default_state: int = 2,
    feature_cols: list = None,
) -> dict:
    """
    Полноценный out-of-time бэктест на нескольких горизонтах.
    Использует готовые модели и признаки (не переобучает их заново).
    """
    # Если feature_cols не задан — определяем автоматически (все колонки из features_df, кроме служебных)
    if feature_cols is None:
        exclude = {"client_id", "month", "state"}
        feature_cols = [c for c in features_df.columns if c not in exclude]

    # Объединяем признаки и панель, чтобы получить статусы
    panel_subset = panel[["client_id", "month", "state"]].copy()
    df_full = features_df.merge(panel_subset, on=["client_id", "month"], how="inner")
    df_full = df_full.sort_values(["client_id", "month"])

    # Разделение по времени
    train = df_full[df_full["month"] < split_month].copy()
    test = df_full[df_full["month"] >= split_month].copy()

    print(f"Train: {len(train):,} записей, Test: {len(test):,} записей")

    # Baseline: частота дефолтов на train
    y_train_baseline = (train["state"] == default_state).astype(int).values
    base_rate = np.mean(y_train_baseline)

    results = {}

    for h in horizons:
        print(f"\n--- Горизонт {h} месяцев ---")

        # Создаём целевую переменную: состояние через h месяцев
        test_h = test.copy()
        test_h["status_next"] = test_h.groupby("client_id")["state"].shift(-h)
        test_h = test_h.dropna(subset=["status_next"])
        test_h["status_next"] = test_h["status_next"].astype(int)

        if len(test_h) == 0:
            print(f"⚠️ Нет данных для горизонта {h}, пропускаем")
            continue

        y_true = (test_h["status_next"] == default_state).astype(int).values
        status_t = test_h["state"].values
        X_test = test_h[feature_cols].values

        # --- Предсказания ---
        # 1. Baseline
        baseline_pred = np.full(len(y_true), base_rate)

        # 2. Статическая модель: P^h
        P_h = np.linalg.matrix_power(P, h)
        static_pred = np.array([P_h[int(s), default_state] for s in status_t], dtype=float)

        # 3. Динамическая модель: итеративный прогноз (упрощённо)
        # Используем predict_proba для одного шага, затем аппроксимируем h шагов
        p_1step = dynamic_model.predict_proba(status_t, X_test)[:, default_state]
        dynamic_pred = 1 - (1 - p_1step) ** h

        # Метрики
        results[h] = {
            "baseline": evaluate_binary(y_true, baseline_pred),
            "static": evaluate_binary(y_true, static_pred),
            "dynamic": evaluate_binary(y_true, dynamic_pred),
        }

    # Вывод результатов
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (MULTI-HORIZON)")
    print("=" * 60)
    for h, res in results.items():
        print(f"\nГоризонт {h} месяцев:")
        table = pd.DataFrame(res).T[["brier", "logloss", "mean_pd", "actual_rate"]]
        print(table.round(4))

    return results


if __name__ == "__main__":
    print("Модуль validation.py готов.")
    print("Доступные функции: run_validation(), backtest()")
