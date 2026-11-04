
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

N_STATES = 3
STATE_NAMES = {0: "current", 1: "delinquent", 2: "default"}



# Оценка матрицы
def count_transitions(
    pairs: pd.DataFrame,
    n_states: int = N_STATES,
) -> np.ndarray:
    """
    Матрица счётов N[i, j] = сколько раз из i перешли в j.
    Дефолт (последняя строка) принудительно [0, 0, N_ii] — поглощение.
    """


    N = np.zeros((n_states, n_states), dtype=float)

    for i in range(n_states - 1):
        mask = pairs["state"] == i
        if mask.sum() == 0:
            continue
        counts = (
            pairs.loc[mask, "next_state"]
            .value_counts()
            .reindex(range(n_states), fill_value=0)
        )
        N[i] = counts.values.astype(float)

    # поглощающее состояние: все наблюдения "из default" остаются в default
    n_def = int((pairs["state"] == n_states - 1).sum())
    N[n_states - 1, n_states - 1] = max(n_def, 1.0)

    return N


def estimate_transition_matrix(
    pairs: pd.DataFrame,
    n_states: int = N_STATES,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    MLE (или additively-smoothed) оценка P.

    smoothing > 0  →  Laplace:  P_ij = (N_ij + α) / (N_i• + α·S)
    Полезно при редких переходах (мало delinquent→current и т.п.).

    Строка default всегда [0, 0, 1].
    """
    
    N = count_transitions(pairs, n_states)
    P = np.zeros_like(N)

    for i in range(n_states - 1):
        row_sum = N[i].sum()
        if row_sum == 0:
            P[i, i] = 1.0
            continue
        P[i] = (N[i] + smoothing) / (row_sum + smoothing * n_states)

    P[n_states - 1, :] = 0.0
    P[n_states - 1, n_states - 1] = 1.0

    _validate_stochastic(P)
    return P


def _validate_stochastic(P: np.ndarray, tol: float = 1e-8) -> None:
    """Строки должны суммироваться в 1 и быть ≥ 0."""
    if np.any(P < -tol):
        raise ValueError("В матрице переходов есть отрицательные элементы")
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError(f"Строки P не суммируются в 1: {row_sums}")


def print_matrix(P: np.ndarray, name: str = "P") -> None:
    header = f"{name:>12}" + "".join(
        f"{'→ ' + STATE_NAMES[j]:>14}" for j in range(N_STATES)
    )

    print(header)
    for i in range(N_STATES):
        row_label = f"{STATE_NAMES[i]:>12}"
        row_vals = "".join(f"{P[i, j]:>14.4f}" for j in range(N_STATES))
        print(row_label + row_vals)


#2 
def bootstrap_transition_ci(
    pairs: pd.DataFrame,
    n_boot: int = 500,
    alpha: float = 0.05,
    n_states: int = N_STATES,
    smoothing: float = 0.0,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap (percentile) доверительные интервалы для каждого P_ij.
    """
    rng = np.random.default_rng(random_state)
    n = len(pairs)
    point = estimate_transition_matrix(pairs, n_states, smoothing)

    boot = np.zeros((n_boot, n_states, n_states))
    idx = np.arange(n)

    for b in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        sample = pairs.iloc[sample_idx]
        boot[b] = estimate_transition_matrix(sample, n_states, smoothing)

    lower = np.quantile(boot, alpha / 2, axis=0)
    upper = np.quantile(boot, 1 - alpha / 2, axis=0)
    mean = boot.mean(axis=0)

    return {
        "point": point,
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "n_boot": n_boot,
        "alpha": alpha,
    }


def print_matrix_with_ci(ci: dict, name: str = "P") -> None:
    # матрица с доверительными интервалами
    P = ci["point"]
    lo = ci["lower"]
    hi = ci["upper"]

    print(f"\n{name} (point estimate + {int((1-ci['alpha'])*100)}% CI)")
    header = f"{'':>12}" + "".join(f"{'→ ' + STATE_NAMES[j]:>22}" for j in range(N_STATES))
    print(header)

    for i in range(N_STATES):
        row = f"{STATE_NAMES[i]:>12}"
        for j in range(N_STATES):
            cell = f"{P[i,j]:.3f} [{lo[i,j]:.3f}; {hi[i,j]:.3f}]"
            row += f"{cell:>22}"
        print(row)

def _mc_lr_pvalue(
    p: pd.DataFrame,
    lr_obs: float,
    n_states: int,
    n_sim: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Monte-Carlo p-value для LR-теста порядка 1 vs 2.
    H0: s_next ⊥ s_prev | state
    → внутри каждого state переставляем s_next.
    """
    rng = np.random.default_rng(random_state)
    base = p[["s_prev", "state", "s_next"]].to_numpy().copy()
    state_col = base[:, 1].astype(int)
    groups = {s: np.where(state_col == s)[0] for s in np.unique(state_col)}

    def _lr(arr: np.ndarray) -> float:
        ll1 = 0.0
        ll2 = 0.0
        for s in np.unique(arr[:, 1]):
            sub = arr[arr[:, 1] == s, 2]
            if sub.size == 0:
                continue
            _, counts = np.unique(sub, return_counts=True)
            n = counts.sum()
            ll1 += np.sum(counts * np.log(counts / n))
        keys = arr[:, 0] * n_states + arr[:, 1]
        for k in np.unique(keys):
            sub = arr[keys == k, 2]
            if sub.size == 0:
                continue
            _, counts = np.unique(sub, return_counts=True)
            n = counts.sum()
            ll2 += np.sum(counts * np.log(counts / n))
        return 2.0 * (ll2 - ll1)

    exceed = 0
    for _ in range(n_sim):
        arr = base.copy()
        for idx in groups.values():
            if len(idx) > 1:
                arr[idx, 2] = rng.permutation(arr[idx, 2])
        if _lr(arr) >= lr_obs - 1e-12:
            exceed += 1
    return (exceed + 1) / (n_sim + 1)

# ---------------------------------------------------------------------------
# 2. Тест марковости 1-го порядка (LR)
# ---------------------------------------------------------------------------
def markov_order_test(
    panel: pd.DataFrame,
    n_states: int = N_STATES,
    min_cell: int = 5,
    sparse_frac_warn: float = 0.25,
    n_mc: int = 1000,
    force_mc: bool = False,
    random_state: int = 42,
) -> dict:
    """
    H0: P(s_{t+1} | s_t, s_{t-1}) = P(s_{t+1} | s_t)

    lr_stat считается как раньше.
    p-value:
      - при достаточных n  → asymptotic χ²
      - при разреженных таблицах (или force_mc=True) → Monte-Carlo
    """
    p = panel.sort_values(["client_id", "month"]).copy()
    p["s_prev"] = p.groupby("client_id")["state"].shift(1)
    p["s_next"] = p.groupby("client_id")["state"].shift(-1)
    p = p.dropna(subset=["s_prev", "s_next"])
    p = p[p["state"] != n_states - 1]
    p[["s_prev", "s_next", "state"]] = p[["s_prev", "s_next", "state"]].astype(int)

    n_obs = len(p)
    if n_obs == 0:
        return {
            "test": "Markov order-1 (LR-test)",
            "error": "нет наблюдений после фильтрации",
            "p_value": None,
            "conclusion": "тест не выполнен",
        }

    # --- диагностика разреженности ---
    g2_sizes = p.groupby(["s_prev", "state"]).size()
    n_groups2 = len(g2_sizes)
    n_small2 = int((g2_sizes < min_cell).sum())
    sparse_fraction = n_small2 / n_groups2 if n_groups2 else 1.0
    min_group2 = int(g2_sizes.min()) if n_groups2 else 0
    sparse = (sparse_fraction >= sparse_frac_warn) or (min_group2 < min_cell)

    def _ll_and_params(groupby_cols: list[str]) -> tuple[float, int]:
        ll = 0.0
        n_params = 0
        for _, sub in p.groupby(groupby_cols):
            counts = sub["s_next"].value_counts()
            n = int(counts.sum())
            if n == 0:
                continue
            for c in counts.values:
                if c > 0:
                    ll += c * np.log(c / n)
            k = int((counts > 0).sum())
            n_params += max(k - 1, 0)
        return ll, n_params

    # --- LR как раньше (не меняем) ---
    ll1, params1 = _ll_and_params(["state"])
    ll2, params2 = _ll_and_params(["s_prev", "state"])
    lr_stat = 2.0 * (ll2 - ll1)
    df = max(params2 - params1, 1)
    p_asymp = float(1 - stats.chi2.cdf(lr_stat, df))

    # --- p-value: χ² или Monte-Carlo ---
    use_mc = force_mc or sparse
    p_mc = None
    if use_mc:
        p_mc = _mc_lr_pvalue(
            p, lr_obs=lr_stat, n_states=n_states,
            n_sim=n_mc, random_state=random_state,
        )
    p_value = p_mc if use_mc else p_asymp
    method = "monte_carlo" if use_mc else "chi2_asymptotic"

    warnings_list = []
    if sparse:
        warnings_list.append(
            f"разреженная таблица: {n_small2}/{n_groups2} групп (s_prev,state) "
            f"с n < {min_cell} (доля {sparse_fraction:.0%}), min n = {min_group2}. "
            f"Asymptotic χ² ненадёжен → использован Monte-Carlo."
        )
    if n_obs < 200:
        warnings_list.append(f"мало наблюдений: n_obs = {n_obs}")

    return {
        "test": "Markov order-1 (LR-test)",
        "method": method,
        "lr_statistic": round(lr_stat, 3),
        "df": df,
        "p_value": round(float(p_value), 6),
        "p_value_asymptotic": round(p_asymp, 6),
        "p_value_mc": round(p_mc, 6) if p_mc is not None else None,
        "n_obs": n_obs,
        "sparse_diagnostics": {
            "n_groups_order2": n_groups2,
            "n_small_groups": n_small2,
            "sparse_fraction": round(sparse_fraction, 3),
            "min_group_size": min_group2,
            "min_cell": min_cell,
        },
        "warnings": warnings_list,
        "conclusion": (
            "H0 отвергнута: процесс показывает память глубже 1 шага"
            if p_value < 0.05
            else "H0 не отвергнута: допущение Маркова первого порядка разумно"
        ),
    }

# 3. Тест однородности во времени

def homogeneity_test(
    pairs: pd.DataFrame,
    split_month: int,
    n_states: int = N_STATES,
) -> dict:

    early = pairs[pairs["month"] < split_month]
    late = pairs[pairs["month"] >= split_month]

    total_stat = 0.0
    total_df = 0

    for s in range(n_states - 1):
        e_counts = (
            early.loc[early["state"] == s, "next_state"]
            .value_counts()
            .reindex(range(n_states), fill_value=0)
            .values.astype(float)
        )
        l_counts = (
            late.loc[late["state"] == s, "next_state"]
            .value_counts()
            .reindex(range(n_states), fill_value=0)
            .values.astype(float)
        )

        table = np.vstack([e_counts, l_counts])
        # убираем столбцы, где оба периода нули
        col_mask = table.sum(axis=0) > 0
        table = table[:, col_mask]

        if table.shape[1] < 2 or table.sum() == 0:
            continue
        # chi2_contingency требует ненулевые маргиналы строк
        if (table.sum(axis=1) == 0).any():
            continue

        chi2, _, dof, _ = stats.chi2_contingency(table)
        total_stat += chi2
        total_df += dof

    if total_df <= 0:
        p_value = float("nan")
        conclusion = "Недостаточно данных для теста однородности"
    else:
        p_value = float(1 - stats.chi2.cdf(total_stat, total_df))
        conclusion = (
            "H0 отвергнута: матрица НЕ однородна, нужна динамическая модель"
            if p_value < 0.05
            else "H0 не отвергнута: матрица стабильна во времени"
        )

    return {
        "test": "Time-homogeneity (chi-square)",
        "split_month": split_month,
        "chi2_statistic": round(total_stat, 3),
        "df": total_df,
        "p_value": round(p_value, 6) if p_value == p_value else None,
        "conclusion": conclusion,
    }

def homogeneity_test_mc(
    pairs: pd.DataFrame,
    split_month: int,
    n_states: int = N_STATES,
    n_sim: int = 1000,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)

    early = pairs[pairs["month"] < split_month]
    late = pairs[pairs["month"] >= split_month]

    def _chi2_stat(e_df, l_df) -> float:
        total = 0.0
        for s in range(n_states - 1):
            e_counts = (
                e_df.loc[e_df["state"] == s, "next_state"]
                .value_counts()
                .reindex(range(n_states), fill_value=0)
                .values.astype(float)
            )
            l_counts = (
                l_df.loc[l_df["state"] == s, "next_state"]
                .value_counts()
                .reindex(range(n_states), fill_value=0)
                .values.astype(float)
            )
            table = np.vstack([e_counts, l_counts])
            col_mask = table.sum(axis=0) > 0
            table = table[:, col_mask]
            if table.shape[1] < 2 or table.sum() == 0:
                continue
            if (table.sum(axis=1) == 0).any():
                continue
            chi2, _, _, _ = stats.chi2_contingency(table)
            total += chi2
        return total

    # наблюдаемая статистика
    obs_stat = _chi2_stat(early, late)

    base = pairs[["state", "next_state", "month"]].to_numpy().copy()
    state_col = base[:, 0].astype(int)
    groups = {s: np.where(state_col == s)[0] for s in range(n_states - 1)}

    exceed = 0
    for _ in range(n_sim):
        arr = base.copy()
        for idx in groups.values():
            if len(idx) > 1:
                # перемешиваем month (тем самым early/late)
                arr[idx, 2] = rng.permutation(arr[idx, 2])
        sim_early = arr[arr[:, 2] < split_month]
        sim_late = arr[arr[:, 2] >= split_month]
        # превращаем обратно в DataFrame-подобный вид для _chi2_stat
        e_df = pd.DataFrame(sim_early, columns=["state", "next_state", "month"])
        l_df = pd.DataFrame(sim_late, columns=["state", "next_state", "month"])
        if _chi2_stat(e_df, l_df) >= obs_stat - 1e-12:
            exceed += 1

    p_value = (exceed + 1) / (n_sim + 1)

    return {
        "test": "Time-homogeneity (Monte-Carlo)",
        "split_month": split_month,
        "chi2_statistic": round(obs_stat, 3),
        "p_value": round(p_value, 6),
        "n_sim": n_sim,
        "conclusion": (
            "H0 отвергнута: матрица НЕ однородна → нужна динамическая модель"
            if p_value < 0.05
            else "H0 не отвергнута: матрица стабильна во времени"
        ),
    }
# 4. Проекция и характеристики absorbing-цепи

def project_matrix(P: np.ndarray, months: int) -> np.ndarray:
    """P^n — вероятности переходов за `months` шагов."""
    _validate_stochastic(P)
    if months < 0:
        raise ValueError("months должен быть ≥ 0")
    if months == 0:
        return np.eye(P.shape[0])
    return np.linalg.matrix_power(P, months)


def absorption_probabilities(P: np.ndarray) -> np.ndarray:
    """
    Для цепи с одним поглощающим состоянием (последнее):
    вероятность когда-либо попасть в default из каждогоtransient-состояния.

    Возвращает вектор длины n_states; для default = 1.
    """
    n = P.shape[0]
    # Q — переходы между transient (все кроме последнего)
    Q = P[: n - 1, : n - 1]
    R = P[: n - 1, n - 1]  # в поглощающее
    I = np.eye(n - 1)
    try:
        N_fund = np.linalg.inv(I - Q)  # фундаментальная матрица
    except np.linalg.LinAlgError:
        # если (I-Q) вырождена — численный fallback
        N_fund = np.linalg.pinv(I - Q)
    B = N_fund @ R  # вероятности поглощения
    out = np.ones(n)
    out[: n - 1] = B
    return out


def mean_time_to_absorption(P: np.ndarray) -> np.ndarray:
    
    """
    Среднее число шагов до default из каждого transient-состояния.
    Для default = 0.
    """
    n = P.shape[0]
    Q = P[: n - 1, : n - 1]
    I = np.eye(n - 1)
    try:
        N_fund = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N_fund = np.linalg.pinv(I - Q)
    t = N_fund @ np.ones(n - 1)
    out = np.zeros(n)
    out[: n - 1] = t
    return out


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Стационарное π: π P = π, sum π = 1.
    Для absorbing-цепи с default — вырожденно π = [0, 0, 1].
    Имеет смысл, если убрать поглощение или смотреть на подцепь current/delinquent.
    """
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1] = 1.0  # условие нормировки
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 0, None)
    pi /= pi.sum()
    return pi



def run_step3_report(
    pairs: pd.DataFrame,
    panel: pd.DataFrame,
    split_month: int,
    horizon_months: int = 12,
    n_boot: int = 500,
    n_mc: int = 1000,
) -> dict:

    #матрица, тесты, годовая проекция риска, время до default.

    pairs_train = pairs[pairs["month"]< split_month]
    P = estimate_transition_matrix(pairs_train)
    N = count_transitions(pairs_train)
    ci = bootstrap_transition_ci(pairs_train, n_boot=n_boot)

    markov = markov_order_test(panel)
    homog_chi2 = homogeneity_test(pairs, split_month=split_month)
    homog_mc = homogeneity_test_mc(pairs, split_month=split_month, n_sim=n_mc)

    P_h = project_matrix(P, horizon_months)
    abs_prob = absorption_probabilities(P)
    mttf = mean_time_to_absorption(P)

    report = {
        "P": P,
        "N": N,
        "P_ci": ci,
        "markov_test": markov,
        "homogeneity_chi2": homog_chi2,
        "homogeneity_mc": homog_mc,
        f"P_{horizon_months}": P_h,
        "default_prob_eventual": {
            STATE_NAMES[i]: round(float(abs_prob[i]), 4) for i in range(N_STATES)
        },
        "mean_months_to_default": {
            STATE_NAMES[i]: round(float(mttf[i]), 2) for i in range(N_STATES - 1)
        },
        "pd_from_current_in_horizon": round(float(P_h[0, 2]), 4),
        "pd_from_delinquent_in_horizon": round(float(P_h[1, 2]), 4),
    }
    return report