"""
Feature engineering: objects × features matrix.

Одна строка = один клиент в один момент времени t.
Все признаки считаются СТРОГО по истории [0, t] — никаких данных из будущего.

Структура признаков:
  Блок A — объём и масштаб операций
  Блок B — динамика (как менялось поведение)
  Блок C — регулярность (ритм платежей)
  Блок D — разнообразие трат (MCC / категории)
  Блок E — риск-сигналы (ночные, крупные, e-commerce)
  Блок F — текущее состояние (утилизация, макро)

Совместимость с пайплайном:
  - Входной panel DataFrame: колонки [client_id, month, state,
    utilization, macro_stress] — то же, что generate_panel().
  - Выходной features DataFrame: колонки [client_id, month, state] +
    все признаки. Этот датафрейм готов как X для dynamic_model.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _safe_mean(arr):
    return float(np.mean(arr)) if len(arr) > 0 else 0.0

def _safe_std(arr):
    return float(np.std(arr)) if len(arr) > 1 else 0.0

def _safe_entropy(counts):
    """Энтропия Шеннона нормализованная к [0, 1]."""
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    h = sp_entropy(probs)
    max_h = np.log(len(counts)) if len(counts) > 1 else 1.0
    return float(h / max_h) if max_h > 0 else 0.0

def _linear_trend(values):
    """Коэффициент наклона линейного тренда. Растёт > 0, падает < 0."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = np.array(values, dtype=float) - np.mean(values)
    denom = np.dot(x, x)
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Ядро: признаки для одного клиента по накопленной истории до месяца t
# ---------------------------------------------------------------------------

def _features_for_window(history: pd.DataFrame, current_row: pd.Series) -> dict:
    """
    history  — строки панели для данного клиента с month <= t
    current_row — строка панели для месяца t (содержит state, utilization, macro_stress)

    Возвращает словарь признаков (один вектор матрицы объекты×признаки).
    """


    feats = {
        'client_id': current_row['client_id'],
        'month':     current_row['month'],
        'state':     current_row['state'],
    }

    # Нам нужна хотя бы одна строка истории
    if len(history) == 0:
        # Заполняем нулями — клиент только появился
        for k in _FEATURE_NAMES:
            feats[k] = 0.0
        feats['utilization']  = current_row['utilization']
        feats['macro_stress'] = current_row['macro_stress']
        return feats

    util_hist  = history['utilization'].values
    state_hist = history['state'].values
    n          = len(history)

    # ------------------------------------------------------------------
    # Блок A: объём и масштаб (на основе utilization как прокси суммы)
    # ------------------------------------------------------------------
    feats['util_mean']  = _safe_mean(util_hist)
    feats['util_std']   = _safe_std(util_hist)
    feats['util_max']   = float(np.max(util_hist))
    feats['util_min']   = float(np.min(util_hist))
    feats['util_range'] = feats['util_max'] - feats['util_min']

    # ------------------------------------------------------------------
    # Блок B: динамика — как менялась утилизация со временем
    # ------------------------------------------------------------------
    feats['util_trend']     = _linear_trend(util_hist)

    # Последние 3 месяца vs предыдущие: ускоряется ли рост нагрузки?
    last3  = util_hist[-3:]  if n >= 3  else util_hist
    prev   = util_hist[:-3]  if n > 3   else np.array([0.0])
    feats['util_mean_last3']     = _safe_mean(last3)
    feats['util_mean_prev']      = _safe_mean(prev)
    feats['util_accel']          = feats['util_mean_last3'] - feats['util_mean_prev']

    # Темп роста — как сильно утилизация сейчас выше начального периода
    first3 = util_hist[:3]
    denom  = _safe_mean(first3) + 1e-6
    feats['util_growth_ratio']   = feats['util_mean_last3'] / denom

    # 3 ритм изменений состояния
    # Сколько раз менялось состояние (0→1, 1→0, 1→2)
    state_changes = int(np.sum(np.diff(state_hist) != 0)) if n > 1 else 0
    feats['n_state_changes']     = state_changes
    feats['state_change_rate']   = state_changes / max(n - 1, 1)

    # Доля месяцев в каждом состоянии
    feats['frac_current']        = float(np.mean(state_hist == 0))
    feats['frac_delinquent']     = float(np.mean(state_hist == 1))
    feats['frac_default']        = float(np.mean(state_hist == 2))

    # Максимальная непрерывная "просрочка" подряд
    max_consec_del = 0
    cur_consec_del = 0
    for s in state_hist:
        if s == 1:
            cur_consec_del += 1
            max_consec_del = max(max_consec_del, cur_consec_del)
        else:
            cur_consec_del = 0
    feats['max_consec_delinquent'] = max_consec_del

    # При наличии, к-во месяцев с момента последней просрочки 
    del_months = np.where(state_hist == 1)[0]
    feats['months_since_last_del'] = (
        int(n - 1 - del_months[-1]) if len(del_months) > 0 else n
    )

    # 4 разнообразие поведения

    # 4.1 Энтропия состояний — насколько история "разнообразна" по статусам
    state_counts = np.bincount(state_hist, minlength=3).astype(float)
    feats['state_entropy'] = _safe_entropy(state_counts)

    # 4.2 Волатильность утилизации CV = std/mean
    mean_u = feats['util_mean'] + 1e-6
    feats['util_cv'] = feats['util_std'] / mean_u

    # 5. риск-сигналы
    # 5.1 Доля месяцев с "высокой" утилизацией (> 0.7)
    feats['high_util_ratio'] = float(np.mean(util_hist > 0.7))

    # 5.2 Флаг: был ли хоть раз в просрочке за всю историю
    feats['ever_delinquent']     = int(np.any(state_hist == 1))

    # 5.3 Флаг: утилизация растёт последние 3 месяца подряд
    if n >= 3:
        feats['util_rising_3m'] = int(
            util_hist[-1] > util_hist[-2] > util_hist[-3]
        )
    else:
        feats['util_rising_3m'] = 0

    # 5.4 Разрыв между текущей утилизацией и её средним — насколько сейчас хуже/лучше нормы
    feats['util_vs_mean_dev']    = float(current_row['utilization'] - feats['util_mean'])


    # 6: текущий снимок (point-in-time, не история)
   
    feats['utilization']         = float(current_row['utilization'])
    feats['macro_stress']        = float(current_row['macro_stress'])

    # Взаимодействие: высокая утилизация В условиях макро-стресса
    feats['util_x_macro']        = feats['utilization'] * feats['macro_stress']

    return feats



# Список всех признаков (client_id / month / state)
_FEATURE_NAMES = [
    # A
    'util_mean', 'util_std', 'util_max', 'util_min', 'util_range',
    # B
    'util_trend', 'util_mean_last3', 'util_mean_prev', 'util_accel',
    'util_growth_ratio',
    # C
    'n_state_changes', 'state_change_rate',
    'frac_current', 'frac_delinquent', 'frac_default',
    'max_consec_delinquent', 'months_since_last_del',
    # D
    'state_entropy', 'util_cv',
    # E
    'high_util_ratio', 'ever_delinquent', 'util_rising_3m',
    'util_vs_mean_dev',
    # F
    'utilization', 'macro_stress', 'util_x_macro',
]

FEATURE_COLS = _FEATURE_NAMES  # публичный alias для импорта в других модулях



# Публичный API

def build_feature_matrix(panel: pd.DataFrame,
                          min_history: int = 2) -> pd.DataFrame:


    #сортировка по возрастанию месяцев
    panel = panel.sort_values(['client_id', 'month']).reset_index(drop=True)
    records = [] #общий словарь для признаков

    for client_id, client_df in panel.groupby('client_id'):
        client_df = client_df.sort_values('month').reset_index(drop=True)

        for i, row in client_df.iterrows():
            t = row['month']
            # история = все месяцы этого клиента ДО t включительно
            history = client_df[client_df['month'] <= t]

            if len(history) < min_history:
                continue  # слишком мало истории — пропускаем

            # признаки на основе этой истории
            feats = _features_for_window(history, row)
            records.append(feats)

    features_df = pd.DataFrame(records)

    # Гарантируем порядок колонок: мета → признаки
    col_order = ['client_id', 'month', 'state'] + FEATURE_COLS
    features_df = features_df[col_order]

    return features_df.reset_index(drop=True)


def split_Xy(features_df: pd.DataFrame, target_col: str = "next_state"):
    X = features_df[FEATURE_COLS].values
    y = features_df[target_col].values
    return X, y



def split_train_test_oot(features_df: pd.DataFrame,
                          split_month: int) -> tuple:
    train = features_df[features_df['month'] <  split_month]
    test  = features_df[features_df['month'] >= split_month]
    return train, test

def add_next_state(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.sort_values(["client_id", "month"]).copy()
    df["next_state"] = df.groupby("client_id")["state"].shift(-1)
    df = df.dropna(subset=["next_state"]).copy()
    df["next_state"] = df["next_state"].astype(int)
    return df.reset_index(drop=True)