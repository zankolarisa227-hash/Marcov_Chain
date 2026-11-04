"""feature_contracts.py — список признаков и anti-leakage правила."""

from __future__ import annotations

# признаки 
FEATURE_COLS: list[str] = [
    # utilization stats
    "util_mean", "util_std", "util_max", "util_min", "util_range", "n_tx", "sum_amnt", "mean_amnt",
    # dynamics
    "util_trend", "util_mean_last3", "util_mean_prev", "util_accel", "util_growth_ratio",
    # state history
    "n_state_changes", "state_change_rate",
    "frac_current", "frac_delinquent", "frac_default",
    "max_consec_delinquent", "months_since_last_del",
    # flags / complexity
    "state_entropy", "util_cv",
    "high_util_ratio", "ever_delinquent", "util_rising_3m", "util_vs_mean_dev",
    # current
    "utilization", "macro_stress", "util_x_macro",
    #объём и масштаб 
]

# нельзя использовать как feature
FORBIDDEN_COLS = {
    "next_state", "target_h", "client_id", "app_id", "month", "flag"
}

# Правило: признак должен быть computable at time t (as-of t)
FEATURE_ASOF_POLICY = {
    "правило": "причинно-следственное",
    "описание": "Признак на месяц t использует только данные с месяцем <= t для того же клиента",
}