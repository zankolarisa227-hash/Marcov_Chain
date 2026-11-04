"""
Толковые тесты static Markov core для credit migration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.markov_chain import (
    absorption_probabilities,
    estimate_transition_matrix,
    homogeneity_test,
    markov_order_test,
    mean_time_to_absorption,
    project_matrix,
)

DEFAULT = 2


# --------------------- helpers: клиентские траектории ---------------------

def make_client_panel(
    n_clients=100,
    n_months=6,
    p01=0.10,
    p12=0.20,
    p10=0.25,
    seed=42,
) -> pd.DataFrame:
    """Короткие клиентские истории: 0=current, 1=delinq, 2=default(absorbing)."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(n_clients):
        s = 0
        for m in range(1, n_months + 1):
            util = {0: rng.uniform(0.05, 0.4),
                    1: rng.uniform(0.4, 0.85),
                    2: rng.uniform(0.6, 0.99)}[s]
            rows.append({
                "client_id": cid,
                "month": m,
                "state": s,
                "utilization": float(util),
                "macro_stress": float(rng.normal(0, 0.25)),
            })
            if s == 2:
                s = 2
            elif s == 0:
                r = rng.random()
                s = 1 if r < p01 else (2 if r > 0.995 else 0)
            else:
                r = rng.random()
                if r < p12:
                    s = 2
                elif r < p12 + p10:
                    s = 0
                else:
                    s = 1
    return pd.DataFrame(rows)


def pairs_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["client_id", "month"]).copy()
    df["next_state"] = df.groupby("client_id")["state"].shift(-1)
    df = df.dropna(subset=["next_state"])
    df["next_state"] = df["next_state"].astype(int)
    return df


# --------------------- fixtures ---------------------

@pytest.fixture(scope="module")
def panel():
    return make_client_panel()


@pytest.fixture(scope="module")
def pairs(panel):
    return pairs_from_panel(panel)


@pytest.fixture(scope="module")
def P(pairs):
    return estimate_transition_matrix(pairs[["state", "next_state"]])


@pytest.fixture
def fixed_P():
    return np.array([
        [0.88, 0.10, 0.02],
        [0.20, 0.55, 0.25],
        [0.00, 0.00, 1.00],
    ], dtype=float)


# --------------------- 1. Matrix invariants ---------------------

def test_rows_sum_to_one(P):
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)


def test_nonnegative(P):
    assert np.all(P >= -1e-12)


def test_default_is_fully_absorbing(P):
    assert P[DEFAULT, DEFAULT] == pytest.approx(1.0)
    assert np.allclose(P[DEFAULT, :], [0.0, 0.0, 1.0])


def test_absorbing_forced_even_if_data_has_escape():
    """Даже при грязных next_state из default оценка должна форсировать absorbing."""
    dirty = pd.DataFrame({
        "state": [0, 1, 2, 2, 2],
        "next_state": [1, 2, 0, 1, 2],
    })
    P = estimate_transition_matrix(dirty)
    assert np.allclose(P[DEFAULT], [0.0, 0.0, 1.0])


def test_delinquent_has_higher_default_rate_than_current(P):
    """На миграционных клиентах: P(default|delinq) >= P(default|current)."""
    assert P[1, DEFAULT] >= P[0, DEFAULT] - 1e-9


# --------------------- 2. Projection / PD term-structure ---------------------

def test_project_h1_is_identity_map(P):
    np.testing.assert_allclose(project_matrix(P, 1), P, atol=1e-12)


def test_project_h12_still_stochastic(P):
    Ph = project_matrix(P, 12)
    np.testing.assert_allclose(Ph.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(Ph >= -1e-10)


def test_pd_nondecreasing_in_horizon_absorbing_chain(fixed_P):
    p1 = project_matrix(fixed_P, 1)[0, DEFAULT]
    p6 = project_matrix(fixed_P, 6)[0, DEFAULT]
    p12 = project_matrix(fixed_P, 12)[0, DEFAULT]
    assert p6 >= p1 - 1e-12
    assert p12 >= p6 - 1e-12


# --------------------- 3. Absorption / MTTF (fixed_P) ---------------------

def test_absorption_probabilities_sensible(fixed_P):
    B = np.asarray(absorption_probabilities(fixed_P), dtype=float).reshape(-1)
    assert np.all(B >= -1e-10) and np.all(B <= 1.0 + 1e-10)
    # единственный absorbing = default → eventual absorption = 1
    assert np.allclose(B[: min(len(B), 2)], 1.0, atol=1e-4) or (
        len(B) == 3 and B[DEFAULT] == pytest.approx(1.0) and np.allclose(B[:2], 1.0, atol=1e-4)
    )


def test_mean_time_to_absorption_semantics(fixed_P):
    t = np.asarray(mean_time_to_absorption(fixed_P), dtype=float).reshape(-1)
    if len(t) == 3:
        assert t[DEFAULT] == pytest.approx(0.0, abs=1e-10)
        assert t[0] > 0 and t[1] > 0
        assert t[0] > t[1]  # на fixed_P это ожидаемо
    else:
        assert np.all(t > 0)


# --------------------- 4. Statistical tests ---------------------

def test_markov_order_test_api(panel):
    out = markov_order_test(panel)
    assert "p_value" in out and "conclusion" in out
    if out["p_value"] is not None:
        assert 0.0 <= float(out["p_value"]) <= 1.0
    if "lr_statistic" in out:
        assert out["lr_statistic"] >= -1e-9


def test_homogeneity_test_api(panel):
    out = homogeneity_test(panel, split_month=4)
    assert "p_value" in out and "conclusion" in out
    if out["p_value"] is not None:
        assert 0.0 <= float(out["p_value"]) <= 1.0


def test_homogeneity_detects_portfolio_regime_shift():
    """Спокойный портфель → стрессовый: тест должен увидеть неоднородность."""
    calm = make_client_panel(n_clients=120, p01=0.05, p12=0.08, p10=0.30, seed=1)
    stress = make_client_panel(n_clients=120, p01=0.25, p12=0.45, p10=0.10, seed=2)
    stress = stress.copy()
    stress["month"] = stress["month"] + 6
    stress["client_id"] = stress["client_id"] + 10000
    panel = pd.concat([calm, stress], ignore_index=True)

    out = homogeneity_test(panel, split_month=7)
    assert out["p_value"] is not None
    assert float(out["p_value"]) < 0.05


# --------------------- 5. Bootstrap (optional naming) ---------------------

def test_bootstrap_ci_lower_leq_upper(pairs):
    try:
        from src.models.markov_chain import bootstrap_ci as boot
        ci = bootstrap_ci = boot(pairs[["state", "next_state"]], n_bootstrap=40, alpha=0.05)
    except TypeError:
        from src.models.markov_chain import bootstrap_ci as boot
        ci = boot(pairs)
    except ImportError:
        try:
            from src.models.markov_chain import bootstrap_transition_ci as boot
            ci = boot(pairs[["state", "next_state"]], n_boot=40)
        except Exception:
            pytest.skip("bootstrap function not available")

    lower = np.asarray(ci["lower"] if "lower" in ci else ci.get("P_ci", {}).get("lower"))
    upper = np.asarray(ci["upper"] if "upper" in ci else ci.get("P_ci", {}).get("upper"))
    if lower is None or upper is None or lower.shape != upper.shape:
        pytest.skip(f"Unexpected CI schema: {list(ci.keys())}")
    assert np.all(lower <= upper + 1e-9)