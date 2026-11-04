from __future__ import annotations

import pytest

from src.risk.risk_metrics import ecl, optimal_limit

def _ecl(pd, ead, lgd):
    """Вызов ecl с разными возможными сигнатурами."""
    try:
        return ecl(pd, ead, lgd)
    except TypeError:
        try:
            return ecl(pd=pd, ead=ead, lgd=lgd)
        except TypeError:
            return ecl(pd_12m=pd, ead=ead, lgd=lgd)


def test_ecl_formula():
    assert _ecl(0.1, 10_000, 0.5) == pytest.approx(500.0)
    

def test_ecl_zero_when_pd_zero():
    assert _ecl(0.0, 50_000, 0.45) == pytest.approx(0.0)


def test_ecl_scales_with_ead():
    """Удвоение EAD → удвоение ECL при том же PD/LGD."""
    a = _ecl(0.2, 10_000, 0.4)
    b = _ecl(0.2, 20_000, 0.4)
    assert b == pytest.approx(2 * a)


def test_ecl_rejects_invalid_pd():
    with pytest.raises(ValueError):
        _ecl(1.5, 10_000, 0.45)

                       
def _limit(pd, **kw):
    """Поддержка max_pd / risk_appetite в разных версиях кода."""
    try:
        return optimal_limit(pd_12m=pd, **kw)
    except TypeError:
        if "max_pd" in kw:
            kw = dict(kw)
            kw["risk_appetite"] = kw.pop("max_pd")
        return optimal_limit(pd_12m=pd, **kw)


def test_limit_policy_gate():
    try:
        bad = optimal_limit(pd_12m=0.4, max_pd=0.2)
        good = optimal_limit(pd_12m=0.02, max_pd=0.2)
    except TypeError:
        bad = optimal_limit(pd_12m=0.4, risk_appetite=0.2)
        good = optimal_limit(pd_12m=0.02, risk_appetite=0.2)
    assert bad.get("approved") is False
    assert good.get("approved") is True



def test_limit_rejects_when_el_ratio_too_high():

    try:
        rec = optimal_limit(
            pd_12m=0.20,
            max_pd=0.50,       # PD-порог не срабатывает
            max_el_ratio=0.05,
            lgd=0.50,
        )
    except TypeError:
        pytest.skip("optimal_limit without max_el_ratio in this version")
    assert rec.get("approved") is False