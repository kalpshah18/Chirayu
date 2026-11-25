import numpy as np

from risk.pricing import price_lease, solve_base_rent, market_value_dcf
from risk.residual import ResidualParams, fit_lambda_from_haircut


def test_base_rent_round_trips_to_acquisition_cost():
    T = 24
    gsu_per_month = np.full(T, 100.0)
    r = 50.0
    terminal_value = 200_000.0
    annual_yield = 0.12
    acquisition_cost = 2_500_000.0

    B = solve_base_rent(acquisition_cost, gsu_per_month, r, terminal_value, annual_yield)

    y_m = annual_yield / 12.0
    months = np.arange(1, T + 1)
    discount = 1.0 / (1 + y_m) ** months
    pv = np.sum((B + r * gsu_per_month) * discount) + terminal_value / (1 + y_m) ** T

    assert pv == pytest_approx(acquisition_cost)


def pytest_approx(x, tol=1.0):
    import pytest
    return pytest.approx(x, abs=tol)


def test_price_lease_runs_and_prices_bursty_higher_than_steady():
    lam = fit_lambda_from_haircut(hours=20000, haircut_low=0.15, haircut_high=0.25, gsu_per_hour=1.0)
    params = ResidualParams(lam=lam, salvage_frac=0.10)

    def p_fail(g):
        return np.clip(g / 1_000_000.0, 0.0, 0.5)

    T = 24
    acquisition_cost = 2_500_000.0
    rental_rate = 150.0
    hours_per_month = 600.0

    steady_gsu = np.full(T, 0.85 * hours_per_month)
    bursty_gsu = np.full(T, 1.25 * hours_per_month)

    steady_result = price_lease(acquisition_cost, rental_rate, hours_per_month, steady_gsu, p_fail, params, 0.12)
    bursty_result = price_lease(acquisition_cost, rental_rate, hours_per_month, bursty_gsu, p_fail, params, 0.12)

    assert steady_result.base_rent_monthly > 0
    assert bursty_result.base_rent_monthly > 0
    # A bursty lessee wears the GPU faster, so its effective rent (base + reserve*GSU)
    # should be at least as high as the steady lessee's, on average over the term.
    assert np.mean(bursty_result.effective_rent_per_month) >= np.mean(steady_result.effective_rent_per_month)


def test_market_value_dcf_positive_and_monotonic_in_rate():
    low = market_value_dcf(100.0, 600.0, 24, 0.12)
    high = market_value_dcf(200.0, 600.0, 24, 0.12)
    assert 0 < low < high
