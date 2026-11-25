import numpy as np

from risk.montecarlo import LeaseSpec, MarketParams, simulate_book


def _hazard(scale=1e-7):
    def f(g):
        return np.clip(g * scale, 0.0, 0.9)
    return f


def test_simulate_book_runs_and_produces_sane_outputs():
    market = MarketParams(drift_normal=-0.10, vol_normal=0.15, drift_shock=-0.35, vol_shock=0.35,
                           p_stay_normal=0.95, p_stay_shock=0.70)
    leases = [
        LeaseSpec(
            acquisition_cost=2_500_000.0, gsu_per_hour_mean=0.85, gsu_per_hour_std=0.1,
            hours_per_month=600.0, term_months=24, base_rent_monthly=80_000.0,
            reserve_rate_per_gsu=50.0, lam_condition=1e-5, salvage_frac=0.10,
            hazard_fn=_hazard(),
        )
        for _ in range(20)
    ]

    result = simulate_book(leases, market, n_paths=2000, market_factor_loading=0.3, seed=42)

    assert result.losses.shape == (2000,)
    assert result.per_lease_losses.shape == (20, 2000)
    assert result.expected_loss >= 0
    assert result.var99 >= result.expected_loss
    assert result.cvar99 >= result.var99


def test_more_wear_increases_expected_loss():
    market = MarketParams(drift_normal=-0.10, vol_normal=0.15, drift_shock=-0.35, vol_shock=0.35,
                           p_stay_normal=0.95, p_stay_shock=0.70)

    def make_leases(gsu_mean):
        return [LeaseSpec(
            acquisition_cost=2_500_000.0, gsu_per_hour_mean=gsu_mean, gsu_per_hour_std=0.05,
            hours_per_month=600.0, term_months=24, base_rent_monthly=80_000.0,
            reserve_rate_per_gsu=50.0, lam_condition=5e-5, salvage_frac=0.10,
            hazard_fn=_hazard(scale=5e-6),
        ) for _ in range(30)]

    low = simulate_book(make_leases(0.5), market, n_paths=4000, market_factor_loading=0.3, seed=1)
    high = simulate_book(make_leases(2.0), market, n_paths=4000, market_factor_loading=0.3, seed=1)

    assert high.expected_loss >= low.expected_loss
