import numpy as np

from risk.tranching import attachment_point, build_tranches, value_of_data


def test_attachment_point_is_a_quantile():
    losses = np.arange(1, 101, dtype=float)  # 1..100
    a = attachment_point(losses, exceedance_prob=0.01)
    assert a == np.quantile(losses, 0.99)


def test_build_tranches_orders_senior_below_mezzanine_attachment():
    rng = np.random.default_rng(0)
    losses = np.abs(rng.normal(1000, 300, size=20000))
    tranches = build_tranches(losses, {"senior": 0.01, "mezzanine": 0.05})
    by_name = {t.name: t.attachment for t in tranches}
    # A 1% exceedance threshold must sit above (be stricter than) a 5% one.
    assert by_name["senior"] > by_name["mezzanine"]


def test_value_of_data_has_expected_sign():
    rng = np.random.default_rng(1)
    # With real condition data, losses are tighter (lower variance) around the same mean.
    losses_with_data = np.abs(rng.normal(1000, 150, size=20000))
    losses_without_data = np.abs(rng.normal(1000, 500, size=20000))

    result = value_of_data(
        losses_with_data, losses_without_data,
        target_exceedance={"senior": 0.01, "mezzanine": 0.05},
        notional=50_000_000.0, spread_senior=0.06, spread_mezzanine=0.12,
    )
    # Tighter loss distribution -> lower senior attachment point needed -> positive delta -> savings.
    assert result["delta_attachment"] > 0
    assert result["funding_saving"] > 0
