from attacks.inject import AttackSpec, inject
from data.synth import make_synthetic_node_month
from ledger.divergence import fit_divergence, score


def test_divergence_fit_recovers_near_unity_gain():
    df = make_synthetic_node_month(n_steps=2000, seed=1)
    fit = fit_divergence(df.iloc[:1000])
    # beta absorbs PSU efficiency and the ~11% offset built into the synthetic
    # generator, so it should land a bit above 1, not at some arbitrary value.
    assert 0.9 < fit.beta < 1.5
    assert fit.residual_std >= 0


def test_power_spoof_raises_an_alarm():
    df = make_synthetic_node_month(n_steps=3000, seed=2)
    fit = fit_divergence(df.iloc[:1000])

    spoofed, mask = inject(df, AttackSpec("underreport_load", 0.6, start_frac=0.5, duration_frac=0.3))
    result = score(spoofed, fit)

    # At least one alarm should fire somewhere during (or shortly after) the attack window.
    attack_region = result["alarm"][mask]
    assert attack_region.any()


def test_clean_data_rarely_alarms():
    df = make_synthetic_node_month(n_steps=3000, seed=3)
    fit = fit_divergence(df.iloc[:1000])
    result = score(df.iloc[1000:], fit, h=5.0)
    false_alarm_rate = result["alarm"].mean()
    assert false_alarm_rate < 0.05
