import numpy as np
import pytest

from wear.gsu import GsuParams, gsu, reference_trace_gsu_per_hour


def test_reference_trace_is_one_gsu_per_hour():
    assert reference_trace_gsu_per_hour() == pytest.approx(1.0, abs=1e-6)


def test_hotter_trace_gives_more_gsu():
    params = GsuParams.from_yaml()
    dt = np.array([1.0, 1.0, 1.0])
    cool = gsu(np.full(3, params.T_ref_c), np.full(3, params.P_ref_w), dt, params=params)
    hot = gsu(np.full(3, params.T_ref_c + 20), np.full(3, params.P_ref_w), dt, params=params)
    assert hot["gsu"] > cool["gsu"]


def test_more_cycled_trace_gives_more_gsu():
    params = GsuParams.from_yaml()
    dt = np.ones(6)
    steady = gsu(np.full(6, params.T_ref_c), np.full(6, params.P_ref_w), dt, params=params)
    cycled_temp = params.T_ref_c + np.array([0, 20, 0, 20, 0, 20], dtype=float)
    cycled = gsu(cycled_temp, np.full(6, params.P_ref_w), dt, params=params)
    assert cycled["gsu"] > steady["gsu"]


def test_higher_power_gives_more_gsu():
    params = GsuParams.from_yaml()
    dt = np.array([1.0, 1.0])
    low = gsu(np.full(2, params.T_ref_c), np.full(2, params.P_ref_w * 0.5), dt, params=params)
    high = gsu(np.full(2, params.T_ref_c), np.full(2, params.P_ref_w * 1.5), dt, params=params)
    assert high["gsu"] > low["gsu"]


def test_memory_error_term_adds_wear():
    params = GsuParams.from_yaml()
    dt = np.array([1.0, 1.0])
    no_err = gsu(np.full(2, params.T_ref_c), np.full(2, params.P_ref_w), dt, remapped_rows_delta=0.0, params=params)
    with_err = gsu(np.full(2, params.T_ref_c), np.full(2, params.P_ref_w), dt, remapped_rows_delta=10.0, params=params)
    assert with_err["gsu"] > no_err["gsu"]
