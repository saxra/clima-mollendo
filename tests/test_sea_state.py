import numpy as np
import pytest

from clima_mollendo.sea_state import (
    Partition,
    beat_period,
    combined_hs,
    prob_exceed,
    rayleigh_stats,
    wave_distribution,
    zero_crossing_waves,
)


def test_rayleigh_ratios():
    s = rayleigh_stats(hs=1.0, tm=10.0, duration_s=7200)
    assert s["h_1_10"] == pytest.approx(1.27)
    assert s["h_1_100"] == pytest.approx(1.67)
    assert 1.7 < s["h_max"] < 2.0


def test_prob_exceed_edges():
    assert prob_exceed(1.0, 0.0) == pytest.approx(1.0)
    assert prob_exceed(1.0, 1.0) == pytest.approx(np.exp(-2))
    assert prob_exceed(0.0, 1.0) == 0.0


def test_combined_hs_adds_energy():
    assert combined_hs([Partition(0.6, 10, 230), Partition(0.8, 6, 190)]) == pytest.approx(1.0)


def test_beat_period():
    assert beat_period(10.0, 10.0) == float("inf")
    assert beat_period(11.0, 8.0) == pytest.approx(29.33, rel=1e-2)


def test_zero_crossing_pure_sine():
    t = np.arange(0, 300, 0.1)
    waves = zero_crossing_waves(t, 0.5 * np.sin(2 * np.pi * t / 8))
    assert waves["height"].mean() == pytest.approx(1.0, abs=0.02)
    assert waves["period"].mean() == pytest.approx(8.0, abs=0.1)


def test_zero_crossing_empty():
    t = np.arange(0, 10, 0.1)
    assert zero_crossing_waves(t, np.ones_like(t)).is_empty()


def test_simulation_reproduces_hs():
    parts = [Partition(0.9, 11.0, 233), Partition(0.5, 6.0, 190)]
    waves, s = wave_distribution(parts, duration_s=7200, seed=1)
    assert s["hs_sim"] == pytest.approx(s["hs_input"], rel=0.1)
    assert s["h_1_3"] == pytest.approx(s["hs_sim"], rel=0.15)
    assert s["h_max"] > s["h_1_10"] > s["h_mean"]
    assert len(waves) > 300
