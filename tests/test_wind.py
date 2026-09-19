import polars as pl
import pytest

from clima_mollendo.spot import MOLLENDO
from clima_mollendo.wind import effective_wind, onshore_factor, score_wind, wind_relation


def test_onshore_factor_geometry():
    assert onshore_factor(225, 225) == pytest.approx(1.0)  # straight onto the beach
    assert onshore_factor(45, 225) == pytest.approx(0.0)  # terral
    assert onshore_factor(135, 225) == pytest.approx(0.5)  # side


def test_wind_relation_labels():
    assert wind_relation(250, 225) == "onshore"
    assert wind_relation(160, 225) == "side-on"
    assert wind_relation(120, 225) == "side-off"
    assert wind_relation(45, 225) == "terral"


def test_effective_wind_is_mean_of_sustained_and_gust():
    assert effective_wind(5.0, 19.0) == 12.0


def test_score_wind_flags_thermal_breeze_and_handles_nulls():
    df = pl.DataFrame(
        {
            "wind_kmh": [5.0, 8.0, None],
            "gust_kmh": [19.0, 16.0, None],
            "wind_dir": [242, 126, None],
        }
    )
    out = score_wind(df, MOLLENDO)
    assert out["wind_rel"].to_list() == ["onshore", "side-off", None]
    assert out["thermal"].to_list() == [True, False, False]
    assert out["wind_eff"][0] > out["wind_eff"][1]
    assert out["wind_eff"][2] is None
