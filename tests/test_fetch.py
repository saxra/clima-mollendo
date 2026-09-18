from datetime import datetime

import polars as pl
import pytest

from clima_mollendo.fetch import parse_tides, track_swells
from clima_mollendo.interactions import swell_interactions
from clima_mollendo.spot import MOLLENDO, sector


def _row(kind: str, clock: str, day: str, height: str, unit: str = "m") -> str:
    return (
        f"<tr><td>{kind} Tide</td><td><b>{clock}</b>"
        f'<span class="tide-day-tides__secondary">({day})</span></td>'
        f'<td class="x"><b class="y">{height} {unit}</b></td></tr>'
    )


TIDE_HTML = (
    "<p>The predicted tide times today on Friday 18 September 2026 for X are:</p>"
    + "".join(
        [
            _row("High", "00:54 AM", "Fri 18 September", "0.79"),
            _row("Low", " 8:20 AM", "Fri 18 September", "0.28"),
            _row("High", "00:54 AM", "Fri 18 September", "0.79"),
            _row("Low", " 1:05 PM", "Sat 3 January", "0.30"),
        ]
    )
)


def test_parse_tides_dedups_and_rolls_year():
    df = parse_tides(TIDE_HTML)
    assert df.height == 3
    assert df["time"][0] == datetime(2026, 9, 18, 0, 54)
    assert df["kind"].to_list() == ["high", "low", "low"]
    assert df["time"][-1] == datetime(2027, 1, 3, 13, 5)


def test_parse_tides_converts_feet():
    html = "tide times today on Friday 18 September 2026 x" + _row(
        "High", "1:17 PM", "Fri 18 September", "1.38", "ft"
    )
    df = parse_tides(html)
    assert df["height_m"][0] == pytest.approx(0.42, abs=0.01)


def test_parse_tides_raises_when_empty():
    with pytest.raises(ValueError):
        parse_tides("tide times today on Friday 18 September 2026 nothing here")


def _marine(rows: list[tuple]) -> pl.DataFrame:
    cols = ["time", "s1_h", "s1_t", "s1_d", "s2_h", "s2_t", "s2_d", "s3_h", "s3_t", "s3_d"]
    return pl.DataFrame(rows, schema=cols, orient="row")


def test_track_swells_keeps_identity_after_rank_swap():
    t0 = datetime(2026, 9, 18, 0)
    marine = _marine(
        [
            (t0, 0.9, 11.0, 233, 0.5, 6.0, 190, None, None, None),
            (t0.replace(hour=1), 0.9, 11.0, 233, 0.6, 5.9, 191, None, None, None),
            (t0.replace(hour=2), 0.95, 5.8, 192, 0.9, 10.9, 234, None, None, None),
        ]
    )
    tracks = track_swells(marine)
    long_swell = tracks.filter(pl.col("t") > 9)
    assert long_swell["track"].n_unique() == 1
    assert tracks.filter(pl.col("t") < 7)["track"].n_unique() == 1
    assert tracks["track"].n_unique() == 2


def test_track_swells_handles_missing_partitions():
    t0 = datetime(2026, 9, 18, 0)
    marine = _marine([(t0, 0.9, 11.0, 233, None, None, None, None, None, None)])
    tracks = track_swells(marine)
    assert tracks.height == 1
    assert tracks["track"][0] == 1


def test_sector_and_interaction_rule():
    assert sector(190) == "S"
    assert sector(233) == "SW"
    assert sector(270) == "W"
    assert sector(None) is None
    t0 = datetime(2026, 9, 18, 0)
    marine = _marine([(t0, 0.9, 11.0, 190, 0.5, 8.0, 270, None, None, None)]).with_columns(
        ww_h=pl.lit(None, dtype=pl.Float64),
        ww_t=pl.lit(None, dtype=pl.Float64),
        ww_d=pl.lit(None, dtype=pl.Float64),
    )
    out = swell_interactions(marine, MOLLENDO)
    assert out["expected_wave"][0] == "lefts"
    assert out["rel_angle_deg"][0] == 80.0
