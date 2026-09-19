"""Wind scoring: what matters for wave shape is gusty wind blowing onto the beach."""

import math

import polars as pl

from clima_mollendo.spot import Spot

THERMAL_GUST_RATIO = 2.5  # gust / sustained above this smells like unresolved sea breeze
THERMAL_MIN_GUST = 15.0  # km/h


def onshore_factor(wind_dir: float, coast_facing: float) -> float:
    """1 = blowing straight onto the beach, 0.5 = side, 0 = straight offshore (terral)."""
    delta = math.radians(wind_dir - coast_facing)
    return (1.0 + math.cos(delta)) / 2.0


def wind_relation(wind_dir: float, coast_facing: float) -> str:
    """Coarse label of the wind relative to the coast."""
    delta = abs(((wind_dir - coast_facing) + 180) % 360 - 180)
    if delta < 45:
        return "onshore"
    if delta < 90:
        return "side-on"
    if delta <= 135:
        return "side-off"
    return "terral"


def effective_wind(sustained: float, gust: float) -> float:
    """Pessimistic speed: the model's sustained wind under-reads at the beach."""
    return (sustained + gust) / 2.0


def score_wind(hourly: pl.DataFrame, spot: Spot) -> pl.DataFrame:
    """Add wind_eff (damaging km/h), wind_rel (onshore/side/terral) and thermal flag."""
    rows = []
    for w, g, d in zip(hourly["wind_kmh"], hourly["gust_kmh"], hourly["wind_dir"], strict=True):
        if w is None or g is None or d is None:
            rows.append((None, None, False))
            continue
        eff = effective_wind(w, g) * onshore_factor(d, spot.coast_facing_deg)
        thermal = g >= THERMAL_MIN_GUST and g / max(w, 0.1) > THERMAL_GUST_RATIO
        rows.append((round(eff, 1), wind_relation(d, spot.coast_facing_deg), thermal))
    extra = pl.DataFrame(
        rows,
        schema={"wind_eff": pl.Float64, "wind_rel": pl.String, "thermal": pl.Boolean},
        orient="row",
    )
    return pl.concat([hourly, extra], how="horizontal")
