"""Swell-to-swell interaction factors used to calibrate wave shape per spot (future work)."""

import polars as pl

from clima_mollendo.sea_state import Partition, beat_period, partitions_from_row
from clima_mollendo.spot import Spot, sector


def _swells(row: dict) -> list[Partition]:
    swells = [p for p in partitions_from_row(row) if p.kind == "swell"]
    return sorted(swells, key=lambda p: p.hs, reverse=True)


def interaction_row(row: dict, spot: Spot) -> dict:
    """Factors for the two most energetic swells of one hourly row."""
    swells = _swells(row)
    out = {
        "time": row["time"],
        "n_swells": len(swells),
        "primary_sector": None,
        "secondary_sector": None,
        "energy_ratio": None,
        "beat_period_s": None,
        "rel_angle_deg": None,
        "expected_wave": None,
    }
    if not swells:
        return out
    p = swells[0]
    out["primary_sector"] = sector(p.direction)
    if len(swells) < 2:
        out["expected_wave"] = "primary only"
        return out
    s = swells[1]
    out["secondary_sector"] = sector(s.direction)
    out["energy_ratio"] = round(s.hs**2 / p.hs**2, 3)
    out["beat_period_s"] = round(beat_period(p.tm, s.tm), 1)
    out["rel_angle_deg"] = round(abs(((s.direction - p.direction) + 180) % 360 - 180), 1)
    out["expected_wave"] = spot.interaction_rules.get(
        (out["primary_sector"], out["secondary_sector"]), "uncalibrated"
    )
    return out


def swell_interactions(marine: pl.DataFrame, spot: Spot) -> pl.DataFrame:
    """One row per hour with the interaction factors of the top two swells."""
    return pl.DataFrame([interaction_row(r, spot) for r in marine.iter_rows(named=True)])
