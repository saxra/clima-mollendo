"""Spot definitions and spot-specific rules (calibration lives here)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spot:
    name: str
    lat: float
    lon: float
    timezone: str
    tide_slug: str
    coast_facing_deg: float
    # (primary sector, secondary sector) -> expected dominant wave. Empty means "unknown yet".
    interaction_rules: dict[tuple[str, str], str] = field(default_factory=dict)


# Direction sectors (degrees the swell comes FROM). Tuned for the Peruvian south coast.
SECTORS: dict[str, tuple[float, float]] = {
    "S": (150.0, 200.0),
    "SW": (200.0, 245.0),
    "W": (245.0, 300.0),
}


def sector(direction_deg: float | None) -> str | None:
    """Map a from-direction in degrees to a coarse sector label."""
    if direction_deg is None:
        return None
    for name, (lo, hi) in SECTORS.items():
        if lo <= direction_deg < hi:
            return name
    return "other"


MOLLENDO = Spot(
    name="Mollendo",
    lat=-17.0231,
    lon=-72.0149,
    timezone="America/Lima",
    tide_slug="Mollendo",
    coast_facing_deg=225.0,
    # User hypothesis (to be calibrated): strong S + secondary W -> lefts, and vice versa.
    interaction_rules={
        ("S", "W"): "lefts",
        ("S", "SW"): "lefts",
        ("W", "S"): "rights",
        ("SW", "S"): "rights",
    },
)
