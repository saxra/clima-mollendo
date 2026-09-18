"""Green -> yellow -> red scales for surf conditions. Thresholds are per variable."""

from dataclasses import dataclass

GREEN = (46, 204, 113)
YELLOW = (241, 196, 15)
RED = (231, 76, 60)


@dataclass(frozen=True)
class Scale:
    """Value at which the colour is green, yellow and red (linear in between)."""

    lo: float
    mid: float
    hi: float
    unit: str = ""

    def frac(self, value: float | None) -> float | None:
        """Position on the scale in [0, 1]; 0.5 is yellow."""
        if value is None:
            return None
        if value <= self.lo:
            return 0.0
        if value >= self.hi:
            return 1.0
        if value <= self.mid:
            return 0.5 * (value - self.lo) / (self.mid - self.lo)
        return 0.5 + 0.5 * (value - self.mid) / (self.hi - self.mid)

    def rgb(self, value: float | None) -> tuple[int, int, int] | None:
        f = self.frac(value)
        if f is None:
            return None
        a, b, t = (GREEN, YELLOW, f * 2) if f <= 0.5 else (YELLOW, RED, (f - 0.5) * 2)
        return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def hex(self, value: float | None, default: str = "#ffffff") -> str:
        rgb = self.rgb(value)
        return default if rgb is None else "#{:02x}{:02x}{:02x}".format(*rgb)

    def css(self, value: float | None) -> str:
        """Inline background style for a table cell (light tint so text stays readable)."""
        rgb = self.rgb(value)
        return "" if rgb is None else f'style="background:rgba({rgb[0]},{rgb[1]},{rgb[2]},0.45)"'

    def label(self, value: float | None) -> str:
        f = self.frac(value)
        if f is None:
            return "—"
        return "chico" if f < 0.34 else "medio" if f < 0.67 else "grande"


# Default scales for a Peruvian beach break. Tune per spot as observations come in.
SCALES: dict[str, Scale] = {
    "hs": Scale(0.5, 1.5, 2.5, "m"),
    "swell_h": Scale(0.3, 1.2, 2.2, "m"),
    "h_1_10": Scale(0.8, 1.8, 3.0, "m"),
    "h_max": Scale(1.0, 2.2, 3.5, "m"),
    "period": Scale(6.0, 11.0, 16.0, "s"),
    "wind_kn": Scale(3.0, 10.0, 18.0, "kn"),
    "gust_kn": Scale(5.0, 15.0, 25.0, "kn"),
    "rain_prob": Scale(10.0, 50.0, 90.0, "%"),
    "p_over_1.5": Scale(0.0, 0.05, 0.2, ""),
}

PLOTLY_SCALE = [[0.0, "#2ecc71"], [0.5, "#f1c40f"], [1.0, "#e74c3c"]]
