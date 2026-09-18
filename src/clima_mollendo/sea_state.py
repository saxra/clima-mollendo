"""Individual-wave statistics: Rayleigh theory and spectral simulation of a multi-swell sea."""

from dataclasses import dataclass

import numpy as np
import polars as pl

G = 9.81


@dataclass(frozen=True)
class Partition:
    """One wave system: significant height (m), mean period (s), from-direction (deg)."""

    hs: float
    tm: float
    direction: float
    kind: str = "swell"  # "swell" or "wind"

    @property
    def tp(self) -> float:
        """Peak period estimated from the mean period (narrow swell vs broad wind sea)."""
        return self.tm * (1.15 if self.kind == "swell" else 1.29)

    @property
    def gamma(self) -> float:
        """JONSWAP peak-enhancement: swells are much narrower than wind seas."""
        return 7.0 if self.kind == "swell" else 3.3


# ---------- Rayleigh (closed form) ----------


def prob_exceed(hs: float, h: float) -> float:
    """P(H > h) for individual waves in a narrow-band sea with significant height hs."""
    return float(np.exp(-2.0 * (h / hs) ** 2)) if hs > 0 else 0.0


def rayleigh_stats(hs: float, tm: float, duration_s: float) -> dict[str, float]:
    """Expected individual-wave heights for a session of `duration_s` seconds."""
    n = max(duration_s / tm, 1.0)
    return {
        "n_waves": n,
        "h_mean": 0.63 * hs,
        "h_1_10": 1.27 * hs,
        "h_1_100": 1.67 * hs,
        "h_max": float(hs * np.sqrt(np.log(n) / 2.0)),
    }


def combined_hs(partitions: list[Partition]) -> float:
    """Total significant height: energies add, not heights."""
    return float(np.sqrt(sum(p.hs**2 for p in partitions)))


def beat_period(t1: float, t2: float) -> float:
    """Period of the set envelope produced by two swells of periods t1 and t2 (s)."""
    return float("inf") if t1 == t2 else 1.0 / abs(1.0 / t1 - 1.0 / t2)


# ---------- Spectral simulation ----------


def jonswap(f: np.ndarray, hs: float, tp: float, gamma: float) -> np.ndarray:
    """JONSWAP variance density (m^2/Hz) scaled so that 4*sqrt(m0) == hs."""
    fp = 1.0 / tp
    sigma = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp) ** 2) / (2.0 * sigma**2 * fp**2))
    shape = f**-5 * np.exp(-1.25 * (fp / f) ** 4) * gamma**r
    m0 = np.trapezoid(shape, f)
    return shape * (hs**2 / 16.0) / m0


def simulate_surface(
    partitions: list[Partition],
    duration_s: float = 7200.0,
    dt: float = 0.5,
    n_freq: int = 400,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear superposition of random-phase components from each partition's spectrum."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt)
    eta = np.zeros_like(t)
    for p in partitions:
        if p.hs <= 0:
            continue
        f = np.linspace(0.3 / p.tp, 3.0 / p.tp, n_freq)
        df = f[1] - f[0]
        amp = np.sqrt(2.0 * jonswap(f, p.hs, p.tp, p.gamma) * df)
        phase = rng.uniform(0.0, 2.0 * np.pi, n_freq)
        eta += (amp[:, None] * np.cos(2.0 * np.pi * f[:, None] * t[None, :] + phase[:, None])).sum(
            0
        )
    return t, eta


def zero_crossing_waves(t: np.ndarray, eta: np.ndarray) -> pl.DataFrame:
    """Split a surface record into individual waves (zero up-crossing method)."""
    up = np.flatnonzero((eta[:-1] < 0) & (eta[1:] >= 0)) + 1
    if len(up) < 2:
        return pl.DataFrame({"t_start": [], "height": [], "period": []})
    starts, ends = up[:-1], up[1:]
    height = np.array([eta[s:e].max() - eta[s:e].min() for s, e in zip(starts, ends, strict=True)])
    return pl.DataFrame({"t_start": t[starts], "height": height, "period": t[ends] - t[starts]})


def detect_sets(waves: pl.DataFrame, hs: float) -> pl.DataFrame:
    """Group consecutive waves above hs into sets; returns one row per set."""
    if waves.is_empty():
        return pl.DataFrame({"set_id": [], "n_waves": [], "h_max": [], "t_start": []})
    flagged = waves.with_columns(big=pl.col("height") > hs)
    flagged = flagged.with_columns(
        set_id=(pl.col("big") & ~pl.col("big").shift(1, fill_value=False)).cum_sum()
    )
    return (
        flagged.filter("big")
        .group_by("set_id")
        .agg(
            pl.len().alias("n_waves"),
            pl.col("height").max().alias("h_max"),
            pl.col("t_start").min(),
        )
        .sort("t_start")
    )


def wave_distribution(
    partitions: list[Partition],
    duration_s: float = 7200.0,
    thresholds: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
    seed: int | None = 0,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Simulate the sea and return every individual wave plus a summary of extremes."""
    t, eta = simulate_surface(partitions, duration_s=duration_s, seed=seed)
    hs_sim = float(4.0 * eta.std())
    # Ripples from the wind sea create spurious zero crossings; drop waves below 0.2 Hs.
    waves = zero_crossing_waves(t, eta).filter(pl.col("height") > 0.2 * hs_sim)
    heights = waves["height"]
    n = len(waves)
    top_third = heights.sort(descending=True).head(max(n // 3, 1)).mean() if n else 0.0
    sets = detect_sets(waves, hs_sim)
    summary = {
        "hs_input": combined_hs(partitions),
        "hs_sim": hs_sim,
        "n_waves": float(n),
        "h_mean": float(heights.mean() or 0.0),
        "h_1_3": float(top_third),
        "h_1_10": float(heights.sort(descending=True).head(max(n // 10, 1)).mean() or 0.0),
        "h_max": float(heights.max() or 0.0),
        "t_mean": float(waves["period"].mean() or 0.0),
        "t_p10": float(waves["period"].quantile(0.1) or 0.0),
        "t_p90": float(waves["period"].quantile(0.9) or 0.0),
        "n_sets": float(len(sets)),
        "set_interval_s": float(sets["t_start"].diff().mean() or 0.0) if len(sets) > 1 else 0.0,
    }
    for h in thresholds:
        summary[f"p_over_{h:.1f}"] = float((heights > h).mean() or 0.0) if n else 0.0
        summary[f"n_over_{h:.1f}"] = float((heights > h).sum()) if n else 0.0
    return waves, summary


def partitions_from_row(row: dict) -> list[Partition]:
    """Build partitions from one hourly marine row (s1..s3 swells + wind sea)."""
    parts = []
    for i in (1, 2, 3):
        h, t, d = row.get(f"s{i}_h"), row.get(f"s{i}_t"), row.get(f"s{i}_d")
        if h and t and h > 0:
            parts.append(Partition(h, t, d, "swell"))
    h, t, d = row.get("ww_h"), row.get("ww_t"), row.get("ww_d")
    if h and t and h > 0.05:
        parts.append(Partition(h, t, d, "wind"))
    return parts
