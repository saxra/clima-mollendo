"""Download hourly weather, marine partitions and tide extremes for a spot."""

import re
from datetime import date, datetime
from pathlib import Path

import polars as pl
import requests

from clima_mollendo.spot import Spot

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
TIDE_URL = "https://www.tide-forecast.com/locations/{slug}/tides/latest"

WEATHER_RENAME = {
    "temperature_2m": "temp",
    "cloud_cover": "cloud",
    "precipitation_probability": "rain_prob",
    "wind_speed_10m": "wind_kn",
    "wind_direction_10m": "wind_dir",
    "wind_gusts_10m": "gust_kn",
}
MARINE_RENAME = {
    "wave_height": "hs",
    "wave_period": "tm",
    "wave_direction": "dir",
    "swell_wave_height": "s1_h",
    "swell_wave_period": "s1_t",
    "swell_wave_direction": "s1_d",
    "secondary_swell_wave_height": "s2_h",
    "secondary_swell_wave_period": "s2_t",
    "secondary_swell_wave_direction": "s2_d",
    "tertiary_swell_wave_height": "s3_h",
    "tertiary_swell_wave_period": "s3_t",
    "tertiary_swell_wave_direction": "s3_d",
    "wind_wave_height": "ww_h",
    "wind_wave_period": "ww_t",
    "wind_wave_direction": "ww_d",
    "sea_surface_temperature": "sst",
}


def _hourly_frame(payload: dict, rename: dict[str, str]) -> pl.DataFrame:
    df = pl.DataFrame(payload["hourly"]).rename(rename)
    return df.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M"))


def fetch_weather(spot: Spot, days: int = 3) -> pl.DataFrame:
    """Hourly atmospheric forecast; wind in knots, directions in degrees (from)."""
    params = {
        "latitude": spot.lat,
        "longitude": spot.lon,
        "hourly": ",".join(WEATHER_RENAME),
        "timezone": spot.timezone,
        "forecast_days": days,
        "wind_speed_unit": "kn",
    }
    payload = requests.get(WEATHER_URL, params=params, timeout=30).json()
    return _hourly_frame(payload, WEATHER_RENAME)


def fetch_marine(spot: Spot, days: int = 3) -> pl.DataFrame:
    """Hourly sea state with up to three swell partitions plus wind sea."""
    params = {
        "latitude": spot.lat,
        "longitude": spot.lon,
        "hourly": ",".join(MARINE_RENAME),
        "timezone": spot.timezone,
        "forecast_days": days,
    }
    payload = requests.get(MARINE_URL, params=params, timeout=30).json()
    return _hourly_frame(payload, MARINE_RENAME)


EMPTY_TIDES = pl.DataFrame(
    schema={"time": pl.Datetime("us"), "kind": pl.String, "height_m": pl.Float64}
)

_TIDE_ROW = re.compile(
    r"<td>(High|Low) Tide</td><td><b>\s*([\d:]+ [AP]M)</b>"
    r'<span class="tide-day-tides__secondary">\(([^)]+)\)</span></td>'
    r"<td[^>]*><b[^>]*>([\d.]+) m</b>"
)
_TIDE_YEAR = re.compile(r"tide times today on \w+ \d+ \w+ (\d{4})")


def parse_tides(html: str) -> pl.DataFrame:
    """Parse tide-forecast.com daily tables into (time, kind, height_m)."""
    year = int(_TIDE_YEAR.search(html).group(1))
    rows, seen = [], set()
    prev_month = None
    for kind, clock, day, height in _TIDE_ROW.findall(html):
        key = (kind, clock, day)
        if key in seen:
            continue
        seen.add(key)
        _, dom, month = day.split()
        clock = clock.replace("00:", "12:", 1)  # site writes 00:54 AM for 12:54 AM
        stamp = datetime.strptime(f"{dom} {month} {year} {clock}", "%d %B %Y %I:%M %p")
        if prev_month is not None and stamp.month < prev_month:
            year += 1
            stamp = stamp.replace(year=year)
        prev_month = stamp.month
        rows.append({"time": stamp, "kind": kind.lower(), "height_m": float(height)})
    return pl.DataFrame(rows).sort("time")


def fetch_tides(spot: Spot) -> pl.DataFrame:
    """Tide extremes (high/low) for the next ~30 days from tide-forecast.com."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    html = requests.get(TIDE_URL.format(slug=spot.tide_slug), headers=headers, timeout=30).text
    return parse_tides(html)


def track_swells(marine: pl.DataFrame, max_cost: float = 1.2, max_gap_h: int = 6) -> pl.DataFrame:
    """Reassign swell partitions to persistent tracks so a swell keeps its id across hours.

    Open-Meteo ranks partitions by energy each hour, so the same swell can jump between
    s1/s2/s3. Partitions are matched hour to hour by closeness in period and direction;
    a track not seen for `max_gap_h` hours is retired.
    Returns a long frame: time, rank, h, t, d, track.
    """
    long = (
        marine.select(
            "time",
            *[pl.struct(h=f"s{i}_h", t=f"s{i}_t", d=f"s{i}_d").alias(f"s{i}") for i in (1, 2, 3)],
        )
        .unpivot(index="time", variable_name="rank", value_name="p")
        .unnest("p")
        .drop_nulls("h")
        .with_columns(pl.col("rank").str.strip_prefix("s").cast(pl.Int8))
        .sort("time", "rank")
    )
    tracks: dict[int, tuple[float, float, datetime]] = {}
    next_id = 1
    assigned: list[int] = []
    for (stamp,), group in long.group_by("time", maintain_order=True):
        tracks = {
            k: v for k, v in tracks.items() if (stamp - v[2]).total_seconds() <= max_gap_h * 3600
        }
        used: set[int] = set()
        for t, d in zip(group["t"], group["d"], strict=True):
            best, best_cost = None, max_cost
            for tid, (pt, pd_, _) in tracks.items():
                if tid in used:
                    continue
                cost = abs(t - pt) / 2.5 + abs(((d - pd_) + 180) % 360 - 180) / 30.0
                if cost < best_cost:
                    best, best_cost = tid, cost
            if best is None:
                best = next_id
                next_id += 1
            tracks[best] = (t, d, stamp)
            used.add(best)
            assigned.append(best)
    return long.with_columns(pl.Series("track", assigned, dtype=pl.Int16))


def save_raw(df: pl.DataFrame, name: str, root: Path = Path("data/raw")) -> Path:
    """Write a Parquet snapshot stamped with today's date."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{date.today():%Y%m%d}.parquet"
    df.write_parquet(path)
    return path
