"""Command line entry point: fetch data, save snapshots and build the HTML report."""

import argparse
from datetime import date
from pathlib import Path

from clima_mollendo.fetch import fetch_marine, fetch_tides, fetch_weather, save_raw, track_swells
from clima_mollendo.interactions import swell_interactions
from clima_mollendo.report import build_report
from clima_mollendo.spot import MOLLENDO


def main() -> None:
    parser = argparse.ArgumentParser(description="Surf forecast report for Mollendo")
    parser.add_argument("--days", type=int, default=3, help="forecast days (1-7)")
    parser.add_argument("--out", type=Path, default=None, help="output HTML path")
    args = parser.parse_args()

    spot = MOLLENDO
    weather = fetch_weather(spot, args.days)
    marine = fetch_marine(spot, args.days)
    tides = fetch_tides(spot)
    hourly = weather.join(marine, on="time", how="inner")
    tracks = track_swells(marine)
    interactions = swell_interactions(marine, spot)

    for df, name in ((hourly, "hourly"), (tides, "tides"), (tracks, "swell_tracks")):
        save_raw(df, name)

    out = args.out or Path("reports") / f"{spot.name.lower()}_{date.today():%Y%m%d}.html"
    path = build_report(spot, hourly, tracks, tides, interactions, out)
    print(f"Report written to {path}")
