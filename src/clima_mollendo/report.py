"""Interactive HTML report: swell tracks, wind, tide and per-hour individual-wave distributions."""

from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots

from clima_mollendo.sea_state import partitions_from_row, wave_distribution
from clima_mollendo.spot import Spot

PLOTLY_CDN = (
    f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{get_plotlyjs_version()}/plotly.min.js"
)
TRACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
DAY_HOURS = range(6, 19)


def tide_curve(tides: pl.DataFrame, times: pl.Series) -> pl.Series:
    """Cosine interpolation of tide height between consecutive extremes."""
    q = times.to_numpy().astype("datetime64[s]").astype(np.int64)
    out = np.full(len(q), np.nan)
    if tides.is_empty():
        return pl.Series("tide_m", out)
    ext_t = tides["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    ext_h = tides["height_m"].to_numpy()
    for i in range(len(ext_t) - 1):
        t1, t2, h1, h2 = ext_t[i], ext_t[i + 1], ext_h[i], ext_h[i + 1]
        mask = (q >= t1) & (q <= t2)
        out[mask] = (h1 + h2) / 2 + (h1 - h2) / 2 * np.cos(np.pi * (q[mask] - t1) / (t2 - t1))
    return pl.Series("tide_m", out)


def _fmt_swell(h: float | None, t: float | None, d: float | None, track: int | None) -> str:
    if h is None:
        return "—"
    return f"{h:.2f} m · {t:.1f} s · {d:.0f}° (#{track})"


def hourly_table(
    hourly: pl.DataFrame, tracks: pl.DataFrame, summaries: dict[datetime, dict]
) -> pl.DataFrame:
    """Daylight rows with top-3 swells, wind, tide and extreme-wave stats."""
    top = (
        tracks.sort("time", "h", descending=[False, True])
        .with_columns(pos=pl.int_range(pl.len()).over("time") + 1)
        .filter(pl.col("pos") <= 3)
        .with_columns(
            label=pl.struct("h", "t", "d", "track").map_elements(
                lambda s: _fmt_swell(s["h"], s["t"], s["d"], s["track"]), return_dtype=pl.String
            )
        )
        .pivot(on="pos", index="time", values="label")
        .rename({"1": "swell_1", "2": "swell_2", "3": "swell_3"}, strict=False)
    )
    for col in ("swell_1", "swell_2", "swell_3"):
        if col not in top.columns:
            top = top.with_columns(pl.lit("—").alias(col))
    stats = pl.DataFrame(
        [
            {"time": k, "h_1_10": v["h_1_10"], "h_max": v["h_max"], "p_over_1.5": v["p_over_1.5"]}
            for k, v in summaries.items()
        ]
    )
    return (
        hourly.join(top, on="time", how="left")
        .join(stats, on="time", how="left")
        .filter(pl.col("time").dt.hour().is_in(list(DAY_HOURS)))
        .select(
            "time",
            "hs",
            "swell_1",
            "swell_2",
            "swell_3",
            "wind_kn",
            "wind_dir",
            "gust_kn",
            "tide_m",
            "h_1_10",
            "h_max",
            "p_over_1.5",
            "temp",
            "cloud",
            "rain_prob",
        )
    )


def overview_figure(hourly: pl.DataFrame, tracks: pl.DataFrame, tides: pl.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Altura de swells (m)",
            "Periodo (s)",
            "Dirección de origen (°)",
            "Viento (kn)",
            "Marea (m)",
        ),
    )
    for tid, grp in tracks.group_by("track", maintain_order=True):
        tid = tid[0]
        color = TRACK_COLORS[(tid - 1) % len(TRACK_COLORS)]
        if len(grp) < 3:  # transient partition from the model, shown but not tracked as a swell
            fig.add_trace(
                go.Scatter(
                    x=grp["time"],
                    y=grp["h"],
                    mode="markers",
                    name="transitorio",
                    marker={"color": "lightgray", "symbol": "diamond"},
                    showlegend=False,
                    text=[
                        f"{h:.2f} m · {t:.1f} s · {d:.0f}°"
                        for h, t, d in zip(grp["h"], grp["t"], grp["d"], strict=True)
                    ],
                    hovertemplate="%{text}",
                ),
                1,
                1,
            )
            continue
        hover = [
            f"{h:.2f} m · {t:.1f} s · {d:.0f}°"
            for h, t, d in zip(grp["h"], grp["t"], grp["d"], strict=True)
        ]
        common = {
            "x": grp["time"],
            "legendgroup": f"s{tid}",
            "line": {"color": color},
            "text": hover,
        }
        fig.add_trace(
            go.Scatter(y=grp["h"], name=f"Swell #{tid}", hovertemplate="%{text}", **common), 1, 1
        )
        fig.add_trace(
            go.Scatter(y=grp["t"], showlegend=False, hovertemplate="%{text}", **common), 2, 1
        )
        fig.add_trace(
            go.Scatter(
                y=grp["d"],
                showlegend=False,
                hovertemplate="%{text}",
                mode="markers",
                x=grp["time"],
                legendgroup=f"s{tid}",
                marker={"color": color},
                text=hover,
            ),
            3,
            1,
        )
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["hs"],
            name="Hs total",
            line={"color": "black", "dash": "dot"},
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["wind_dir"],
            name="Viento dir",
            mode="markers",
            marker={"color": "gray", "symbol": "x"},
        ),
        3,
        1,
    )
    fig.add_trace(
        go.Scatter(x=hourly["time"], y=hourly["wind_kn"], name="Viento", line={"color": "gray"}),
        4,
        1,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["gust_kn"],
            name="Ráfagas",
            line={"color": "gray", "dash": "dot"},
        ),
        4,
        1,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["tide_m"],
            name="Marea",
            line={"color": "#0077b6"},
            fill="tozeroy",
        ),
        5,
        1,
    )
    window = tides.filter(pl.col("time").is_between(hourly["time"].min(), hourly["time"].max()))
    fig.add_trace(
        go.Scatter(
            x=window["time"],
            y=window["height_m"],
            mode="markers+text",
            name="Alta/baja",
            text=[
                f"{k[0].upper()} {h:.2f}"
                for k, h in zip(window["kind"], window["height_m"], strict=True)
            ],
            textposition="top center",
            marker={"color": "#0077b6", "size": 8},
        ),
        5,
        1,
    )
    fig.update_yaxes(range=[0, 360], dtick=90, row=3, col=1)
    fig.update_layout(height=1100, hovermode="x unified", legend={"orientation": "h", "y": -0.03})
    return fig


def distribution_figure(sims: dict[datetime, tuple[pl.DataFrame, dict]]) -> go.Figure:
    """Histograms of individual wave height and period per hour, switched with a dropdown."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Altura de olas individuales (m)",
            "Periodo de olas individuales (s)",
            "Altura vs periodo",
        ),
    )
    keys = list(sims)
    for i, key in enumerate(keys):
        waves, s = sims[key]
        vis = i == 0
        fig.add_trace(
            go.Histogram(
                x=waves["height"],
                xbins={"size": 0.1},
                name="H",
                marker_color="#1f77b4",
                visible=vis,
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Histogram(
                x=waves["period"],
                xbins={"size": 1.0},
                name="T",
                marker_color="#2ca02c",
                visible=vis,
            ),
            1,
            2,
        )
        fig.add_trace(
            go.Scatter(
                x=waves["period"],
                y=waves["height"],
                mode="markers",
                name="olas",
                marker={"size": 4, "opacity": 0.4, "color": "#d62728"},
                visible=vis,
            ),
            1,
            3,
        )
    n_tr = 3
    buttons = []
    for i, key in enumerate(keys):
        s = sims[key][1]
        visible = [False] * (n_tr * len(keys))
        for j in range(n_tr):
            visible[i * n_tr + j] = True
        title = (
            f"{key:%a %d %H:%M} — Hs {s['hs_sim']:.2f} m · H1/10 {s['h_1_10']:.2f} m · "
            f"Hmax {s['h_max']:.2f} m · P(H>1.5) {s['p_over_1.5'] * 100:.1f}% · "
            f"sets cada {s['set_interval_s']:.0f} s · T {s['t_p10']:.0f}–{s['t_p90']:.0f} s"
        )
        buttons.append(
            {
                "label": f"{key:%a %d %H:%M}",
                "method": "update",
                "args": [{"visible": visible}, {"title": title}],
            }
        )
    fig.update_layout(
        height=450,
        showlegend=False,
        title=buttons[0]["args"][1]["title"] if buttons else "",
        updatemenus=[
            {"buttons": buttons, "direction": "down", "x": 0, "y": 1.25, "xanchor": "left"}
        ],
    )
    fig.update_xaxes(title_text="m", row=1, col=1)
    fig.update_xaxes(title_text="s", row=1, col=2)
    fig.update_xaxes(title_text="periodo (s)", row=1, col=3)
    fig.update_yaxes(title_text="altura (m)", row=1, col=3)
    return fig


def _html_table(df: pl.DataFrame, title: str, formats: dict[str, str] | None = None) -> str:
    """Plain HTML table; wraps long cells and starts a new day block visually."""
    formats = formats or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for row in df.iter_rows(named=True):
        cells = []
        for col, v in row.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                text = "—"
            elif col == "time":
                text = v.strftime("%a %d %H:%M")
            elif col in formats:
                text = formats[col].format(v)
            else:
                text = str(v)
            cells.append(f"<td>{text}</td>")
        cls = ' class="day-start"' if row.get("time") and row["time"].hour == DAY_HOURS[0] else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    body = "".join(rows)
    return f"<h2>{title}</h2><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def simulate_hours(
    hourly: pl.DataFrame, duration_s: float = 7200.0
) -> dict[datetime, tuple[pl.DataFrame, dict]]:
    """Run the sea simulation for each daylight hour."""
    sims = {}
    for row in hourly.filter(pl.col("time").dt.hour().is_in(list(DAY_HOURS))).iter_rows(named=True):
        parts = partitions_from_row(row)
        if parts:
            sims[row["time"]] = wave_distribution(
                parts, duration_s=duration_s, seed=hash(row["time"]) & 0xFFFF
            )
    return sims


def build_report(
    spot: Spot,
    hourly: pl.DataFrame,
    tracks: pl.DataFrame,
    tides: pl.DataFrame,
    interactions: pl.DataFrame,
    out: Path,
) -> Path:
    """Assemble all figures into one self-contained HTML file."""
    hourly = hourly.with_columns(tide_curve(tides, hourly["time"]))
    sims = simulate_hours(hourly)
    summaries = {k: v[1] for k, v in sims.items()}
    table = hourly_table(hourly, tracks, summaries)
    fmt = {
        "hs": "{:.2f}",
        "wind_kn": "{:.0f}",
        "wind_dir": "{:.0f}°",
        "gust_kn": "{:.0f}",
        "tide_m": "{:.2f}",
        "h_1_10": "{:.2f}",
        "h_max": "{:.2f}",
        "p_over_1.5": "{:.1%}",
        "temp": "{:.0f}",
        "cloud": "{:.0f}",
        "rain_prob": "{:.0f}",
    }
    ifmt = {"energy_ratio": "{:.2f}", "beat_period_s": "{:.0f}", "rel_angle_deg": "{:.0f}°"}
    intro = (
        f"<h1>{spot.name} — pronóstico de surf</h1>"
        f"<p>Generado {datetime.now():%Y-%m-%d %H:%M}. Fuentes: Open-Meteo (atmósfera + mar, "
        "3 particiones de swell), tide-forecast.com (marea). Direcciones en grados: de dónde "
        "viene. Sesión simulada: 2 h.</p>"
    )
    dist_intro = (
        "<h2>Distribución de olas individuales (elige la hora)</h2>"
        "<p>Simulación espectral (JONSWAP por partición, fases aleatorias, cruce por cero). "
        "Hs es el promedio del tercio más alto; las olas grandes de un set superan H1/10 y "
        "Hmax es la mayor esperada en la sesión.</p>"
    )
    day_interactions = interactions.filter(pl.col("time").dt.hour().is_in(list(DAY_HOURS)))
    parts = [
        intro,
        overview_figure(hourly, tracks, tides).to_html(full_html=False, include_plotlyjs=False),
        _html_table(table, "Tabla horaria (horas de luz)", fmt),
        dist_intro,
        distribution_figure(sims).to_html(full_html=False, include_plotlyjs=False),
        _html_table(
            day_interactions, "Interacción de swells (factores para calibrar el spot)", ifmt
        ),
    ]
    style = (
        "body{font-family:sans-serif;max-width:1400px;margin:auto;padding:0 16px}"
        "table{border-collapse:collapse;font-size:13px;width:100%}"
        "th,td{border:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}"
        "th{background:#f0f0f0;position:sticky;top:0}"
        "tr.day-start td{border-top:3px solid #0077b6}"
    )
    html = (
        f"<html><head><meta charset='utf-8'><title>{spot.name} surf</title>"
        f"<script src='{PLOTLY_CDN}'></script><style>{style}</style></head><body>"
        + "\n".join(parts)
        + "</body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
