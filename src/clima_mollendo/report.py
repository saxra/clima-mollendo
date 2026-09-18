"""Interactive HTML report: swell tracks, wind, tide and per-hour individual-wave distributions."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots

from clima_mollendo.colors import PLOTLY_SCALE, SCALES, Scale
from clima_mollendo.sea_state import partitions_from_row, wave_distribution
from clima_mollendo.spot import Spot

PLOTLY_CDN = (
    f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{get_plotlyjs_version()}/plotly.min.js"
)
TRACK_COLORS = ["#1f77b4", "#8e44ad", "#16a085", "#d35400", "#7f8c8d", "#8c564b", "#17becf"]
DAY_HOURS = range(6, 19)
DAYS_ES = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
DAYS_ES_LONG = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def _stamp(t: datetime) -> str:
    return f"{DAYS_ES[t.weekday()]} {t:%d %H:%M}"


SESSIONS = {"Mañana": range(6, 10), "Mediodía": range(10, 14), "Late": range(14, 19)}


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
    ranked = (
        tracks.sort("time", "h", descending=[False, True])
        .with_columns(pos=pl.int_range(pl.len()).over("time") + 1)
        .filter(pl.col("pos") <= 3)
        .with_columns(
            label=pl.struct("h", "t", "d", "track").map_elements(
                lambda s: _fmt_swell(s["h"], s["t"], s["d"], s["track"]), return_dtype=pl.String
            )
        )
    )
    labels = ranked.pivot(on="pos", index="time", values="label").rename(
        {"1": "swell_1", "2": "swell_2", "3": "swell_3"}, strict=False
    )
    heights = ranked.pivot(on="pos", index="time", values="h").rename(
        {"1": "swell_1_h", "2": "swell_2_h", "3": "swell_3_h"}, strict=False
    )
    top = labels.join(heights, on="time")
    for i in (1, 2, 3):
        if f"swell_{i}" not in top.columns:
            top = top.with_columns(
                pl.lit("—").alias(f"swell_{i}"), pl.lit(None, pl.Float64).alias(f"swell_{i}_h")
            )
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
            "h_1_10",
            "h_max",
            "p_over_1.5",
            "wind_kmh",
            "wind_dir",
            "gust_kmh",
            "tide_m",
            "temp",
            "cloud",
            "rain_prob",
            "swell_1_h",
            "swell_2_h",
            "swell_3_h",
        )
    )


def _add_bands(fig: go.Figure, scale: Scale, row: int, top: float) -> None:
    """Green / yellow / red horizontal bands behind a subplot."""
    edges = [0.0, (scale.lo + scale.mid) / 2, (scale.mid + scale.hi) / 2, max(top, scale.hi)]
    for (y0, y1), color in zip(
        zip(edges[:-1], edges[1:], strict=True), ["#2ecc71", "#f1c40f", "#e74c3c"], strict=True
    ):
        fig.add_hrect(
            y0=y0, y1=y1, fillcolor=color, opacity=0.12, line_width=0, layer="below", row=row, col=1
        )


def _shade_nights(fig: go.Figure, hourly: pl.DataFrame, rows: int) -> None:
    start, end = hourly["time"].min(), hourly["time"].max()
    day = start.replace(hour=0, minute=0)
    while day <= end:
        for r in range(1, rows + 1):
            fig.add_vrect(
                x0=day,
                x1=day + timedelta(hours=DAY_HOURS[0]),
                fillcolor="#2c3e50",
                opacity=0.06,
                line_width=0,
                layer="below",
                row=r,
                col=1,
            )
            fig.add_vrect(
                x0=day + timedelta(hours=DAY_HOURS[-1] + 1),
                x1=day + timedelta(days=1),
                fillcolor="#2c3e50",
                opacity=0.06,
                line_width=0,
                layer="below",
                row=r,
                col=1,
            )
        day += timedelta(days=1)


def overview_figure(
    hourly: pl.DataFrame, tracks: pl.DataFrame, tides: pl.DataFrame, summaries: dict[datetime, dict]
) -> go.Figure:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.3, 0.17, 0.17, 0.18, 0.18],
        subplot_titles=(
            "Altura (m): swells, Hs total y Hmax esperada — fondo = chico/medio/grande",
            "Periodo (s)",
            "Dirección de origen (°)",
            "Viento (km/h) — fondo = flojo/medio/fuerte",
            "Marea (m)",
        ),
    )
    hs_scale, wind_scale = SCALES["hs"], SCALES["wind_kmh"]
    h_max = [summaries[t]["h_max"] if t in summaries else None for t in hourly["time"]]
    top_h = max([v for v in h_max if v is not None] + [float(hourly["hs"].max())]) * 1.15
    for tid, grp in tracks.group_by("track", maintain_order=True):
        tid = tid[0]
        color = TRACK_COLORS[(tid - 1) % len(TRACK_COLORS)]
        hover = [
            f"{h:.2f} m · {t:.1f} s · {d:.0f}°"
            for h, t, d in zip(grp["h"], grp["t"], grp["d"], strict=True)
        ]
        if len(grp) < 3:  # transient partition from the model, shown but not tracked as a swell
            fig.add_trace(
                go.Scatter(
                    x=grp["time"],
                    y=grp["h"],
                    mode="markers",
                    name="transitorio",
                    marker={"color": "lightgray", "symbol": "diamond"},
                    showlegend=False,
                    text=hover,
                    hovertemplate="%{text}",
                ),
                1,
                1,
            )
            continue
        common = {"x": grp["time"], "legendgroup": f"s{tid}", "text": hover}
        fig.add_trace(
            go.Scatter(
                y=grp["h"],
                name=f"Swell #{tid}",
                hovertemplate="%{text}",
                line={"color": color, "width": 2},
                **common,
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Scatter(
                y=grp["t"],
                showlegend=False,
                hovertemplate="%{text}",
                line={"color": color, "width": 2},
                **common,
            ),
            2,
            1,
        )
        fig.add_trace(
            go.Scatter(
                y=grp["d"],
                showlegend=False,
                hovertemplate="%{text}",
                mode="markers",
                marker={"color": color, "size": 6},
                **common,
            ),
            3,
            1,
        )

    marker_scale = {
        "colorscale": PLOTLY_SCALE,
        "cmin": hs_scale.lo,
        "cmax": hs_scale.hi,
        "size": 9,
        "line": {"color": "white", "width": 1},
    }
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["hs"],
            name="Hs total",
            mode="lines+markers",
            line={"color": "#2c3e50", "width": 3},
            marker={"color": hourly["hs"], **marker_scale},
            hovertemplate="Hs %{y:.2f} m",
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=h_max,
            name="Hmax esperada (sesión 2 h)",
            mode="markers",
            marker={"symbol": "triangle-up", "size": 9, "color": "#c0392b"},
            hovertemplate="Hmax %{y:.2f} m",
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
        go.Scatter(
            x=hourly["time"],
            y=hourly["wind_kmh"],
            name="Viento",
            mode="lines+markers",
            line={"color": "#2c3e50", "width": 2},
            marker={
                "color": hourly["wind_kmh"],
                "colorscale": PLOTLY_SCALE,
                "cmin": wind_scale.lo,
                "cmax": wind_scale.hi,
                "size": 7,
            },
            hovertemplate="%{y:.0f} km/h",
        ),
        4,
        1,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["gust_kmh"],
            name="Ráfagas",
            line={"color": "gray", "dash": "dot"},
            hovertemplate="ráfaga %{y:.0f} km/h",
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
            fillcolor="rgba(0,119,182,0.25)",
            hovertemplate="marea %{y:.2f} m",
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
                f"{'▲' if k == 'high' else '▼'} {h:.2f}"
                for k, h in zip(window["kind"], window["height_m"], strict=True)
            ],
            textposition="top center",
            marker={"color": "#0077b6", "size": 9},
            hovertemplate="%{text} m · %{x|%H:%M}",
        ),
        5,
        1,
    )
    # Shapes must be added after the traces: add_hrect/add_vrect skip subplots with no data.
    _add_bands(fig, hs_scale, 1, top_h)
    _add_bands(fig, wind_scale, 4, float(hourly["gust_kmh"].max()) * 1.15)
    _shade_nights(fig, hourly, 5)
    fig.update_yaxes(range=[0, top_h], row=1, col=1)
    fig.update_yaxes(range=[0, 360], dtick=90, row=3, col=1)
    fig.update_yaxes(range=[0, float(hourly["tide_m"].max() or 1.0) * 1.3], row=5, col=1)
    fig.update_layout(
        template="plotly_white",
        height=1250,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.03},
        margin={"l": 50, "r": 20, "t": 40, "b": 20},
    )
    return fig


def distribution_figure(sims: dict[datetime, tuple[pl.DataFrame, dict]]) -> go.Figure:
    """Coloured histograms of individual wave height and period per hour (dropdown by hour)."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Altura de olas individuales (m)",
            "Periodo de olas individuales (s)",
            "Altura vs periodo",
        ),
    )
    scale = SCALES["hs"]
    keys = list(sims)
    h_edges = np.arange(0.0, 4.05, 0.1)
    t_edges = np.arange(0.0, 25.0, 1.0)
    frames = []
    for i, key in enumerate(keys):
        waves, s = sims[key]
        vis = i == 0
        h_counts, _ = np.histogram(waves["height"].to_numpy(), bins=h_edges)
        h_centers = (h_edges[:-1] + h_edges[1:]) / 2
        t_counts, _ = np.histogram(waves["period"].to_numpy(), bins=t_edges)
        t_centers = (t_edges[:-1] + t_edges[1:]) / 2
        fig.add_trace(
            go.Bar(
                x=h_centers,
                y=h_counts,
                width=0.1,
                marker={"color": [scale.hex(c) for c in h_centers], "line": {"width": 0}},
                name="H",
                visible=vis,
                hovertemplate="%{x:.1f} m: %{y} olas",
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Bar(
                x=t_centers,
                y=t_counts,
                width=1.0,
                marker={"color": "#3498db", "line": {"width": 0}},
                name="T",
                visible=vis,
                hovertemplate="%{x:.0f} s: %{y} olas",
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
                marker={
                    "size": 5,
                    "opacity": 0.6,
                    "color": waves["height"],
                    "colorscale": PLOTLY_SCALE,
                    "cmin": scale.lo,
                    "cmax": scale.hi,
                },
                visible=vis,
                hovertemplate="%{y:.2f} m · %{x:.1f} s",
            ),
            1,
            3,
        )
        frames.append(s)
    n_tr = 3
    buttons = []

    def _lines(s: dict) -> list[dict]:
        out = []
        for value, color, dash in (
            (s["hs_sim"], "#2c3e50", "solid"),
            (s["h_1_10"], "#e67e22", "dash"),
            (s["h_max"], "#c0392b", "dot"),
        ):
            out.append(
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "paper",
                    "x0": value,
                    "x1": value,
                    "y0": 0,
                    "y1": 1,
                    "line": {"color": color, "width": 2, "dash": dash},
                }
            )
        return out

    for i, key in enumerate(keys):
        s = frames[i]
        visible = [False] * (n_tr * len(keys))
        for j in range(n_tr):
            visible[i * n_tr + j] = True
        title = (
            f"{_stamp(key)} — Hs {s['hs_sim']:.2f} m · H1/10 {s['h_1_10']:.2f} m · "
            f"Hmax {s['h_max']:.2f} m · P(H>1.5) {s['p_over_1.5'] * 100:.1f}% · "
            f"sets cada {s['set_interval_s']:.0f} s · T {s['t_p10']:.0f}–{s['t_p90']:.0f} s"
        )
        buttons.append(
            {
                "label": _stamp(key),
                "method": "update",
                "args": [{"visible": visible}, {"title": title, "shapes": _lines(s)}],
            }
        )
    fig.update_layout(
        template="plotly_white",
        height=460,
        showlegend=False,
        title={"text": buttons[0]["args"][1]["title"] if buttons else "", "x": 0, "y": 0.93},
        shapes=_lines(frames[0]) if frames else [],
        updatemenus=[
            {"buttons": buttons, "direction": "down", "x": 1, "y": 1.3, "xanchor": "right"}
        ],
        margin={"t": 110},
    )
    fig.update_xaxes(title_text="m  (líneas: Hs · H1/10 · Hmax)", row=1, col=1)
    fig.update_xaxes(title_text="s", row=1, col=2)
    fig.update_xaxes(title_text="periodo (s)", row=1, col=3)
    fig.update_yaxes(title_text="altura (m)", row=1, col=3)
    return fig


def session_cards(
    hourly: pl.DataFrame, summaries: dict[datetime, dict], interactions: pl.DataFrame
) -> str:
    """One card per day and session (mañana / mediodía / late) with colour-coded numbers."""
    stats = pl.DataFrame(
        [{"time": k, "h_max": v["h_max"], "h_1_10": v["h_1_10"]} for k, v in summaries.items()]
    )
    df = hourly.join(stats, on="time", how="left").join(
        interactions.select("time", "expected_wave"), on="time", how="left"
    )
    cards = []
    for (day,), grp in df.group_by(pl.col("time").dt.date(), maintain_order=True):
        title = f"{DAYS_ES_LONG[day.weekday()]} {day:%d/%m}"
        cards.append(f"<div class='day'><h3>{title}</h3><div class='cards'>")
        for name, hours in SESSIONS.items():
            sess = grp.filter(pl.col("time").dt.hour().is_in(list(hours)))
            if sess.is_empty():
                continue
            hs = float(sess["hs"].mean())
            hmax = sess["h_max"].max()
            wind = float(sess["wind_kmh"].mean())
            wind_dir = float(sess["wind_dir"].mean())
            gust = float(sess["gust_kmh"].max())
            rain = float(sess["rain_prob"].max())
            tide = sess["tide_m"].drop_nulls()
            tide_txt = (
                f"{tide[0]:.2f}→{tide[-1]:.2f} m"
                if len(tide) > 1
                else f"{tide[0]:.2f} m"
                if len(tide)
                else "—"
            )
            wave = sess["expected_wave"].drop_nulls().mode()
            wave_txt = wave[0] if len(wave) else "—"
            cards.append(
                f"<div class='card' style='border-left:8px solid {SCALES['hs'].hex(hs)}'>"
                f"<div class='card-title'>{name} {hours[0]:02d}–{hours[-1] + 1:02d} h</div>"
                f"<div class='kpi' {SCALES['hs'].css(hs)}>Hs <b>{hs:.2f} m</b> "
                f"<small>{SCALES['hs'].label(hs)}</small></div>"
                f"<div class='kpi' {SCALES['h_max'].css(hmax)}>Hmax <b>{hmax:.2f} m</b></div>"
                f"<div class='kpi' {SCALES['wind_kmh'].css(wind)}>Viento <b>{wind:.0f} km/h</b> "
                f"{wind_dir:.0f}° <small>ráf. {gust:.0f}</small></div>"
                f"<div class='kpi' {SCALES['rain_prob'].css(rain)}>Lluvia <b>{rain:.0f} %</b></div>"
                f"<div class='kpi'>Marea {tide_txt}</div>"
                f"<div class='kpi'>Ola esperada: <b>{wave_txt}</b></div>"
                "</div>"
            )
        cards.append("</div></div>")
    legend = (
        "<p class='legend'><span style='background:#2ecc71'></span> chico / flojo &nbsp; "
        "<span style='background:#f1c40f'></span> medio &nbsp; "
        "<span style='background:#e74c3c'></span> grande / fuerte</p>"
    )
    return "<h2>Resumen por sesión</h2>" + legend + "".join(cards)


WIND_LEVELS = [
    (0, 6, "Calma", "Espejo, el mar queda perfecto."),
    (6, 18, "Suave", "Ideal; si es terral (de tierra) peina la ola."),
    (18, 33, "Moderado", "Empieza a texturar; onshore ya molesta."),
    (33, 50, "Fuerte", "Mar picado, sesión difícil."),
    (50, 999, "Muy fuerte", "No se surfea."),
]


def reading_guide() -> str:
    """Glossary of the numbers in the report and a wind reference table (km/h)."""
    glossary = [
        (
            "Hs",
            "Altura significativa: promedio del tercio de olas más grandes. Es la que dan "
            "todos los pronósticos y la que 'se ve' desde la orilla en un set normal.",
        ),
        (
            "H1/10",
            "Promedio del 10 % de olas más grandes: el tamaño típico de las olas buenas de un set.",
        ),
        (
            "Hmax",
            "La ola más grande que se espera en una sesión de 2 h. Si Hmax es mucho mayor "
            "que Hs, el día es 'chico pero traicionero': hay sets que vienen de la nada.",
        ),
        ("P(H>1.5)", "Porcentaje de olas de la sesión que superan 1.5 m."),
        (
            "Swell #n",
            "Cada tren de olas que llega al spot: altura · periodo · dirección de "
            "origen. Se numeran para seguirlos en el tiempo; el modelo puede reportar hasta 3.",
        ),
        (
            "Periodo",
            "Segundos entre olas. Más periodo = más energía y olas más ordenadas: "
            "&lt; 8 s mar de viento, 9–12 s swell normal, &gt; 13 s swell de fondo potente.",
        ),
        (
            "Dirección",
            "De dónde viene, en grados: 180° = del sur, 225° = del suroeste, 270° = del oeste.",
        ),
        (
            "Marea",
            "Altura del mar sobre el nivel de referencia; ▲ alta, ▼ baja. En Mollendo el "
            "rango es chico (≈0.5 m).",
        ),
        (
            "Ola esperada",
            "Izquierdas / derechas según la combinación de swells. Es una "
            "hipótesis en calibración; 'uncalibrated' = combinación aún sin regla.",
        ),
    ]
    items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in glossary)
    scale = SCALES["wind_kmh"]
    rows = "".join(
        f"<tr><td {scale.css(min(lo + 4, 60))}>{lo}–{hi if hi < 999 else '…'} km/h</td>"
        f"<td><b>{name}</b></td><td>{note}</td></tr>"
        for lo, hi, name, note in WIND_LEVELS
    )
    wind = (
        "<h3>Viento (km/h)</h3><table class='ref'><thead><tr><th>Velocidad</th><th>Nivel</th>"
        f"<th>Qué significa para surfear</th></tr></thead><tbody>{rows}</tbody></table>"
        "<p>Las ráfagas son picos de segundos; el valor principal es el viento sostenido. "
        "Con la playa mirando al suroeste, viento del este/noreste es terral (bueno) y del "
        "oeste/suroeste es onshore (malo).</p>"
    )
    return (
        "<details class='guide'><summary><b>Cómo leer este reporte</b> (glosario y escala de "
        f"viento)</summary><dl>{items}</dl>{wind}</details>"
    )


def _html_table(
    df: pl.DataFrame,
    title: str,
    formats: dict[str, str] | None = None,
    color_by: dict[str, tuple[str, Scale]] | None = None,
    hide: tuple[str, ...] = (),
) -> str:
    """HTML table with per-cell colour. `color_by` maps column -> (numeric source column, scale)."""
    formats = formats or {}
    color_by = color_by or {}
    cols = [c for c in df.columns if c not in hide]
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for row in df.iter_rows(named=True):
        cells = []
        for col in cols:
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                text = "—"
            elif col == "time":
                text = _stamp(v)
            elif col in formats:
                text = formats[col].format(v)
            else:
                text = str(v)
            style = ""
            if col in color_by:
                src, scale = color_by[col]
                style = " " + scale.css(row[src])
            cells.append(f"<td{style}>{text}</td>")
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
            sims[row["time"]] = wave_distribution(parts, duration_s=duration_s, seed=0)
    return sims


STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:1400px;margin:auto;padding:0 16px;"
    "color:#222;background:#fff}"
    "h1{margin-bottom:0}h2{margin-top:32px}"
    "table{border-collapse:collapse;font-size:13px;width:100%}"
    "th,td{border:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}"
    "th{background:#f0f0f0;position:sticky;top:0}"
    "tr.day-start td{border-top:3px solid #0077b6}"
    ".day h3{margin:12px 0 6px}.cards{display:flex;gap:12px;flex-wrap:wrap}"
    ".card{flex:1 1 280px;background:#fafafa;border:1px solid #e5e5e5;border-radius:8px;"
    "padding:10px 12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}"
    ".card-title{font-weight:600;margin-bottom:6px}"
    ".kpi{padding:3px 6px;margin:3px 0;border-radius:4px;font-size:14px}"
    ".kpi small{color:#555}"
    ".legend span{display:inline-block;width:14px;height:14px;border-radius:3px;"
    "vertical-align:middle}"
    ".guide{background:#f7f9fb;border:1px solid #e1e6ea;border-radius:8px;padding:8px 14px;"
    "margin:12px 0}.guide summary{cursor:pointer}.guide dt{font-weight:600;margin-top:8px}"
    ".guide dd{margin:2px 0 0 0}.guide table.ref{width:auto;margin-top:6px}"
)


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
        "wind_kmh": "{:.0f}",
        "wind_dir": "{:.0f}°",
        "gust_kmh": "{:.0f}",
        "tide_m": "{:.2f}",
        "h_1_10": "{:.2f}",
        "h_max": "{:.2f}",
        "p_over_1.5": "{:.1%}",
        "temp": "{:.0f}",
        "cloud": "{:.0f}",
        "rain_prob": "{:.0f}",
    }
    color_by = {
        "hs": ("hs", SCALES["hs"]),
        "swell_1": ("swell_1_h", SCALES["swell_h"]),
        "swell_2": ("swell_2_h", SCALES["swell_h"]),
        "swell_3": ("swell_3_h", SCALES["swell_h"]),
        "h_1_10": ("h_1_10", SCALES["h_1_10"]),
        "h_max": ("h_max", SCALES["h_max"]),
        "p_over_1.5": ("p_over_1.5", SCALES["p_over_1.5"]),
        "wind_kmh": ("wind_kmh", SCALES["wind_kmh"]),
        "gust_kmh": ("gust_kmh", SCALES["gust_kmh"]),
        "rain_prob": ("rain_prob", SCALES["rain_prob"]),
    }
    ifmt = {"energy_ratio": "{:.2f}", "beat_period_s": "{:.0f}", "rel_angle_deg": "{:.0f}°"}
    tide_note = "" if not tides.is_empty() else " <b>Marea no disponible en esta corrida.</b>"
    intro = (
        f"<h1>{spot.name} — pronóstico de surf</h1>"
        f"<p>Generado {datetime.now():%Y-%m-%d %H:%M}. Fuentes: Open-Meteo (atmósfera + mar, "
        "3 particiones de swell), tide-forecast.com (marea). Direcciones en grados: de dónde "
        f"viene. Sesión simulada: 2 h.{tide_note}</p>"
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
        reading_guide(),
        session_cards(hourly, summaries, interactions),
        "<h2>Evolución horaria</h2>",
        overview_figure(hourly, tracks, tides, summaries).to_html(
            full_html=False, include_plotlyjs=False
        ),
        _html_table(
            table,
            "Tabla horaria (horas de luz)",
            fmt,
            color_by,
            hide=("swell_1_h", "swell_2_h", "swell_3_h"),
        ),
        dist_intro,
        distribution_figure(sims).to_html(full_html=False, include_plotlyjs=False),
        _html_table(
            day_interactions, "Interacción de swells (factores para calibrar el spot)", ifmt
        ),
    ]
    html = (
        f"<html><head><meta charset='utf-8'><title>{spot.name} surf</title>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<script src='{PLOTLY_CDN}'></script><style>{STYLE}</style></head><body>"
        + "\n".join(parts)
        + "</body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
