from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

PALETTE = [
    "#2364aa", "#3da35d", "#f25f5c", "#7b2cbf", "#f6ae2d", "#008b8b",
    "#d81159", "#6a994e", "#5f0f40", "#2ec4b6", "#9a031e", "#4361ee",
]


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def continuous_color(t: float) -> str:
    t = min(1.0, max(0.0, float(t)))
    stops = [(35, 100, 170), (46, 196, 182), (246, 174, 45), (216, 17, 89)]
    scaled = t * (len(stops) - 1)
    idx = min(int(scaled), len(stops) - 2)
    local = scaled - idx
    a = stops[idx]
    b = stops[idx + 1]
    rgb = tuple(round(a[i] + (b[i] - a[i]) * local) for i in range(3))
    return "#%02x%02x%02x" % rgb


def numeric_legend_bits(x: float, y: float, label: str, c_min: float, c_max: float) -> list[str]:
    gradient_id = f"grad_{safe_name(label)}"
    width = 145
    height = 14
    mid = (c_min + c_max) / 2
    return [
        f'<defs><linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">'
        '<stop offset="0%" stop-color="#2364aa"/>'
        '<stop offset="33%" stop-color="#2ec4b6"/>'
        '<stop offset="66%" stop-color="#f6ae2d"/>'
        '<stop offset="100%" stop-color="#d81159"/>'
        '</linearGradient></defs>',
        f'<text x="{x}" y="{y - 8}" font-size="13" font-family="Arial, sans-serif" font-weight="700">{html.escape(label)}</text>',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="url(#{gradient_id})" stroke="#64748b"/>',
        f'<text x="{x}" y="{y + height + 16}" font-size="11" font-family="Arial, sans-serif">{c_min:.3g}</text>',
        f'<text x="{x + width / 2 - 18}" y="{y + height + 16}" font-size="11" font-family="Arial, sans-serif">{mid:.3g}</text>',
        f'<text x="{x + width - 36}" y="{y + height + 16}" font-size="11" font-family="Arial, sans-serif">{c_max:.3g}</text>',
    ]


def env_shape(cx: float, cy: float, fill: str, env: object, r: float = 3.0, opacity: float = 0.72) -> str:
    env_name = "unknown" if pd.isna(env) else str(env)
    stroke = "#1f2937"
    if env_name == "Marine":
        pts = [(cx, cy - r * 1.25), (cx - r * 1.1, cy + r), (cx + r * 1.1, cy + r)]
        point_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        return f'<polygon points="{point_str}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-opacity="0.42" stroke-width="0.45" />'
    if env_name == "Soil":
        return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-opacity="0.35" stroke-width="0.45" />'
    side = r * 1.8
    return f'<rect x="{cx - side / 2:.2f}" y="{cy - side / 2:.2f}" width="{side:.2f}" height="{side:.2f}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-opacity="0.35" stroke-width="0.45" />'


def env_shape_legend_bits(x: float, y: float) -> list[str]:
    return [
        f'<text x="{x}" y="{y}" font-size="13" font-family="Arial, sans-serif" font-weight="700">Environment shape</text>',
        env_shape(x + 12, y + 22, "#6b7280", "Soil", r=4.2, opacity=0.8),
        f'<text x="{x + 28}" y="{y + 26}" font-size="12" font-family="Arial, sans-serif">Soil</text>',
        env_shape(x + 12, y + 44, "#6b7280", "Marine", r=4.2, opacity=0.8),
        f'<text x="{x + 28}" y="{y + 48}" font-size="12" font-family="Arial, sans-serif">Marine</text>',
        env_shape(x + 12, y + 66, "#6b7280", "missing", r=4.2, opacity=0.8),
        f'<text x="{x + 28}" y="{y + 70}" font-size="12" font-family="Arial, sans-serif">missing</text>',
    ]


def write_svg_scatter(coords: pd.DataFrame, x_col: str, y_col: str, color_col: str, outpath: Path, title: str) -> None:
    needed = [x_col, y_col, color_col]
    use_env_shape = color_col != "Environment" and "Environment" in coords.columns
    if use_env_shape:
        needed.append("Environment")
    sub = coords[needed].dropna(subset=[x_col, y_col])
    if sub.empty:
        return

    width, height = 980, 740
    left, right, top, bottom = 70, 250, 48, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = sub[x_col].to_numpy(dtype=float)
    ys = sub[y_col].to_numpy(dtype=float)
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1

    def sx(value: float) -> float:
        return left + (float(value) - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (1 - (float(value) - y_min) / (y_max - y_min)) * plot_h

    numeric = pd.api.types.is_numeric_dtype(sub[color_col])
    legend_bits = []
    if numeric:
        values = pd.to_numeric(sub[color_col], errors="coerce")
        c_min = float(values.min())
        c_max = float(values.max())
        if c_min == c_max:
            c_min, c_max = c_min - 1, c_max + 1

        def color(value: object) -> str:
            if pd.isna(value):
                return "#b8b8b8"
            return continuous_color((float(value) - c_min) / (c_max - c_min))

        legend_bits.extend(numeric_legend_bits(width - right + 12, top + 22, color_col, c_min, c_max))
        shape_y = top + 112
    else:
        counts = sub[color_col].fillna("missing").astype(str).value_counts()
        categories = list(counts.index[: len(PALETTE)])
        color_map = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(categories)}

        def color(value: object) -> str:
            cat = "missing" if pd.isna(value) else str(value)
            return color_map.get(cat, "#b8b8b8")

        legend_bits.append(f'<text x="{width - right + 12}" y="{top}" font-size="13" font-family="Arial, sans-serif" font-weight="700">{html.escape(color_col)}</text>')
        for i, cat in enumerate(categories[:11]):
            y = top + 24 + i * 22
            legend_bits.append(f'<circle cx="{width - right + 30}" cy="{y}" r="5" fill="{color_map[cat]}" />')
            legend_bits.append(f'<text x="{width - right + 44}" y="{y + 4}" font-size="12" font-family="Arial, sans-serif">{html.escape(cat)} (n={counts[cat]})</text>')
        shape_y = top + 292

    if use_env_shape:
        legend_bits.extend(env_shape_legend_bits(width - right + 12, shape_y))

    points = []
    for _, row in sub.iterrows():
        cx = sx(row[x_col])
        cy = sy(row[y_col])
        fill = color(row[color_col])
        if use_env_shape:
            points.append(env_shape(cx, cy, fill, row.get("Environment"), r=2.7, opacity=0.70))
        else:
            points.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.7" fill="{fill}" fill-opacity="0.72" />')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="28" font-size="18" font-family="Arial, sans-serif" font-weight="700">{html.escape(title)}</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>
{''.join(points)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
<text x="{left + plot_w / 2 - 30}" y="{height - 24}" font-size="13" font-family="Arial, sans-serif">{html.escape(x_col)}</text>
<text transform="translate(22 {top + plot_h / 2 + 40}) rotate(-90)" font-size="13" font-family="Arial, sans-serif">{html.escape(y_col)}</text>
{''.join(legend_bits)}
</svg>
'''
    outpath.write_text(svg)


def write_svg_boxplot(coords: pd.DataFrame, group_col: str, value_col: str, outpath: Path, title: str) -> None:
    cols = [group_col, value_col]
    if "Environment" in coords.columns and "Environment" not in cols:
        cols.append("Environment")
    sub = coords[cols].dropna(subset=[group_col, value_col])
    if sub.empty:
        return
    counts = sub[group_col].astype(str).value_counts()
    groups = list(counts.index[:12])
    stats = []
    y_values = []
    for group in groups:
        vals = pd.to_numeric(sub.loc[sub[group_col].astype(str) == group, value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        low = np.min(vals[vals >= q1 - 1.5 * iqr])
        high = np.max(vals[vals <= q3 + 1.5 * iqr])
        stats.append((group, vals.size, low, q1, med, q3, high))
        y_values.extend(vals.tolist())
    if not stats:
        return

    rng = np.random.default_rng(7)
    width, height = 1080, 680
    left, right, top, bottom = 90, 120, 48, 165
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = min(y_values), max(y_values)
    pad = (y_max - y_min) * 0.06 or 1
    raw_min, raw_max = y_min, y_max
    y_min, y_max = y_min - pad, y_max + pad

    def sy(value: float) -> float:
        return top + (1 - (float(value) - y_min) / (y_max - y_min)) * plot_h

    step = plot_w / len(stats)
    bits = []
    for i, (group, n, low, q1, med, q3, high) in enumerate(stats):
        cx = left + step * (i + 0.5)
        box_w = min(48, step * 0.52)
        group_rows = sub[sub[group_col].astype(str) == group]
        if len(group_rows) > 0:
            jitter_n = max(1, int(round(len(group_rows) * 0.10)))
            jitter_rows = group_rows.sample(n=jitter_n, random_state=7) if jitter_n < len(group_rows) else group_rows
        else:
            jitter_rows = group_rows
        for _, row in jitter_rows.iterrows():
            jitter = rng.uniform(-box_w * 0.78, box_w * 0.78)
            fill = "#334155"
            if "Environment" in row:
                fill = {"Soil": "#2364aa", "Marine": "#f25f5c"}.get(str(row["Environment"]), "#64748b")
            bits.append(f'<circle cx="{cx + jitter:.2f}" cy="{sy(row[value_col]):.2f}" r="2.0" fill="{fill}" fill-opacity="0.18" />')
        bits.append(f'<line x1="{cx:.2f}" y1="{sy(low):.2f}" x2="{cx:.2f}" y2="{sy(high):.2f}" stroke="#334155" stroke-width="1.4"/>')
        bits.append(f'<line x1="{cx - box_w / 3:.2f}" y1="{sy(low):.2f}" x2="{cx + box_w / 3:.2f}" y2="{sy(low):.2f}" stroke="#334155" stroke-width="1.4"/>')
        bits.append(f'<line x1="{cx - box_w / 3:.2f}" y1="{sy(high):.2f}" x2="{cx + box_w / 3:.2f}" y2="{sy(high):.2f}" stroke="#334155" stroke-width="1.4"/>')
        bits.append(f'<rect x="{cx - box_w / 2:.2f}" y="{sy(q3):.2f}" width="{box_w:.2f}" height="{max(1, sy(q1) - sy(q3)):.2f}" fill="{PALETTE[i % len(PALETTE)]}" fill-opacity="0.34" stroke="#111827"/>')
        bits.append(f'<line x1="{cx - box_w / 2:.2f}" y1="{sy(med):.2f}" x2="{cx + box_w / 2:.2f}" y2="{sy(med):.2f}" stroke="#111827" stroke-width="2"/>')
        label = f"{group} (n={n})"
        bits.append(f'<text transform="translate({cx - 5:.2f} {height - 34}) rotate(-45)" font-size="11" font-family="Arial, sans-serif">{html.escape(label)}</text>')

    if "Environment" in sub.columns:
        bits.append('<circle cx="970" cy="80" r="4" fill="#2364aa" fill-opacity="0.5"/><text x="982" y="84" font-size="12" font-family="Arial, sans-serif">Soil jitter, 10%</text>')
        bits.append('<circle cx="970" cy="102" r="4" fill="#f25f5c" fill-opacity="0.5"/><text x="982" y="106" font-size="12" font-family="Arial, sans-serif">Marine jitter, 10%</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="28" font-size="18" font-family="Arial, sans-serif" font-weight="700">{html.escape(title)}</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>
{''.join(bits)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
<text x="{left - 62}" y="{sy(raw_max) + 4:.2f}" font-size="12" font-family="Arial, sans-serif">{raw_max:.4g}</text>
<text x="{left - 62}" y="{sy(raw_min) + 4:.2f}" font-size="12" font-family="Arial, sans-serif">{raw_min:.4g}</text>
<text transform="translate(24 {top + plot_h / 2 + 42}) rotate(-90)" font-size="13" font-family="Arial, sans-serif">{html.escape(value_col)}</text>
</svg>
'''
    outpath.write_text(svg)
