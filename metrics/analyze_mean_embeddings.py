#!/usr/bin/env python
"""Explore DNABERT mean bag embeddings against curated metadata.

The mean embedding files contain one 768-dimensional vector per FASTQ bag. This
script joins those bag-level means to sample metadata, computes low-dimensional
views, and summarizes the stored read-level embedding variance.

Common commands
---------------
Recompute all sample-level reductions and SVGs, including PCA, PCoA, and UMAP::

    env OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    /home/projects/zeevid/ofirlev/miniconda3/envs/dnabert_s/bin/python \
      llm/metrics/analyze_mean_embeddings.py \
      --max-files 0 \
      --workers 8 \
      --aggregate-sample \
      --pcoa-max 3000 \
      --outdir llm/results/mean_dnabert_2_all_samples

Recompute all bag-level PCA and SVGs. Full all-bag PCoA/UMAP is intentionally
skipped because 172,500 bags is too large for exact PCoA and heavy for UMAP::

    env OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    /home/projects/zeevid/ofirlev/miniconda3/envs/dnabert_s/bin/python \
      llm/metrics/analyze_mean_embeddings.py \
      --max-files 0 \
      --workers 8 \
      --skip-neighbors \
      --skip-pcoa \
      --skip-umap \
      --outdir llm/results/mean_dnabert_2_all_bags \
      --plot-columns Environment,Project,Temperature,Depth_m,variance_mean,read_spread_l2

SVG-only replotting does not recalculate PCA/PCoA/UMAP. It reads each existing
``mean_embedding_reductions.csv`` and rewrites SVGs with the current plotting
style. Run::

    /home/projects/zeevid/ofirlev/miniconda3/envs/dnabert_s/bin/python - <<'PY'
    from pathlib import Path
    import sys
    import pandas as pd

    sys.path.insert(0, 'llm/metrics')
    from svg_plotting_styles import safe_name, write_svg_boxplot, write_svg_scatter
    from analyze_mean_embeddings import write_svg_histogram

    def regenerate(outdir, plot_cols):
        base = Path(outdir)
        df = pd.read_csv(base / 'mean_embedding_reductions.csv', low_memory=False)
        df = df.loc[:, ~df.columns.duplicated()]
        if 'variance_mean' in df:
            write_svg_histogram(df, 'variance_mean', base / 'histogram_variance_mean.svg', 'Histogram of variance_mean')
        if 'mean_nn_distance' in df and df['mean_nn_distance'].notna().any():
            write_svg_histogram(df, 'mean_nn_distance', base / 'histogram_mean_nn_distance.svg', 'Histogram of mean_nn_distance')
        if {'Environment', 'variance_mean'}.issubset(df.columns):
            write_svg_boxplot(df, 'Environment', 'variance_mean', base / 'boxplot_variance_mean_by_environment.svg', 'variance_mean by Environment')
        if {'Project', 'read_spread_l2'}.issubset(df.columns):
            write_svg_boxplot(df, 'Project', 'read_spread_l2', base / 'boxplot_read_spread_l2_by_project.svg', 'read_spread_l2 by Project')
        for method, x_col, y_col in [('pca','pca_1','pca_2'), ('pcoa','pcoa_1','pcoa_2'), ('umap','umap_1','umap_2')]:
            if x_col not in df or y_col not in df or df[[x_col, y_col]].dropna().empty:
                continue
            for color_col in plot_cols:
                if color_col in df and df[color_col].notna().any():
                    write_svg_scatter(df, x_col, y_col, color_col, base / f'{method}_colored_by_{safe_name(color_col)}.svg', f'{method.upper()} colored by {color_col}')

    sample_cols = ['Environment', 'Project', 'Temperature', 'Elevation', 'Depth_m', 'lat_band', 'world_region', 'GeoLocName', 'Biome', 'UsageInAnalysis', 'variance_mean', 'read_spread_l2', 'spread_to_nn_ratio', 'sem_to_nn_ratio', 'mean_norm', 'mean_nn_distance']
    bag_cols = ['Environment', 'Project', 'Temperature', 'Depth_m', 'variance_mean', 'read_spread_l2']
    regenerate('llm/results/mean_dnabert_2_all_samples', sample_cols)
    regenerate('llm/results/mean_dnabert_2_all_bags', bag_cols)
    PY
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import glob
import html
import math
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DEFAULT_EMBEDDING_DIR = "llm/data/mean_dnabert_2"
DEFAULT_METADATA = (
    "/home/projects/zeevid/tomerant/EbG/DocumentsForSubmission/data/metadata/"
    "TableS1_total_curated_metadata_and_measurements_of_environmental_parameters.csv"
)
DEFAULT_OUTDIR = "llm/results/mean_dnabert_2_exploration"

COLOR_COLUMNS = [
    "Environment",
    "Project",
    "Temperature",
    "Elevation",
    "Depth_m",
    "lat_band",
    "world_region",
    "GeoLocName",
    "Biome",
    "UsageInAnalysis",
    "variance_mean",
    "read_spread_l2",
    "spread_to_nn_ratio",
    "sem_to_nn_ratio",
    "mean_norm",
    "mean_nn_distance",
]

PALETTE = [
    "#2364aa",
    "#3da35d",
    "#f25f5c",
    "#7b2cbf",
    "#f6ae2d",
    "#008b8b",
    "#d81159",
    "#6a994e",
    "#5f0f40",
    "#2ec4b6",
    "#9a031e",
    "#4361ee",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--max-files", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=1, help="Parallel HDF5 reader processes.")
    parser.add_argument("--pcoa-max", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-svg", action="store_true", help="Skip SVG scatter plot generation.")
    parser.add_argument("--skip-pcoa", action="store_true", help="Skip PCoA/MDS computation.")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP computation.")
    parser.add_argument("--skip-neighbors", action="store_true", help="Skip nearest-neighbor spread-ratio diagnostics.")
    parser.add_argument(
        "--plot-columns",
        default=",".join(COLOR_COLUMNS),
        help="Comma-separated metadata/value columns to use for reduction scatter SVGs.",
    )
    parser.add_argument(
        "--neighbor-mode",
        choices=["all", "different_sample"],
        default="different_sample",
        help="How to choose nearest mean-vector neighbors for spread ratios.",
    )
    parser.add_argument(
        "--aggregate-sample",
        action="store_true",
        help="Average bag means per sample before dimensionality reduction.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score embedding dimensions before dimensionality reduction.",
    )
    return parser.parse_args()


def sample_paths(paths: list[str], max_files: int | None, seed: int) -> list[str]:
    if not max_files or max_files >= len(paths):
        return paths
    rng = random.Random(seed)
    selected = rng.sample(paths, max_files)
    return sorted(selected)


def read_one_embedding(path: str) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        mean = np.asarray(handle["mean_embedding"], dtype=np.float32)
        variance = np.asarray(handle.attrs["embedding_variance"], dtype=np.float32)
        source_fastq = handle.attrs.get("source_fastq", "")
        sample_id = str(handle.attrs.get("sample_id", Path(path).name.split("-R1.part_")[0]))
        num_reads = int(handle.attrs.get("num_reads", 0))

    row = {
        "embedding_file": path,
        "bag_id": Path(path).stem,
        "sample_id": sample_id,
        "source_fastq": source_fastq,
        "num_reads": num_reads,
        "mean_norm": float(np.linalg.norm(mean)),
        "variance_mean": float(np.mean(variance)),
        "variance_median": float(np.median(variance)),
        "variance_max": float(np.max(variance)),
        "variance_l2": float(np.linalg.norm(variance)),
        "variance_trace": float(np.sum(variance)),
        "read_spread_l2": float(math.sqrt(np.sum(variance))),
        "sem_l2_estimate": float(math.sqrt(np.sum(variance) / max(num_reads, 1))),
    }
    return row, mean, variance


def load_embeddings(
    paths: list[str],
    workers: int = 1,
    progress_every: int = 5000,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    if workers <= 1:
        for idx, path in enumerate(paths, start=1):
            row, mean, variance = read_one_embedding(path)
            rows.append(row)
            means.append(mean)
            variances.append(variance)
            if idx % progress_every == 0 or idx == len(paths):
                print(f"  read {idx}/{len(paths)} HDF5 files", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, (row, mean, variance) in enumerate(
                executor.map(read_one_embedding, paths, chunksize=64),
                start=1,
            ):
                rows.append(row)
                means.append(mean)
                variances.append(variance)
                if idx % progress_every == 0 or idx == len(paths):
                    print(f"  read {idx}/{len(paths)} HDF5 files", flush=True)
    return pd.DataFrame(rows), np.vstack(means), np.vstack(variances)


def add_metadata(rows: pd.DataFrame, metadata_path: str) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    merged = rows.merge(metadata, left_on="sample_id", right_on="UnifiedIndex", how="left")
    for col in ["Latitude", "Longitude", "Temperature", "Elevation", "Depth_m"]:
        if col in merged:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["lat_band"] = merged["Latitude"].apply(lat_band)
    merged["world_region"] = merged.apply(world_region, axis=1)
    merged["vertical_context"] = merged.apply(vertical_context, axis=1)
    return merged


def lat_band(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value <= -66.5:
        return "antarctic"
    if value < -23.5:
        return "south_temperate"
    if value <= 23.5:
        return "tropical"
    if value < 66.5:
        return "north_temperate"
    return "arctic"


def world_region(row: pd.Series) -> str:
    geo = row.get("GeoLocName")
    if isinstance(geo, str) and geo.strip():
        return geo.strip()
    country = row.get("Country")
    if isinstance(country, str) and country.strip():
        return country.strip()
    lon = row.get("Longitude")
    if pd.isna(lon):
        return "unknown"
    if -170 <= lon < -30:
        return "Americas/Atlantic-Pacific"
    if -30 <= lon < 60:
        return "Europe-Africa/Atlantic-Indian"
    return "Asia-Pacific"


def vertical_context(row: pd.Series) -> str:
    env = row.get("Environment")
    elev = row.get("Elevation")
    depth = row.get("Depth_m")
    if env == "Soil":
        return "soil_depth" if not pd.isna(depth) else "soil_no_depth"
    if env == "Marine":
        return "water_depth" if not pd.isna(depth) else "marine_no_depth"
    if not pd.isna(elev):
        return "elevation_available"
    if not pd.isna(depth):
        return "depth_available"
    return "unknown"


def aggregate_by_sample(
    table: pd.DataFrame, means: np.ndarray, variances: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mean_cols = [f"embedding_{i}" for i in range(means.shape[1])]
    var_cols = [f"variance_{i}" for i in range(variances.shape[1])]
    expanded = pd.concat(
        [
            table.reset_index(drop=True),
            pd.DataFrame(means, columns=mean_cols),
            pd.DataFrame(variances, columns=var_cols),
        ],
        axis=1,
    )
    grouped = expanded.groupby("sample_id", sort=True)
    keep_cols = [c for c in table.columns if c not in {"embedding_file", "bag_id", "source_fastq"}]
    sample_table = grouped[keep_cols].first().reset_index(drop=True)
    sample_table["bag_count"] = grouped.size().to_numpy()
    sample_table["variance_mean"] = grouped["variance_mean"].mean().to_numpy()
    sample_table["variance_median"] = grouped["variance_median"].mean().to_numpy()
    sample_table["variance_max"] = grouped["variance_max"].max().to_numpy()
    sample_table["variance_l2"] = grouped["variance_l2"].mean().to_numpy()
    sample_table["variance_trace"] = grouped["variance_trace"].mean().to_numpy()
    sample_table["read_spread_l2"] = np.sqrt(sample_table["variance_trace"])
    sample_table["sem_l2_estimate"] = grouped["sem_l2_estimate"].mean().to_numpy()
    sample_means = grouped[mean_cols].mean().to_numpy(dtype=np.float32)
    sample_variances = grouped[var_cols].mean().to_numpy(dtype=np.float32)
    sample_table["mean_norm"] = np.linalg.norm(sample_means, axis=1)
    return sample_table, sample_means, sample_variances


def add_neighbor_diagnostics(
    table: pd.DataFrame,
    embeddings: np.ndarray,
    neighbor_mode: str,
) -> pd.DataFrame:
    table = table.copy()
    if len(table) < 2:
        table["mean_nn_distance"] = np.nan
        table["sem_to_nn_ratio"] = np.nan
        table["spread_to_nn_ratio"] = np.nan
        return table

    n_neighbors = 2 if neighbor_mode == "all" else min(len(table), 128)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    sample_ids = table["sample_id"].astype(str).to_numpy()
    nearest = np.full(len(table), np.nan, dtype=float)
    nearest_index = np.full(len(table), -1, dtype=int)
    for row_idx in range(len(table)):
        for dist, neighbor_idx in zip(distances[row_idx, 1:], indices[row_idx, 1:]):
            if neighbor_mode == "all" or sample_ids[neighbor_idx] != sample_ids[row_idx]:
                nearest[row_idx] = dist
                nearest_index[row_idx] = neighbor_idx
                break
    table["mean_nn_mode"] = neighbor_mode
    table["mean_nn_distance"] = nearest
    table["mean_nn_index"] = nearest_index
    table["mean_nn_sample_id"] = np.where(
        nearest_index >= 0,
        table.iloc[np.maximum(nearest_index, 0)]["sample_id"].to_numpy(),
        "",
    )
    table["sem_to_nn_ratio"] = table["sem_l2_estimate"] / nearest
    table["spread_to_nn_ratio"] = table["read_spread_l2"] / nearest
    return table


def compute_reductions(
    table: pd.DataFrame,
    embeddings: np.ndarray,
    pcoa_max: int,
    seed: int,
    standardize: bool,
    skip_pcoa: bool,
    skip_umap: bool,
) -> pd.DataFrame:
    x = embeddings.astype(np.float32, copy=False)
    if standardize:
        x = StandardScaler().fit_transform(x)

    coords = table.copy()
    pca = PCA(n_components=10, random_state=seed)
    pca_coords = pca.fit_transform(x)
    for i in range(2):
        coords[f"pca_{i + 1}"] = pca_coords[:, i]
    coords["pca_explained_1"] = pca.explained_variance_ratio_[0]
    coords["pca_explained_2"] = pca.explained_variance_ratio_[1]

    coords["pcoa_1"] = np.nan
    coords["pcoa_2"] = np.nan
    if not skip_pcoa:
        if len(x) <= pcoa_max:
            pcoa_index = np.arange(len(x))
        else:
            rng = np.random.default_rng(seed)
            pcoa_index = np.sort(rng.choice(len(x), size=pcoa_max, replace=False))

        dists = pairwise_distances(x[pcoa_index], metric="euclidean")
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=seed,
            max_iter=300,
            n_init=1,
            normalized_stress="auto",
        )
        pcoa_coords = mds.fit_transform(dists)
        coords.loc[coords.index[pcoa_index], "pcoa_1"] = pcoa_coords[:, 0]
        coords.loc[coords.index[pcoa_index], "pcoa_2"] = pcoa_coords[:, 1]

    coords["umap_1"] = np.nan
    coords["umap_2"] = np.nan
    if not skip_umap:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed)
            umap_coords = reducer.fit_transform(x)
            coords["umap_1"] = umap_coords[:, 0]
            coords["umap_2"] = umap_coords[:, 1]
        except Exception as exc:
            coords.attrs["umap_error"] = f"{type(exc).__name__}: {exc}"

    return coords


def write_correlations(coords: pd.DataFrame, outpath: Path) -> None:
    axes = [c for c in ["pca_1", "pca_2", "pcoa_1", "pcoa_2", "umap_1", "umap_2"] if c in coords]
    targets = [
        "Temperature",
        "Elevation",
        "Depth_m",
        "Latitude",
        "Longitude",
        "variance_mean",
        "variance_max",
        "read_spread_l2",
        "sem_l2_estimate",
        "spread_to_nn_ratio",
        "sem_to_nn_ratio",
        "mean_norm",
    ]
    rows = []
    for axis in axes:
        for target in targets:
            if target in coords:
                sub = coords[[axis, target]].dropna()
                if len(sub) >= 3:
                    rows.append(
                        {
                            "axis": axis,
                            "target": target,
                            "n": len(sub),
                            "spearman": sub[axis].corr(sub[target], method="spearman"),
                            "pearson": sub[axis].corr(sub[target], method="pearson"),
                        }
                    )
    pd.DataFrame(rows).to_csv(outpath, index=False)


def write_svg_scatter(
    coords: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    outpath: Path,
    title: str,
) -> None:
    sub = coords[[x_col, y_col, color_col]].dropna(subset=[x_col, y_col])
    if sub.empty:
        return

    width, height = 900, 720
    left, right, top, bottom = 70, 210, 48, 70
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
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_h

    numeric = pd.api.types.is_numeric_dtype(sub[color_col])
    legend = []
    if numeric:
        values = pd.to_numeric(sub[color_col], errors="coerce")
        c_min = float(values.min())
        c_max = float(values.max())
        if c_min == c_max:
            c_min, c_max = c_min - 1, c_max + 1

        def color(value: object) -> str:
            if pd.isna(value):
                return "#b8b8b8"
            t = (float(value) - c_min) / (c_max - c_min)
            return continuous_color(t)

        legend = [f"{color_col}: {c_min:.3g} to {c_max:.3g}"]
    else:
        counts = sub[color_col].fillna("missing").astype(str).value_counts()
        categories = list(counts.index[: len(PALETTE)])
        color_map = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(categories)}

        def color(value: object) -> str:
            cat = "missing" if pd.isna(value) else str(value)
            return color_map.get(cat, "#b8b8b8")

        legend = [f"{cat} (n={counts[cat]})" for cat in categories]

    points = []
    for _, row in sub.iterrows():
        points.append(
            f'<circle cx="{sx(row[x_col]):.2f}" cy="{sy(row[y_col]):.2f}" r="2.2" '
            f'fill="{color(row[color_col])}" fill-opacity="0.68" />'
        )

    legend_bits = numeric_legend_bits(width - right + 8, top + 22, color_col, c_min, c_max) if numeric else []
    if not numeric:
        for i, item in enumerate(legend[:14]):
            y = top + 24 + i * 22
            fill = PALETTE[i % len(PALETTE)]
            legend_bits.append(f'<circle cx="{width - right + 28}" cy="{y}" r="5" fill="{fill}" />')
            legend_bits.append(
                f'<text x="{width - right + 40}" y="{y + 4}" font-size="12">{html.escape(item)}</text>'
            )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="28" font-size="18" font-family="Arial, sans-serif" font-weight="700">{html.escape(title)}</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>
{''.join(points)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
<text x="{left + plot_w / 2 - 30}" y="{height - 24}" font-size="13" font-family="Arial, sans-serif">{html.escape(x_col)}</text>
<text transform="translate(22 {top + plot_h / 2 + 40}) rotate(-90)" font-size="13" font-family="Arial, sans-serif">{html.escape(y_col)}</text>
<text x="{width - right + 8}" y="{top}" font-size="13" font-family="Arial, sans-serif" font-weight="700">{html.escape(color_col)}</text>
{''.join(legend_bits)}
</svg>
'''
    outpath.write_text(svg)


def numeric_legend_bits(x: float, y: float, label: str, c_min: float, c_max: float) -> list[str]:
    gradient_id = f"grad_{safe_name(label)}"
    width = 145
    height = 14
    bits = [
        f'<defs><linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">'
        '<stop offset="0%" stop-color="#2364aa"/>'
        '<stop offset="33%" stop-color="#2ec4b6"/>'
        '<stop offset="66%" stop-color="#f6ae2d"/>'
        '<stop offset="100%" stop-color="#d81159"/>'
        '</linearGradient></defs>',
        f'<text x="{x}" y="{y - 8}" font-size="13" font-family="Arial, sans-serif" font-weight="700">{html.escape(label)}</text>',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="url(#{gradient_id})" stroke="#64748b"/>',
        f'<text x="{x}" y="{y + height + 16}" font-size="11" font-family="Arial, sans-serif">{c_min:.3g}</text>',
        f'<text x="{x + width - 36}" y="{y + height + 16}" font-size="11" font-family="Arial, sans-serif">{c_max:.3g}</text>',
    ]
    mid = (c_min + c_max) / 2
    bits.append(f'<text x="{x + width / 2 - 18}" y="{y + height + 16}" font-size="11" font-family="Arial, sans-serif">{mid:.3g}</text>')
    return bits


def continuous_color(t: float) -> str:
    t = min(1.0, max(0.0, t))
    stops = [(35, 100, 170), (46, 196, 182), (246, 174, 45), (216, 17, 89)]
    scaled = t * (len(stops) - 1)
    idx = min(int(scaled), len(stops) - 2)
    local = scaled - idx
    a = stops[idx]
    b = stops[idx + 1]
    rgb = tuple(round(a[i] + (b[i] - a[i]) * local) for i in range(3))
    return "#%02x%02x%02x" % rgb


def write_variance_diagnostic_plots(coords: pd.DataFrame, outdir: Path) -> None:
    write_svg_histogram(
        coords,
        "variance_mean",
        outdir / "histogram_variance_mean.svg",
        "Histogram of variance_mean",
    )
    write_svg_boxplot(
        coords,
        "Environment",
        "variance_mean",
        outdir / "boxplot_variance_mean_by_environment.svg",
        "variance_mean by Environment",
    )
    write_svg_boxplot(
        coords,
        "Project",
        "read_spread_l2",
        outdir / "boxplot_read_spread_l2_by_project.svg",
        "read_spread_l2 by Project",
    )
    write_svg_scatter(
        coords,
        "Depth_m",
        "variance_mean",
        "Environment",
        outdir / "scatter_variance_mean_vs_depth_m.svg",
        "variance_mean vs Depth_m",
    )
    write_svg_scatter(
        coords,
        "Temperature",
        "variance_mean",
        "Environment",
        outdir / "scatter_variance_mean_vs_temperature.svg",
        "variance_mean vs Temperature",
    )


def write_svg_histogram(coords: pd.DataFrame, value_col: str, outpath: Path, title: str) -> None:
    values = pd.to_numeric(coords[value_col], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return

    counts, edges = np.histogram(values, bins=40)
    width, height = 900, 560
    left, right, top, bottom = 70, 36, 48, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_count = max(int(counts.max()), 1)
    bars = []
    for count, x0, x1 in zip(counts, edges[:-1], edges[1:]):
        x = left + (x0 - edges[0]) / (edges[-1] - edges[0]) * plot_w
        bar_w = max(1.0, (x1 - x0) / (edges[-1] - edges[0]) * plot_w - 1)
        bar_h = count / max_count * plot_h
        y = top + plot_h - bar_h
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#2364aa" fill-opacity="0.78"/>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="28" font-size="18" font-family="Arial, sans-serif" font-weight="700">{html.escape(title)}</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>
{''.join(bars)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
<text x="{left}" y="{height - 28}" font-size="12" font-family="Arial, sans-serif">{edges[0]:.4g}</text>
<text x="{left + plot_w - 50}" y="{height - 28}" font-size="12" font-family="Arial, sans-serif">{edges[-1]:.4g}</text>
<text x="{left + plot_w / 2 - 42}" y="{height - 28}" font-size="12" font-family="Arial, sans-serif">{html.escape(value_col)}</text>
<text transform="translate(22 {top + plot_h / 2 + 24}) rotate(-90)" font-size="13" font-family="Arial, sans-serif">bag count</text>
<text x="{left + 6}" y="{top + 18}" font-size="12" font-family="Arial, sans-serif">max bin n={max_count}</text>
</svg>
'''
    outpath.write_text(svg)


def write_svg_boxplot(coords: pd.DataFrame, group_col: str, value_col: str, outpath: Path, title: str) -> None:
    sub = coords[[group_col, value_col]].dropna()
    if sub.empty:
        return
    counts = sub[group_col].astype(str).value_counts()
    groups = list(counts.index[:12])
    stats = []
    for group in groups:
        vals = pd.to_numeric(sub.loc[sub[group_col].astype(str) == group, value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        low = np.min(vals[vals >= q1 - 1.5 * iqr]) if vals.size else q1
        high = np.max(vals[vals <= q3 + 1.5 * iqr]) if vals.size else q3
        stats.append((group, vals.size, low, q1, med, q3, high))
    if not stats:
        return

    width, height = 980, 620
    left, right, top, bottom = 90, 36, 48, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min = min(s[2] for s in stats)
    y_max = max(s[6] for s in stats)
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1

    def sy(value: float) -> float:
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_h

    step = plot_w / len(stats)
    bits = []
    for i, (group, n, low, q1, med, q3, high) in enumerate(stats):
        cx = left + step * (i + 0.5)
        box_w = min(48, step * 0.58)
        bits.append(f'<line x1="{cx:.2f}" y1="{sy(low):.2f}" x2="{cx:.2f}" y2="{sy(high):.2f}" stroke="#334155" stroke-width="1.4"/>')
        bits.append(f'<line x1="{cx - box_w / 3:.2f}" y1="{sy(low):.2f}" x2="{cx + box_w / 3:.2f}" y2="{sy(low):.2f}" stroke="#334155" stroke-width="1.4"/>')
        bits.append(f'<line x1="{cx - box_w / 3:.2f}" y1="{sy(high):.2f}" x2="{cx + box_w / 3:.2f}" y2="{sy(high):.2f}" stroke="#334155" stroke-width="1.4"/>')
        bits.append(f'<rect x="{cx - box_w / 2:.2f}" y="{sy(q3):.2f}" width="{box_w:.2f}" height="{max(1, sy(q1) - sy(q3)):.2f}" fill="{PALETTE[i % len(PALETTE)]}" fill-opacity="0.62" stroke="#334155"/>')
        bits.append(f'<line x1="{cx - box_w / 2:.2f}" y1="{sy(med):.2f}" x2="{cx + box_w / 2:.2f}" y2="{sy(med):.2f}" stroke="#111827" stroke-width="2"/>')
        label = f"{group} (n={n})"
        bits.append(f'<text transform="translate({cx - 4:.2f} {height - 32}) rotate(-45)" font-size="11" font-family="Arial, sans-serif">{html.escape(label)}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="28" font-size="18" font-family="Arial, sans-serif" font-weight="700">{html.escape(title)}</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>
{''.join(bits)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
<text x="{left - 62}" y="{sy(y_max) + 4:.2f}" font-size="12" font-family="Arial, sans-serif">{y_max:.4g}</text>
<text x="{left - 62}" y="{sy(y_min) + 4:.2f}" font-size="12" font-family="Arial, sans-serif">{y_min:.4g}</text>
<text transform="translate(24 {top + plot_h / 2 + 42}) rotate(-90)" font-size="13" font-family="Arial, sans-serif">{html.escape(value_col)}</text>
</svg>
'''
    outpath.write_text(svg)


def write_summary(
    outpath: Path,
    paths_total: int,
    paths_used: int,
    table: pd.DataFrame,
    coords: pd.DataFrame,
    aggregate_sample: bool,
    skip_pcoa: bool,
    skip_umap: bool,
    skip_neighbors: bool,
) -> None:
    lines = [
        "# Mean DNABERT-2 Embedding Exploration",
        "",
        f"- HDF5 files discovered: {paths_total}",
        f"- Embeddings analyzed: {paths_used}",
        f"- Analysis level: {'sample' if aggregate_sample else 'bag'}",
        f"- Metadata matches: {int(table['UnifiedIndex'].notna().sum())}/{len(table)}",
        f"- PCA explained variance: PC1={coords['pca_explained_1'].iloc[0]:.4f}, PC2={coords['pca_explained_2'].iloc[0]:.4f}",
    ]
    if skip_pcoa:
        lines.append("- PCoA skipped intentionally for this run.")
    if skip_umap:
        lines.append("- UMAP skipped intentionally for this run.")
    elif coords["umap_1"].isna().all():
        lines.append(f"- UMAP skipped: {coords.attrs.get('umap_error', 'umap-learn is unavailable')}")
    if skip_neighbors:
        lines.append("- Nearest-neighbor spread-ratio diagnostics skipped intentionally for this run.")
    lines.extend(
        [
            "",
            "## Column Notes",
            "",
            "- `Elevation` is site altitude relative to sea level and is populated mainly for NEON soil samples.",
            "- `Depth_m` is sampling depth: soil core depth for soil samples and water-column depth for marine samples.",
            "- `embedding_variance` is the per-DNABERT-coordinate variance across the reads in that one bag.",
            "- `read_spread_l2 = sqrt(sum(embedding_variance))` is the typical read-level spread around the bag mean in DNABERT space.",
            "- `sem_l2_estimate = sqrt(sum(embedding_variance) / num_reads)` estimates uncertainty of the bag mean, not ecological heterogeneity by itself.",
            "- `spread_to_nn_ratio` compares read-level spread to the distance from this bag mean to its nearest other bag mean; large values flag means that may average over a broad/multimodal read cloud.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(glob.glob(os.path.join(args.embedding_dir, "*.h5")))
    paths = sample_paths(all_paths, args.max_files, args.seed)
    print(f"Discovered {len(all_paths)} files; reading {len(paths)} files", flush=True)
    table, embeddings, variances = load_embeddings(paths, workers=args.workers)
    print(f"Loaded embeddings matrix {embeddings.shape}", flush=True)
    table = add_metadata(table, args.metadata)
    print(f"Metadata matches: {table['UnifiedIndex'].notna().sum()}/{len(table)}", flush=True)

    if args.aggregate_sample:
        print("Aggregating bags by sample", flush=True)
        table, embeddings, variances = aggregate_by_sample(table, embeddings, variances)

    if args.skip_neighbors:
        table = table.copy()
        table["mean_nn_mode"] = "skipped"
        table["mean_nn_distance"] = np.nan
        table["mean_nn_index"] = -1
        table["mean_nn_sample_id"] = ""
        table["sem_to_nn_ratio"] = np.nan
        table["spread_to_nn_ratio"] = np.nan
    else:
        table = add_neighbor_diagnostics(table, embeddings, args.neighbor_mode)
    print("Computing PCA/PCoA/optional UMAP", flush=True)
    coords = compute_reductions(
        table,
        embeddings,
        args.pcoa_max,
        args.seed,
        args.standardize,
        args.skip_pcoa,
        args.skip_umap,
    )
    print("Writing tables", flush=True)
    table.to_csv(outdir / "mean_embedding_metadata_table.csv", index=False)
    coords.to_csv(outdir / "mean_embedding_reductions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(variances).describe().to_csv(outdir / "embedding_variance_dimension_summary.csv")
    write_correlations(coords, outdir / "axis_metadata_correlations.csv")
    write_summary(
        outdir / "README.md",
        len(all_paths),
        len(paths),
        table,
        coords,
        args.aggregate_sample,
        args.skip_pcoa,
        args.skip_umap,
        args.skip_neighbors,
    )

    if not args.no_svg:
        print("Writing SVG plots", flush=True)
        plot_columns = [col.strip() for col in args.plot_columns.split(",") if col.strip()]
        reductions = [("pca", "pca_1", "pca_2"), ("pcoa", "pcoa_1", "pcoa_2"), ("umap", "umap_1", "umap_2")]
        for method, x_col, y_col in reductions:
            if x_col not in coords or coords[x_col].isna().all():
                continue
            for color_col in plot_columns:
                if color_col in coords and coords[color_col].notna().any():
                    write_svg_scatter(
                        coords,
                        x_col,
                        y_col,
                        color_col,
                        outdir / f"{method}_colored_by_{safe_name(color_col)}.svg",
                        f"{method.upper()} colored by {color_col}",
                    )
        write_variance_diagnostic_plots(coords, outdir)

    print(f"Wrote analysis to {outdir}")


try:
    from svg_plotting_styles import write_svg_boxplot, write_svg_scatter
except ImportError:
    pass


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


if __name__ == "__main__":
    main()
