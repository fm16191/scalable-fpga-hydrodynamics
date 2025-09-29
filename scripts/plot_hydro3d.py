#!/usr/bin/env python3

from pathlib import Path
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from itertools import repeat
import matplotlib.ticker as mticker
import numpy as np

RE = re.compile(r"output_(\d+)_(\d+(?:\.\d+)?).csv")
FIGSIZE = (16, 12)
DPI = 100
COLORBAR_FMT = "%.1f"
CHUNKSIZE = 5_000_000  # rows per chunk when scanning

def parse_filename(p: Path):
    m = RE.match(p.name)
    if m:
        return int(m.group(1)), float(m.group(2))
    return float('inf'), None

def compute_global_range(files, var):
    vmin, vmax = float('inf'), float('-inf')
    for f in files:
        for chunk in pd.read_csv(f, usecols=[var], chunksize=CHUNKSIZE):
            vals = chunk[var]
            vmin = min(vmin, vals.min())
            vmax = max(vmax, vals.max())
    return vmin, vmax

def render_csv_to_png(src: str, out_dir: str, var: str, vmin: float, vmax: float, axis: str = 'z'):
    srcp = Path(src)
    it, ts = parse_filename(srcp)
    if ts is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frame_{axis}_{it:05d}_{ts:.6f}.png"

    df = pd.read_csv(srcp)
    
    # Handle 3D data by selecting a slice based on the axis
    if 'z' in df.columns:
        if axis == 'x':
            _value = int(df['x'].max() / 2)
            df = df[df['x'] == _value]
            pivot_index, pivot_columns = 'z', 'y'
            xlabel, ylabel = 'Y', 'Z'
            axis_label = f", x={_value}"
        elif axis == 'y':
            _value = int(df['y'].max() / 2)
            df = df[df['y'] == _value]
            pivot_index, pivot_columns = 'z', 'x'
            xlabel, ylabel = 'X', 'Z'
            axis_label = f", y={_value}"
        else:  # 'z' or default
            _value = int(df['z'].max() / 2)
            df = df[df['z'] == _value]
            pivot_index, pivot_columns = 'y', 'x'
            xlabel, ylabel = 'X', 'Y'
            axis_label = f", z={_value}"
    else:
        # Handle 2D data (no 'z' column)
        pivot_index, pivot_columns = 'y', 'x'
        xlabel, ylabel = 'X', 'Y'
        axis_label = ""
    
    if not {pivot_index, pivot_columns, var} <= set(df.columns):
        return None

    grid = df.pivot(index=pivot_index, columns=pivot_columns, values=var)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    heatmap = sns.heatmap(
        grid,
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"format": mticker.FormatStrFormatter(COLORBAR_FMT)},
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label(var, rotation=270, labelpad=15)

    ax.set_title(f"Hydrodynamics : {var} at timestep {ts:.4e} (iteration={it}){axis_label}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.subplots_adjust(left=0.05, right=0.96, top=0.95, bottom=0.05)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return str(out_path)

def main():
    p = argparse.ArgumentParser(description="Render CSV grid frames to PNG for 3D data.")
    p.add_argument("files", nargs="+", help="CSV files (output_ITER_TIMESTEP.csv)")
    p.add_argument("-o", "--out", default="frames", help="output directory")
    p.add_argument("-v", "--var", default="rho", help="variable/column to plot")
    p.add_argument("-j", "--jobs", type=int, default=min(4, multiprocessing.cpu_count()), help="parallel workers")
    p.add_argument("-s", "--global-scale", action="store_true",
                   help="Compute global min/max across all frames for fixed color scale")
    p.add_argument("-a", "--axis", default="z", choices=["x", "y", "z"], 
                   help="Axis to slice for 3D data (default: z)")
    args = p.parse_args()

    files = [Path(f) for f in args.files]
    sorted_files = sorted(files, key=lambda f: parse_filename(f)[0])

    total_frames = len(sorted_files)
    print(f"Total frames : {total_frames}")

    if args.global_scale:
        print("Computing global color scale range...")
        vmin, vmax = compute_global_range(sorted_files, args.var)
        print(f"Global range: vmin={vmin}, vmax={vmax}")
    else:
        vmin = vmax = None  # per-frame scaling

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = ex.map(render_csv_to_png, map(str, sorted_files), repeat(args.out), repeat(args.var), repeat(vmin), repeat(vmax), repeat(args.axis))
        for i, out in enumerate(results, start=1):
            print(f"Processing {i}/{total_frames}", end='\r')

    print("\nExporting animation frames complete.")

if __name__ == "__main__":
    main()
