#!/usr/bin/env python3
"""
analyze_results.py
==================
Reads results.csv produced by ./autotune and generates:
  - Per-size GFLOPS heatmaps (tile vs CF)
  - Line plots: GFLOPS vs CF for each tile size, per matrix size
  - A summary table of the best config per matrix size
  - A bar chart comparing best auto-tuned config vs cuBLAS and Your Best kernel

Usage:
    python3 analyze_results.py [results.csv]

Outputs:
    heatmap_<SIZE>.png   — one per matrix size
    lineplot_<SIZE>.png  — one per matrix size
    comparison.png       — bar chart
    summary.txt          — best configs table
"""

import sys
import csv
import math
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib/numpy not found; only text summary will be produced.")

# ── Load CSV ──────────────────────────────────────────────────────────────────

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
if not os.path.exists(csv_path):
    print(f"ERROR: {csv_path} not found.  Run  ./autotune > results.csv  first.")
    sys.exit(1)

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Separate cuBLAS rows from tuned-kernel rows
cublas_gf  = {}   # size -> gflops
tuned_rows = []

for r in rows:
    sz = int(r["MatSize"])
    if r["Tile"] == "cuBLAS":
        cublas_gf[sz] = float(r["GFLOPS"])
    else:
        tuned_rows.append({
            "size": sz,
            "tile": int(r["Tile"]),
            "cf":   int(r["CF"]),
            "ms":   float(r["Time_ms"]),
            "gf":   float(r["GFLOPS"]),
            "ok":   int(r["Correct"]),
        })

sizes = sorted(set(r["size"] for r in tuned_rows))
tiles = sorted(set(r["tile"] for r in tuned_rows))
cfs   = sorted(set(r["cf"]   for r in tuned_rows))

# Index: (size, tile, cf) -> row
data = {(r["size"], r["tile"], r["cf"]): r for r in tuned_rows}

# ── Text summary ──────────────────────────────────────────────────────────────

summary_lines = []
summary_lines.append("=" * 72)
summary_lines.append("AUTO-TUNING SUMMARY: Best Configuration per Matrix Size")
summary_lines.append("=" * 72)
summary_lines.append(f"{'Size':>6}  {'Tile':>5}  {'CF':>4}  {'Block':>7}  {'OutTile':>8}  "
                     f"{'GFLOPS':>9}  {'vs cuBLAS':>10}")
summary_lines.append("-" * 72)

best_per_size = {}
for sz in sizes:
    valid = [r for r in tuned_rows if r["size"] == sz and r["ok"] == 1]
    if not valid:
        continue
    best = max(valid, key=lambda r: r["gf"])
    best_per_size[sz] = best
    vs = best["gf"] / cublas_gf.get(sz, 1.0) * 100
    summary_lines.append(
        f"{sz:6d}  {best['tile']:5d}  {best['cf']:4d}  "
        f"{best['tile']}x{best['tile']:1d}  {best['tile']*best['cf']:8d}  "
        f"{best['gf']:9.1f}  {vs:9.1f}%"
    )

summary_lines.append("=" * 72)
summary_lines.append("")

# Full table: GFLOPS for every (tile, CF) per size
for sz in sizes:
    summary_lines.append(f"\n── Matrix size {sz}x{sz} (GFLOPS) ──")
    header = "Tile/CF".ljust(8) + "".join(f"{cf:>9}" for cf in cfs)
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    for tile in tiles:
        row_str = f"{tile:<8}"
        for cf in cfs:
            key = (sz, tile, cf)
            if key in data and data[key]["ok"]:
                row_str += f"{data[key]['gf']:9.0f}"
            else:
                row_str += f"{'—':>9}"
        summary_lines.append(row_str)

txt = "\n".join(summary_lines)
print(txt)
with open("summary.txt", "w") as f:
    f.write(txt + "\n")
print("\nWrote summary.txt")

# ── Plots ─────────────────────────────────────────────────────────────────────

if not HAS_MPL:
    sys.exit(0)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]   # one per tile size

# 1. Heatmaps (tile vs CF), one per size
for sz in sizes:
    matrix = np.full((len(tiles), len(cfs)), np.nan)
    for ti, tile in enumerate(tiles):
        for ci, cf in enumerate(cfs):
            key = (sz, tile, cf)
            if key in data and data[key]["ok"]:
                matrix[ti, ci] = data[key]["gf"]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cfs)));  ax.set_xticklabels([str(c) for c in cfs])
    ax.set_yticks(range(len(tiles))); ax.set_yticklabels([str(t) for t in tiles])
    ax.set_xlabel("Coarsening Factor (CF)", fontsize=12)
    ax.set_ylabel("Tile Size", fontsize=12)
    ax.set_title(f"GFLOPS Heatmap — {sz}×{sz} matrix", fontsize=13)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("GFLOPS", fontsize=11)

    # Annotate cells
    for ti in range(len(tiles)):
        for ci in range(len(cfs)):
            val = matrix[ti, ci]
            if not np.isnan(val):
                ax.text(ci, ti, f"{val:.0f}", ha="center", va="center",
                        color="white" if val < matrix[~np.isnan(matrix)].mean() else "black",
                        fontsize=9, fontweight="bold")

    fig.tight_layout()
    fname = f"heatmap_{sz}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Wrote {fname}")

# 2. Line plots: GFLOPS vs CF, grouped by tile, one figure per size
for sz in sizes:
    fig, ax = plt.subplots(figsize=(8, 5))

    for ti, tile in enumerate(tiles):
        xs, ys = [], []
        for cf in cfs:
            key = (sz, tile, cf)
            if key in data and data[key]["ok"]:
                xs.append(cf)
                ys.append(data[key]["gf"])
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2,
                    color=COLORS[ti % len(COLORS)],
                    label=f"Tile {tile}×{tile}")

    # cuBLAS reference line
    if sz in cublas_gf:
        ax.axhline(cublas_gf[sz], color="red", linestyle="--",
                   linewidth=1.5, label="cuBLAS")

    ax.set_xlabel("Coarsening Factor (CF)", fontsize=12)
    ax.set_ylabel("GFLOPS", fontsize=12)
    ax.set_title(f"GFLOPS vs Coarsening Factor — {sz}×{sz} matrix", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xticks(cfs)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"lineplot_{sz}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Wrote {fname}")

# 3. Bar chart: cuBLAS vs best auto-tuned, per size
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(sizes))
width = 0.35

bars_cublas = [cublas_gf.get(sz, 0) for sz in sizes]
bars_best   = [best_per_size[sz]["gf"] if sz in best_per_size else 0 for sz in sizes]

b1 = ax.bar(x - width/2, bars_cublas, width, label="cuBLAS", color="#4C72B0")
b2 = ax.bar(x + width/2, bars_best,   width, label="Best auto-tuned",  color="#55A868")

ax.set_xlabel("Matrix Size", fontsize=12)
ax.set_ylabel("GFLOPS", fontsize=12)
ax.set_title("Best Auto-tuned Config vs cuBLAS", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f"{sz}×{sz}" for sz in sizes])
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.3)

# Annotate with % of cuBLAS
for i, sz in enumerate(sizes):
    if sz in best_per_size and cublas_gf.get(sz, 0) > 0:
        pct = best_per_size[sz]["gf"] / cublas_gf[sz] * 100
        ax.text(i + width/2, best_per_size[sz]["gf"] + 200,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig("comparison.png", dpi=150)
plt.close(fig)
print("Wrote comparison.png")

# 4. Scaling plot: GFLOPS vs matrix size for each (tile, cf) config
fig, ax = plt.subplots(figsize=(9, 5))
configs_to_plot = [(16, 1), (16, 2), (16, 4), (16, 8),
                   (8, 4),  (32, 4)]
style = ["-o", "-s", "-^", "-D", "--o", "--s"]
for idx, (tile, cf) in enumerate(configs_to_plot):
    xs, ys = [], []
    for sz in sizes:
        key = (sz, tile, cf)
        if key in data and data[key]["ok"]:
            xs.append(sz)
            ys.append(data[key]["gf"])
    if xs:
        ax.plot(xs, ys, style[idx % len(style)],
                linewidth=2, markersize=7,
                label=f"Tile={tile}, CF={cf}")

# cuBLAS
cbl_x = [sz for sz in sizes if sz in cublas_gf]
cbl_y = [cublas_gf[sz] for sz in cbl_x]
ax.plot(cbl_x, cbl_y, "r--x", linewidth=2, markersize=8, label="cuBLAS")

ax.set_xlabel("Matrix Size (N×N)", fontsize=12)
ax.set_ylabel("GFLOPS", fontsize=12)
ax.set_title("GFLOPS vs Matrix Size for Selected Configurations", fontsize=13)
ax.set_xticks(sizes)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scaling.png", dpi=150)
plt.close(fig)
print("Wrote scaling.png")

print("\nDone.")