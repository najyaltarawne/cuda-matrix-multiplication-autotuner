# cuda-matrix-multiplication-autotuner
CUDA matrix multiplication benchmark and autotuner that compares tile sizes, coarsening factors, and optimization strategies against cuBLAS.
# CUDA GEMM Auto-Tuning Study

## File Overview

| File | Purpose |
|------|---------|
| `template.cu` | All CUDA kernels: Assignment-2 kernels (1–5) + auto-tuned parameterized kernels |
| `benchmark.cpp` | Assignment-2 harness — runs kernels 1–5 for sizes 1024/2048/4096 |
| `autotune.cpp` | Auto-tuning harness — sweeps all (tile, CF) combos, outputs CSV |
| `analyze_results.py` | Python post-processing: plots heatmaps, line charts, tables |
| `Makefile` | Builds both `matmul` and `autotune` |

---

## Build

```bash
make            # builds both: matmul  and  autotune
make clean      # removes binaries and results.csv
```

Default NVCC path: `/usr/local/cuda-12.6/bin/nvcc`  
To override: `make NVCC=/path/to/nvcc`

GPU target is `sm_86` (RTX 3080). Change in Makefile if needed.

---

## Run Assignment-2 Benchmark

```bash
./matmul
```

Runs the 5 kernels (Naive, Coalesced, 2×2 Coarsened, Tiled, Best) for
matrix sizes 1024, 2048, 4096 and prints a GFLOPS table.

---

## Run Auto-Tuning Sweep

```bash
./autotune > results.csv      # runs sweep, saves CSV, prints progress to stderr
```

or via Make:

```bash
make tune
```

This sweeps:
- **Tile sizes**: 8, 16, 32  
- **Coarsening factors**: 1, 2, 4, 8  
- **Matrix sizes**: 512, 1024, 2048, 4096  

Each kernel is warmed up once, then timed and verified against cuBLAS.

### CSV columns
```
MatSize, Tile, CF, BlockDim, OutputTile, Time_ms, GFLOPS, Correct
```

---

## Analyze Results

```bash
python3 analyze_results.py results.csv
```

Produces:
- `heatmap_<N>.png` — GFLOPS heatmap (tile vs CF) for each matrix size
- `lineplot_<N>.png` — GFLOPS vs CF, one line per tile size
- `comparison.png` — bar chart: best auto-tuned vs cuBLAS
- `scaling.png` — GFLOPS vs matrix size for selected configs
- `summary.txt` — text table of best config per size

Requires: `matplotlib`, `numpy`  
Install: `pip install matplotlib numpy`

---

## Kernel Design Notes

### Auto-tuning Kernel (`DEFINE_TILED_CF_KERNEL` macro)

The macro generates a specialized kernel for each `(TILE, CF)` pair:
- **Block size**: `TILE × TILE` threads  
- **Each thread computes**: `CF × CF` output elements  
- **Shared memory**: `A` tile is `(TILE·CF) × TILE`, `B` tile is `TILE × (TILE·CF)`  
- **Output tile**: `(TILE·CF) × (TILE·CF)` per thread block  

This unifies the tiled and coarsened designs into one parameterized kernel,
allowing systematic comparison.

### Combinations tested

| Tile\CF |  1  |  2  |  4  |  8  |
|---------|-----|-----|-----|-----|
| 8       | ✓   | ✓   | ✓   | ✓   |
| 16      | ✓   | ✓   | ✓   | ✓   |
| 32      | ✓   | ✓   | ✓   | ✓   |

Note: `tile=32` requires 1024 threads/block (the CUDA maximum); combinations
exceeding 1024 are automatically skipped.

