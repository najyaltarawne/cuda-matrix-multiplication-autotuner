#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// A: m x k, B: k x n, C: m x n (row-major)

// =============================================================================
// Kernel 1: Naive row-indexing
// =============================================================================
__global__ void matmul1_naive(float *A, float *B, float *C, int M, int N, int K) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // row index
    int j = threadIdx.y + blockIdx.y * blockDim.y; // column index
    if (i >= M || j >= N) return;
    float c = 0.0f;
    for (int k = 0; k < K; k++) {
        c += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = c;
}

// =============================================================================
// Kernel 2: Coalesced indexing (swapped thread indices)
// =============================================================================
__global__ void matmul2_coalesced(float *A, float *B, float *C, int M, int N, int K) {
    int j = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int i = threadIdx.y + blockIdx.y * blockDim.y; // row index
    if (i >= M || j >= N) return;
    float c = 0.0f;
    for (int k = 0; k < K; k++) {
        c += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = c;
}

// =============================================================================
// Kernel 3: Coalesced, coarsened 2x2
// Each thread computes a 2x2 patch of C.
// =============================================================================
__global__ void coarsened_matmul2x2(float *A, float *B, float *C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = blockIdx.x * blockDim.x * 2 + tx * 2; // column index
    int i = blockIdx.y * blockDim.y * 2 + ty * 2; // row index

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    for (int k = 0; k < K; k++) {
        float a0 = A[i*K + k];
        float a1 = A[(i+1)*K + k];
        float b0 = B[k*N + j];
        float b1 = B[k*N + j + 1];
        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
    }

    if (i < M   && j < N)   C[i*N + j]         = c00;
    if (i < M   && j+1 < N) C[i*N + j + 1]     = c01;
    if (i+1 < M && j < N)   C[(i+1)*N + j]     = c10;
    if (i+1 < M && j+1 < N) C[(i+1)*N + j + 1] = c11;
}

// =============================================================================
// Kernel 4: Shared memory tiled (TS=16)
// =============================================================================
#define TS 16
__global__ void MatMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= M || j >= N) return;

    float c = 0.0f;

    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        As[threadIdx.y][threadIdx.x] =
            (i < M && t*TS + threadIdx.x < K) ? A[i*K + t*TS + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (t*TS + threadIdx.y < K && j < N) ? B[(t*TS + threadIdx.y)*N + j] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TS; k++) {
            c += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[i*N + j] = c;
}

// =============================================================================
// Kernel 5: Best kernel — tiled + 4x4 thread coarsening (BS=16, coarsen=4)
// =============================================================================
#define BS 16
__global__ void MatmulBest(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[BS * 4][BS];
    __shared__ float Bs[BS][BS * 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * (BS * 4) + ty * 4;
    int col = blockIdx.x * (BS * 4) + tx * 4;

    float c[4][4] = {0.0f};

    for (int t = 0; t < (K + BS - 1) / BS; t++) {
        int tiledK0 = t * BS + tx;
        int tiledK1 = t * BS + ty;

        As[ty * 4 + 0][tx] = (row + 0 < M && tiledK0 < K) ? A[(row + 0) * K + tiledK0] : 0.0f;
        As[ty * 4 + 1][tx] = (row + 1 < M && tiledK0 < K) ? A[(row + 1) * K + tiledK0] : 0.0f;
        As[ty * 4 + 2][tx] = (row + 2 < M && tiledK0 < K) ? A[(row + 2) * K + tiledK0] : 0.0f;
        As[ty * 4 + 3][tx] = (row + 3 < M && tiledK0 < K) ? A[(row + 3) * K + tiledK0] : 0.0f;

        Bs[ty][tx * 4 + 0] = (tiledK1 < K && col + 0 < N) ? B[tiledK1 * N + col + 0] : 0.0f;
        Bs[ty][tx * 4 + 1] = (tiledK1 < K && col + 1 < N) ? B[tiledK1 * N + col + 1] : 0.0f;
        Bs[ty][tx * 4 + 2] = (tiledK1 < K && col + 2 < N) ? B[tiledK1 * N + col + 2] : 0.0f;
        Bs[ty][tx * 4 + 3] = (tiledK1 < K && col + 3 < N) ? B[tiledK1 * N + col + 3] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BS; k++) {
            float a[4], b[4];
            for (int r = 0; r < 4; r++)
                a[r] = As[ty * 4 + r][k];
            for (int cc = 0; cc < 4; cc++)
                b[cc] = Bs[k][tx * 4 + cc];
            for (int r = 0; r < 4; r++)
                for (int cc = 0; cc < 4; cc++)
                    c[r][cc] += a[r] * b[cc];
        }
        __syncthreads();
    }

    for (int r = 0; r < 4; r++)
        for (int cc = 0; cc < 4; cc++)
            if (row + r < M && col + cc < N)
                C[(row + r) * N + (col + cc)] = c[r][cc];
}

// =============================================================================
// AUTO-TUNING KERNELS
// Parameterized tiled GEMM with compile-time tile size (TILE) and
// thread coarsening factor (CF). Each thread computes CF x CF elements.
//
// We generate specialized versions for the combinations we want to sweep:
//   TILE in {8, 16, 32}
//   CF   in {1, 2, 4, 8}
//
// Naming: matmul_tiled_<TILE>_cf<CF>
// =============================================================================

// Helper macro to define one kernel variant
// TILE: shared-memory tile edge length (= blockDim.x = blockDim.y)
// CF:   coarsening factor (each thread computes CF x CF outputs)
#define DEFINE_TILED_CF_KERNEL(TILE, CF)                                          \
__global__ void matmul_tiled_##TILE##_cf##CF(                                     \
        float * __restrict__ A,                                                   \
        float * __restrict__ B,                                                   \
        float * __restrict__ C,                                                   \
        int M, int N, int K)                                                      \
{                                                                                 \
    __shared__ float As[TILE * CF][TILE];                                         \
    __shared__ float Bs[TILE][TILE * CF];                                         \
                                                                                  \
    int tx  = threadIdx.x;                                                        \
    int ty  = threadIdx.y;                                                        \
    int row = blockIdx.y * (TILE * CF) + ty * CF;                                 \
    int col = blockIdx.x * (TILE * CF) + tx * CF;                                 \
                                                                                  \
    float acc[CF][CF];                                                            \
    for (int r = 0; r < CF; r++)                                                  \
        for (int c = 0; c < CF; c++)                                              \
            acc[r][c] = 0.0f;                                                     \
                                                                                  \
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {                            \
        int kBase0 = t * TILE + tx; /* used to index K when loading A */          \
        int kBase1 = t * TILE + ty; /* used to index K when loading B */          \
        for (int r = 0; r < CF; r++) {                                            \
            As[ty * CF + r][tx] =                                                 \
                (row + r < M && kBase0 < K) ? A[(row + r) * K + kBase0] : 0.0f;  \
        }                                                                         \
        for (int c = 0; c < CF; c++) {                                            \
            Bs[ty][tx * CF + c] =                                                 \
                (kBase1 < K && col + c < N) ? B[kBase1 * N + col + c] : 0.0f;    \
        }                                                                         \
        __syncthreads();                                                           \
                                                                                  \
        for (int k = 0; k < TILE; k++) {                                          \
            float a[CF], b[CF];                                                   \
            for (int r = 0; r < CF; r++) a[r] = As[ty * CF + r][k];              \
            for (int c = 0; c < CF; c++) b[c] = Bs[k][tx * CF + c];              \
            for (int r = 0; r < CF; r++)                                          \
                for (int c = 0; c < CF; c++)                                      \
                    acc[r][c] += a[r] * b[c];                                     \
        }                                                                         \
        __syncthreads();                                                           \
    }                                                                             \
                                                                                  \
    for (int r = 0; r < CF; r++)                                                  \
        for (int c = 0; c < CF; c++)                                              \
            if (row + r < M && col + c < N)                                       \
                C[(row + r) * N + (col + c)] = acc[r][c];                         \
}

// Instantiate all 12 combinations we need
// CF=1 (no coarsening — standard tiled)
DEFINE_TILED_CF_KERNEL(8,  1)
DEFINE_TILED_CF_KERNEL(16, 1)
DEFINE_TILED_CF_KERNEL(32, 1)

// CF=2
DEFINE_TILED_CF_KERNEL(8,  2)
DEFINE_TILED_CF_KERNEL(16, 2)
DEFINE_TILED_CF_KERNEL(32, 2)

// CF=4
DEFINE_TILED_CF_KERNEL(8,  4)
DEFINE_TILED_CF_KERNEL(16, 4)
DEFINE_TILED_CF_KERNEL(32, 4)

// CF=8
DEFINE_TILED_CF_KERNEL(8,  8)
DEFINE_TILED_CF_KERNEL(16, 8)
DEFINE_TILED_CF_KERNEL(32, 8)

// =============================================================================
// Auto-tuning launcher: called from the benchmark with (tile, cf) parameters
// =============================================================================
extern "C" void launchAutoTuneKernel(int tile, int cf,
                                     float *A, float *B, float *C,
                                     int M, int N, int K)
{
    // block is always tile x tile threads; each thread covers cf x cf outputs
    dim3 blockDim(tile, tile);
    int outTile = tile * cf;
    dim3 gridDim((N + outTile - 1) / outTile,
                 (M + outTile - 1) / outTile);

#define DISPATCH(T, CF) \
    if (tile == T && cf == CF) { \
        matmul_tiled_##T##_cf##CF<<<gridDim, blockDim>>>(A, B, C, M, N, K); \
        cudaDeviceSynchronize(); \
        return; \
    }

    DISPATCH(8,  1) DISPATCH(16, 1) DISPATCH(32, 1)
    DISPATCH(8,  2) DISPATCH(16, 2) DISPATCH(32, 2)
    DISPATCH(8,  4) DISPATCH(16, 4) DISPATCH(32, 4)
    DISPATCH(8,  8) DISPATCH(16, 8) DISPATCH(32, 8)

    printf("WARNING: unsupported (tile=%d, cf=%d) — skipping\n", tile, cf);
}

// =============================================================================
// Original assignment-2 launcher (kernelId 1-5)
// =============================================================================
extern "C" void launchMatMulKernel(int kernelId,
                                   float *A, float *B, float *C,
                                   int M, int N, int K)
{
    if (kernelId == 1) {
        dim3 blockDim(16, 16);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        matmul1_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 2) {
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);
        matmul2_coalesced<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 3) {
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 31) / 32, (M + 31) / 32);
        coarsened_matmul2x2<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 4) {
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 15) / 16, (M + 15) / 16);
        MatMulTiled<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 5) {
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 63) / 64, (M + 63) / 64);
        MatmulBest<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else {
        dim3 blockDim(16, 16);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        matmul1_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
}
