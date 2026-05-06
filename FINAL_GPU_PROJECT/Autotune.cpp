// autotune.cpp
// Benchmarks all combinations of tile size (8, 16, 32) x coarsening factor
// (1, 2, 4, 8) across matrix sizes (512, 1024, 2048, 4096).
// Outputs CSV results to stdout (redirect to results.csv) and a
// summary table to stderr so you can watch progress.
//
// Usage:
//   ./autotune              # prints CSV to stdout, progress to stderr
//   ./autotune > results.csv

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif
void launchAutoTuneKernel(int tile, int cf,
                          float *A, float *B, float *C,
                          int M, int N, int K);
void launchMatMulKernel(int kernelId,
                        float *A, float *B, float *C,
                        int M, int N, int K);
#ifdef __cplusplus
}
#endif

// ---- helpers ----------------------------------------------------------------

static int verifyResult(const float *ref, const float *out, int m, int n,
                        float tol, const char *tag)
{
    int errs = 0;
    for (int i = 0; i < m && errs < 5; i++)
        for (int j = 0; j < n && errs < 5; j++) {
            if (fabsf(ref[i*n+j] - out[i*n+j]) > tol) {
                fprintf(stderr, "  MISMATCH %s at (%d,%d): got %.4f expected %.4f\n",
                        tag, i, j, out[i*n+j], ref[i*n+j]);
                errs++;
            }
        }
    return errs > 0 ? 1 : 0;
}

// Time one kernel call (already-warm GPU, events created outside)
static float timeKernel(cudaEvent_t ev0, cudaEvent_t ev1,
                        int tile, int cf,
                        float *A, float *B, float *C,
                        int M, int N, int K)
{
    cudaMemset(C, 0, (size_t)M * N * sizeof(float));
    cudaEventRecord(ev0);
    launchAutoTuneKernel(tile, cf, A, B, C, M, N, K);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    return ms;
}

// ---- main -------------------------------------------------------------------

int main(void)
{
    // Parameter space
    int tiles[] = {8, 16, 32};
    int cfs[]   = {1, 2, 4, 8};
    int nTiles  = 3, nCFs = 4;

    int sizes[]   = {512, 1024, 2048, 4096};
    int nSizes    = 4;

    float tol   = 1e-2f;
    int   fails  = 0;

    // ---- CSV header (stdout) ----
    printf("MatSize,Tile,CF,BlockDim,OutputTile,Time_ms,GFLOPS,Correct\n");

    for (int s = 0; s < nSizes; s++) {
        int SZ = sizes[s];
        int M = SZ, N = SZ, K = SZ;

        fprintf(stderr,
                "\n=== Matrix size %d x %d ===\n"
                "  %-8s %-4s %-10s %-12s %-12s %s\n",
                M, N,
                "Tile", "CF", "BlockDim", "Time(ms)", "GFLOPS", "OK?");

        // Host buffers
        float *A_h = (float *)malloc((size_t)M * K * sizeof(float));
        float *B_h = (float *)malloc((size_t)K * N * sizeof(float));
        float *R_h = (float *)malloc((size_t)M * N * sizeof(float)); // reference
        float *O_h = (float *)malloc((size_t)M * N * sizeof(float)); // output

        srand(42);
        for (int i = 0; i < M * K; i++) A_h[i] = 2.0f*((float)rand()/RAND_MAX)-1.0f;
        for (int i = 0; i < K * N; i++) B_h[i] = 2.0f*((float)rand()/RAND_MAX)-1.0f;

        // Device buffers
        float *A_d, *B_d, *C_d;
        cudaMalloc(&A_d, (size_t)M * K * sizeof(float));
        cudaMalloc(&B_d, (size_t)K * N * sizeof(float));
        cudaMalloc(&C_d, (size_t)M * N * sizeof(float));
        cudaMemcpy(A_d, A_h, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice);

        // Timing events
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);

        // --- cuBLAS reference ---
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;

        // Warmup
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, B_d, N, A_d, K, &beta, C_d, N);
        cudaDeviceSynchronize();

        cudaEventRecord(ev0);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, B_d, N, A_d, K, &beta, C_d, N);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float cublas_ms = 0.0f;
        cudaEventElapsedTime(&cublas_ms, ev0, ev1);
        double cublas_gf = (2.0 * M * N * K) / (cublas_ms * 1e6);
        fprintf(stderr, "  cuBLAS: %.2f ms  %.0f GFLOPS\n", cublas_ms, cublas_gf);
        printf("%d,cuBLAS,cuBLAS,—,—,%.4f,%.2f,1\n", SZ, cublas_ms, cublas_gf);

        cudaMemcpy(R_h, C_d, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);

        // --- Sweep all (tile, cf) combinations ---
        for (int ti = 0; ti < nTiles; ti++) {
            int tile = tiles[ti];
            for (int ci = 0; ci < nCFs; ci++) {
                int cf = cfs[ci];
                if (tile * cf > 128) continue;

                // Skip combinations where blockDim > 1024 (CUDA limit)
                // blockDim.x * blockDim.y = tile * tile
                if ((long long)tile * tile > 1024) {
                    fprintf(stderr, "  tile=%-2d cf=%-2d  SKIP (block too large)\n",
                            tile, cf);
                    continue;
                }

                // Warmup run (don't time it)
                cudaMemset(C_d, 0, (size_t)M * N * sizeof(float));
                launchAutoTuneKernel(tile, cf, A_d, B_d, C_d, M, N, K);

                // Timed run
                float ms = timeKernel(ev0, ev1, tile, cf,
                                      A_d, B_d, C_d, M, N, K);
                double gf = (2.0 * M * N * K) / (ms * 1e6);

                // Correctness
                cudaMemcpy(O_h, C_d, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
                char tag[64];
                snprintf(tag, sizeof(tag), "T%d_CF%d", tile, cf);
                int ok = !verifyResult(R_h, O_h, M, N, tol, tag);
                fails += ok ? 0 : 1;

                int outTile = tile * cf;
                fprintf(stderr,
                        "  tile=%-2d cf=%-2d  block=%dx%d  outTile=%3d  "
                        "%7.2f ms  %8.0f GFLOPS  %s\n",
                        tile, cf, tile, tile, outTile, ms, gf,
                        ok ? "OK" : "FAIL");

                printf("%d,%d,%d,%dx%d,%d,%.4f,%.2f,%d\n",
                       SZ, tile, cf, tile, tile, outTile, ms, gf, ok ? 1 : 0);
                fflush(stdout);
            }
        }

        free(A_h); free(B_h); free(R_h); free(O_h);
        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
        cublasDestroy(handle);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }

    fprintf(stderr, "\n%s  (%d failures)\n",
            fails == 0 ? "All kernels PASSED" : "Some kernels FAILED", fails);

    return fails > 0 ? 1 : 0;
}