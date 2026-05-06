#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


#ifdef __cplusplus
extern "C" {
#endif

// Launch the student matrix‐multiply kernel.
//
// Parameters:
//   kernelId: identifier for the kernel version to run (e.g. 1 for naive, 2 for coalesced, etc.).
//   A, B, C: device pointers to matrices (in row‑major order)
//   M, N, K: dimensions such that A is M×K, B is K×N, and C is M×N.
void launchMatMulKernel(int kernelId, float *A, float *B, float *C, int M, int N, int K);

#ifdef __cplusplus
}
#endif




// Verification helper: compare kernel result to reference
// return 1 for fail to pass verification; 0 otherwise
int verifyResult(const float* reference, const float* result, int m, int n, float tolerance, const char* kernelName) {
    int errors = 0;
    for (int i = 0; i < m && errors < 10; i++) {
        for (int j = 0; j < n && errors < 10; j++) {
            float refVal = reference[i*n + j];
            float outVal = result[i*n + j];
            if (fabs(refVal - outVal) > tolerance) {
                printf("%s error at (%d,%d): %.4f vs %.4f\n", kernelName, i, j, outVal, refVal);
                errors++;
            }
        }
    }
    return errors>0 ? 1 : 0; 
}

int main() {
    // Matrix sizes to test
    int sizes[] = {1024, 2048, 4096};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    int fails = 0; 
    // Loop over each matrix size
    for (int s = 0; s < numSizes; s++) {
        int LL = sizes[s];
        int m = LL, n = LL, k = LL;
        printf("===================================================\n");
        printf("Matrix multiplication for size: %d x %d x %d\n", m, n, k);
        printf("---------------------------------------------------\n");
        printf("%-20s %15s %15s\n", "Kernel", "Time (ms)", "GFLOPS");
        printf("---------------------------------------------------\n");

        float *A_d, *B_d, *C_d;
        float *A_h, *B_h, *result_h, *ref_h;

        // Allocate host memory
        A_h      = (float*) malloc(m * k * sizeof(float));
        B_h      = (float*) malloc(k * n * sizeof(float));
        result_h = (float*) malloc(m * n * sizeof(float));
        ref_h    = (float*) malloc(m * n * sizeof(float));

        // Initialize matrices A and B with random values in [-1,1]
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                A_h[i*k + j] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            }
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                B_h[i*n + j] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            }
        }

        // Allocate device memory
        cudaMalloc(&A_d, m * k * sizeof(float));
        cudaMalloc(&B_d, k * n * sizeof(float));
        cudaMalloc(&C_d, m * n * sizeof(float));

        // Copy host data to device
        cudaMemcpy(A_d, A_h, m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, k * n * sizeof(float), cudaMemcpyHostToDevice);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        // Create cuBLAS handle and set up SGEMM parameters
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;

        // Warmup run 
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k, &alpha,
                    B_d, n, A_d, k, &beta,
                    C_d, n);

        // Compute reference result using cuBLAS SGEMM.
        cudaEventRecord(start);
	// (cuBLAS expects column‑major, so we swap A and B)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k, &alpha,
                    B_d, n, A_d, k, &beta,
                    C_d, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        float gflops = (2.0 * m * n * k) / (ms * 1e6);
        printf("%-20s %15.2f %15.2f\n", "cuBLAS SGEMM", ms, gflops);

        // Copy reference result back to host
        cudaMemcpy(ref_h, C_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);

        float tolerance = 1e-2;
        const char* kernelNames[6] = {"", "Naive", "Coalesced", "2x2 Coarsened", "Tiled", "Your Best"};


        // Run student kernels (IDs 1-5)
        for (int kernelToRun = 1; kernelToRun <= 5; kernelToRun++) {
            // Reset device memory for C
            cudaMemset(C_d, 0, m * n * sizeof(float));

            // Run the student's kernel and time it
            cudaEventRecord(start);
            launchMatMulKernel(kernelToRun, A_d, B_d, C_d, m, n, k);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            gflops = (2.0 * m * n * k) / (ms * 1e6);

            // Print result as a table row
            printf("%-20s %15.2f %15.2f\n", kernelNames[kernelToRun], ms, gflops);

            // Copy the result from device to host and verify correctness
            cudaMemcpy(result_h, C_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            fails += verifyResult(ref_h, result_h, m, n, tolerance, kernelNames[kernelToRun]);
        }

        printf("===================================================\n\n");

        // Cleanup resources for this matrix size
        free(A_h);
        free(B_h);
        free(result_h);
        free(ref_h);
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    if (fails > 0) {
	exit(1);
    }
    return 0;
}
