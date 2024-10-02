#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// Function prototypes
void InitializeMatrix(float* matrix, int N);
__global__ void MatrixMultiplyKernel(float* A, float* B, float* C, int N);

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <MatrixSize>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);
    size_t matrixSize = N * N * sizeof(float);

    // Allocate memory on host
    float *h_A = (float *)malloc(matrixSize);
    float *h_B = (float *)malloc(matrixSize);
    float *h_C = (float *)malloc(matrixSize);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        printf("Memory Allocation Failed\n");
        return -1;
    }

    // Initialize matrices A and B with random values
    srand(time(NULL));
    InitializeMatrix(h_A, N);
    InitializeMatrix(h_B, N);

    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Launch the matrix multiplication kernel
    MatrixMultiplyKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // Print results for small matrices (optional)
    if (N <= 5) {
        printf("Matrix C (Result):\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%0.2f ", h_C[i * N + j]);
            }
            printf("\n");
        }
    }

    // Free host and device memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Matrix multiplication of size %d x %d completed.\n", N, N);
    return 0;
}

// Function to initialize a matrix with random numbers
void InitializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)(rand() % (MAXNUM - MINNUM + 1) + MINNUM);
    }
}
