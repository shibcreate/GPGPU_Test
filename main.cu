#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"

int main (int argc, char *argv[]) {
    float *A_h, *B_h, *C_h;   // Host matrices
    float *A_d, *B_d, *C_d;   // Device matrices
    int MatSize;
   
    // Accept matrix size from the user
    if (argc == 1) {
        MatSize = 16;  // Default size
    } else if (argc == 2) {
        MatSize = atoi(argv[1]);  // User-specified size
    } else {
        printf("Usage: ./matMul <Size>\n");
        exit(0);
    }

    int size = MatSize * MatSize * sizeof(float);

    // Allocate memory on the host
    A_h = (float*) malloc(size);
    B_h = (float*) malloc(size);
    C_h = (float*) malloc(size);

    // Initialize the host matrices
    for (int i = 0; i < MatSize * MatSize; i++) {
        A_h[i] = rand() % 100;  // Random values for demo
        B_h[i] = rand() % 100;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);  // 16x16 threads per block
    dim3 blocksPerGrid((MatSize + 15) / 16, (MatSize + 15) / 16);

    // Launch the kernel
    MatMul<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, MatSize);

    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Print a small portion of the result matrix for verification
    for (int i = 0; i < MatSize && i < 4; i++) {
        for (int j = 0; j < MatSize && j < 4; j++) {
            printf("%f ", C_h[i * MatSize + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // Free host memory
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}
