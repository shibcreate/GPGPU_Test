#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000

__global__ void vector_add_cuda(float *out, float *a, float *b, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (id < n) {
        out[id] = a[id] + b[id];
    }
}

int main() {
    // Number of bytes to allocate for N floats
    size_t bytes = N * sizeof(float);

    float *h_a, *h_b, *h_out, *d_a, *d_b, *d_out;

    // Allocate data in host pointers
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_out, bytes);

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;        
    }

    // Allocate device memory for d_a, d_b, d_out
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int THREADS = 256;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Kernel function
    vector_add_cuda <<< BLOCKS, THREADS >>> (d_out, d_a, d_b, N);

    // Transfer computed data from device to host memory
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Check/validate results
    for (int i = 0; i < N; i++) {
        if (i < 10) {
            printf("h_a[%d] = %.5f, h_out[%d] = %.5f\n", i, h_a[i], i, h_out[i]);
        }

        // Change 'out[i]' to expected result (1.0 + 2.0)
        if (h_out[i] != 3.0f) {
            printf("Error at %d, h_out[%d] = %.5f, expected = 3.0\n", i, i, h_out[i]);
            break;
        }
    }

    // Cleanup after kernel execution
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_out);

    return 0;
}
