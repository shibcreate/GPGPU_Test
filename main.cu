#include <stdio.h>

#define N 1000


__global__ void vector_add_cuda(float *out, float *a, float *b, int n) {
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(id < N)
    {
        out[id] = a[id] + b[id];
    }
}

int main(){

    // Number of bytes to allocate for N doubles
    size_t bytes = N * sizeof(float);

    float *h_a, *h_b, *out, *h_out, *d_a, *d_b, *d_out; 

    // Allocate data in host pointer
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_out, bytes);

    // Initialize array
    for(int i = 0; i < N; i++){
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
    for(int i = 0; i < N; i++)
    {
        if (i < 10)
        {
            printf("h_a[%d] = %.5lf, h_out[%d] = %.5lf\n", i, h_a[i], i, h_out[i]);
        }

        if (out[i] != h_out[i])
        {
            printf(" Error at %d, h_out[%d] = %.5lf, out[%d] = %.5lf\n", i, i, h_out[i], i, out[i]);
            break;
        }

    }

    // Cleanup after kernel execution
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_out);

    return 0;
}
