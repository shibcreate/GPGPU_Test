#include "kernel.cu"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;    
    unsigned VecSize;

    if (argc != 2) {
        printf("Usage: ./vecAdd <Size>\n");
        exit(0);
    }

    VecSize = atoi(argv[1]);
    printf("Running with vector size: %u\n", VecSize);

    A_h = (float*) malloc(sizeof(float) * VecSize);
    B_h = (float*) malloc(sizeof(float) * VecSize);
    C_h = (float*) malloc(sizeof(float) * VecSize);

    for (unsigned int i = 0; i < VecSize; i++) {
        A_h[i] = i;
        B_h[i] = i;
    }    
    
    cudaMalloc((void**) &A_d, sizeof(float) * VecSize);
    cudaMalloc((void**) &B_d, sizeof(float) * VecSize);
    cudaMalloc((void**) &C_d, sizeof(float) * VecSize);

    int totalThreadsOptions[] = {128, 512, 2048}; /* For testing the variants in the option sets */

    for (int i = 0; i < sizeof(totalThreadsOptions) / sizeof(totalThreadsOptions[0]); i++) {

        int totalThreads = totalThreadsOptions[i];

        for (int threadsPerBlock = 1; threadsPerBlock <= totalThreads; threadsPerBlock *= 2) {
            int numBlocks = totalThreads / threadsPerBlock;

            if (numBlocks == 0){
		    break;
	    }

            clock_t start = clock();

            VecAdd<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, VecSize);

            cudaDeviceSynchronize();

            clock_t end = clock();
            double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

            cudaMemcpy(C_h, C_d, sizeof(float) * VecSize, cudaMemcpyDeviceToHost);

            printf("Total Threads: %d, Threads per block: %d, Blocks: %d, Time taken: %f seconds\n",
                   totalThreads, threadsPerBlock, numBlocks, time_taken);
        }
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
