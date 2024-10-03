#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"

int main (int argc, char *argv[]){

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;    
    unsigned VecSize;
   
    if (argc == 1) {
        VecSize = 256;
    } else if (argc == 2) {
      VecSize = atoi(argv[1]);
    } else {
        printf("Usage: ./vecAdd <Size>");
        exit(0);
    }
	
    A_h = (float*) malloc( sizeof(float) * VecSize );
	B_h = (float*) malloc( sizeof(float) * VecSize );
	C_h = (float*) malloc( sizeof(float) * VecSize );
	
    for (unsigned int i=0; i < VecSize; i++) {
		A_h[i] = i;
		B_h[i] = i;
	}    

    cudaDeviceSynchronize();
    
    //INSERT Memory CODE HERE

    cudaDeviceSynchronize();

    //INSERT kernel launch CODE HERE

    cudaDeviceSynchronize();

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT Memory CODE HERE

    return 0;
}
