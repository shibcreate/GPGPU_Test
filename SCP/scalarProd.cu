
/*
 * This sample calculates scalar products of a 
 * given set of input vector pairs
 */



#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cutil_inline.h>


#include "polybenchUtilFuncts.h"

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
void scalarProdCPU(
    float *h_C,
    float *h_A,
    float *h_B,
    int vectorN,
    int elementN
);



///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////
#include "scalarProd_kernel.cu"



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

//Total number of input vector pairs; arbitrary
long  int VECTOR_N = 256; //SpM made them non-const
//Number of elements per vector; arbitrary, 
//but strongly preferred to be a multiple of warp size
//to meet memory coalescing constraints
long  int ELEMENT_N = 4096; //SpM made them non-const
//Total number of data elements 



///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    createOutputFiles("SCP");
  float *h_A, *h_B, *h_C_CPU, *h_C_GPU;
    float *d_A, *d_B, *d_C;
    double delta, ref, sum_delta, sum_ref, L1norm;
    unsigned int hTimer;
    int i;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    cutilCheckError( cutCreateTimer(&hTimer) );
	int VECTOR_N1 = VECTOR_N;
     if (cutGetCmdLineArgumenti(argc, (const char**)argv, "vector_n", &VECTOR_N1))
        {
	    VECTOR_N = VECTOR_N1;
            if (VECTOR_N < 100)
            {
                printf("VECTOR_N should be greater than 100 \n");
            }
           printf(" Found VECTOR_N = %d\n", VECTOR_N);
        }
	int ELEMENT_N1 = ELEMENT_N;
      if (cutGetCmdLineArgumenti(argc, (const char**)argv, "element_n", &ELEMENT_N1))
        {   
  	   ELEMENT_N = ELEMENT_N1;
           printf(" Found ELEMENT_N = %d\n", ELEMENT_N);
        }  
     long int    DATA_N = VECTOR_N * ELEMENT_N;

     long int   DATA_SZ = DATA_N * sizeof(float);
     long int RESULT_SZ = VECTOR_N  * sizeof(float);
printf(" DATA_N %ld DATA_SZ %ld RESULT_SZ %ld \n",DATA_N, DATA_SZ, RESULT_SZ);
    printf("Initializing data...\n");
        printf("...allocating CPU memory.\n");
        h_A     = (float *)malloc(DATA_SZ);
        h_B     = (float *)malloc(DATA_SZ);
        h_C_CPU = (float *)malloc(RESULT_SZ);
        h_C_GPU = (float *)malloc(RESULT_SZ);

        printf("...allocating GPU memory.\n");
        cutilSafeCall( cudaMalloc((void **)&d_A, DATA_SZ)   );
        cutilSafeCall( cudaMalloc((void **)&d_B, DATA_SZ)   );
        cutilSafeCall( cudaMalloc((void **)&d_C, RESULT_SZ) );

        printf("...generating input data in CPU mem.\n");
        srand(123);
        //Generating input data on CPU		
        for(i = 0; i < DATA_N; i++){
		    //////haonan
            //h_A[i] = RandFloat(0.0f, 1.0f);
            //h_B[i] = RandFloat(0.0f, 1.0f);
			
			//h_A[i] = floor( (i) / 16 );
            //h_B[i] = floor( (i) / 16 );
			h_A[i] = i;
            h_B[i] = i;
			//////haonan
        }

        printf("...copying input data to GPU mem.\n");
        //Copy options data to GPU memory for further processing 
        cutilSafeCall( cudaMemcpy(d_A, h_A, DATA_SZ, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy(d_B, h_B, DATA_SZ, cudaMemcpyHostToDevice) );
    printf("Data init done.\n");


    printf("Executing GPU kernel...\n");
        cutilSafeCall( cudaThreadSynchronize() );
        cutilCheckError( cutResetTimer(hTimer) );
        cutilCheckError( cutStartTimer(hTimer) );
		
		//////haonan
        //scalarProdGPU<<<128, 256>>>(d_C, d_A, d_B, VECTOR_N, ELEMENT_N);///////////////////////////////////////////////////////////////////
		scalarProdGPU<<<32, 1024>>>(d_C, d_A, d_B, VECTOR_N, ELEMENT_N);///////////////////////////////////////////////////////////////////
		//////haonan
		
        cutilCheckMsg("scalarProdGPU() execution failed\n");
        cutilSafeCall( cudaThreadSynchronize() );
        cutilCheckError( cutStopTimer(hTimer) );
    printf("GPU time: %f msecs.\n", cutGetTimerValue(hTimer));

    printf("Reading back GPU result...\n");
        //Read back GPU results to compare them to CPU results
        cutilSafeCall( cudaMemcpy(h_C_GPU, d_C, RESULT_SZ, cudaMemcpyDeviceToHost) );


    printf("Checking GPU results...\n");
        printf("..running CPU scalar product calculation\n");
        scalarProdCPU(h_C_CPU, h_A, h_B, VECTOR_N, ELEMENT_N);///////////////////////////////////////////////////////////////////

        printf("...comparing the results\n");
        //Calculate max absolute difference and L1 distance
        //between CPU and GPU results
        sum_delta = 0;
        sum_ref   = 0;
        double nIncluded2=0, sumRelativeError_noskip = 0, sumRelativeError_skipzero = 0;
        double nIncluded=0, nDifferent =0, sumDifferent=0, sumRelativeError=0;

         fprintf(CPUoutputFile, "1\n");
            fprintf(GPUoutputFile, "1\n");	 
        //double percErrorSum=0;
	for(i = 0; i < VECTOR_N; i++){
	      fprintf(CPUoutputFile, "%f ",h_C_CPU[i]);
              fprintf(GPUoutputFile, "%f ", h_C_GPU[i]);
            delta = fabs(h_C_GPU[i] - h_C_CPU[i]);
            ref   = h_C_CPU[i];
            sum_delta += delta;
            sum_ref   += ref;
             if (delta > MIN_EPSILON_ERROR)
                   nDifferent++;
              bool include =true;
            double temp = percentDiffOurs(h_C_CPU[i] ,h_C_GPU[i],include);
                if(include) {
                   sumRelativeError += temp;
                   nIncluded++;
                }
                bool include2 =true;
                double temp2 = percentDiff_skipzero(h_C_CPU[i] ,h_C_GPU[i], include2);
                if(include2) {
                   sumRelativeError_skipzero += temp2;
                   nIncluded2++;
                }

                double temp3 = percentDiff_noskip(h_C_CPU[i] ,h_C_GPU[i]);
                sumRelativeError_noskip += temp3;

      //    percErrorSum += delta*100.0/ref;
	}
	  fprintf(CPUoutputFile, "\n");
	     fprintf(GPUoutputFile, "\n");

        L1norm = sum_delta / sum_ref;
    printf("L1 error: %E\n", L1norm);
    printf((L1norm < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n");
   // double finalPercError = percErrorSum/VECTOR_N;
    //printf("ACT_percLossInQoR %E\n", finalPercError);
        double finalPercError = (100.0*nDifferent)/VECTOR_N;
        //printf("ACT_percLossInQoR %E\n", finalPercError);
        printf("ACT_percLossInQoR %E\n", finalPercError);
        printf("ACT_percSumDifferent %E\n", (sumDifferent*100.0)/VECTOR_N);
        printf("ACT_percRelativeError %E\n", (sumRelativeError*100.0)/nIncluded);
        printf("ACT_percRelativeError_skipzero %E\n", (sumRelativeError_skipzero*100.0)/nIncluded2);
        printf("ACT_percRelativeError_noskip %E\n", (sumRelativeError_noskip*100.0)/VECTOR_N);

    printf("Shutting down...\n");
        cutilSafeCall( cudaFree(d_C) );
        cutilSafeCall( cudaFree(d_B)   );
        cutilSafeCall( cudaFree(d_A)   );
        free(h_C_GPU);
        free(h_C_CPU);
        free(h_B);
        free(h_A);
        cutilCheckError( cutDeleteTimer(hTimer) );

    //cudaThreadExit();

    //cutilExit(argc, argv);

}
