//polybenchUtilFuncts.h
//Scott Grauer-Gray (sgrauerg@gmail.com)
//Functions used across hmpp codes

#ifndef POLYBENCH_UTIL_FUNCTS_H
#define POLYBENCH_UTIL_FUNCTS_H

FILE * CPUoutputFile;
FILE * GPUoutputFile;

//void createOutputFiles(char * BENCHname, FILE * CPUoutputFile, FILE * GPUoutputFile)
void createOutputFiles(char * BENCHname)
{
    char filenameCPU[128], filenameGPU[128];
    char* CPUextension = "_CPU.txt";
    char* GPUextension = "_GPU.txt";

    strncpy(filenameCPU, BENCHname, sizeof(filenameCPU));
    strncat(filenameCPU, CPUextension, (sizeof(filenameCPU) - strlen(filenameCPU)) );
    printf("FilenameCPU is %s\n", filenameCPU);

    strncpy(filenameGPU, BENCHname, sizeof(filenameGPU));
    strncat(filenameGPU, GPUextension, (sizeof(filenameGPU) - strlen(filenameGPU)) );
    printf("FilenameGPU is %s\n", filenameGPU);

    CPUoutputFile = fopen(filenameCPU, "w");
    GPUoutputFile = fopen(filenameGPU, "w");
}

//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f
#define MIN_EPSILON_ERROR 1e-3f //written by SpM
#include <sys/time.h>
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


float absVal(float a)
{
	if(a < 0)
	{
		return (a * -1);
	}
   	else
	{ 
		return a;
	}
}



float percentDiffOurs(double val1, double val2, bool & shouldInclude)
{
	if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
	{
		shouldInclude=false;
		return 0.0f;
	}

	else
	{
		shouldInclude=true;
    		return (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
	}
} 
float percentDiff(double val1, double val2)
{
	if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
	{
		return 0.0f;
	}

	else
	{
    		return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
	}
} 

float percentDiff_noskip(double val1, double val2)
{
        if (absVal(val1) < 0.01 )
        {
                return 1.0f;
        }
        else
        {
                return (absVal(absVal(val1 - val2) / absVal(val1)));
        }
}

float percentDiff_skipzero(double val1, double val2, bool & shouldInclude)
{
        if (absVal(val1) < 0.01)
        {
                shouldInclude=false;
                return 0.0f;
        }
        else
        {
                shouldInclude=true;
                return (absVal(absVal(val1 - val2) / absVal(val1)));
        }
}
#endif //POLYBENCH_UTIL_FUNCTS_H
