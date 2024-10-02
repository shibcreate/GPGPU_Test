# Makefile for CUDA matrix multiplication project with sm_20 architecture

NVCC = /usr/local/cuda/bin/nvcc
TARGET = matrix_mul

# Specify the architecture for sm_20 (Fermi)
ARCH = -gencode arch=compute_20,code=sm_20

all: $(TARGET)

$(TARGET): main.o kernel.o
	$(NVCC) $(ARCH) -o $(TARGET) main.o kernel.o

main.o: main.cu
	$(NVCC) $(ARCH) -c main.cu

kernel.o: kernel.cu
	$(NVCC) $(ARCH) -c kernel.cu

clean:
	rm -f *.o $(TARGET)
