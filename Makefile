CC=/usr/local/cuda/bin/nvcc
ARCH=-arch=sm_52
SOURCES=main.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=matrixMulReg

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(ARCH) $(OBJECTS) -o $@

%.o: %.cu
	$(CC) -c $(ARCH) $< -o $@

.PHONY: clean
clean:
	-rm -f $(EXECUTABLE) $(OBJECTS)
