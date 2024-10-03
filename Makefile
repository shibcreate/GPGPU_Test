CC=nvcc
ARCH=-arch=sm_52
SOURCES=main.cu
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=vecadd
all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(ARCH) $(OBJECTS) -o $@
.PHONY : clean
clean :
	-rm $(EXECUTABLE)

