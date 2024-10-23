CPP = g++ -lboost_system $(SNOW)
CC = gcc -lboost_system $(SNOW)

INCFLAGEXTRA ?=
CFLAGS     ?=
CCFLAGS    ?=
CUFLAGS    ?=
NEWLIBDIR  ?=
LIB        ?= $(OTHER_LIBS)
SRCDIR     ?=
ROOTDIR    ?=
ROOTBINDIR ?= bin
BINDIR     ?= $(ROOTBINDIR)
ROOTOBJDIR ?= obj
LIBDIR     := "$(NVIDIA_CUDA_SDK_LOCATION)/C/lib"
COMMONDIR  := "$(NVIDIA_CUDA_SDK_LOCATION)/C/common"
SDKINCDIR  := "$(NVIDIA_CUDA_SDK_LOCATION)/C/common/inc/"
LIB += -lm -lz -lGL
CUTIL:=cutil_x86_64

INTERMED_FILES := *.cpp*.i *.cpp*.ii *.cu.c *.cudafe*.* *.fatbin.c *.cu.cpp *.linkinfo *.cpp_o core
MY_CC_VERSION := $(shell gcc --version | head -1 | awk '{for(i=1;i<=NF;i++){ if(match($$i,/^[0-9]\.[0-9]\.[0-9]$$/))  {print $$i; exit 0 }}}')
MY_CUDA_VERSION_STRING:=$(shell $(CUDA_INSTALL_PATH)/bin/nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//')
MY_CUDART_VERSION:=$(shell echo $(MY_CUDA_VERSION_STRING) | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')
GCC_VERSION ?= gcc-$(MY_CC_VERSION)

SIM_OBJDIR :=
SIM_OBJS +=  $(patsubst %.cpp,$(SIM_OBJDIR)%.cpp_o,$(CCFILES))
SIM_OBJS +=  $(patsubst %.c,$(SIM_OBJDIR)%.c_o,$(CFILES))
SIM_OBJS +=  $(patsubst %.cu,$(SIM_OBJDIR)%.cu_o,$(CUFILES))

.SUFFIXES:

gpgpu_ptx_sim__$(EXECUTABLE): $(SIM_OBJS)
	$(CPP) $(CFLAGS) -g $(notdir $(SIM_OBJS)) \
		-L$(LIBDIR) -l$(CUTIL) -L$(GPGPUSIM_ROOT)/lib/$(GCC_VERSION)/cuda-$(MY_CUDART_VERSION)/release -lcudart \
		$(NEWLIBDIR) $(LIB) -o gpgpu_ptx_sim__$(EXECUTABLE)
		rm -rf $(INTERMED_FILES) *.cubin cubin.bin *_o *.hash $(EXECUTABLE) 

%.cpp_o: %.cpp
	$(CPP) $(CCFLAGS) $(INCFLAGEXTRA) -I$(CUDAHOME)/include -I$(SDKINCDIR) -L$(LIBDIR) -g -c $< -o $(notdir $@) 

%.c_o: %.c
	$(CC) $(CFLAGS) $(INCFLAGEXTRA)  -I$(CUDAHOME)/include -I$(SDKINCDIR) -L$(LIBDIR) -g -c $< -o $(notdir $@)

%.cu_o: %.cu
	nvcc $(CUFLAGS) -c --keep --compiler-options -fno-strict-aliasing \
		-I. -I$(CUDAHOME)/include/ -I$(SDKINCDIR) -l$(CUTIL) \
		$(INCFLAGEXTRA) -L$(LIBDIR) -DUNIX $< -o $(EXECUTABLE)

	$(CC) -g -c $(notdir $<.cpp.ii) -o $(notdir $@)

%.cu: %.cu.c

clean:
	rm -f $(INTERMED_FILES) *.cubin *.fatbin *.o *_o *.hash *.ptx *.ptxinfo _cuobjdump* cubin.bin $(EXECUTABLE) gpgpu_ptx_sim__$(EXECUTABLE)

