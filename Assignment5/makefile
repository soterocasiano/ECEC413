CUDA_PATH ?= /usr/local/cuda-10.1
HOST_COMPILER ?= g++
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCCFLAGS       := -m64 -O3 -gencode arch=compute_75,code=compute_75

all: trap

trap: trap.cu trap_kernel.cu trap_gold.cpp
        $(NVCC) -o trap trap.cu trap_gold.cpp $(NVCCFLAGS)

clean:
        rm trap
