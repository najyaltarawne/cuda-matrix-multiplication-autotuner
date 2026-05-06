NVCC = /usr/local/cuda-12.6/bin/nvcc
FLAGS = -O3 -arch=sm_86 --use_fast_math

all: matmul autotune

matmul:
	$(NVCC) $(FLAGS) benchmark.cpp template.cu -lcublas -o matmul

autotune:
	$(NVCC) $(FLAGS) Autotune.cpp template.cu -lcublas -o autotune

clean:
	rm -f matmul autotune results.csv