#
CXX = nvcc
INCLUDE=-I/usr/local/cuda/include

all: matrix_perf

run:
	./matrix_perf $(ARGS)

clean:
	rm -f matrix_perf output*.txt 

matrix_perf: matrix_perf.cu matrix_common.cu
	$(CXX) matrix_perf.cu matrix_common.cu -O2 -o matrix_perf ${INCLUDE} -lcusparse

test_common: test_common.cu matrix_common.cu
	$(CXX) -g test_common.cu matrix_common.cu -o test_common ${INCLUDE} -lcusparse
