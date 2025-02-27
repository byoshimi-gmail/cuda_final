#
CXX = nvcc

all: matrix_perf

run:
	./matrix_perf $(ARGS)

clean:
	rm -f matrix_perf output*.txt 

matrix_perf: matrix_perf.cu matrix_common.cu
	$(CXX) matrix_perf.cu matrix_common.cu -O2 -o matrix_perf -lcusparse

test_common: test_common.cu matrix_common.cu
	$(CXX) -g test_common.cu matrix_common.cu -o test_common -lcusparse
