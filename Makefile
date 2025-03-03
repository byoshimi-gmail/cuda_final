#
CXX = nvcc
INCLUDES=-I/usr/local/cuda/include --Wno-deprecated-declarations -L/usr/local/cuda/lib64
LIBS=-lcusparse -lcublas
all: test_common

run:
	./matrix_perf $(ARGS)

clean:
	rm -f matrix_perf output*.txt 

matrix_perf: matrix_perf.cu matrix_common.cu
	$(CXX) matrix_perf.cu matrix_common.cu -O2 -o matrix_perf ${INCLUDES} ${LIBS}

test_common: test_common.cu matrix_common.cu
	$(CXX) -g test_common.cu matrix_common.cu -o test_common ${INCLUDES} ${LIBS}