#
CXX = nvcc
DEBUG = # -g
INCLUDES=-I/usr/local/cuda/include --Wno-deprecated-declarations -L/usr/local/cuda/lib64
LIBS=-lcusparse -lcublas
all: test_common

run:
	./test_common >output.txt

clean:
	rm -f matrix_perf output*.txt 

matrix_perf: matrix_perf.cu matrix_common.cu
	$(CXX) ${DEBUG} matrix_perf.cu matrix_common.cu -O2 -o matrix_perf ${INCLUDES} ${LIBS}

test_common: test_common.cu matrix_common.cu
	$(CXX) ${DEBUG} test_common.cu matrix_common.cu -o test_common ${INCLUDES} ${LIBS}